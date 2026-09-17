//! Opt-in measurement primitives. Preparation is graph ordered but outside
//! event brackets; every sample restores declared writes to its pre-image.
//! No samples are discarded. Profiling never changes the serving path.
//! Like every runtime entry point, a probe's binds the runtime's context
//! on entry: a rank is measured from whichever thread holds it.
//!
//! This lives inside the runtime, not beside `kern test`, because a sample
//! is a captured range of the compiled launch list bracketed by external
//! events, with pre-images copied on the runtime's own stream: none of
//! that is reachable through the harness surface.
use std::collections::{BTreeMap, BTreeSet};

use cudarc::driver::{sys, CudaFunction, LaunchConfig, PushKernelArg};
use kern_manifest::types::{Arg, BufferKind, Dir};

use crate::compile::{Dense, RVal};
use crate::cublas::gemm_bf16_tn;
use crate::device::Events;
use crate::{alloc, cuda_check, DeviceBuf, Result, Runtime};

impl Events {
    fn captured_record(&self, i: usize, rt: &Runtime) -> Result<()> {
        // Default captured events express dependencies without updating an
        // externally readable timestamp. Explicit event nodes retain timing.
        cuda_check(
            unsafe {
                sys::cuEventRecordWithFlags(
                    self.0[i],
                    rt.stream.cu_stream(),
                    sys::CUevent_record_flags::CU_EVENT_RECORD_EXTERNAL as u32,
                )
            },
            "profile event record",
        )
    }
}

struct Graph(sys::CUgraphExec);
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { sys::cuGraphExecDestroy(self.0) };
    }
}

fn capture(rt: &Runtime, body: impl FnOnce() -> Result<()>) -> Result<Graph> {
    rt.ctx.bind_to_thread()?;
    cuda_check(
        unsafe {
            sys::cuStreamBeginCapture_v2(
                rt.stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        },
        "profile capture",
    )?;
    let result = body();
    let mut graph = std::ptr::null_mut();
    let ended = unsafe { sys::cuStreamEndCapture(rt.stream.cu_stream(), &mut graph) };
    if let Err(e) = result {
        if !graph.is_null() {
            unsafe { sys::cuGraphDestroy(graph) };
        }
        return Err(e);
    }
    cuda_check(ended, "profile end capture")?;
    let mut exec = std::ptr::null_mut();
    let result = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
    unsafe { sys::cuGraphDestroy(graph) };
    cuda_check(result, "profile instantiate")?;
    Ok(Graph(exec))
}

fn replay(rt: &Runtime, graph: &Graph) -> Result<()> {
    cuda_check(unsafe { sys::cuGraphLaunch(graph.0, rt.stream.cu_stream()) }, "profile replay")?;
    rt.stream.synchronize()?;
    Ok(())
}

fn copy(rt: &Runtime, dst: u64, src: u64, bytes: u64) -> Result<()> {
    cuda_check(unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes as usize, rt.stream.cu_stream()) }, "profile copy")
}

/// GPU-resident pre-images. Whole state allocations are deliberately used:
/// the manifest does not declare a kernel's state read/write byte ranges.
struct Snapshot(Vec<(u64, DeviceBuf)>);
impl Snapshot {
    fn new(rt: &Runtime, buffers: BTreeSet<String>, states: BTreeSet<String>) -> Result<Self> {
        let mut saved = Vec::new();
        for (name, state) in buffers.into_iter().map(|n| (n, false)).chain(states.into_iter().map(|n| (n, true))) {
            let src = if state {
                rt.whole_state(&name)?;
                &rt.states[&name]
            } else {
                &rt.buffers[&name]
            };
            let dst = alloc(&rt.stream, src.bytes)?;
            copy(rt, dst.ptr, src.ptr, src.bytes)?;
            saved.push((src.ptr, dst));
        }
        rt.stream.synchronize()?;
        Ok(Self(saved))
    }
    fn restore(&self, rt: &Runtime) -> Result<()> {
        self.0.iter().try_for_each(|(dst, src)| copy(rt, *dst, src.ptr, src.bytes))
    }
}

/// The buffers and states calls `calls` of `program` declare they write:
/// what a sample restores to its pre-image.
fn declared_writes(rt: &Runtime, program: &str, calls: std::ops::Range<usize>) -> (BTreeSet<String>, BTreeSet<String>) {
    let m = &rt.manifest;
    let (mut buffers, mut states) = (BTreeSet::new(), BTreeSet::new());
    for call in &m.programs[program].calls[calls] {
        for (a, p) in call.args.iter().zip(&m.ops[&call.op].params) {
            if matches!(p.dir(), Some(Dir::Out | Dir::InOut)) {
                match a {
                    Arg::Buf { buf, .. } => {
                        buffers.insert(buf.clone());
                    }
                    Arg::State { state, .. } => {
                        states.insert(state.clone());
                    }
                    _ => {}
                }
            }
        }
    }
    (buffers, states)
}

pub struct Probe {
    pub device: String,
    pub l2_bytes: u64,
    pub sm_count: u32,
    pub driver_version: i32,
    pub eviction_bytes: u64,
    a: DeviceBuf,
    b: DeviceBuf,
    read: CudaFunction,
    copy: CudaFunction,
    empty: CudaFunction,
    markers: BTreeMap<&'static str, CudaFunction>,
}

/// All entries are single executions in chronological order, in microseconds.
pub struct CallSamples {
    pub cold_us: Vec<f64>,
    pub warm_us: Vec<f64>,
}

pub struct ProgramSamples {
    pub graph_us: Vec<f64>,
    pub attributed_us: Vec<Vec<f64>>,
    pub instrumented_us: Vec<f64>,
}

pub struct Anchor {
    pub name: &'static str,
    pub bytes: u64,
    pub traffic_bytes: u64,
    pub flops: u64,
    pub samples_us: Vec<f64>,
}

impl Probe {
    pub fn new(rt: &Runtime) -> Result<Self> {
        rt.ctx.bind_to_thread()?;
        let l2_bytes = rt.ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)? as u64;
        let sm_count = rt.ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let eviction_bytes = (l2_bytes * 8).max(256 << 20);
        // Bundled PTX is generated from profile.cu with CUDA 13.0:
        // nvcc --ptx -arch=compute_80 profile.cu -o profile.ptx
        // Measurement must not depend on the host's NVRTC toolchain.
        let ptx = cudarc::nvrtc::Ptx::from_src(include_str!("profile.ptx"));
        let module = rt.ctx.load_module(ptx)?;
        let a = alloc(&rt.stream, eviction_bytes)?;
        let b = alloc(&rt.stream, eviction_bytes)?;
        let seed = module.load_function("seed_data")?;
        for buf in [&a, &b] {
            unsafe {
                rt.stream.launch_builder(&seed).arg(&buf.ptr).arg(&(buf.bytes / 16)).launch(LaunchConfig {
                    grid_dim: (sm_count * 8, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
        }
        rt.stream.synchronize()?;
        let mut driver_version = 0;
        cuda_check(unsafe { sys::cuDriverGetVersion(&mut driver_version) }, "driver version")?;
        Ok(Self {
            device: rt.ctx.name()?,
            l2_bytes,
            sm_count,
            driver_version,
            eviction_bytes,
            a,
            b,
            read: module.load_function("stream_read")?,
            copy: module.load_function("stream_copy")?,
            empty: module.load_function("empty_probe")?,
            markers: [
                "profile_cold_start",
                "profile_warm_start",
                "profile_program_start",
                "profile_anchor_start",
                "profile_end",
            ]
            .into_iter()
            .map(|n| module.load_function(n).map(|f| (n, f)))
            .collect::<std::result::Result<_, _>>()?,
        })
    }

    fn kernel(&self, rt: &Runtime, f: &CudaFunction, bytes: u64) -> Result<()> {
        unsafe {
            rt.stream.launch_builder(f).arg(&self.a.ptr).arg(&self.b.ptr).arg(&(bytes / 16)).launch(LaunchConfig {
                grid_dim: (self.sm_count * 8, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }

    fn evict(&self, rt: &Runtime) -> Result<()> {
        self.kernel(rt, &self.copy, self.eviction_bytes)
    }

    fn mark(&self, rt: &Runtime, name: &str) -> Result<()> {
        unsafe { rt.stream.launch_builder(&self.markers[name]).launch(LaunchConfig::for_num_elems(1)) }?;
        Ok(())
    }

    /// Local-device calibration. Traffic includes both reads and writes;
    /// the copy payload is retained separately to avoid a factor-of-two error.
    pub fn calibrate(&self, rt: &Runtime, samples: usize) -> Result<Vec<Anchor>> {
        rt.ctx.bind_to_thread()?;
        let mut anchors = Vec::new();
        let l2_work = (self.l2_bytes / 4).max(1 << 20);
        for (name, bytes, traffic, kind) in [
            ("d2d_copy", self.eviction_bytes, self.eviction_bytes * 2, 0),
            ("sm_copy", self.eviction_bytes, self.eviction_bytes * 2, 1),
            ("sm_read", self.eviction_bytes, self.eviction_bytes, 2),
            ("l2_read", l2_work, l2_work, 2),
            ("empty_kernel", 0, 0, 3),
        ] {
            let events = Events::new(samples * 2)?;
            let operation = || match kind {
                0 => copy(rt, self.b.ptr, self.a.ptr, bytes),
                1 => self.kernel(rt, &self.copy, bytes),
                2 => self.kernel(rt, &self.read, bytes),
                _ => {
                    unsafe { rt.stream.launch_builder(&self.empty).launch(LaunchConfig::for_num_elems(1)) }?;
                    Ok(())
                }
            };
            let graph = capture(rt, || {
                for i in 0..samples {
                    operation()?;
                    self.mark(rt, "profile_anchor_start")?;
                    events.captured_record(2 * i, rt)?;
                    operation()?;
                    events.captured_record(2 * i + 1, rt)?;
                    self.mark(rt, "profile_end")?;
                }
                Ok(())
            })?;
            for _ in 0..4 {
                replay(rt, &graph)?;
            }
            let samples_us = (0..samples)
                .map(|i| events.elapsed_ms(2 * i, 2 * i + 1).map(|t| t as f64 * 1000.))
                .collect::<Result<_>>()?;
            anchors.push(Anchor { name, bytes, traffic_bytes: traffic, flops: 0, samples_us });
        }
        // Confirm that the working set is affected by eviction; this is an
        // empirical cache-sensitivity check, not a claim to flush every cache.
        let events = Events::new(samples * 2)?;
        let graph = capture(rt, || {
            for i in 0..samples {
                self.evict(rt)?;
                self.mark(rt, "profile_anchor_start")?;
                events.captured_record(i * 2, rt)?;
                self.kernel(rt, &self.read, l2_work)?;
                events.captured_record(i * 2 + 1, rt)?;
                self.mark(rt, "profile_end")?;
            }
            Ok(())
        })?;
        replay(rt, &graph)?;
        anchors.push(Anchor {
            name: "evicted_read",
            bytes: l2_work,
            traffic_bytes: l2_work,
            flops: 0,
            samples_us: (0..samples)
                .map(|i| events.elapsed_ms(i * 2, i * 2 + 1).map(|t| t as f64 * 1000.))
                .collect::<Result<_>>()?,
        });
        let size = 4096u64;
        let bytes = size * size * 2;
        let a = alloc(&rt.stream, bytes)?;
        let b = alloc(&rt.stream, bytes)?;
        let c = alloc(&rt.stream, bytes)?;
        for buffer in [&a, &b] {
            cuda_check(
                unsafe { sys::cuMemsetD16Async(buffer.ptr, 0x3f00, (bytes / 2) as usize, rt.stream.cu_stream()) },
                "GEMM probe fill",
            )?;
        }
        let args = [a.ptr, b.ptr, c.ptr, size, size, size].map(|val| RVal { val, bytes });
        gemm_bf16_tn(&rt.blt, &rt.stream, &args, 0.)?;
        rt.stream.synchronize()?;
        let events = Events::new(samples * 2)?;
        let graph = capture(rt, || {
            for i in 0..samples {
                self.mark(rt, "profile_anchor_start")?;
                events.captured_record(i * 2, rt)?;
                gemm_bf16_tn(&rt.blt, &rt.stream, &args, 0.)?;
                events.captured_record(i * 2 + 1, rt)?;
                self.mark(rt, "profile_end")?;
            }
            Ok(())
        })?;
        for _ in 0..4 {
            replay(rt, &graph)?;
        }
        anchors.push(Anchor {
            name: "bf16_gemm_4096",
            bytes,
            traffic_bytes: 0,
            flops: 2 * size * size * size,
            samples_us: (0..samples)
                .map(|i| events.elapsed_ms(i * 2, i * 2 + 1).map(|t| t as f64 * 1000.))
                .collect::<Result<_>>()?,
        });
        Ok(anchors)
    }

    /// A call is measured where it occurs in the program. The caller then
    /// executes it once to continue the original trajectory.
    pub fn call(
        &self,
        rt: &Runtime,
        program: &str,
        vars: &BTreeMap<String, u64>,
        index: usize,
        samples: usize,
    ) -> Result<CallSamples> {
        rt.ctx.bind_to_thread()?;
        let (buffers, states) = declared_writes(rt, program, index..index + 1);
        let snapshot = Snapshot::new(rt, buffers, states)?;
        let prog = &rt.programs[program];
        let dense = Dense::check(&rt.manifest, vars, &prog.vars)?;
        let (lo, hi) = prog.call_ranges[index];
        let run = || prog.launches[lo..hi].iter().try_for_each(|l| rt.launch(l, &dense));
        // Prime libraries and kernel code before capture, then undo the write.
        for _ in 0..3 {
            snapshot.restore(rt)?;
            run()?;
        }
        snapshot.restore(rt)?;
        rt.stream.synchronize()?;
        let events = Events::new(samples * 4)?;
        let graph = capture(rt, || {
            for i in 0..samples {
                // Alternate the order across samples to expose order effects.
                for cold in if i % 2 == 0 { [true, false] } else { [false, true] } {
                    snapshot.restore(rt)?;
                    if cold {
                        self.evict(rt)?;
                    } else {
                        run()?;
                        snapshot.restore(rt)?;
                    }
                    let e = i * 4 + if cold { 0 } else { 2 };
                    self.mark(rt, if cold { "profile_cold_start" } else { "profile_warm_start" })?;
                    events.captured_record(e, rt)?;
                    run()?;
                    events.captured_record(e + 1, rt)?;
                    self.mark(rt, "profile_end")?;
                }
            }
            snapshot.restore(rt)
        })?;
        let measured = replay(rt, &graph);
        snapshot.restore(rt)?;
        rt.stream.synchronize()?;
        measured?;
        let times = |offset| {
            (0..samples)
                .map(|i| events.elapsed_ms(i * 4 + offset, i * 4 + offset + 1).map(|t| t as f64 * 1000.))
                .collect::<Result<Vec<_>>>()
        };
        Ok(CallSamples { cold_us: times(0)?, warm_us: times(2)? })
    }

    /// Whole graph timings and separately instrumented per-call timings.
    /// Each replay restores the same state/carry; normal inter-call locality
    /// is preserved. Instrumentation inflation is exposed, never hidden.
    pub fn program(
        &self,
        rt: &Runtime,
        program: &str,
        vars: &BTreeMap<String, u64>,
        samples: usize,
    ) -> Result<ProgramSamples> {
        rt.ctx.bind_to_thread()?;
        // The program's pre-image is what outlives a run and it writes: its
        // carries and states. Workspace is written before it is read within
        // a run, so it is left alone; snapshotting every carry and state of
        // a large manifest would not fit beside them. An exported carry is
        // not this rank's to restore either: peers write into it on their
        // own clock, and the barrier epochs inside it must only advance.
        let n_calls = rt.manifest.programs[program].calls.len();
        let (written, states) = declared_writes(rt, program, 0..n_calls);
        let carries = written
            .into_iter()
            .filter(|b| {
                let buffer = &rt.manifest.buffers[b];
                buffer.kind == BufferKind::Carry && !buffer.export
            })
            .collect();
        let snapshot = Snapshot::new(rt, carries, states)?;
        let prog = &rt.programs[program];
        let dense = Dense::check(&rt.manifest, vars, &prog.vars)?;
        let n = prog.call_ranges.len();
        let instrumented = Events::new(n + 1)?;
        let whole = Events::new(2)?;
        let graph = capture(rt, || {
            self.mark(rt, "profile_program_start")?;
            whole.captured_record(0, rt)?;
            rt.replay(prog, &dense)?;
            whole.captured_record(1, rt)?;
            self.mark(rt, "profile_end")
        })?;
        let attributed = capture(rt, || {
            instrumented.captured_record(0, rt)?;
            for (i, &(lo, hi)) in prog.call_ranges.iter().enumerate() {
                for l in &prog.launches[lo..hi] {
                    rt.launch(l, &dense)?;
                }
                instrumented.captured_record(i + 1, rt)?;
            }
            Ok(())
        })?;
        let mut result =
            ProgramSamples { graph_us: Vec::new(), attributed_us: vec![Vec::new(); n], instrumented_us: Vec::new() };
        for i in 0..samples + 4 {
            snapshot.restore(rt)?;
            self.evict(rt)?;
            replay(rt, &graph)?;
            if i >= 4 {
                result.graph_us.push(whole.elapsed_ms(0, 1)? as f64 * 1000.);
            }
            if i >= 4 && (i - 4) % 4 == 0 {
                snapshot.restore(rt)?;
                self.evict(rt)?;
                replay(rt, &attributed)?;
                for j in 0..n {
                    result.attributed_us[j].push(instrumented.elapsed_ms(j, j + 1)? as f64 * 1000.);
                }
                result.instrumented_us.push(instrumented.elapsed_ms(0, n)? as f64 * 1000.);
            }
        }
        snapshot.restore(rt)?;
        rt.stream.synchronize()?;
        Ok(result)
    }
}
