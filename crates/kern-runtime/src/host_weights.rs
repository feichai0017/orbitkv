//! Immutable mapped host weights shared by runtimes in one explicit scope.
//!
//! A scope represents one checkpoint snapshot, not a process-global name cache.
//! Its allocations exist before program compilation, so their GPU aliases never
//! need rebinding when the checkpoint is populated. Only the first successful
//! initializer writes bytes; later ranks share those bytes without another copy.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use cudarc::driver::{sys, CudaContext};
use kern_manifest::types::Buffer;

use crate::error::{bail, cuda_check};
use crate::{Error, Result};

const HUGE_PAGE: usize = 512 << 20;
// smaps_rollup counts the whole process. Serialize fault/verification and
// release so another host-weight allocation cannot mask a small-page fallback.
static HOST_MAPPING_LOCK: Mutex<()> = Mutex::new(());

struct Mapping {
    base: *mut c_void,
    len: usize,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base, self.len) };
    }
}

fn mapping_lengths(bytes: usize) -> Result<(usize, usize)> {
    let size = bytes.max(1).checked_add(HUGE_PAGE - 1).map(|n| n & !(HUGE_PAGE - 1));
    size.and_then(|size| size.checked_add(HUGE_PAGE).map(|reserved| (size, reserved)))
        .ok_or_else(|| Error::Api("host weight huge-page mapping exceeds address space".into()))
}

fn parse_anon_huge_pages(report: &str) -> Result<u64> {
    report
        .lines()
        .find_map(|line| line.strip_prefix("AnonHugePages:"))
        .and_then(|value| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kb| kb.checked_mul(1024))
        .ok_or_else(|| Error::Api("missing or invalid AnonHugePages in smaps_rollup".into()))
}

fn anon_huge_pages(name: &str) -> Result<u64> {
    let report = std::fs::read_to_string("/proc/self/smaps_rollup")
        .map_err(|e| Error::Api(format!("host weight `{name}`: cannot read smaps_rollup: {e}")))?;
    parse_anon_huge_pages(&report).map_err(|e| Error::Api(format!("host weight `{name}`: {e}")))
}

fn require_huge_pages(name: &str, expected: u64, before: u64, after: u64) -> Result<u64> {
    let got = after.saturating_sub(before);
    if got < expected {
        bail!(Api, "host weight `{name}`: insufficient AnonHugePages: expected {expected} bytes, got {got} bytes; 512 MiB THP is required");
    }
    Ok(got)
}

fn touch_huge_pages(ptr: usize, size: usize) {
    let pages = size / HUGE_PAGE;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16).min(pages.max(1));
    std::thread::scope(|s| {
        for t in 0..threads {
            s.spawn(move || {
                for p in (t..pages).step_by(threads) {
                    unsafe { std::ptr::write_volatile((ptr + p * HUGE_PAGE) as *mut u8, 0) };
                }
            });
        }
    });
}

/// A checkpoint-scoped owner of immutable, portable mapped host allocations.
///
/// Share one `Arc<HostWeights>` across the runtimes that load the same checkpoint
/// snapshot. Use a new scope when loading a different snapshot, even when tensor
/// names and shapes are unchanged. This is an ownership boundary, not a cache.
#[derive(Default)]
pub struct HostWeights {
    entries: Mutex<BTreeMap<String, Arc<HostWeight>>>,
}

impl HostWeights {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate once by name, refusing differently shaped/bound aliases.
    pub(crate) fn acquire(
        &self,
        name: &str,
        buffer: &Buffer,
        bytes: u64,
        ctx: &Arc<CudaContext>,
    ) -> Result<Arc<HostWeight>> {
        let contract = serde_json::to_string(buffer).map_err(|e| Error::Manifest(e.to_string()))?;
        let mut entries = self.entries.lock().map_err(|_| Error::Api("host-weight scope lock poisoned".into()))?;
        if let Some(weight) = entries.get(name) {
            if weight.bytes != bytes || weight.contract != contract {
                bail!(WeightArtifact, "host weight `{name}` differs from the binding already in this checkpoint scope");
            }
            return Ok(weight.clone());
        }
        let weight = Arc::new(HostWeight::allocate(name, bytes, contract, ctx)?);
        entries.insert(name.to_owned(), weight.clone());
        Ok(weight)
    }

    /// Logical host-weight bytes, counted once across ranks; excludes THP padding.
    pub fn allocated_bytes(&self) -> Result<u64> {
        let entries = self.entries.lock().map_err(|_| Error::Api("host-weight scope lock poisoned".into()))?;
        entries.values().try_fold(0u64, |n, w| {
            n.checked_add(w.bytes).ok_or_else(|| Error::Api("host-weight byte count overflow".into()))
        })
    }
}

/// A single allocation. Mutation is confined to the initialization lock; all
/// GPU users retain an Arc through DeviceBuf, and Runtime drops synchronize its
/// streams before releasing those buffers. Keeping the allocator context alive
/// also allows the last reference to be dropped from another thread/context.
pub(crate) struct HostWeight {
    ptr: *mut c_void,
    mapping_base: *mut c_void,
    mapping_len: usize,
    pub(crate) bytes: u64,
    contract: String,
    ready: Mutex<bool>,
    ctx: Arc<CudaContext>,
}

// Host pointer is portable across CUDA contexts. The only CPU mutation is
// serialized by ready, and the allocation cannot be freed while a runtime owns
// its Arc. Kernel interfaces for this backing must be immutable weight inputs.
unsafe impl Send for HostWeight {}
unsafe impl Sync for HostWeight {}

impl HostWeight {
    fn allocate(name: &str, bytes: u64, contract: String, ctx: &Arc<CudaContext>) -> Result<Self> {
        ctx.bind_to_thread()?;
        let size = usize::try_from(bytes.max(1)).map_err(|_| Error::Api("host weight exceeds address space".into()))?;
        let (size, reserved) = mapping_lengths(size)?;
        let _guard = HOST_MAPPING_LOCK.lock().map_err(|_| Error::Api("host-weight mapping lock poisoned".into()))?;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserved,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            bail!(Api, "host weight `{name}`: mmap: {}", std::io::Error::last_os_error());
        }
        let mapping = Mapping { base, len: reserved };
        let offset = (HUGE_PAGE - (base as usize & (HUGE_PAGE - 1))) & (HUGE_PAGE - 1);
        let ptr = unsafe { base.cast::<u8>().add(offset).cast::<c_void>() };
        if unsafe { libc::madvise(ptr, size, libc::MADV_HUGEPAGE) } != 0 {
            bail!(Api, "host weight `{name}`: madvise(MADV_HUGEPAGE): {}", std::io::Error::last_os_error());
        }
        // The GPUs read this mapping through ATS (registered, not pinned),
        // so NUMA balancing sees their reads as hinting faults serviced from
        // a kworker, and migrates the 512 MiB page to that worker's node,
        // then again to the next one's. A reading kernel stalls for the
        // whole copy each time, and a chain of them outlasts the 60 s of a
        // collective's timeout. A policy of the VMA's own carries no
        // migrate-on-fault flag, so the balancer skips the range;
        // allocation stays first-touch local.
        let bound = unsafe {
            libc::syscall(libc::SYS_mbind, ptr, size, libc::MPOL_LOCAL, std::ptr::null::<u64>(), 0usize, 0u32)
        };
        if bound != 0 {
            bail!(Api, "host weight `{name}`: mbind(MPOL_LOCAL): {}", std::io::Error::last_os_error());
        }
        let before = anon_huge_pages(name)?;
        // First touch must follow madvise. Fault complete aligned huge pages,
        // including the final padded page; expose only the logical bytes below.
        // The kernel zeroes a huge page as it faults it, so one byte per page
        // is the whole touch, and the faults spread over cores.
        touch_huge_pages(ptr as usize, size);
        let huge_bytes = require_huge_pages(name, size as u64, before, anon_huge_pages(name)?)?;
        cuda_check(
            unsafe {
                sys::cuMemHostRegister_v2(
                    ptr,
                    size,
                    sys::CU_MEMHOSTREGISTER_PORTABLE | sys::CU_MEMHOSTREGISTER_DEVICEMAP,
                )
            },
            &format!("cuMemHostRegister(host weight `{name}`)"),
        )?;
        tracing::info!(
            buffer = name,
            weight_bytes = bytes,
            mapped_bytes = size,
            anon_huge_bytes = huge_bytes,
            "registered huge-page host weight"
        );
        let weight = Self {
            ptr,
            mapping_base: base,
            mapping_len: reserved,
            bytes,
            contract,
            ready: Mutex::new(false),
            ctx: ctx.clone(),
        };
        std::mem::forget(mapping);
        Ok(weight)
    }

    /// Stable mapped address in the calling context (the caller binds it).
    pub(crate) fn device_pointer(&self) -> Result<u64> {
        let mut device = 0;
        cuda_check(
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, self.ptr, 0) },
            "cuMemHostGetDevicePointer(host weight)",
        )?;
        Ok(device)
    }

    /// Populate the snapshot once. The callback must cover the complete buffer;
    /// the existing weight-plan verifier guarantees that for bound tensors.
    /// A failed initializer leaves the allocation retryable and unready.
    pub(crate) fn initialize(&self, fill: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<bool> {
        let mut ready = self.ready.lock().map_err(|_| Error::Api("host-weight initialization lock poisoned".into()))?;
        if *ready {
            return Ok(false);
        }
        let bytes = unsafe { std::slice::from_raw_parts_mut(self.ptr.cast::<u8>(), self.bytes as usize) };
        fill(bytes)?;
        *ready = true;
        Ok(true)
    }
}

impl Drop for HostWeight {
    fn drop(&mut self) {
        let _guard = HOST_MAPPING_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.ctx.bind_to_thread();
        unsafe {
            sys::cuMemHostUnregister(self.ptr);
            libc::munmap(self.mapping_base, self.mapping_len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn huge_page_padding_covers_small_and_partial_final_pages() {
        assert_eq!(mapping_lengths(0).unwrap(), (HUGE_PAGE, HUGE_PAGE * 2));
        assert_eq!(mapping_lengths(64).unwrap(), (HUGE_PAGE, HUGE_PAGE * 2));
        assert_eq!(mapping_lengths(HUGE_PAGE).unwrap(), (HUGE_PAGE, HUGE_PAGE * 2));
        assert_eq!(mapping_lengths(HUGE_PAGE + 1).unwrap(), (HUGE_PAGE * 2, HUGE_PAGE * 3));
        assert!(mapping_lengths(usize::MAX).is_err());
    }

    #[test]
    fn huge_page_verification_rejects_fallback_even_with_existing_huge_weights() {
        let before = parse_anon_huge_pages("Rss: 900000 kB\nAnonHugePages:    524288 kB\n").unwrap();
        assert_eq!(before, HUGE_PAGE as u64);
        assert!(parse_anon_huge_pages("AnonHugePages: invalid kB\n").is_err());
        assert!(parse_anon_huge_pages("Rss: 100 kB\n").is_err());
        let error = require_huge_pages("table", HUGE_PAGE as u64, before, before).unwrap_err().to_string();
        assert!(error.contains("table") && error.contains("expected 536870912 bytes, got 0 bytes"));
        assert_eq!(require_huge_pages("table", HUGE_PAGE as u64, before, before * 2).unwrap(), before);
    }

    fn descriptor() -> Buffer {
        serde_json::from_value(serde_json::json!({
            "dtype": "u8", "shape": [64], "kind": "weight", "bind": [{"tensor": "table"}]
        }))
        .unwrap()
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn immutable_host_scope_initializes_once_and_outlives_registry() {
        let ctx = CudaContext::new(0).unwrap();
        let scope = Arc::new(HostWeights::new());
        let weight = scope.acquire("table", &descriptor(), 64, &ctx).unwrap();
        assert!(Arc::ptr_eq(&weight, &scope.acquire("table", &descriptor(), 64, &ctx).unwrap()));
        let count = AtomicUsize::new(0);
        std::thread::scope(|threads| {
            for _ in 0..4 {
                threads.spawn(|| {
                    weight
                        .initialize(|bytes| {
                            count.fetch_add(1, Ordering::SeqCst);
                            for (i, byte) in bytes.iter_mut().enumerate() {
                                *byte = i as u8;
                            }
                            Ok(())
                        })
                        .unwrap();
                });
            }
        });
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(scope.allocated_bytes().unwrap(), 64);
        let mut different = descriptor();
        different.shape = vec![kern_manifest::types::Dim::Const(32)];
        assert!(scope.acquire("table", &different, 32, &ctx).is_err());
        drop(scope);
        ctx.bind_to_thread().unwrap();
        let mut output = [0u8; 64];
        cuda_check(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), weight.device_pointer().unwrap(), output.len()) },
            "host weight test copy",
        )
        .unwrap();
        assert_eq!(output, std::array::from_fn(|i| i as u8));
    }

    #[test]
    #[ignore = "requires four CUDA GPUs"]
    fn portable_host_weight_is_shared_by_four_contexts() {
        let scope = HostWeights::new();
        let first = CudaContext::new(0).unwrap();
        let weight = scope.acquire("table", &descriptor(), 64, &first).unwrap();
        weight
            .initialize(|dst| {
                dst.fill(73);
                Ok(())
            })
            .unwrap();
        let buffers: Vec<_> = (0..4)
            .map(|gpu| {
                let ctx = CudaContext::new(gpu).unwrap();
                let same = scope.acquire("table", &descriptor(), 64, &ctx).unwrap();
                assert!(Arc::ptr_eq(&weight, &same));
                let buffer = crate::device::alloc_host(&ctx.default_stream(), same).unwrap();
                (ctx, buffer)
            })
            .collect();
        assert_eq!(scope.allocated_bytes().unwrap(), 64);
        drop(weight);
        drop(scope);
        drop(first);
        for (ctx, buffer) in buffers {
            ctx.bind_to_thread().unwrap();
            let mut output = [0u8; 64];
            cuda_check(
                unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), buffer.ptr, output.len()) },
                "portable host weight read",
            )
            .unwrap();
            assert_eq!(output, [73; 64]);
        }
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn failed_host_initializer_can_retry() {
        let ctx = CudaContext::new(0).unwrap();
        let scope = HostWeights::new();
        let weight = scope.acquire("table", &descriptor(), 64, &ctx).unwrap();
        assert!(weight.initialize(|_| Err(Error::WeightArtifact("incomplete input".into()))).is_err());
        weight
            .initialize(|dst| {
                dst.fill(19);
                Ok(())
            })
            .unwrap();
        let mut output = [0u8; 64];
        cuda_check(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), weight.device_pointer().unwrap(), output.len()) },
            "host weight retry copy",
        )
        .unwrap();
        assert_eq!(output, [19; 64]);
    }
}
