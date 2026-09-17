//! What the runtime's allocations cost, by phase, so the pool's chunk size
//! and the host weights' first touch are chosen on numbers: `cuMemCreate`
//! of chunks (one thread and several), `cuMemMap`, `cuMemSetAccess` per
//! chunk versus once over the range, whether one physical allocation can be
//! mapped piecewise, the same pool built on several GPUs at once, and
//! first-touching huge-page host memory (memset versus one byte per page).
//!
//! `alloc_bench [gpu] [chunks] [host GiB] [chunk MiB]`
//! `alloc_bench pool <chunks> <chunk MiB> <gpu>...`
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys;
use cudarc::driver::CudaContext;

const HUGE: usize = 512 << 20;

fn check(r: sys::CUresult, what: &str) {
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "{what}: {r:?}");
}

fn prop(dev: i32) -> sys::CUmemAllocationProp {
    let mut p: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    p.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    p.requestedHandleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    p.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = dev;
    p
}

fn access(dev: i32) -> sys::CUmemAccessDesc {
    let mut a: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    a.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    a.location.id = dev;
    a.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    a
}

fn create(dev: i32, chunk: usize, n: usize) -> Vec<sys::CUmemGenericAllocationHandle> {
    let p = prop(dev);
    (0..n)
        .map(|_| {
            let mut h = 0;
            check(unsafe { sys::cuMemCreate(&mut h, chunk, &p, 0) }, "cuMemCreate");
            h
        })
        .collect()
}

fn create_threaded(
    ctx: &Arc<CudaContext>,
    dev: i32,
    chunk: usize,
    n: usize,
    threads: usize,
) -> Vec<sys::CUmemGenericAllocationHandle> {
    std::thread::scope(|s| {
        let parts: Vec<_> = (0..threads)
            .map(|t| {
                let ctx = Arc::clone(ctx);
                let share = n / threads + usize::from(t < n % threads);
                s.spawn(move || {
                    ctx.bind_to_thread().unwrap();
                    create(dev, chunk, share)
                })
            })
            .collect();
        parts.into_iter().flat_map(|h| h.join().unwrap()).collect()
    })
}

fn release(hs: &[sys::CUmemGenericAllocationHandle]) {
    for &h in hs {
        unsafe { sys::cuMemRelease(h) };
    }
}

fn map_all(va: u64, chunk: usize, hs: &[sys::CUmemGenericAllocationHandle]) {
    for (i, &h) in hs.iter().enumerate() {
        check(unsafe { sys::cuMemMap(va + (i * chunk) as u64, chunk, 0, h, 0) }, "cuMemMap");
    }
}

fn unmap_all(va: u64, chunk: usize, n: usize) {
    for i in 0..n {
        check(unsafe { sys::cuMemUnmap(va + (i * chunk) as u64, chunk) }, "cuMemUnmap");
    }
}

fn granularities(dev: i32) {
    let p = prop(dev);
    for (name, opt) in [
        ("minimum", sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        ("recommended", sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
    ] {
        let mut g = 0usize;
        check(unsafe { sys::cuMemGetAllocationGranularity(&mut g, &p, opt) }, "granularity");
        println!("granularity {name}: {} MiB", g >> 20);
    }
}

/// Whether one physical allocation can be mapped piecewise: `cuMemMap`
/// of its second half at a non-zero offset, and of its first half alone.
fn suballocate(dev: i32) {
    let (mut h, mut va) = (0, 0);
    check(unsafe { sys::cuMemCreate(&mut h, 4 << 20, &prop(dev), 0) }, "cuMemCreate");
    check(unsafe { sys::cuMemAddressReserve(&mut va, 4 << 20, 0, 0, 0) }, "reserve");
    let r = unsafe { sys::cuMemMap(va, 2 << 20, 2 << 20, h, 0) };
    println!("cuMemMap 2 MiB at offset 2 MiB of a 4 MiB handle: {r:?}");
    let r = unsafe { sys::cuMemMap(va, 2 << 20, 0, h, 0) };
    println!("cuMemMap 2 MiB at offset 0 of a 4 MiB handle (partial size): {r:?}");
    unsafe {
        sys::cuMemUnmap(va, 2 << 20);
        sys::cuMemAddressFree(va, 4 << 20);
        sys::cuMemRelease(h);
    }
}

fn device(ctx: &Arc<CudaContext>, dev: i32, chunk: usize, n: usize) {
    for threads in [1, 4, 8, 16] {
        let t = Instant::now();
        let hs = create_threaded(ctx, dev, chunk, n, threads);
        let create_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        release(&hs);
        println!(
            "cuMemCreate {n} x {} MiB, {threads} threads: {create_s:.2}s ({:.0}/s); release {:.2}s",
            chunk >> 20,
            n as f64 / create_s,
            t.elapsed().as_secs_f64()
        );
    }
    let hs = create(dev, chunk, n);
    let mut va = 0;
    check(unsafe { sys::cuMemAddressReserve(&mut va, n * chunk, 0, 0, 0) }, "reserve");
    let t = Instant::now();
    map_all(va, chunk, &hs);
    println!("cuMemMap {n}: {:.2}s", t.elapsed().as_secs_f64());
    let a = access(dev);
    let t = Instant::now();
    for i in 0..n {
        check(unsafe { sys::cuMemSetAccess(va + (i * chunk) as u64, chunk, &a, 1) }, "cuMemSetAccess");
    }
    println!("cuMemSetAccess per chunk {n}: {:.2}s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    unmap_all(va, chunk, n);
    println!("cuMemUnmap {n}: {:.2}s", t.elapsed().as_secs_f64());
    map_all(va, chunk, &hs);
    let t = Instant::now();
    check(unsafe { sys::cuMemSetAccess(va, n * chunk, &a, 1) }, "cuMemSetAccess");
    println!("cuMemSetAccess once over {n}: {:.2}s", t.elapsed().as_secs_f64());
    unmap_all(va, chunk, n);
    unsafe { sys::cuMemAddressFree(va, n * chunk) };
    release(&hs);
}

/// Create, map and grant `n` chunks: a rank's state pool.
fn pool(dev: i32, chunk: usize, n: usize) -> (f64, f64) {
    let t = Instant::now();
    let hs = create(dev, chunk, n);
    let created = t.elapsed().as_secs_f64();
    let mut va = 0;
    check(unsafe { sys::cuMemAddressReserve(&mut va, n * chunk, 0, 0, 0) }, "reserve");
    let t = Instant::now();
    map_all(va, chunk, &hs);
    check(unsafe { sys::cuMemSetAccess(va, n * chunk, &access(dev), 1) }, "cuMemSetAccess");
    (created, t.elapsed().as_secs_f64())
}

/// The pool on every GPU at once, one thread per GPU in this process:
/// what a tray's ranks do together.
fn pools(chunk: usize, n: usize, gpus: &[usize]) {
    let t = Instant::now();
    std::thread::scope(|s| {
        for &g in gpus {
            s.spawn(move || {
                let ctx = CudaContext::new(g).unwrap();
                ctx.bind_to_thread().unwrap();
                let (c, m) = pool(g as i32, chunk, n);
                println!(
                    "gpu {g}: {n} chunks of {} MiB created in {c:.2}s, mapped and granted in {m:.2}s",
                    chunk >> 20
                );
            });
        }
    });
    println!("{} GPUs in one process: {:.2}s", gpus.len(), t.elapsed().as_secs_f64());
}

fn huge_mmap(size: usize) -> *mut u8 {
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size + HUGE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(base, libc::MAP_FAILED);
    let ptr = ((base as usize + HUGE - 1) & !(HUGE - 1)) as *mut u8;
    assert_eq!(unsafe { libc::madvise(ptr.cast(), size, libc::MADV_HUGEPAGE) }, 0);
    assert_eq!(
        unsafe { libc::syscall(libc::SYS_mbind, ptr, size, libc::MPOL_LOCAL, std::ptr::null::<u64>(), 0usize, 0u32) },
        0
    );
    ptr
}

fn anon_huge() -> u64 {
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("AnonHugePages:"))
        .map(|v| v.trim().trim_end_matches(" kB").parse::<u64>().unwrap() << 10)
        .unwrap()
}

fn host(size: usize) {
    let pages = size / HUGE;
    let ptr = huge_mmap(size);
    let (before, t) = (anon_huge(), Instant::now());
    unsafe { std::ptr::write_bytes(ptr, 0, size) };
    println!(
        "host memset {} GiB: {:.2}s, huge {} GiB",
        size >> 30,
        t.elapsed().as_secs_f64(),
        (anon_huge() - before) >> 30
    );
    unsafe { libc::munmap(ptr.cast(), size) };
    for threads in [1, 4, 8, 16, 32] {
        let ptr = huge_mmap(size) as usize;
        let (before, t) = (anon_huge(), Instant::now());
        std::thread::scope(|s| {
            for t in 0..threads {
                s.spawn(move || {
                    for p in (t..pages).step_by(threads) {
                        unsafe { std::ptr::write_volatile((ptr + p * HUGE) as *mut u8, 0) };
                    }
                });
            }
        });
        println!(
            "host touch one byte per page {} GiB, {threads} threads: {:.2}s, huge {} GiB",
            size >> 30,
            t.elapsed().as_secs_f64(),
            (anon_huge() - before) >> 30
        );
        unsafe { libc::munmap(ptr as *mut _, size) };
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: usize| args.get(i).map_or(default, |a| a.parse().unwrap());
    if args.get(1).map(String::as_str) == Some("pool") {
        let gpus: Vec<usize> = args[4..].iter().map(|a| a.parse().unwrap()).collect();
        return pools(arg(3, 2) << 20, arg(2, 10000), &gpus);
    }
    let (gpu, chunks, host_gib, chunk) = (arg(1, 0), arg(2, 10000), arg(3, 16), arg(4, 2) << 20);
    let ctx = CudaContext::new(gpu).unwrap();
    ctx.bind_to_thread().unwrap();
    granularities(gpu as i32);
    suballocate(gpu as i32);
    device(&ctx, gpu as i32, chunk, chunks);
    host(host_gib << 30);
}
