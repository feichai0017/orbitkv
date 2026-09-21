use super::*;
use crate::transfer::MemcpyBackend;
use cudarc::driver::sys;

#[derive(Clone, Copy)]
struct MappedHost {
    host: *mut u8,
    device: u64,
}

/// Allocate `len` bytes of mapped pinned host memory.
fn alloc_mapped_host(len: usize) -> MappedHost {
    let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
    let r = unsafe { sys::cuMemHostAlloc(&mut p, len, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostAlloc");

    let mut device: sys::CUdeviceptr = 0;
    let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, p, 0) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer");

    MappedHost {
        host: p as *mut u8,
        device,
    }
}

fn alloc_device(len: usize) -> u64 {
    let mut d: sys::CUdeviceptr = 0;
    let r = unsafe { sys::cuMemAlloc_v2(&mut d, len) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemAlloc");
    d
}

/// Build `n` descriptors of `seg` bytes each over contiguous device/host
/// regions. `host_base` is reinterpreted per direction by the backend.
fn descs(device_base: u64, host_base: MappedHost, n: usize, seg: usize) -> Vec<CopyDesc> {
    (0..n)
        .map(|k| CopyDesc {
            device: device_base + (k * seg) as u64,
            // SAFETY: within the [host_base, host_base+n*seg) allocation.
            host: unsafe { host_base.host.add(k * seg) },
            host_device: host_base.device + (k * seg) as u64,
            size: seg,
            device_allocation: 0,
            host_allocation: 0,
        })
        .collect()
}

/// The kernel backend must move bytes identically to the direct backend, in
/// both directions, over mapped pinned host memory.
#[test]
#[ignore = "requires a CUDA GPU"]
fn kernel_matches_direct_both_directions() {
    const N: usize = 257; // odd, > one grid of blocks would merge nothing
    const SEG: usize = 4096 + 16; // non-power-of-two, 16B-aligned
    let total = N * SEG;

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernel = KernelBackend::new(&ctx).expect("kernel backend");
    let memcpy = MemcpyBackend;

    let host = alloc_mapped_host(total);
    let device = alloc_device(total);

    // Distinct pattern per byte so a misrouted copy is caught.
    let pattern: Vec<u8> = (0..total).map(|i| (i * 31 + 7) as u8).collect();
    let host_slice = unsafe { std::slice::from_raw_parts_mut(host.host, total) };

    let read_device = |out: &mut [u8]| {
        let r = unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, device, total) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "DtoH");
    };
    let clear_device = || {
        let zeros = vec![0u8; total];
        let r = unsafe { sys::cuMemcpyHtoD_v2(device, zeros.as_ptr() as *const _, total) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "HtoD clear");
    };

    let d = descs(device, host, N, SEG);

    // H2D via each backend, then verify device holds the pattern.
    for backend in [&kernel as &dyn TransferBackend, &memcpy] {
        host_slice.copy_from_slice(&pattern);
        clear_device();
        backend.h2d(&d, &stream).expect("h2d");
        stream.synchronize().expect("sync");

        let mut out = vec![0u8; total];
        read_device(&mut out);
        assert_eq!(out, pattern, "h2d mismatch for backend {}", backend.name());
    }

    // D2H via each backend: device holds the pattern, host is cleared first.
    for backend in [&kernel as &dyn TransferBackend, &memcpy] {
        let r = unsafe { sys::cuMemcpyHtoD_v2(device, pattern.as_ptr() as *const _, total) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "HtoD seed");
        host_slice.fill(0);
        backend.d2h(&d, &stream).expect("d2h");
        stream.synchronize().expect("sync");
        assert_eq!(
            host_slice,
            &pattern[..],
            "d2h mismatch for backend {}",
            backend.name()
        );
    }

    unsafe {
        sys::cuMemFree_v2(device);
        sys::cuMemFreeHost(host.host as *mut _);
    }
}
