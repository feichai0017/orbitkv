use std::{ptr, time::Instant};

use cudarc::driver::CudaContext;
use cudarc::runtime::sys as rt;

const GB: usize = 1024 * 1024 * 1024;
const MB_2: usize = 2 * 1024 * 1024;

// RTX 5070ti epyc 7402
// === mmap(MAP_POPULATE) + cudaHostRegister ===
//   mmap:        16360.05ms
//   register:     3049.01ms
//   TOTAL:       19409.06ms

// === mmap(MAP_HUGETLB 2MB) + cudaHostRegister ===
//   mmap:            0.06ms
//   register:     5730.77ms
//   TOTAL:        5730.83ms
fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    ctx.bind_to_thread().expect("bind CUDA context");

    let size = 30 * GB;

    println!("Benchmarking 30GB pinned memory allocation...\n");

    // === mmap + MAP_POPULATE + cudaHostRegister ===
    println!("=== mmap(MAP_POPULATE) + cudaHostRegister ===");
    {
        let start = Instant::now();
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
                -1,
                0,
            )
        };
        let mmap_elapsed = start.elapsed();

        if ptr == libc::MAP_FAILED {
            println!("  mmap FAILED");
        } else {
            println!(
                "  mmap:        {:8.2}ms",
                mmap_elapsed.as_secs_f64() * 1000.0
            );

            let reg_start = Instant::now();
            let result = unsafe { rt::cudaHostRegister(ptr, size, rt::cudaHostRegisterDefault) };
            let reg_elapsed = reg_start.elapsed();

            if result == rt::cudaError::cudaSuccess {
                println!(
                    "  register:    {:8.2}ms",
                    reg_elapsed.as_secs_f64() * 1000.0
                );
                println!(
                    "  TOTAL:       {:8.2}ms",
                    (mmap_elapsed + reg_elapsed).as_secs_f64() * 1000.0
                );
                unsafe { rt::cudaHostUnregister(ptr) };
            } else {
                println!("  cudaHostRegister FAILED: {:?}", result);
            }

            unsafe { libc::munmap(ptr, size) };
        }
    }

    // === 2MB Huge Pages + cudaHostRegister ===
    println!("\n=== mmap(MAP_HUGETLB 2MB) + cudaHostRegister ===");
    {
        let aligned_size = (size + MB_2 - 1) & !(MB_2 - 1);

        let start = Instant::now();
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                -1,
                0,
            )
        };
        let mmap_elapsed = start.elapsed();

        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            println!("  mmap FAILED: {}", err);
            println!("  Run: sudo sh -c 'echo 15360 > /proc/sys/vm/nr_hugepages'");
        } else {
            println!(
                "  mmap:        {:8.2}ms",
                mmap_elapsed.as_secs_f64() * 1000.0
            );

            let reg_start = Instant::now();
            let result =
                unsafe { rt::cudaHostRegister(ptr, aligned_size, rt::cudaHostRegisterDefault) };
            let reg_elapsed = reg_start.elapsed();

            if result == rt::cudaError::cudaSuccess {
                println!(
                    "  register:    {:8.2}ms",
                    reg_elapsed.as_secs_f64() * 1000.0
                );
                println!(
                    "  TOTAL:       {:8.2}ms",
                    (mmap_elapsed + reg_elapsed).as_secs_f64() * 1000.0
                );
                unsafe { rt::cudaHostUnregister(ptr) };
            } else {
                println!("  cudaHostRegister FAILED: {:?}", result);
            }

            unsafe { libc::munmap(ptr, aligned_size) };
        }
    }
}
