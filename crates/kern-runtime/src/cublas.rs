//! The extern ops the runtime provides itself, all cuBLAS: the kernels a
//! manifest names as `extern:` instead of shipping. `Blas` is the handle
//! and workspace the f32-result GEMM needs beside the cublasLt one.

use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublas;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use half::bf16;

use crate::compile::RVal;
use crate::error::{bail, Error, Result};

/// Raw device pointer presented as a `DevicePtr<bf16>`/`DevicePtrMut<bf16>`
/// for the cublasLt extern op.
struct RawBf16 {
    ptr: sys::CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
}

impl DeviceSlice<bf16> for RawBf16 {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl DevicePtr<bf16> for RawBf16 {
    fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl DevicePtrMut<bf16> for RawBf16 {
    fn device_ptr_mut<'a>(&'a mut self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

/// `extern:cublaslt_bf16_tn`: row-major `C[m,n] = A[m,k] @ W[n,k]^T`,
/// resolved args `[a, w, c, m, n, k]`. Column-major mapping: compute
/// `C_cm[n,m] = W_cm^T[n,k] x A_cm[k,m]` -> transa=T on W (lda=k),
/// transb=N on A (ldb=k), m'=n, n'=m, ldc=n.
/// `extern:cublaslt_bf16_tn_acc` is the same with beta=1: `C += A @ W^T`.
pub(crate) fn gemm_bf16_tn(blt: &CudaBlasLT, stream: &Arc<CudaStream>, args: &[RVal], beta: f32) -> Result<()> {
    // `c[m, n] (+)= a[m, k] @ w[n, k]^T`; an optional 7th arg is C's row
    // stride in elements (default n), and an optional 8th arg is A's row
    // stride (default k). This permits group views without transposing A.
    let (a, w, c, m, n, k, ldc, a_stride) = match args {
        [a, w, c, m, n, k] => (a, w, c, m.val, n.val, k.val, n.val, k.val),
        [a, w, c, m, n, k, ldc] => (a, w, c, m.val, n.val, k.val, ldc.val, k.val),
        [a, w, c, m, n, k, ldc, a_stride] => (a, w, c, m.val, n.val, k.val, ldc.val, a_stride.val),
        _ => bail!(Manifest, "gemm expects 6, 7 or 8 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "gemm: ldc {ldc} < n {n}");
    }
    if a_stride < k {
        bail!(Manifest, "gemm: A row stride {a_stride} < k {k}");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if a.bytes < ((m - 1) * a_stride + k) * 2 || w.bytes < n * k * 2 || c.bytes < ((m - 1) * ldc + n) * 2 {
        bail!(Manifest, "gemm: operands too small for m={m} n={n} k={k} ldc={ldc} A row stride={a_stride}");
    }
    let view = |rv: &RVal| RawBf16 { ptr: rv.val, len: (rv.bytes / 2) as usize, stream: stream.clone() };
    let cfg = MatmulConfig {
        transa: true,
        transb: false,
        transc: false,
        m: n,
        n: m,
        k,
        alpha: 1.0,
        beta,
        lda: k as i64,
        ldb: a_stride as i64,
        ldc: ldc as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };
    let mut out = view(c);
    unsafe {
        blt.matmul(cfg, &view(w), &view(a), &mut out, None, None)
            .map_err(|e| Error::Cuda(format!("cublasLt matmul (m={m} n={n} k={k}): {e:?}")))?;
    }
    Ok(())
}

/// A cuBLAS handle bound to the runtime's stream with its own workspace, for
/// the f32-result GEMM built-in (`cublasGemmEx`; cublasLt's typed `Matmul`
/// only lands in the operand type). Kept separate from the Lt handle so the
/// two never share a workspace.
pub(crate) struct Blas {
    handle: cublas::sys::cublasHandle_t,
    _workspace: CudaSlice<u8>,
}

// The handle is only ever used from the runtime's own thread, on its stream.
unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Blas {
    const WORKSPACE: usize = 32 << 20;

    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Blas> {
        let handle = cublas::result::create_handle().map_err(|e| Error::Cuda(format!("cublasCreate: {e:?}")))?;
        let workspace: CudaSlice<u8> = stream.alloc_zeros(Self::WORKSPACE)?;
        unsafe {
            cublas::result::set_stream(handle, stream.cu_stream() as *mut _)
                .map_err(|e| Error::Cuda(format!("cublasSetStream: {e:?}")))?;
            let (ws, _g) = workspace.device_ptr(stream);
            cublas::sys::cublasSetWorkspace_v2(handle, ws as *mut c_void, Self::WORKSPACE)
                .result()
                .map_err(|e| Error::Cuda(format!("cublasSetWorkspace: {e:?}")))?;
            cublas::sys::cublasSetMathMode(handle, cublas::sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH)
                .result()
                .map_err(|e| Error::Cuda(format!("cublasSetMathMode: {e:?}")))?;
        }
        Ok(Blas { handle, _workspace: workspace })
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            let _ = cublas::sys::cublasDestroy_v2(self.handle);
        }
    }
}

/// `extern:cublas_bf16_tn_f32`: row-major `C[m,n] = A[m,k] @ W[n,k]^T` with
/// bf16 operands, f32 accumulation and an **f32** result — cublasGemmEx with
/// `CUBLAS_COMPUTE_32F` / `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, the call an
/// engine that lands its own bf16 partials makes. Args as
/// [`gemm_bf16_tn`]: `[a, w, c, m, n, k]` or 7 with C's row stride.
pub(crate) fn gemm_bf16_tn_f32(blas: &Blas, args: &[RVal]) -> Result<()> {
    let (a, w, c, m, n, k, ldc) = match args {
        [a, w, c, m, n, k] => (a, w, c, m.val, n.val, k.val, n.val),
        [a, w, c, m, n, k, ldc] => (a, w, c, m.val, n.val, k.val, ldc.val),
        _ => bail!(Manifest, "gemm expects 6 or 7 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "gemm: ldc {ldc} < n {n}");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if a.bytes < m * k * 2 || w.bytes < n * k * 2 || c.bytes < ((m - 1) * ldc + n) * 4 {
        bail!(
            Manifest,
            "gemm f32: operands too small for m={m} n={n} k={k} ldc={ldc}: a {} B, w {} B, c {} B",
            a.bytes,
            w.bytes,
            c.bytes
        );
    }
    let dim = |v: u64| i32::try_from(v).map_err(|_| Error::Manifest(format!("gemm f32: dimension {v} exceeds i32")));
    let (alpha, beta) = (1.0f32, 0.0f32);
    use cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType};
    unsafe {
        cublas::result::gemm_ex(
            blas.handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            dim(n)?,
            dim(m)?,
            dim(k)?,
            &alpha as *const f32 as *const c_void,
            w.val as *const c_void,
            cudaDataType::CUDA_R_16BF,
            dim(k)?,
            a.val as *const c_void,
            cudaDataType::CUDA_R_16BF,
            dim(k)?,
            &beta as *const f32 as *const c_void,
            c.val as *mut c_void,
            cudaDataType::CUDA_R_32F,
            dim(ldc)?,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .map_err(|e| Error::Cuda(format!("cublasGemmEx f32 (m={m} n={n} k={k}): {e:?}")))?;
    }
    Ok(())
}
