//! Safe CUDA dispatch for logits-processing contracts.

use crate::cuda_backend::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::{CudaExecutorError, RowStridedLayout};
use half::{bf16, f16};
use loom_kernels::{DType, LogitsPreprocessSpec, MinPFilterSpec};

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Applies mask, sparse bias/suppression, and temperature in one F32 pass.
    #[allow(clippy::too_many_arguments)]
    pub fn logits_preprocess_f32(
        &self,
        logits: &mut impl CudaDeviceWrite<f32>,
        temperatures: &impl CudaDeviceRead<f32>,
        blocked_mask: Option<&dyn CudaDeviceRead<u8>>,
        bias_row_ids: Option<&dyn CudaDeviceRead<i32>>,
        bias_token_ids: Option<&dyn CudaDeviceRead<i32>>,
        bias_values: Option<&dyn CudaDeviceRead<f32>>,
        suppressed_row_ids: Option<&dyn CudaDeviceRead<i32>>,
        suppressed_token_ids: Option<&dyn CudaDeviceRead<i32>>,
        spec: LogitsPreprocessSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        logits.require_len(
            layout.storage_elements(spec.rows(), spec.vocab_size())?,
            "logits-preprocessing logits",
        )?;
        temperatures.require_len(spec.rows(), "logits-preprocessing temperatures")?;
        validate_optional_buffer(
            blocked_mask,
            spec.blocked_mask_numel(),
            "logits-preprocessing blocked mask",
        )?;
        validate_optional_buffer(
            bias_row_ids,
            spec.bias_count(),
            "logits-preprocessing bias row IDs",
        )?;
        validate_optional_buffer(
            bias_token_ids,
            spec.bias_count(),
            "logits-preprocessing bias token IDs",
        )?;
        validate_optional_buffer(
            bias_values,
            spec.bias_count(),
            "logits-preprocessing bias values",
        )?;
        validate_optional_buffer(
            suppressed_row_ids,
            spec.suppression_count(),
            "logits-preprocessing suppressed row IDs",
        )?;
        validate_optional_buffer(
            suppressed_token_ids,
            spec.suppression_count(),
            "logits-preprocessing suppressed token IDs",
        )?;

        let rows = i32::try_from(spec.rows()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "logits-preprocessing rows exceed the int32 sparse-index ABI".into(),
            )
        })? as u32;
        let vocab_size = i32::try_from(spec.vocab_size()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "logits-preprocessing vocabulary exceeds the int32 sparse-index ABI".into(),
            )
        })? as u32;
        let bias_count = u32::try_from(spec.bias_count()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "logits-preprocessing bias count exceeds the CUDA ABI".into(),
            )
        })?;
        let suppression_count = u32::try_from(spec.suppression_count()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "logits-preprocessing suppression count exceeds the CUDA ABI".into(),
            )
        })?;
        let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "logits-preprocessing row stride exceeds the CUDA ABI".into(),
            )
        })?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_logits_preprocess_f32(
                logits.as_mut_ptr(),
                temperatures.as_ptr(),
                optional_pointer(blocked_mask),
                optional_pointer(bias_row_ids),
                optional_pointer(bias_token_ids),
                optional_pointer(bias_values),
                bias_count,
                optional_pointer(suppressed_row_ids),
                optional_pointer(suppressed_token_ids),
                suppression_count,
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Applies in-place F32 min-p filtering without materializing softmax.
    pub fn min_p_filter_f32(
        &self,
        logits: &mut impl CudaDeviceWrite<f32>,
        min_p: &impl CudaDeviceRead<f32>,
        spec: MinPFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F32)?;
        let (rows, vocab_size, row_stride) = validate_buffers(logits, min_p, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_min_p_filter_f32(
                logits.as_mut_ptr(),
                min_p.as_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Applies in-place FP16 min-p filtering without materializing softmax.
    pub fn min_p_filter_f16(
        &self,
        logits: &mut impl CudaDeviceWrite<f16>,
        min_p: &impl CudaDeviceRead<f32>,
        spec: MinPFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F16)?;
        let (rows, vocab_size, row_stride) = validate_buffers(logits, min_p, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_min_p_filter_f16(
                logits.as_mut_ptr().cast::<u16>(),
                min_p.as_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Applies in-place BF16 min-p filtering without materializing softmax.
    pub fn min_p_filter_bf16(
        &self,
        logits: &mut impl CudaDeviceWrite<bf16>,
        min_p: &impl CudaDeviceRead<f32>,
        spec: MinPFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::Bf16)?;
        let (rows, vocab_size, row_stride) = validate_buffers(logits, min_p, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_min_p_filter_bf16(
                logits.as_mut_ptr().cast::<u16>(),
                min_p.as_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }
}

fn validate_optional_buffer<T: Copy>(
    buffer: Option<&dyn CudaDeviceRead<T>>,
    expected: usize,
    name: &str,
) -> Result<(), CudaExecutorError> {
    match (buffer, expected) {
        (None, 0) => Ok(()),
        (Some(buffer), expected) if expected > 0 => buffer.require_len(expected, name),
        (None, expected) => Err(CudaExecutorError::InvalidContract(format!(
            "{name} is absent, expected {expected} elements"
        ))),
        (Some(_), 0) => Err(CudaExecutorError::InvalidContract(format!(
            "{name} must be absent when its element count is zero"
        ))),
        (Some(_), _) => unreachable!(),
    }
}

fn optional_pointer<T: Copy>(buffer: Option<&dyn CudaDeviceRead<T>>) -> *const T {
    buffer.map_or(std::ptr::null(), CudaDeviceRead::as_ptr)
}

fn require_dtype(spec: MinPFilterSpec, expected: DType) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "min-p filtering for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    min_p: &impl CudaDeviceRead<f32>,
    spec: MinPFilterSpec,
    layout: RowStridedLayout,
) -> Result<(u32, u32, u64), CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "min-p logits",
    )?;
    min_p.require_len(spec.rows(), "min-p probabilities")?;
    let rows = u32::try_from(spec.rows())
        .map_err(|_| CudaExecutorError::InvalidContract("min-p rows exceed the CUDA ABI".into()))?;
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("min-p vocabulary exceeds the CUDA ABI".into())
    })?;
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract("min-p row stride exceeds the CUDA ABI".into())
    })?;
    Ok((rows, vocab_size, row_stride))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceBuffer;
    use loom_kernels::{logits_preprocess_f32_reference, min_p_filter_f32_reference};

    #[test]
    fn safe_logits_preprocessing_matches_the_cpu_oracle() {
        let spec = LogitsPreprocessSpec::new(2, 5, true, 2, 2).unwrap();
        let original = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, //
            -1.0, -2.0, -3.0, -4.0, -5.0,
        ];
        let temperatures = [0.0_f32, 0.5];
        let blocked_mask = [0_u8, 1, 0, 0, 0, 0, 0, 0, 1, 0];
        let bias_row_ids = [0_i32, 1];
        let bias_token_ids = [0_i32, 4];
        let bias_values = [0.25_f32, 1.0];
        let suppressed_row_ids = [0_i32, 1];
        let suppressed_token_ids = [3_i32, 0];
        let mut expected = original;
        logits_preprocess_f32_reference(
            &mut expected,
            &temperatures,
            Some(&blocked_mask),
            &bias_row_ids,
            &bias_token_ids,
            &bias_values,
            &suppressed_row_ids,
            &suppressed_token_ids,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let mut logits = DeviceBuffer::from_slice(&original).unwrap();
        let temperatures = DeviceBuffer::from_slice(&temperatures).unwrap();
        let blocked_mask = DeviceBuffer::from_slice(&blocked_mask).unwrap();
        let bias_row_ids = DeviceBuffer::from_slice(&bias_row_ids).unwrap();
        let bias_token_ids = DeviceBuffer::from_slice(&bias_token_ids).unwrap();
        let bias_values = DeviceBuffer::from_slice(&bias_values).unwrap();
        let suppressed_row_ids = DeviceBuffer::from_slice(&suppressed_row_ids).unwrap();
        let suppressed_token_ids = DeviceBuffer::from_slice(&suppressed_token_ids).unwrap();
        backend
            .logits_preprocess_f32(
                &mut logits,
                &temperatures,
                Some(&blocked_mask),
                Some(&bias_row_ids),
                Some(&bias_token_ids),
                Some(&bias_values),
                Some(&suppressed_row_ids),
                Some(&suppressed_token_ids),
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(logits.copy_to_vec().unwrap(), expected);
    }

    #[test]
    fn safe_rust_wrapper_matches_the_cpu_oracle() {
        let spec = MinPFilterSpec::new(3, 5, DType::F32).unwrap();
        let original = [
            1.0_f32, 3.0, 2.0, -1.0, 0.5, //
            -2.0, -1.0, 2.0, 0.0, 1.0, //
            4.0, 4.0, 3.0, -8.0, 0.0,
        ];
        let probabilities = [0.0_f32, 0.2, 1.0];
        let mut expected = original;
        min_p_filter_f32_reference(&mut expected, &probabilities, spec).unwrap();

        let backend = CudaBackend::new().unwrap();
        let mut logits = DeviceBuffer::from_slice(&original).unwrap();
        let min_p = DeviceBuffer::from_slice(&probabilities).unwrap();
        backend
            .min_p_filter_f32(
                &mut logits,
                &min_p,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(logits.copy_to_vec().unwrap(), expected);
    }
}
