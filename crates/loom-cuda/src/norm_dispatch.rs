//! Safe CUDA dispatch for normalization contracts.

use crate::cuda_backend::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::CudaExecutorError;
use half::{bf16, f16};
use loom_kernels::{
    AddRmsNormSpec, DType, RmsNormDynamicFp8Spec, RmsNormDynamicInt8Spec, RmsNormSpec,
};

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Launches F32 RMSNorm asynchronously on this backend's stream.
    pub fn rms_norm_f32(
        &self,
        input: &impl CudaDeviceRead<f32>,
        weight: &impl CudaDeviceRead<f32>,
        output: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F32 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "F32 RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_buffers(input, weight, output, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_f32(
                input.as_ptr(),
                weight.as_ptr(),
                output.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Launches pair-vectorized FP16 RMSNorm asynchronously on this stream.
    pub fn rms_norm_f16(
        &self,
        input: &impl CudaDeviceRead<f16>,
        weight: &impl CudaDeviceRead<f16>,
        output: &mut impl CudaDeviceWrite<f16>,
        spec: RmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "FP16 RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_buffers(input, weight, output, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_f16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                output.as_mut_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Launches pair-vectorized BF16 RMSNorm asynchronously on this stream.
    pub fn rms_norm_bf16(
        &self,
        input: &impl CudaDeviceRead<bf16>,
        weight: &impl CudaDeviceRead<bf16>,
        output: &mut impl CudaDeviceWrite<bf16>,
        spec: RmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::Bf16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "BF16 RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_buffers(input, weight, output, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_bf16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                output.as_mut_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses F32 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_f32(
        &self,
        input: &impl CudaDeviceRead<f32>,
        weight: &impl CudaDeviceRead<f32>,
        mut residual: Option<&mut dyn CudaDeviceWrite<f32>>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::F32 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "F32 RMSNorm+FP8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let residual_read = residual
            .as_deref()
            .map(|values| values as &dyn CudaDeviceRead<f32>);
        let (rows, hidden_size) = validate_rms_norm_dynamic_fp8_buffers(
            input,
            weight,
            residual_read,
            output,
            scales,
            spec,
        )?;
        let residual_pointer = residual
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), CudaDeviceWrite::as_mut_ptr);

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_fp8_f32(
                input.as_ptr(),
                weight.as_ptr(),
                residual_pointer,
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses FP16 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_f16(
        &self,
        input: &impl CudaDeviceRead<f16>,
        weight: &impl CudaDeviceRead<f16>,
        mut residual: Option<&mut dyn CudaDeviceWrite<f16>>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::F16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "FP16 RMSNorm+FP8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let residual_read = residual
            .as_deref()
            .map(|values| values as &dyn CudaDeviceRead<f16>);
        let (rows, hidden_size) = validate_rms_norm_dynamic_fp8_buffers(
            input,
            weight,
            residual_read,
            output,
            scales,
            spec,
        )?;
        let residual_pointer = residual
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), CudaDeviceWrite::as_mut_ptr);

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_fp8_f16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                residual_pointer.cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses BF16 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_bf16(
        &self,
        input: &impl CudaDeviceRead<bf16>,
        weight: &impl CudaDeviceRead<bf16>,
        mut residual: Option<&mut dyn CudaDeviceWrite<bf16>>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::Bf16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "BF16 RMSNorm+FP8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let residual_read = residual
            .as_deref()
            .map(|values| values as &dyn CudaDeviceRead<bf16>);
        let (rows, hidden_size) = validate_rms_norm_dynamic_fp8_buffers(
            input,
            weight,
            residual_read,
            output,
            scales,
            spec,
        )?;
        let residual_pointer = residual
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), CudaDeviceWrite::as_mut_ptr);

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_fp8_bf16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                residual_pointer.cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses F32 RMSNorm with symmetric dynamic per-token INT8 quantization.
    pub fn rms_norm_dynamic_int8_f32(
        &self,
        input: &impl CudaDeviceRead<f32>,
        weight: &impl CudaDeviceRead<f32>,
        mut residual: Option<&mut dyn CudaDeviceWrite<f32>>,
        output: &mut impl CudaDeviceWrite<i8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicInt8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::F32 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "F32 RMSNorm+INT8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let residual_read = residual
            .as_deref()
            .map(|values| values as &dyn CudaDeviceRead<f32>);
        let (rows, hidden_size) = validate_rms_norm_dynamic_int8_buffers(
            input,
            weight,
            residual_read,
            output,
            scales,
            spec,
        )?;
        let residual_pointer = residual
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), CudaDeviceWrite::as_mut_ptr);

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_int8_f32(
                input.as_ptr(),
                weight.as_ptr(),
                residual_pointer,
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses FP16 RMSNorm with symmetric dynamic per-token INT8 quantization.
    pub fn rms_norm_dynamic_int8_f16(
        &self,
        input: &impl CudaDeviceRead<f16>,
        weight: &impl CudaDeviceRead<f16>,
        mut residual: Option<&mut dyn CudaDeviceWrite<f16>>,
        output: &mut impl CudaDeviceWrite<i8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicInt8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::F16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "FP16 RMSNorm+INT8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let residual_read = residual
            .as_deref()
            .map(|values| values as &dyn CudaDeviceRead<f16>);
        let (rows, hidden_size) = validate_rms_norm_dynamic_int8_buffers(
            input,
            weight,
            residual_read,
            output,
            scales,
            spec,
        )?;
        let residual_pointer = residual
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), CudaDeviceWrite::as_mut_ptr);

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_int8_f16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                residual_pointer.cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses BF16 RMSNorm with symmetric dynamic per-token INT8 quantization.
    pub fn rms_norm_dynamic_int8_bf16(
        &self,
        input: &impl CudaDeviceRead<bf16>,
        weight: &impl CudaDeviceRead<bf16>,
        mut residual: Option<&mut dyn CudaDeviceWrite<bf16>>,
        output: &mut impl CudaDeviceWrite<i8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicInt8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::Bf16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "BF16 RMSNorm+INT8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let residual_read = residual
            .as_deref()
            .map(|values| values as &dyn CudaDeviceRead<bf16>);
        let (rows, hidden_size) = validate_rms_norm_dynamic_int8_buffers(
            input,
            weight,
            residual_read,
            output,
            scales,
            spec,
        )?;
        let residual_pointer = residual
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), CudaDeviceWrite::as_mut_ptr);

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_int8_bf16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                residual_pointer.cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses F32 residual addition and RMSNorm, updating both buffers in place.
    pub fn add_rms_norm_f32(
        &self,
        input: &mut impl CudaDeviceWrite<f32>,
        residual: &mut impl CudaDeviceWrite<f32>,
        weight: &impl CudaDeviceRead<f32>,
        spec: AddRmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F32 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "F32 Add+RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_add_rms_norm_buffers(input, residual, weight, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_add_rms_norm_f32(
                input.as_mut_ptr(),
                residual.as_mut_ptr(),
                weight.as_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses pair-vectorized FP16 residual addition and RMSNorm in place.
    pub fn add_rms_norm_f16(
        &self,
        input: &mut impl CudaDeviceWrite<f16>,
        residual: &mut impl CudaDeviceWrite<f16>,
        weight: &impl CudaDeviceRead<f16>,
        spec: AddRmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "FP16 Add+RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_add_rms_norm_buffers(input, residual, weight, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_add_rms_norm_f16(
                input.as_mut_ptr().cast::<u16>(),
                residual.as_mut_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses pair-vectorized BF16 residual addition and RMSNorm in place.
    pub fn add_rms_norm_bf16(
        &self,
        input: &mut impl CudaDeviceWrite<bf16>,
        residual: &mut impl CudaDeviceWrite<bf16>,
        weight: &impl CudaDeviceRead<bf16>,
        spec: AddRmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::Bf16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "BF16 Add+RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_add_rms_norm_buffers(input, residual, weight, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_add_rms_norm_bf16(
                input.as_mut_ptr().cast::<u16>(),
                residual.as_mut_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }
}

fn validate_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    weight: &impl CudaDeviceRead<T>,
    output: &impl CudaDeviceRead<T>,
    spec: RmsNormSpec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.numel(), "RMSNorm input")?;
    weight.require_len(spec.hidden_size(), "RMSNorm weight")?;
    output.require_len(spec.numel(), "RMSNorm output")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm rows exceed the CUDA ABI".into())
    })?;
    let hidden_size = u32::try_from(spec.hidden_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm hidden size exceeds the CUDA ABI".into())
    })?;
    Ok((rows, hidden_size))
}

fn validate_add_rms_norm_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    residual: &impl CudaDeviceRead<T>,
    weight: &impl CudaDeviceRead<T>,
    spec: AddRmsNormSpec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.numel(), "Add+RMSNorm input")?;
    residual.require_len(spec.numel(), "Add+RMSNorm residual")?;
    weight.require_len(spec.hidden_size(), "Add+RMSNorm weight")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("Add+RMSNorm rows exceed the CUDA ABI".into())
    })?;
    let hidden_size = u32::try_from(spec.hidden_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("Add+RMSNorm hidden size exceeds the CUDA ABI".into())
    })?;
    Ok((rows, hidden_size))
}

fn validate_rms_norm_dynamic_fp8_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    weight: &impl CudaDeviceRead<T>,
    residual: Option<&dyn CudaDeviceRead<T>>,
    output: &impl CudaDeviceRead<u8>,
    scales: &impl CudaDeviceRead<f32>,
    spec: RmsNormDynamicFp8Spec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.numel(), "RMSNorm+FP8 input")?;
    weight.require_len(spec.hidden_size(), "RMSNorm+FP8 weight")?;
    if let Some(values) = residual {
        values.require_len(spec.numel(), "RMSNorm+FP8 residual")?;
    }
    output.require_len(spec.numel(), "RMSNorm+FP8 output")?;
    scales.require_len(spec.scale_count(), "RMSNorm+FP8 scales")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm+FP8 rows exceed the CUDA ABI".into())
    })?;
    let hidden_size = u32::try_from(spec.hidden_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm+FP8 hidden size exceeds the CUDA ABI".into())
    })?;
    Ok((rows, hidden_size))
}

fn validate_rms_norm_dynamic_int8_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    weight: &impl CudaDeviceRead<T>,
    residual: Option<&dyn CudaDeviceRead<T>>,
    output: &impl CudaDeviceRead<i8>,
    scales: &impl CudaDeviceRead<f32>,
    spec: RmsNormDynamicInt8Spec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.numel(), "RMSNorm+INT8 input")?;
    weight.require_len(spec.hidden_size(), "RMSNorm+INT8 weight")?;
    if let Some(values) = residual {
        values.require_len(spec.numel(), "RMSNorm+INT8 residual")?;
    }
    output.require_len(spec.numel(), "RMSNorm+INT8 output")?;
    scales.require_len(spec.scale_count(), "RMSNorm+INT8 scales")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm+INT8 rows exceed the CUDA ABI".into())
    })?;
    let hidden_size = u32::try_from(spec.hidden_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm+INT8 hidden size exceeds the CUDA ABI".into())
    })?;
    Ok((rows, hidden_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceBuffer;
    use loom_kernels::{rms_norm_dynamic_fp8_f32_reference, rms_norm_dynamic_int8_f32_reference};

    #[test]
    fn residual_dynamic_fp8_wrapper_matches_the_cpu_oracle() {
        let spec = RmsNormDynamicFp8Spec::new(2, 4, 1.0e-5, DType::F32).unwrap();
        let input = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -2.0];
        let residual = vec![1.0, 0.25, -0.5, 2.0, 0.5, -0.25, 1.0, 0.75];
        let weight = vec![1.0, 0.75, 1.25, 0.5];
        let mut expected_residual = residual.clone();
        let mut expected_output = vec![0_u8; spec.numel()];
        let mut expected_scales = vec![0.0_f32; spec.scale_count()];
        rms_norm_dynamic_fp8_f32_reference(
            &input,
            &weight,
            &mut expected_output,
            &mut expected_scales,
            Some(&mut expected_residual),
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let device_input = DeviceBuffer::from_slice(&input).unwrap();
        let device_weight = DeviceBuffer::from_slice(&weight).unwrap();
        let mut device_residual = DeviceBuffer::from_slice(&residual).unwrap();
        let mut device_output = DeviceBuffer::<u8>::uninitialized(spec.numel()).unwrap();
        let mut device_scales = DeviceBuffer::<f32>::uninitialized(spec.scale_count()).unwrap();
        backend
            .rms_norm_dynamic_fp8_f32(
                &device_input,
                &device_weight,
                Some(&mut device_residual),
                &mut device_output,
                &mut device_scales,
                spec,
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(device_residual.copy_to_vec().unwrap(), expected_residual);
        assert_eq!(device_output.copy_to_vec().unwrap(), expected_output);
        for (actual, expected) in device_scales
            .copy_to_vec()
            .unwrap()
            .iter()
            .zip(expected_scales)
        {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn residual_dynamic_int8_wrapper_matches_the_cpu_oracle() {
        let spec = RmsNormDynamicInt8Spec::new(2, 4, 1.0e-5, DType::F32).unwrap();
        let input = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -2.0];
        let residual = vec![1.0, 0.25, -0.5, 2.0, 0.5, -0.25, 1.0, 0.75];
        let weight = vec![1.0, 0.75, 1.25, 0.5];
        let mut expected_residual = residual.clone();
        let mut expected_output = vec![0_i8; spec.numel()];
        let mut expected_scales = vec![0.0_f32; spec.scale_count()];
        rms_norm_dynamic_int8_f32_reference(
            &input,
            &weight,
            &mut expected_output,
            &mut expected_scales,
            Some(&mut expected_residual),
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let device_input = DeviceBuffer::from_slice(&input).unwrap();
        let device_weight = DeviceBuffer::from_slice(&weight).unwrap();
        let mut device_residual = DeviceBuffer::from_slice(&residual).unwrap();
        let mut device_output = DeviceBuffer::<i8>::uninitialized(spec.numel()).unwrap();
        let mut device_scales = DeviceBuffer::<f32>::uninitialized(spec.scale_count()).unwrap();
        backend
            .rms_norm_dynamic_int8_f32(
                &device_input,
                &device_weight,
                Some(&mut device_residual),
                &mut device_output,
                &mut device_scales,
                spec,
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(device_residual.copy_to_vec().unwrap(), expected_residual);
        assert_eq!(device_output.copy_to_vec().unwrap(), expected_output);
        for (actual, expected) in device_scales
            .copy_to_vec()
            .unwrap()
            .iter()
            .zip(expected_scales)
        {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
}
