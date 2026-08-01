//! Safe CUDA dispatch for MoE token movement around vendor grouped GEMM.

use half::{bf16, f16};
use loom_kernels::{DType, MoeCombineSpec, MoePermuteSpec};

use crate::cuda_backend::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::CudaExecutorError;

fn cuda_u32(value: usize, name: &str) -> Result<u32, CudaExecutorError> {
    u32::try_from(value)
        .map_err(|_| CudaExecutorError::InvalidContract(format!("{name} exceeds the CUDA ABI")))
}

/// Returns the exact caller-owned byte workspace for stable radix sorting.
pub fn moe_permute_workspace_bytes(spec: MoePermuteSpec) -> Result<usize, CudaExecutorError> {
    let assignments = cuda_u32(spec.assignment_count(), "MoE assignment count")?;
    let num_experts = cuda_u32(spec.num_experts(), "MoE expert count")?;
    let mut workspace_bytes = 0_u64;
    loom_status_result(unsafe {
        loom_cuda_sys::loom_cuda_moe_permute_workspace_size(
            assignments,
            num_experts,
            &mut workspace_bytes,
        )
    })?;
    usize::try_from(workspace_bytes).map_err(|_| {
        CudaExecutorError::InvalidContract("MoE workspace exceeds the host address space".into())
    })
}

fn validate_expert_map(
    expert_map: Option<&dyn CudaDeviceRead<i32>>,
    spec: MoePermuteSpec,
) -> Result<*const i32, CudaExecutorError> {
    match (spec.has_expert_map(), expert_map) {
        (false, None) => Ok(std::ptr::null()),
        (true, Some(mapping)) => {
            mapping.require_len(spec.num_experts(), "MoE expert map")?;
            Ok(mapping.as_ptr())
        }
        _ => Err(CudaExecutorError::InvalidContract(format!(
            "MoE expert-map presence does not match the contract; expected={}",
            spec.has_expert_map()
        ))),
    }
}

macro_rules! impl_moe_permute {
    ($method:ident, $element:ty, $dtype:expr, $launch:path) => {
        impl<S: CudaStreamHandle> CudaBackend<S> {
            /// Stably groups token assignments and gathers expert-major rows.
            #[allow(clippy::too_many_arguments)]
            pub fn $method(
                &self,
                hidden_states: &impl CudaDeviceRead<$element>,
                topk_ids: &impl CudaDeviceRead<i32>,
                expert_map: Option<&dyn CudaDeviceRead<i32>>,
                permuted_hidden_states: &mut impl CudaDeviceWrite<$element>,
                expert_offsets: &mut impl CudaDeviceWrite<i64>,
                inverse_permutation: &mut impl CudaDeviceWrite<i32>,
                permuted_assignment_ids: &mut impl CudaDeviceWrite<i32>,
                workspace: &mut impl CudaDeviceWrite<u8>,
                spec: MoePermuteSpec,
            ) -> Result<(), CudaExecutorError> {
                if spec.dtype() != $dtype {
                    return Err(CudaExecutorError::InvalidContract(format!(
                        "MoE permutation method expects {:?}, got {:?}",
                        $dtype,
                        spec.dtype()
                    )));
                }
                hidden_states.require_len(spec.hidden_numel(), "MoE hidden states")?;
                topk_ids.require_len(spec.assignment_count(), "MoE top-k expert IDs")?;
                permuted_hidden_states
                    .require_len(spec.permuted_hidden_numel(), "MoE permuted hidden states")?;
                expert_offsets.require_len(spec.expert_offset_count(), "MoE expert offsets")?;
                inverse_permutation
                    .require_len(spec.assignment_count(), "MoE inverse permutation")?;
                permuted_assignment_ids
                    .require_len(spec.assignment_count(), "MoE permuted assignment IDs")?;
                let expected_workspace = moe_permute_workspace_bytes(spec)?;
                workspace.require_len(expected_workspace, "MoE permutation workspace")?;
                let expert_map = validate_expert_map(expert_map, spec)?;

                loom_status_result(unsafe {
                    $launch(
                        hidden_states.as_ptr().cast(),
                        topk_ids.as_ptr(),
                        expert_map,
                        permuted_hidden_states.as_mut_ptr().cast(),
                        expert_offsets.as_mut_ptr(),
                        inverse_permutation.as_mut_ptr(),
                        permuted_assignment_ids.as_mut_ptr(),
                        workspace.as_mut_ptr(),
                        u64::try_from(workspace.len()).map_err(|_| {
                            CudaExecutorError::InvalidContract(
                                "MoE workspace byte count exceeds uint64".into(),
                            )
                        })?,
                        cuda_u32(spec.tokens(), "MoE token count")?,
                        cuda_u32(spec.hidden_size(), "MoE hidden size")?,
                        cuda_u32(spec.top_k(), "MoE top-k")?,
                        cuda_u32(spec.num_experts(), "MoE expert count")?,
                        cuda_u32(spec.num_local_experts(), "MoE local expert count")?,
                        self.raw_stream(),
                    )
                })
            }
        }
    };
}

impl_moe_permute!(
    moe_permute_f32,
    f32,
    DType::F32,
    loom_cuda_sys::loom_cuda_moe_permute_f32
);
impl_moe_permute!(
    moe_permute_f16,
    f16,
    DType::F16,
    loom_cuda_sys::loom_cuda_moe_permute_f16
);
impl_moe_permute!(
    moe_permute_bf16,
    bf16,
    DType::Bf16,
    loom_cuda_sys::loom_cuda_moe_permute_bf16
);
impl_moe_permute!(
    moe_permute_fp8_e4m3fn,
    u8,
    DType::Fp8E4M3Fn,
    loom_cuda_sys::loom_cuda_moe_permute_fp8_e4m3fn
);

macro_rules! impl_moe_combine {
    ($method:ident, $element:ty, $dtype:expr, $launch:path) => {
        impl<S: CudaStreamHandle> CudaBackend<S> {
            /// Inverts expert-major movement and applies F32 routing weights.
            #[allow(clippy::too_many_arguments)]
            pub fn $method(
                &self,
                expert_outputs: &impl CudaDeviceRead<$element>,
                routing_weights: &impl CudaDeviceRead<f32>,
                inverse_permutation: &impl CudaDeviceRead<i32>,
                expert_offsets: &impl CudaDeviceRead<i64>,
                output: &mut impl CudaDeviceWrite<$element>,
                spec: MoeCombineSpec,
            ) -> Result<(), CudaExecutorError> {
                if spec.dtype() != $dtype {
                    return Err(CudaExecutorError::InvalidContract(format!(
                        "MoE combine method expects {:?}, got {:?}",
                        $dtype,
                        spec.dtype()
                    )));
                }
                expert_outputs.require_len(spec.expert_output_numel(), "MoE expert outputs")?;
                routing_weights.require_len(spec.assignment_count(), "MoE routing weights")?;
                inverse_permutation
                    .require_len(spec.assignment_count(), "MoE inverse permutation")?;
                expert_offsets.require_len(spec.expert_offset_count(), "MoE expert offsets")?;
                output.require_len(spec.output_numel(), "MoE combined output")?;

                loom_status_result(unsafe {
                    $launch(
                        expert_outputs.as_ptr().cast(),
                        routing_weights.as_ptr(),
                        inverse_permutation.as_ptr(),
                        expert_offsets.as_ptr(),
                        output.as_mut_ptr().cast(),
                        cuda_u32(spec.tokens(), "MoE token count")?,
                        cuda_u32(spec.hidden_size(), "MoE hidden size")?,
                        cuda_u32(spec.top_k(), "MoE top-k")?,
                        cuda_u32(spec.num_local_experts(), "MoE local expert count")?,
                        self.raw_stream(),
                    )
                })
            }
        }
    };
}

impl_moe_combine!(
    moe_combine_f32,
    f32,
    DType::F32,
    loom_cuda_sys::loom_cuda_moe_combine_f32
);
impl_moe_combine!(
    moe_combine_f16,
    f16,
    DType::F16,
    loom_cuda_sys::loom_cuda_moe_combine_f16
);
impl_moe_combine!(
    moe_combine_bf16,
    bf16,
    DType::Bf16,
    loom_cuda_sys::loom_cuda_moe_combine_bf16
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceBuffer;
    use loom_kernels::{moe_combine_f32_reference, moe_permute_f32_reference, DType};

    #[test]
    fn safe_rust_moe_movement_matches_cpu_oracles() {
        let permute_spec = MoePermuteSpec::new(4, 4, 2, 3, 3, false, DType::F32).unwrap();
        let hidden: Vec<f32> = (0..16).map(|value| value as f32).collect();
        let ids = [2_i32, 0, 1, 2, 0, 1, 2, 1];
        let mut expected_permuted = vec![0.0_f32; permute_spec.permuted_hidden_numel()];
        let mut expected_offsets = vec![0_i64; permute_spec.expert_offset_count()];
        let mut expected_inverse = vec![0_i32; permute_spec.assignment_count()];
        let mut expected_assignments = vec![0_i32; permute_spec.assignment_count()];
        moe_permute_f32_reference(
            &hidden,
            &ids,
            None,
            &mut expected_permuted,
            &mut expected_offsets,
            &mut expected_inverse,
            &mut expected_assignments,
            permute_spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let hidden_device = DeviceBuffer::from_slice(&hidden).unwrap();
        let ids_device = DeviceBuffer::from_slice(&ids).unwrap();
        let mut permuted_device =
            DeviceBuffer::from_slice(&vec![17.0_f32; permute_spec.permuted_hidden_numel()])
                .unwrap();
        let mut offsets_device =
            DeviceBuffer::from_slice(&vec![17_i64; permute_spec.expert_offset_count()]).unwrap();
        let mut inverse_device =
            DeviceBuffer::from_slice(&vec![17_i32; permute_spec.assignment_count()]).unwrap();
        let mut assignments_device =
            DeviceBuffer::from_slice(&vec![17_i32; permute_spec.assignment_count()]).unwrap();
        let mut workspace =
            DeviceBuffer::from_slice(&vec![
                0_u8;
                moe_permute_workspace_bytes(permute_spec).unwrap()
            ])
            .unwrap();
        backend
            .moe_permute_f32(
                &hidden_device,
                &ids_device,
                None,
                &mut permuted_device,
                &mut offsets_device,
                &mut inverse_device,
                &mut assignments_device,
                &mut workspace,
                permute_spec,
            )
            .unwrap();
        backend.stream().synchronize().unwrap();
        assert_eq!(permuted_device.copy_to_vec().unwrap(), expected_permuted);
        assert_eq!(offsets_device.copy_to_vec().unwrap(), expected_offsets);
        assert_eq!(inverse_device.copy_to_vec().unwrap(), expected_inverse);
        assert_eq!(
            assignments_device.copy_to_vec().unwrap(),
            expected_assignments
        );

        let combine_spec = MoeCombineSpec::new(4, 4, 2, 3, DType::F32).unwrap();
        let weights = [0.7_f32, 0.3, 0.2, 0.8, 0.6, 0.4, 0.9, 0.1];
        let mut expected_output = vec![0.0_f32; combine_spec.output_numel()];
        moe_combine_f32_reference(
            &expected_permuted,
            &weights,
            &expected_inverse,
            &expected_offsets,
            &mut expected_output,
            combine_spec,
        )
        .unwrap();
        let weights_device = DeviceBuffer::from_slice(&weights).unwrap();
        let mut output_device =
            DeviceBuffer::from_slice(&vec![17.0_f32; combine_spec.output_numel()]).unwrap();
        backend
            .moe_combine_f32(
                &permuted_device,
                &weights_device,
                &inverse_device,
                &offsets_device,
                &mut output_device,
                combine_spec,
            )
            .unwrap();
        backend.stream().synchronize().unwrap();
        let actual = output_device.copy_to_vec().unwrap();
        for (actual, expected) in actual.iter().zip(expected_output) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
}
