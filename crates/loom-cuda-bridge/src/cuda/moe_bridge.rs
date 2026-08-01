//! Checked C bridge entrypoints for MoE movement around vendor grouped GEMM.

use super::*;

trait MoePermuteScalar: Copy {
    const DTYPE: DType;

    #[allow(clippy::too_many_arguments)]
    fn permute<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        hidden_states: &DeviceSlice<'_, Self>,
        topk_ids: &DeviceSlice<'_, i32>,
        expert_map: Option<&dyn CudaDeviceRead<i32>>,
        permuted_hidden_states: &mut DeviceSliceMut<'_, Self>,
        expert_offsets: &mut DeviceSliceMut<'_, i64>,
        inverse_permutation: &mut DeviceSliceMut<'_, i32>,
        permuted_assignment_ids: &mut DeviceSliceMut<'_, i32>,
        workspace: &mut DeviceSliceMut<'_, u8>,
        spec: MoePermuteSpec,
    ) -> Result<(), CudaExecutorError>;
}

trait MoeCombineScalar: MoePermuteScalar {
    #[allow(clippy::too_many_arguments)]
    fn combine<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        expert_outputs: &DeviceSlice<'_, Self>,
        routing_weights: &DeviceSlice<'_, f32>,
        inverse_permutation: &DeviceSlice<'_, i32>,
        expert_offsets: &DeviceSlice<'_, i64>,
        output: &mut DeviceSliceMut<'_, Self>,
        spec: MoeCombineSpec,
    ) -> Result<(), CudaExecutorError>;
}

macro_rules! impl_moe_permute_scalar {
    ($scalar:ty, $dtype:expr, $permute:ident) => {
        impl MoePermuteScalar for $scalar {
            const DTYPE: DType = $dtype;

            fn permute<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                hidden_states: &DeviceSlice<'_, Self>,
                topk_ids: &DeviceSlice<'_, i32>,
                expert_map: Option<&dyn CudaDeviceRead<i32>>,
                permuted_hidden_states: &mut DeviceSliceMut<'_, Self>,
                expert_offsets: &mut DeviceSliceMut<'_, i64>,
                inverse_permutation: &mut DeviceSliceMut<'_, i32>,
                permuted_assignment_ids: &mut DeviceSliceMut<'_, i32>,
                workspace: &mut DeviceSliceMut<'_, u8>,
                spec: MoePermuteSpec,
            ) -> Result<(), CudaExecutorError> {
                backend.$permute(
                    hidden_states,
                    topk_ids,
                    expert_map,
                    permuted_hidden_states,
                    expert_offsets,
                    inverse_permutation,
                    permuted_assignment_ids,
                    workspace,
                    spec,
                )
            }
        }
    };
}

macro_rules! impl_moe_combine_scalar {
    ($scalar:ty, $combine:ident) => {
        impl MoeCombineScalar for $scalar {
            fn combine<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                expert_outputs: &DeviceSlice<'_, Self>,
                routing_weights: &DeviceSlice<'_, f32>,
                inverse_permutation: &DeviceSlice<'_, i32>,
                expert_offsets: &DeviceSlice<'_, i64>,
                output: &mut DeviceSliceMut<'_, Self>,
                spec: MoeCombineSpec,
            ) -> Result<(), CudaExecutorError> {
                backend.$combine(
                    expert_outputs,
                    routing_weights,
                    inverse_permutation,
                    expert_offsets,
                    output,
                    spec,
                )
            }
        }
    };
}

impl_moe_permute_scalar!(f32, DType::F32, moe_permute_f32);
impl_moe_permute_scalar!(f16, DType::F16, moe_permute_f16);
impl_moe_permute_scalar!(bf16, DType::Bf16, moe_permute_bf16);
impl_moe_permute_scalar!(u8, DType::Fp8E4M3Fn, moe_permute_fp8_e4m3fn);
impl_moe_combine_scalar!(f32, moe_combine_f32);
impl_moe_combine_scalar!(f16, moe_combine_f16);
impl_moe_combine_scalar!(bf16, moe_combine_bf16);

#[derive(Clone, Copy)]
enum MoePermuteKind {
    F32,
    F16,
    Bf16,
    Fp8E4M3Fn,
}

fn moe_permute_kind(dtype: u32) -> Result<MoePermuteKind, CudaExecutorError> {
    match dtype {
        DTYPE_F32 => Ok(MoePermuteKind::F32),
        DTYPE_F16 => Ok(MoePermuteKind::F16),
        DTYPE_BF16 => Ok(MoePermuteKind::Bf16),
        DTYPE_FP8_E4M3FN => Ok(MoePermuteKind::Fp8E4M3Fn),
        _ => Err(CudaExecutorError::InvalidContract(format!(
            "unknown MoE permutation dtype code {dtype}"
        ))),
    }
}

macro_rules! dispatch_moe_permute {
    ($kind:expr, $function:ident ( $($argument:expr),* $(,)? )) => {
        match $kind {
            MoePermuteKind::F32 => unsafe { $function::<f32>($($argument),*) },
            MoePermuteKind::F16 => unsafe { $function::<f16>($($argument),*) },
            MoePermuteKind::Bf16 => unsafe { $function::<bf16>($($argument),*) },
            MoePermuteKind::Fp8E4M3Fn => unsafe { $function::<u8>($($argument),*) },
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn permute_spec(
    dtype: DType,
    tokens: u32,
    hidden_size: u32,
    top_k: u32,
    num_experts: u32,
    num_local_experts: u32,
    has_expert_map: bool,
) -> Result<MoePermuteSpec, CudaExecutorError> {
    MoePermuteSpec::new(
        tokens as usize,
        hidden_size as usize,
        top_k as usize,
        num_experts as usize,
        num_local_experts as usize,
        has_expert_map,
        dtype,
    )
    .map_err(invalid_contract)
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_moe_permute<T: MoePermuteScalar>(
    hidden_states: *const T,
    hidden_state_elements: u64,
    topk_ids: *const i32,
    topk_id_elements: u64,
    expert_map: *const i32,
    expert_map_elements: u64,
    permuted_hidden_states: *mut T,
    permuted_hidden_state_elements: u64,
    expert_offsets: *mut i64,
    expert_offset_elements: u64,
    inverse_permutation: *mut i32,
    inverse_permutation_elements: u64,
    permuted_assignment_ids: *mut i32,
    permuted_assignment_id_elements: u64,
    workspace: *mut u8,
    workspace_bytes: u64,
    tokens: u32,
    hidden_size: u32,
    top_k: u32,
    num_experts: u32,
    num_local_experts: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (hidden_states, hidden_range) =
        unsafe { read_slice(hidden_states, hidden_state_elements, "MoE hidden states") }?;
    let (topk_ids, topk_range) =
        unsafe { read_slice(topk_ids, topk_id_elements, "MoE top-k expert IDs") }?;
    let (expert_map, expert_map_range) =
        unsafe { read_optional_slice(expert_map, expert_map_elements, "MoE expert map") }?;
    let (mut permuted_hidden_states, permuted_range) = unsafe {
        write_slice(
            permuted_hidden_states,
            permuted_hidden_state_elements,
            "MoE permuted hidden states",
        )
    }?;
    let (mut expert_offsets, offset_range) =
        unsafe { write_slice(expert_offsets, expert_offset_elements, "MoE expert offsets") }?;
    let (mut inverse_permutation, inverse_range) = unsafe {
        write_slice(
            inverse_permutation,
            inverse_permutation_elements,
            "MoE inverse permutation",
        )
    }?;
    let (mut permuted_assignment_ids, assignment_range) = unsafe {
        write_slice(
            permuted_assignment_ids,
            permuted_assignment_id_elements,
            "MoE permuted assignment IDs",
        )
    }?;
    let (mut workspace, workspace_range) =
        unsafe { write_slice(workspace, workspace_bytes, "MoE permutation workspace") }?;

    let mut regions = vec![
        ("hidden states", hidden_range),
        ("top-k expert IDs", topk_range),
        ("permuted hidden states", permuted_range),
        ("expert offsets", offset_range),
        ("inverse permutation", inverse_range),
        ("permuted assignment IDs", assignment_range),
        ("workspace", workspace_range),
    ];
    if let Some(range) = expert_map_range {
        regions.push(("expert map", range));
    }
    require_disjoint(&regions, "MoE permutation")?;

    let spec = permute_spec(
        T::DTYPE,
        tokens,
        hidden_size,
        top_k,
        num_experts,
        num_local_experts,
        expert_map.is_some(),
    )?;
    T::permute(
        &stream_backend(stream),
        &hidden_states,
        &topk_ids,
        expert_map
            .as_ref()
            .map(|mapping| mapping as &dyn CudaDeviceRead<i32>),
        &mut permuted_hidden_states,
        &mut expert_offsets,
        &mut inverse_permutation,
        &mut permuted_assignment_ids,
        &mut workspace,
        spec,
    )?;
    record_launch(OP_MOE_PERMUTE);
    Ok(())
}

/// Return exact caller-owned MoE radix-sort workspace bytes.
///
/// # Safety
///
/// `workspace_bytes` must be a valid aligned writable host pointer.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_moe_permute_workspace_size(
    dtype: u32,
    tokens: u32,
    hidden_size: u32,
    top_k: u32,
    num_experts: u32,
    num_local_experts: u32,
    has_expert_map: u32,
    workspace_bytes: *mut u64,
) -> c_int {
    bridge_call(|| {
        if workspace_bytes.is_null()
            || !(workspace_bytes as usize).is_multiple_of(align_of::<u64>())
            || has_expert_map > 1
        {
            return Err(CudaExecutorError::InvalidContract(
                "MoE workspace output pointer or expert-map flag is invalid".into(),
            ));
        }
        let kind = moe_permute_kind(dtype)?;
        let dtype = match kind {
            MoePermuteKind::F32 => DType::F32,
            MoePermuteKind::F16 => DType::F16,
            MoePermuteKind::Bf16 => DType::Bf16,
            MoePermuteKind::Fp8E4M3Fn => DType::Fp8E4M3Fn,
        };
        let spec = permute_spec(
            dtype,
            tokens,
            hidden_size,
            top_k,
            num_experts,
            num_local_experts,
            has_expert_map != 0,
        )?;
        let bytes = u64::try_from(moe_permute_workspace_bytes(spec)?).map_err(|_| {
            CudaExecutorError::InvalidContract("MoE workspace exceeds uint64".into())
        })?;
        unsafe {
            *workspace_bytes = bytes;
        }
        Ok(())
    })
}

/// Stable expert-major MoE activation movement.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_moe_permute(
    dtype: u32,
    hidden_states: *const c_void,
    hidden_state_elements: u64,
    topk_ids: *const i32,
    topk_id_elements: u64,
    expert_map: *const i32,
    expert_map_elements: u64,
    permuted_hidden_states: *mut c_void,
    permuted_hidden_state_elements: u64,
    expert_offsets: *mut i64,
    expert_offset_elements: u64,
    inverse_permutation: *mut i32,
    inverse_permutation_elements: u64,
    permuted_assignment_ids: *mut i32,
    permuted_assignment_id_elements: u64,
    workspace: *mut u8,
    workspace_bytes: u64,
    tokens: u32,
    hidden_size: u32,
    top_k: u32,
    num_experts: u32,
    num_local_experts: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = moe_permute_kind(dtype)?;
        dispatch_moe_permute!(
            kind,
            launch_moe_permute(
                hidden_states.cast(),
                hidden_state_elements,
                topk_ids,
                topk_id_elements,
                expert_map,
                expert_map_elements,
                permuted_hidden_states.cast(),
                permuted_hidden_state_elements,
                expert_offsets,
                expert_offset_elements,
                inverse_permutation,
                inverse_permutation_elements,
                permuted_assignment_ids,
                permuted_assignment_id_elements,
                workspace,
                workspace_bytes,
                tokens,
                hidden_size,
                top_k,
                num_experts,
                num_local_experts,
                stream,
            )
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_moe_combine<T: MoeCombineScalar>(
    expert_outputs: *const T,
    expert_output_elements: u64,
    routing_weights: *const f32,
    routing_weight_elements: u64,
    inverse_permutation: *const i32,
    inverse_permutation_elements: u64,
    expert_offsets: *const i64,
    expert_offset_elements: u64,
    output: *mut T,
    output_elements: u64,
    tokens: u32,
    hidden_size: u32,
    top_k: u32,
    num_local_experts: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (expert_outputs, expert_output_range) =
        unsafe { read_slice(expert_outputs, expert_output_elements, "MoE expert outputs") }?;
    let (routing_weights, weight_range) = unsafe {
        read_slice(
            routing_weights,
            routing_weight_elements,
            "MoE routing weights",
        )
    }?;
    let (inverse_permutation, inverse_range) = unsafe {
        read_slice(
            inverse_permutation,
            inverse_permutation_elements,
            "MoE inverse permutation",
        )
    }?;
    let (expert_offsets, offset_range) =
        unsafe { read_slice(expert_offsets, expert_offset_elements, "MoE expert offsets") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "MoE combined output") }?;
    require_disjoint(
        &[
            ("expert outputs", expert_output_range),
            ("routing weights", weight_range),
            ("inverse permutation", inverse_range),
            ("expert offsets", offset_range),
            ("combined output", output_range),
        ],
        "MoE combine",
    )?;
    let spec = MoeCombineSpec::new(
        tokens as usize,
        hidden_size as usize,
        top_k as usize,
        num_local_experts as usize,
        T::DTYPE,
    )
    .map_err(invalid_contract)?;
    T::combine(
        &stream_backend(stream),
        &expert_outputs,
        &routing_weights,
        &inverse_permutation,
        &expert_offsets,
        &mut output,
        spec,
    )?;
    record_launch(OP_MOE_COMBINE);
    Ok(())
}

/// Weighted inverse MoE permutation after vendor grouped GEMM.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_moe_combine(
    dtype: u32,
    expert_outputs: *const c_void,
    expert_output_elements: u64,
    routing_weights: *const f32,
    routing_weight_elements: u64,
    inverse_permutation: *const i32,
    inverse_permutation_elements: u64,
    expert_offsets: *const i64,
    expert_offset_elements: u64,
    output: *mut c_void,
    output_elements: u64,
    tokens: u32,
    hidden_size: u32,
    top_k: u32,
    num_local_experts: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_moe_combine(
                expert_outputs.cast(),
                expert_output_elements,
                routing_weights,
                routing_weight_elements,
                inverse_permutation,
                inverse_permutation_elements,
                expert_offsets,
                expert_offset_elements,
                output.cast(),
                output_elements,
                tokens,
                hidden_size,
                top_k,
                num_local_experts,
                stream,
            )
        )
    })
}
