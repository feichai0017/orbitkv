//! Safe CUDA dispatch for token-selection and logprob contracts.

use crate::cuda_backend::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::{CudaExecutorError, RowStridedLayout};
use half::{bf16, f16};
use loom_kernels::{
    DType, GreedySampleLogprobsSpec, SelectedTokenLogprobsSpec, TokenPenaltiesSpec, TopKFilterSpec,
    TopKSampledLogprobsSpec,
};
use std::mem::align_of;

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Fuses F32 greedy selection with the sampled token's logprob and rank.
    pub fn greedy_sample_logprobs_f32(
        &self,
        logits: &impl CudaDeviceRead<f32>,
        token_ids: &mut impl CudaDeviceWrite<i32>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: GreedySampleLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F32)?;
        let (rows, vocab_size, row_stride) =
            validate_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_greedy_sample_logprobs_f32(
                logits.as_ptr(),
                token_ids.as_mut_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Fuses FP16 greedy selection with an F32 sampled-token logprob.
    pub fn greedy_sample_logprobs_f16(
        &self,
        logits: &impl CudaDeviceRead<f16>,
        token_ids: &mut impl CudaDeviceWrite<i32>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: GreedySampleLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F16)?;
        let (rows, vocab_size, row_stride) =
            validate_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_greedy_sample_logprobs_f16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_mut_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Fuses BF16 greedy selection with an F32 sampled-token logprob.
    pub fn greedy_sample_logprobs_bf16(
        &self,
        logits: &impl CudaDeviceRead<bf16>,
        token_ids: &mut impl CudaDeviceWrite<i32>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: GreedySampleLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::Bf16)?;
        let (rows, vocab_size, row_stride) =
            validate_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_greedy_sample_logprobs_bf16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_mut_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Computes selected-token F32 logprobs and ranks from F32 logits.
    pub fn selected_token_logprobs_f32(
        &self,
        logits: &impl CudaDeviceRead<f32>,
        token_ids: &impl CudaDeviceRead<i64>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: SelectedTokenLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_selected_dtype(spec, DType::F32)?;
        let (rows, vocab_size, row_stride) =
            validate_selected_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_selected_token_logprobs_f32(
                logits.as_ptr(),
                token_ids.as_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Computes selected-token F32 logprobs and ranks from FP16 logits.
    pub fn selected_token_logprobs_f16(
        &self,
        logits: &impl CudaDeviceRead<f16>,
        token_ids: &impl CudaDeviceRead<i64>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: SelectedTokenLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_selected_dtype(spec, DType::F16)?;
        let (rows, vocab_size, row_stride) =
            validate_selected_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_selected_token_logprobs_f16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Computes selected-token F32 logprobs and ranks from BF16 logits.
    pub fn selected_token_logprobs_bf16(
        &self,
        logits: &impl CudaDeviceRead<bf16>,
        token_ids: &impl CudaDeviceRead<i64>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: SelectedTokenLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_selected_dtype(spec, DType::Bf16)?;
        let (rows, vocab_size, row_stride) =
            validate_selected_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_selected_token_logprobs_bf16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Applies exact per-row top-k filtering to F32 logits in place.
    pub fn top_k_filter_f32(
        &self,
        logits: &mut impl CudaDeviceWrite<f32>,
        top_ks: &impl CudaDeviceRead<i32>,
        workspace: &mut impl CudaDeviceWrite<u32>,
        spec: TopKFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_top_k_filter_dtype(spec, DType::F32)?;
        let launch = validate_top_k_filter_buffers(logits, top_ks, workspace, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_top_k_filter_f32(
                logits.as_mut_ptr(),
                top_ks.as_ptr(),
                workspace.as_mut_ptr(),
                launch.workspace_elements,
                launch.rows,
                launch.vocab_size,
                launch.row_stride,
                launch.partitions,
                self.raw_stream(),
            )
        })
    }

    /// Applies exact per-row top-k filtering to FP16 logits in place.
    pub fn top_k_filter_f16(
        &self,
        logits: &mut impl CudaDeviceWrite<f16>,
        top_ks: &impl CudaDeviceRead<i32>,
        workspace: &mut impl CudaDeviceWrite<u32>,
        spec: TopKFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_top_k_filter_dtype(spec, DType::F16)?;
        let launch = validate_top_k_filter_buffers(logits, top_ks, workspace, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_top_k_filter_f16(
                logits.as_mut_ptr().cast::<u16>(),
                top_ks.as_ptr(),
                workspace.as_mut_ptr(),
                launch.workspace_elements,
                launch.rows,
                launch.vocab_size,
                launch.row_stride,
                launch.partitions,
                self.raw_stream(),
            )
        })
    }

    /// Applies exact per-row top-k filtering to BF16 logits in place.
    pub fn top_k_filter_bf16(
        &self,
        logits: &mut impl CudaDeviceWrite<bf16>,
        top_ks: &impl CudaDeviceRead<i32>,
        workspace: &mut impl CudaDeviceWrite<u32>,
        spec: TopKFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_top_k_filter_dtype(spec, DType::Bf16)?;
        let launch = validate_top_k_filter_buffers(logits, top_ks, workspace, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_top_k_filter_bf16(
                logits.as_mut_ptr().cast::<u16>(),
                top_ks.as_ptr(),
                workspace.as_mut_ptr(),
                launch.workspace_elements,
                launch.rows,
                launch.vocab_size,
                launch.row_stride,
                launch.partitions,
                self.raw_stream(),
            )
        })
    }

    /// Fuses F32 normalization, sampled-token rank, and deterministic top-k.
    #[allow(clippy::too_many_arguments)]
    pub fn topk_sampled_logprobs_f32(
        &self,
        logits: &impl CudaDeviceRead<f32>,
        sampled_token_ids: &impl CudaDeviceRead<i64>,
        output_token_ids: &mut impl CudaDeviceWrite<i32>,
        output_logprobs: &mut impl CudaDeviceWrite<f32>,
        sampled_token_ranks: &mut impl CudaDeviceWrite<i64>,
        workspace: &mut impl CudaDeviceWrite<u8>,
        spec: TopKSampledLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_topk_dtype(spec, DType::F32)?;
        let launch = validate_topk_buffers(
            logits,
            sampled_token_ids,
            output_token_ids,
            output_logprobs,
            sampled_token_ranks,
            workspace,
            spec,
            layout,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_topk_sampled_logprobs_f32(
                logits.as_ptr(),
                sampled_token_ids.as_ptr(),
                output_token_ids.as_mut_ptr(),
                output_logprobs.as_mut_ptr(),
                sampled_token_ranks.as_mut_ptr(),
                launch.rows,
                launch.vocab_size,
                launch.top_k,
                launch.row_stride,
                workspace.as_mut_ptr(),
                launch.workspace_bytes,
                launch.partitions,
                self.raw_stream(),
            )
        })
    }

    /// Fuses FP16 normalization, sampled-token rank, and deterministic top-k.
    #[allow(clippy::too_many_arguments)]
    pub fn topk_sampled_logprobs_f16(
        &self,
        logits: &impl CudaDeviceRead<f16>,
        sampled_token_ids: &impl CudaDeviceRead<i64>,
        output_token_ids: &mut impl CudaDeviceWrite<i32>,
        output_logprobs: &mut impl CudaDeviceWrite<f32>,
        sampled_token_ranks: &mut impl CudaDeviceWrite<i64>,
        workspace: &mut impl CudaDeviceWrite<u8>,
        spec: TopKSampledLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_topk_dtype(spec, DType::F16)?;
        let launch = validate_topk_buffers(
            logits,
            sampled_token_ids,
            output_token_ids,
            output_logprobs,
            sampled_token_ranks,
            workspace,
            spec,
            layout,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_topk_sampled_logprobs_f16(
                logits.as_ptr().cast::<u16>(),
                sampled_token_ids.as_ptr(),
                output_token_ids.as_mut_ptr(),
                output_logprobs.as_mut_ptr(),
                sampled_token_ranks.as_mut_ptr(),
                launch.rows,
                launch.vocab_size,
                launch.top_k,
                launch.row_stride,
                workspace.as_mut_ptr(),
                launch.workspace_bytes,
                launch.partitions,
                self.raw_stream(),
            )
        })
    }

    /// Fuses BF16 normalization, sampled-token rank, and deterministic top-k.
    #[allow(clippy::too_many_arguments)]
    pub fn topk_sampled_logprobs_bf16(
        &self,
        logits: &impl CudaDeviceRead<bf16>,
        sampled_token_ids: &impl CudaDeviceRead<i64>,
        output_token_ids: &mut impl CudaDeviceWrite<i32>,
        output_logprobs: &mut impl CudaDeviceWrite<f32>,
        sampled_token_ranks: &mut impl CudaDeviceWrite<i64>,
        workspace: &mut impl CudaDeviceWrite<u8>,
        spec: TopKSampledLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_topk_dtype(spec, DType::Bf16)?;
        let launch = validate_topk_buffers(
            logits,
            sampled_token_ids,
            output_token_ids,
            output_logprobs,
            sampled_token_ranks,
            workspace,
            spec,
            layout,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_topk_sampled_logprobs_bf16(
                logits.as_ptr().cast::<u16>(),
                sampled_token_ids.as_ptr(),
                output_token_ids.as_mut_ptr(),
                output_logprobs.as_mut_ptr(),
                sampled_token_ranks.as_mut_ptr(),
                launch.rows,
                launch.vocab_size,
                launch.top_k,
                launch.row_stride,
                workspace.as_mut_ptr(),
                launch.workspace_bytes,
                launch.partitions,
                self.raw_stream(),
            )
        })
    }

    /// Applies sparse F32 repetition, frequency, and presence penalties.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_token_penalties_f32(
        &self,
        logits: &mut impl CudaDeviceWrite<f32>,
        prompt_token_ids: &impl CudaDeviceRead<i64>,
        output_token_ids: &impl CudaDeviceRead<i64>,
        presence_penalties: &impl CudaDeviceRead<f32>,
        frequency_penalties: &impl CudaDeviceRead<f32>,
        repetition_penalties: &impl CudaDeviceRead<f32>,
        workspace: &mut impl CudaDeviceWrite<u64>,
        spec: TokenPenaltiesSpec,
        logits_layout: RowStridedLayout,
        prompt_layout: RowStridedLayout,
        output_layout: RowStridedLayout,
        workspace_layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        let launch = validate_token_penalty_buffers(
            logits,
            prompt_token_ids,
            output_token_ids,
            presence_penalties,
            frequency_penalties,
            repetition_penalties,
            workspace,
            spec,
            logits_layout,
            prompt_layout,
            output_layout,
            workspace_layout,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_apply_token_penalties_f32(
                logits.as_mut_ptr(),
                prompt_token_ids.as_ptr(),
                output_token_ids.as_ptr(),
                presence_penalties.as_ptr(),
                frequency_penalties.as_ptr(),
                repetition_penalties.as_ptr(),
                workspace.as_mut_ptr(),
                launch.rows,
                launch.vocab_size,
                launch.prompt_tokens,
                launch.output_tokens,
                launch.workspace_capacity,
                launch.logits_row_stride,
                launch.prompt_row_stride,
                launch.output_row_stride,
                launch.workspace_row_stride,
                self.raw_stream(),
            )
        })
    }
}

struct TokenPenaltiesLaunch {
    rows: u32,
    vocab_size: u32,
    prompt_tokens: u32,
    output_tokens: u32,
    workspace_capacity: u32,
    logits_row_stride: u64,
    prompt_row_stride: u64,
    output_row_stride: u64,
    workspace_row_stride: u64,
}

struct TopKSampledLogprobsLaunch {
    rows: u32,
    vocab_size: u32,
    top_k: u32,
    row_stride: u64,
    workspace_bytes: u64,
    partitions: u32,
}

struct TopKFilterLaunch {
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    workspace_elements: u64,
    partitions: u32,
}

fn require_top_k_filter_dtype(
    spec: TopKFilterSpec,
    expected: DType,
) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "top-k filtering for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_top_k_filter_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    top_ks: &impl CudaDeviceRead<i32>,
    workspace: &impl CudaDeviceRead<u32>,
    spec: TopKFilterSpec,
    layout: RowStridedLayout,
) -> Result<TopKFilterLaunch, CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "top-k filter logits",
    )?;
    top_ks.require_len(spec.rows(), "top-k filter values")?;
    workspace.require_len(spec.workspace_elements(), "top-k filter workspace")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("top-k filter rows exceed the CUDA ABI".into())
    })?;
    let partitions = u32::try_from(spec.workspace_partitions()).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "top-k filter workspace partitions exceed the CUDA ABI".into(),
        )
    })?;
    if rows > (i32::MAX as u32) / partitions {
        return Err(CudaExecutorError::InvalidContract(
            "top-k filter launch grid exceeds the CUDA ABI".into(),
        ));
    }
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("top-k filter vocabulary exceeds the CUDA ABI".into())
    })?;
    if vocab_size > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "top-k filter vocabulary exceeds int32 metadata".into(),
        ));
    }
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract("top-k filter row stride exceeds the CUDA ABI".into())
    })?;
    let workspace_elements = u64::try_from(spec.workspace_elements()).map_err(|_| {
        CudaExecutorError::InvalidContract("top-k filter workspace exceeds the CUDA ABI".into())
    })?;
    Ok(TopKFilterLaunch {
        rows,
        vocab_size,
        row_stride,
        workspace_elements,
        partitions,
    })
}

fn require_topk_dtype(
    spec: TopKSampledLogprobsSpec,
    expected: DType,
) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "top-k sampled logprobs for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_topk_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    sampled_token_ids: &impl CudaDeviceRead<i64>,
    output_token_ids: &impl CudaDeviceRead<i32>,
    output_logprobs: &impl CudaDeviceRead<f32>,
    sampled_token_ranks: &impl CudaDeviceRead<i64>,
    workspace: &impl CudaDeviceRead<u8>,
    spec: TopKSampledLogprobsSpec,
    layout: RowStridedLayout,
) -> Result<TopKSampledLogprobsLaunch, CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "top-k sampled-logprob logits",
    )?;
    sampled_token_ids.require_len(spec.rows(), "sampled token IDs")?;
    output_token_ids.require_len(spec.output_numel(), "top-k output token IDs")?;
    output_logprobs.require_len(spec.output_numel(), "top-k output logprobs")?;
    sampled_token_ranks.require_len(spec.rows(), "sampled token ranks")?;
    workspace.require_len(spec.workspace_bytes(), "top-k sampled-logprob workspace")?;
    if !(workspace.as_ptr() as usize).is_multiple_of(align_of::<f32>()) {
        return Err(CudaExecutorError::InvalidContract(
            "top-k sampled-logprob workspace must be aligned to 4 bytes".into(),
        ));
    }

    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("top-k sampled-logprob rows exceed the CUDA ABI".into())
    })?;
    if rows > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "top-k sampled-logprob rows exceed the CUDA grid".into(),
        ));
    }
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "top-k sampled-logprob vocabulary exceeds the CUDA ABI".into(),
        )
    })?;
    if vocab_size > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "top-k sampled-logprob vocabulary exceeds int32 token IDs".into(),
        ));
    }
    let top_k = u32::try_from(spec.top_k()).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "top-k sampled-logprob width exceeds the CUDA ABI".into(),
        )
    })?;
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "top-k sampled-logprob row stride exceeds the CUDA ABI".into(),
        )
    })?;
    let workspace_bytes = u64::try_from(spec.workspace_bytes()).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "top-k sampled-logprob workspace exceeds the CUDA ABI".into(),
        )
    })?;
    let partitions = u32::try_from(spec.workspace_partitions()).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "top-k sampled-logprob partitions exceed the CUDA ABI".into(),
        )
    })?;
    if rows > i32::MAX as u32 / partitions {
        return Err(CudaExecutorError::InvalidContract(
            "top-k sampled-logprob partition grid exceeds CUDA grid.x".into(),
        ));
    }
    Ok(TopKSampledLogprobsLaunch {
        rows,
        vocab_size,
        top_k,
        row_stride,
        workspace_bytes,
        partitions,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_token_penalty_buffers(
    logits: &impl CudaDeviceRead<f32>,
    prompt_token_ids: &impl CudaDeviceRead<i64>,
    output_token_ids: &impl CudaDeviceRead<i64>,
    presence_penalties: &impl CudaDeviceRead<f32>,
    frequency_penalties: &impl CudaDeviceRead<f32>,
    repetition_penalties: &impl CudaDeviceRead<f32>,
    workspace: &impl CudaDeviceRead<u64>,
    spec: TokenPenaltiesSpec,
    logits_layout: RowStridedLayout,
    prompt_layout: RowStridedLayout,
    output_layout: RowStridedLayout,
    workspace_layout: RowStridedLayout,
) -> Result<TokenPenaltiesLaunch, CudaExecutorError> {
    logits.require_len(
        logits_layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "token-penalty logits",
    )?;
    prompt_token_ids.require_len(
        prompt_layout.storage_elements(spec.rows(), spec.prompt_tokens())?,
        "token-penalty prompt IDs",
    )?;
    output_token_ids.require_len(
        output_layout.storage_elements(spec.rows(), spec.output_tokens())?,
        "token-penalty output IDs",
    )?;
    presence_penalties.require_len(spec.rows(), "presence penalties")?;
    frequency_penalties.require_len(spec.rows(), "frequency penalties")?;
    repetition_penalties.require_len(spec.rows(), "repetition penalties")?;
    workspace.require_len(
        workspace_layout.storage_elements(spec.rows(), spec.workspace_capacity())?,
        "token-penalty workspace",
    )?;

    let convert_u32 = |value: usize, name: &str| {
        u32::try_from(value).map_err(|_| {
            CudaExecutorError::InvalidContract(format!("token-penalty {name} exceeds the CUDA ABI"))
        })
    };
    let convert_u64 = |value: usize, name: &str| {
        u64::try_from(value).map_err(|_| {
            CudaExecutorError::InvalidContract(format!("token-penalty {name} exceeds the CUDA ABI"))
        })
    };
    let rows = convert_u32(spec.rows(), "rows")?;
    if rows > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "token-penalty rows exceed the CUDA grid".into(),
        ));
    }
    let vocab_size = convert_u32(spec.vocab_size(), "vocabulary")?;
    if vocab_size > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "token-penalty vocabulary exceeds int32 hash keys".into(),
        ));
    }
    let output_tokens = convert_u32(spec.output_tokens(), "output width")?;
    if output_tokens > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "token-penalty output count exceeds the packed hash state".into(),
        ));
    }

    Ok(TokenPenaltiesLaunch {
        rows,
        vocab_size,
        prompt_tokens: convert_u32(spec.prompt_tokens(), "prompt width")?,
        output_tokens,
        workspace_capacity: convert_u32(spec.workspace_capacity(), "workspace capacity")?,
        logits_row_stride: convert_u64(logits_layout.row_stride(), "logits row stride")?,
        prompt_row_stride: convert_u64(prompt_layout.row_stride(), "prompt row stride")?,
        output_row_stride: convert_u64(output_layout.row_stride(), "output row stride")?,
        workspace_row_stride: convert_u64(workspace_layout.row_stride(), "workspace row stride")?,
    })
}

fn require_dtype(spec: GreedySampleLogprobsSpec, expected: DType) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "greedy sampling for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    token_ids: &impl CudaDeviceRead<i32>,
    logprobs: &impl CudaDeviceRead<f32>,
    ranks: &impl CudaDeviceRead<i64>,
    spec: GreedySampleLogprobsSpec,
    layout: RowStridedLayout,
) -> Result<(u32, u32, u64), CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "greedy-sampling logits",
    )?;
    token_ids.require_len(spec.rows(), "greedy-sampling token IDs")?;
    logprobs.require_len(spec.rows(), "greedy-sampling logprobs")?;
    ranks.require_len(spec.rows(), "greedy-sampling ranks")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("greedy-sampling rows exceed the CUDA ABI".into())
    })?;
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("greedy-sampling vocabulary exceeds the CUDA ABI".into())
    })?;
    if vocab_size > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "greedy-sampling vocabulary exceeds int32 token IDs".into(),
        ));
    }
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract("greedy-sampling row stride exceeds the CUDA ABI".into())
    })?;
    Ok((rows, vocab_size, row_stride))
}

fn require_selected_dtype(
    spec: SelectedTokenLogprobsSpec,
    expected: DType,
) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "selected-token logprobs for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_selected_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    token_ids: &impl CudaDeviceRead<i64>,
    logprobs: &impl CudaDeviceRead<f32>,
    ranks: &impl CudaDeviceRead<i64>,
    spec: SelectedTokenLogprobsSpec,
    layout: RowStridedLayout,
) -> Result<(u32, u32, u64), CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "selected-token logits",
    )?;
    token_ids.require_len(spec.rows(), "selected token IDs")?;
    logprobs.require_len(spec.rows(), "selected-token logprobs")?;
    ranks.require_len(spec.rows(), "selected-token ranks")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("selected-token rows exceed the CUDA ABI".into())
    })?;
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("selected-token vocabulary exceeds the CUDA ABI".into())
    })?;
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract("selected-token row stride exceeds the CUDA ABI".into())
    })?;
    Ok((rows, vocab_size, row_stride))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{DeviceBuffer, DeviceSliceMut};
    use loom_kernels::{
        apply_token_penalties_f32_reference, greedy_sample_logprobs_f32_reference,
        selected_token_logprobs_f32_reference, top_k_filter_f32_reference,
        topk_sampled_logprobs_f32_reference,
    };

    #[test]
    fn safe_rust_wrapper_matches_the_cpu_oracle() {
        let spec = GreedySampleLogprobsSpec::new(2, 5, DType::F32).unwrap();
        let logits = [1.0_f32, 3.0, 3.0, -1.0, 0.5, -2.0, -1.0, 2.0, 0.0, 1.0];
        let mut expected_ids = [u32::MAX; 2];
        let mut expected_logprobs = [0.0_f32; 2];
        greedy_sample_logprobs_f32_reference(
            &logits,
            &mut expected_ids,
            &mut expected_logprobs,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let logits_device = DeviceBuffer::from_slice(&logits).unwrap();
        let mut ids_device = DeviceBuffer::from_slice(&[-1_i32; 2]).unwrap();
        let mut logprobs_device = DeviceBuffer::from_slice(&[0.0_f32; 2]).unwrap();
        let mut ranks_device = DeviceBuffer::from_slice(&[0_i64; 2]).unwrap();
        backend
            .greedy_sample_logprobs_f32(
                &logits_device,
                &mut ids_device,
                &mut logprobs_device,
                &mut ranks_device,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        let actual_ids = ids_device.copy_to_vec().unwrap();
        assert_eq!(actual_ids, expected_ids.map(|value| value as i32));
        for (actual, expected) in logprobs_device
            .copy_to_vec()
            .unwrap()
            .iter()
            .zip(expected_logprobs)
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        assert_eq!(ranks_device.copy_to_vec().unwrap(), vec![2_i64, 1_i64]);
    }

    #[test]
    fn selected_token_wrapper_matches_the_cpu_oracle() {
        let spec = SelectedTokenLogprobsSpec::new(2, 5, DType::F32).unwrap();
        let logits = [1.0_f32, 3.0, 3.0, -1.0, 0.5, -2.0, -1.0, 2.0, 0.0, 1.0];
        let token_ids = [0_i64, 4_i64];
        let mut expected_logprobs = [0.0_f32; 2];
        let mut expected_ranks = [0_i64; 2];
        selected_token_logprobs_f32_reference(
            &logits,
            &token_ids,
            &mut expected_logprobs,
            &mut expected_ranks,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let logits_device = DeviceBuffer::from_slice(&logits).unwrap();
        let ids_device = DeviceBuffer::from_slice(&token_ids).unwrap();
        let mut logprobs_device = DeviceBuffer::from_slice(&[0.0_f32; 2]).unwrap();
        let mut ranks_device = DeviceBuffer::from_slice(&[0_i64; 2]).unwrap();
        backend
            .selected_token_logprobs_f32(
                &logits_device,
                &ids_device,
                &mut logprobs_device,
                &mut ranks_device,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        for (actual, expected) in logprobs_device
            .copy_to_vec()
            .unwrap()
            .iter()
            .zip(expected_logprobs)
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        assert_eq!(ranks_device.copy_to_vec().unwrap(), expected_ranks);
    }

    #[test]
    fn top_k_filter_wrapper_matches_the_cpu_oracle() {
        let spec = TopKFilterSpec::new(3, 5, DType::F32).unwrap();
        let source = [
            5.0_f32, 4.0, 4.0, 1.0, -1.0, //
            -2.0, 3.0, 0.0, 1.0, 2.0, //
            7.0, 7.0, 6.0, 5.0, 4.0,
        ];
        let top_ks = [2_i32, 5, 1];
        let mut expected = source;
        top_k_filter_f32_reference(&mut expected, &top_ks, spec).unwrap();

        let backend = CudaBackend::new().unwrap();
        let mut logits_device = DeviceBuffer::from_slice(&source).unwrap();
        let top_ks_device = DeviceBuffer::from_slice(&top_ks).unwrap();
        let mut workspace_device =
            DeviceBuffer::from_slice(&vec![0_u32; spec.workspace_elements()]).unwrap();
        backend
            .top_k_filter_f32(
                &mut logits_device,
                &top_ks_device,
                &mut workspace_device,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(logits_device.copy_to_vec().unwrap(), expected);
    }

    #[test]
    fn topk_sampled_logprobs_wrapper_matches_the_cpu_oracle() {
        let spec = TopKSampledLogprobsSpec::new(2, 5, 3, DType::F32).unwrap();
        let logits = [1.0_f32, 3.0, 3.0, -1.0, 0.5, -2.0, -1.0, 2.0, 0.0, 1.0];
        let sampled_token_ids = [0_i64, 4_i64];
        let mut expected_ids = [-1_i32; 8];
        let mut expected_logprobs = [0.0_f32; 8];
        let mut expected_ranks = [0_i64; 2];
        topk_sampled_logprobs_f32_reference(
            &logits,
            &sampled_token_ids,
            &mut expected_ids,
            &mut expected_logprobs,
            &mut expected_ranks,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let logits_device = DeviceBuffer::from_slice(&logits).unwrap();
        let sampled_ids_device = DeviceBuffer::from_slice(&sampled_token_ids).unwrap();
        let mut ids_device = DeviceBuffer::from_slice(&[-1_i32; 8]).unwrap();
        let mut logprobs_device = DeviceBuffer::from_slice(&[0.0_f32; 8]).unwrap();
        let mut ranks_device = DeviceBuffer::from_slice(&[0_i64; 2]).unwrap();
        let mut workspace_device =
            DeviceBuffer::from_slice(&vec![0_u8; spec.workspace_bytes()]).unwrap();
        backend
            .topk_sampled_logprobs_f32(
                &logits_device,
                &sampled_ids_device,
                &mut ids_device,
                &mut logprobs_device,
                &mut ranks_device,
                &mut workspace_device,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(ids_device.copy_to_vec().unwrap(), expected_ids);
        for (actual, expected) in logprobs_device
            .copy_to_vec()
            .unwrap()
            .iter()
            .zip(expected_logprobs)
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        assert_eq!(ranks_device.copy_to_vec().unwrap(), expected_ranks);

        let mut workspace_storage =
            DeviceBuffer::from_slice(&vec![0_u8; spec.workspace_bytes() + 1]).unwrap();
        let mut misaligned_workspace = unsafe {
            DeviceSliceMut::from_raw_parts(
                workspace_storage.as_mut_ptr().add(1),
                spec.workspace_bytes(),
            )
        }
        .unwrap();
        let error = backend
            .topk_sampled_logprobs_f32(
                &logits_device,
                &sampled_ids_device,
                &mut ids_device,
                &mut logprobs_device,
                &mut ranks_device,
                &mut misaligned_workspace,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .expect_err("misaligned workspace must be rejected");
        assert!(
            matches!(error, CudaExecutorError::InvalidContract(message) if message.contains("aligned to 4 bytes"))
        );
    }

    #[test]
    fn token_penalty_wrapper_matches_the_cpu_oracle() {
        let spec = TokenPenaltiesSpec::new(2, 6, 5, 4, 32).unwrap();
        let source = [
            2.0_f32, -2.0, 0.0, 4.0, -4.0, 1.0, -1.0, 3.0, -3.0, 2.0, -2.0, 0.5,
        ];
        let prompt = [0_i64, 1, 1, -1, 6, 1, 2, 5, 8, -1];
        let output = [1_i64, 1, 2, 6, 2, 2, 3, -1];
        let presence = [0.4_f32, 0.25];
        let frequency = [0.2_f32, -0.5];
        let repetition = [2.0_f32, 1.25];
        let mut expected = source;
        apply_token_penalties_f32_reference(
            &mut expected,
            &prompt,
            &output,
            &presence,
            &frequency,
            &repetition,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let mut logits_device = DeviceBuffer::from_slice(&source).unwrap();
        let prompt_device = DeviceBuffer::from_slice(&prompt).unwrap();
        let output_device = DeviceBuffer::from_slice(&output).unwrap();
        let presence_device = DeviceBuffer::from_slice(&presence).unwrap();
        let frequency_device = DeviceBuffer::from_slice(&frequency).unwrap();
        let repetition_device = DeviceBuffer::from_slice(&repetition).unwrap();
        let mut workspace_device =
            DeviceBuffer::from_slice(&vec![0_u64; spec.workspace_numel()]).unwrap();
        backend
            .apply_token_penalties_f32(
                &mut logits_device,
                &prompt_device,
                &output_device,
                &presence_device,
                &frequency_device,
                &repetition_device,
                &mut workspace_device,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
                RowStridedLayout::contiguous(spec.prompt_tokens()),
                RowStridedLayout::contiguous(spec.output_tokens()),
                RowStridedLayout::contiguous(spec.workspace_capacity()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(logits_device.copy_to_vec().unwrap(), expected);
    }
}
