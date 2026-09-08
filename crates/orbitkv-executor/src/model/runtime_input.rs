use super::{
    DecodeCaptureSignature, DecodeClassCaptureSignature, DecoderClassDimensions, DecoderClassStep,
    DecoderCompileConfig, DecoderError, DecoderStep,
};

impl DecodeCaptureSignature {
    pub(super) fn from_step(step: DecoderStep<'_>) -> Result<Self, DecoderError> {
        let Some(first_class) = step.classes.first() else {
            return Err(DecoderError::CaptureRequiresDecode);
        };
        let batch_size = first_class
            .attention
            .query_indptr
            .len()
            .checked_sub(1)
            .ok_or(DecoderError::InputCapacity)?;
        let decode_indptr = first_class.attention.query_indptr.first() == Some(&0)
            && first_class.attention.query_indptr.last().copied() == i32::try_from(batch_size).ok()
            && first_class
                .attention
                .query_indptr
                .windows(2)
                .all(|row| row[1] == row[0] + 1);
        if batch_size == 0
            || step.tokens.len() != batch_size
            || step.positions.len() != batch_size
            || !decode_indptr
            || step.classes.iter().any(|class| {
                class.write_slots.len() != batch_size
                    || class.attention.query_indptr != first_class.attention.query_indptr
            })
        {
            return Err(DecoderError::CaptureRequiresDecode);
        }
        Ok(Self {
            query_tokens: step.tokens.len(),
            batch_size,
            classes: step
                .classes
                .iter()
                .map(|class| DecodeClassCaptureSignature {
                    class_id: class.class_id,
                    context_pages: class.attention.page_indices.len(),
                    page_tokens: class.attention.page_tokens,
                    query_indptr: class.attention.query_indptr.clone(),
                    page_indptr: class.attention.page_indptr.clone(),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    }
}

pub(super) fn validate_step(
    step: DecoderStep<'_>,
    compile: DecoderCompileConfig,
    classes: &[DecoderClassDimensions],
    page_tokens: usize,
    vocabulary_size: usize,
) -> Result<(), DecoderError> {
    let Some(first_class) = step.classes.first() else {
        return Err(DecoderError::InputCapacity);
    };
    let batch = first_class
        .attention
        .query_indptr
        .len()
        .checked_sub(1)
        .ok_or(DecoderError::InputCapacity)?;
    if step.tokens.is_empty()
        || step.classes.len() != classes.len()
        || step.tokens.len() != step.positions.len()
        || step.tokens.len() > compile.maximum_query_tokens
        || step
            .tokens
            .iter()
            .any(|&token| usize::try_from(token).map_or(true, |token| token >= vocabulary_size))
        || batch == 0
        || batch > compile.maximum_batch_size
    {
        return Err(DecoderError::InputCapacity);
    }
    for (class_step, class) in step.classes.iter().zip(classes) {
        if !valid_class_step(
            class_step,
            class,
            step.tokens.len(),
            batch,
            compile,
            page_tokens,
        ) || class_step.attention.query_indptr != first_class.attention.query_indptr
        {
            return Err(DecoderError::InputCapacity);
        }
    }
    Ok(())
}

fn valid_class_step(
    step: &DecoderClassStep<'_>,
    class: &DecoderClassDimensions,
    query_tokens: usize,
    batch: usize,
    compile: DecoderCompileConfig,
    page_tokens: usize,
) -> bool {
    let execution_page_tokens = match usize::try_from(step.attention.page_tokens) {
        Ok(1) if class.supports_token_selection => 1,
        Ok(value) if value == class.storage_page_tokens => value,
        _ => return false,
    };
    let Some(expansion) = class
        .storage_page_tokens
        .checked_div(execution_page_tokens)
        .filter(|ratio| *ratio > 0)
    else {
        return false;
    };
    let expansion_u64 = u64::try_from(expansion).unwrap_or(u64::MAX);
    let Some(page_begin) = class.backend_base_index.checked_mul(expansion_u64) else {
        return false;
    };
    let Some(page_end) = class
        .backend_base_index
        .checked_add(u64::from(class.page_count))
        .and_then(|end| end.checked_mul(expansion_u64))
    else {
        return false;
    };
    let Some(slot_begin) = class.backend_base_index.checked_mul(page_tokens as u64) else {
        return false;
    };
    let Some(slot_end) = class
        .backend_base_index
        .checked_add(u64::from(class.page_count))
        .and_then(|end| end.checked_mul(page_tokens as u64))
    else {
        return false;
    };
    step.class_id == class.class_id
        && step.attention.class_id == class.class_id
        && step.write_slots.len() == query_tokens
        && step
            .write_slots
            .iter()
            .all(|&slot| slot >= slot_begin && slot < slot_end)
        && !step.attention.page_indices.is_empty()
        && step.attention.page_indices.len()
            <= compile.maximum_context_pages.saturating_mul(expansion)
        && step.attention.page_indices.iter().all(|&page| {
            u64::try_from(page).is_ok_and(|page| page >= page_begin && page < page_end)
        })
        && step.attention.query_indptr.first() == Some(&0)
        && step.attention.query_indptr.last().copied() == i32::try_from(query_tokens).ok()
        && step.attention.page_indptr.first() == Some(&0)
        && step.attention.page_indptr.last().copied()
            == i32::try_from(step.attention.page_indices.len()).ok()
        && step.attention.page_indptr.len() == batch + 1
        && step.attention.last_page_len.len() == batch
        && step.attention.last_page_len.iter().all(|&tokens| {
            tokens > 0
                && usize::try_from(tokens).is_ok_and(|tokens| tokens <= execution_page_tokens)
        })
        && step
            .attention
            .query_indptr
            .windows(2)
            .all(|row| row[0] < row[1])
        && step
            .attention
            .page_indptr
            .windows(2)
            .all(|row| row[0] < row[1])
}
