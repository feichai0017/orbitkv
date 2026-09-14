//! Feasible joint shape representatives for packed decoder requests.

// Luminal symbols are stable interned dimension identities.
#![allow(clippy::mutable_key_type)]

use luminal::prelude::{CompileOptions, DimBucket, DynMap, Symbol};

use super::{
    DecoderCompileConfig, DecoderTuningProfile, FIRST_MULTI_TOKEN_QUERY, SINGLE_QUERY_TOKEN,
};
use crate::model::{DecoderError, DecoderGraph};

fn dimension_buckets(
    values: &[usize],
    fallback: usize,
    minimum: usize,
    maximum: usize,
) -> Result<Vec<DimBucket>, DecoderError> {
    let defaults = [fallback];
    let values = if values.is_empty() { &defaults } else { values };
    if values
        .iter()
        .any(|&value| !(minimum..=maximum).contains(&value))
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(DecoderError::InvalidGeometry("tuning representatives"));
    }
    Ok(values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let min = if index == 0 { minimum } else { value };
            let max = values.get(index + 1).map_or(maximum, |next| next - 1);
            DimBucket::new(min, max).representative(value)
        })
        .collect())
}

pub(in crate::model) fn decoder_compile_options(
    decoder: &DecoderGraph,
    compile: DecoderCompileConfig,
    tuning: &DecoderTuningProfile,
    page_tokens: usize,
) -> Result<CompileOptions, DecoderError> {
    compile.validate()?;
    if page_tokens == 0 {
        return Err(DecoderError::InvalidGeometry("page tokens"));
    }
    tuning.validate(compile)?;
    let mut query = vec![DimBucket::new(SINGLE_QUERY_TOKEN, SINGLE_QUERY_TOKEN)];
    query.extend(dimension_buckets(
        &tuning.prefill_tokens,
        compile.representative_prefill_tokens,
        FIRST_MULTI_TOKEN_QUERY,
        compile.maximum_query_tokens,
    )?);
    let mut batch = dimension_buckets(&tuning.batch_sizes, 1, 1, compile.maximum_batch_size)?;
    if !tuning.batch_sizes.is_empty() && batch[0].max > 1 {
        let first = batch.remove(0);
        // Preserve the singleton batch interval when explicit workload
        // representatives would otherwise merge it into a larger interval.
        let next_batch_size = first.min + 1;
        batch.insert(
            0,
            DimBucket::new(next_batch_size, first.max)
                .representative(first.representative_value().max(next_batch_size)),
        );
        batch.insert(0, DimBucket::new(first.min, first.min));
    }
    let context = dimension_buckets(
        &tuning.context_pages,
        compile.representative_context_pages,
        1,
        compile.maximum_context_pages,
    )?;
    let mut options = CompileOptions::default()
        .dim_buckets('s', &query)
        .dim_buckets('b', &batch)
        .search_graph_limit(compile.search_graphs)
        .initial_population(tuning.initial_candidates)
        .keep_best(tuning.keep_best)
        .trials(tuning.trials);
    let mut combinations = query
        .len()
        .checked_mul(batch.len())
        .ok_or(DecoderError::InvalidGeometry("tuning bucket count"))?;
    for class in &decoder.class_dimensions {
        combinations = combinations
            .checked_mul(context.len())
            .ok_or(DecoderError::InvalidGeometry("tuning bucket count"))?;
        options = options.dim_buckets(class.context_pages, &context);
    }
    // Bound allocation and egglog work before constructing the Cartesian product.
    if combinations > tuning.maximum_buckets {
        return Err(DecoderError::InvalidGeometry("tuning bucket count"));
    }
    let representatives = feasible_representatives(decoder, &options, page_tokens)?;
    if let Some(ms) = tuning.search_time_limit_ms {
        options = options.search_time_limit(std::time::Duration::from_millis(ms));
    }
    Ok(options.bucket_representatives(representatives))
}

/// Independent dimension intervals can include impossible packed batches. Keep
/// one legal point per feasible interval combination so the compiler never
/// profiles invalid CSR metadata merely to cover a Cartesian grid.
fn feasible_representatives(
    decoder: &DecoderGraph,
    options: &CompileOptions,
    page_tokens: usize,
) -> Result<Vec<DynMap>, DecoderError> {
    let mut representatives = Vec::new();
    for indices in luminal::search::bucket_index_combinations(&options.dim_buckets) {
        let bucket = |dim| &options.dim_buckets[&dim][indices[&dim]];
        let sb = bucket(Symbol::from('s'));
        let bb = bucket(Symbol::from('b'));
        // A shared context policy may extend beyond a smaller attention
        // arena. Those intervals have no runtime points in that arena.
        if decoder
            .class_dimensions
            .iter()
            .any(|class| bucket(class.context_pages).min > class.page_count as usize)
        {
            continue;
        }
        let available_pages = decoder
            .class_dimensions
            .iter()
            .map(|class| {
                let cb = bucket(class.context_pages);
                cb.max.min(class.page_count as usize)
            })
            .min()
            .ok_or(DecoderError::UnsupportedPlan)?;
        let max_batch = bb.max.min(sb.max).min(available_pages);
        let max_query = sb.max.min(available_pages.saturating_mul(page_tokens));
        if max_batch < bb.min || max_query < sb.min.max(bb.min) {
            continue;
        }
        let b = bb.representative_value().clamp(bb.min, max_batch);
        let s = sb.representative_value().clamp(sb.min.max(b), max_query);
        let mut representative = DynMap::default();
        representative.insert(Symbol::from('s'), s);
        representative.insert(Symbol::from('b'), b);
        for class in &decoder.class_dimensions {
            let cb = bucket(class.context_pages);
            let max_pages = cb.max.min(class.page_count as usize);
            let min_pages = cb.min.max(b).max(s.div_ceil(page_tokens));
            if min_pages > max_pages {
                return Err(DecoderError::InvalidGeometry(
                    "tuning context exceeds physical arena",
                ));
            }
            representative.insert(
                class.context_pages,
                cb.representative_value().clamp(min_pages, max_pages),
            );
        }
        representatives.push(representative);
    }
    if representatives.is_empty() {
        return Err(DecoderError::InvalidGeometry("no feasible tuning buckets"));
    }
    Ok(representatives)
}
