//! Valid representative metadata for compiler measurement and startup preparation.
//!
//! This fixture obeys the decoder metadata ABI but does not model a request
//! distribution or shared-prefix page placement. Runtime metadata is supplied
//! by the KV manager. Startup may use these descriptors to prepare graphs, but
//! must never execute the model against manager-owned state with them.

// OrbitKV symbols are stable interned dimension identities.
#![allow(clippy::mutable_key_type)]

use orbitkv_compiler::prelude::{DynMap, NodeIndex, Symbol};
use orbitkv_cuda::runtime::CudaRuntime;

use super::DecoderCompileConfig;

const INDEX_BYTES: usize = std::mem::size_of::<i32>();
use crate::model::{DecoderClassDimensions, DecoderGraph};

/// The graph binding and maximum allocation belong together. Compiler setup
/// must not infer capacities by inspecting node identities or attention classes.
#[derive(Clone, Copy)]
struct RepresentativeTensor {
    id: NodeIndex,
    capacity_bytes: usize,
}

impl RepresentativeTensor {
    fn new(id: NodeIndex, maximum_elements: usize) -> Self {
        Self {
            id,
            capacity_bytes: maximum_elements
                .checked_mul(INDEX_BYTES)
                .expect("validated profile allocation size"),
        }
    }
}

pub(in crate::model) struct RepresentativeInputs {
    token_ids: RepresentativeTensor,
    positions: RepresentativeTensor,
    query_indptr: RepresentativeTensor,
    classes: Vec<(RepresentativeClassInputs, DecoderClassDimensions)>,
    fixed_slots: Vec<RepresentativeTensor>,
    page_tokens: usize,
}

struct RepresentativeClassInputs {
    write_slots: RepresentativeTensor,
    page_indices: RepresentativeTensor,
    page_indptr: RepresentativeTensor,
    last_page_len: RepresentativeTensor,
}

/// Split query rows against the tightest per-row page capacity shared by all
/// attention classes. Each nonempty request receives at least one token.
fn query_rows(tokens: usize, batch: usize, pages: usize, page_tokens: usize) -> Vec<usize> {
    assert!(tokens >= batch && pages >= batch && tokens <= pages * page_tokens);
    let mut remaining = tokens - batch;
    (0..batch)
        .map(|row| {
            let row_pages = pages / batch + usize::from(row < pages % batch);
            let extra = remaining.min(row_pages * page_tokens - 1);
            remaining -= extra;
            1 + extra
        })
        .collect()
}

fn indptr(lengths: &[usize]) -> Vec<i32> {
    std::iter::once(0)
        .chain(lengths.iter().scan(0, |sum, &length| {
            *sum += length;
            Some(i32::try_from(*sum).expect("validated profile index range"))
        }))
        .collect()
}

impl RepresentativeInputs {
    /// Captures graph bindings after compile geometry and arena validation.
    pub(in crate::model) fn new(
        decoder: &DecoderGraph,
        compile: DecoderCompileConfig,
        page_tokens: usize,
    ) -> Self {
        let query_capacity = compile.maximum_query_tokens;
        let batch_capacity = compile.maximum_batch_size;
        // CSR stores a terminal offset in addition to one offset per request.
        let indptr_capacity = batch_capacity + 1;
        Self {
            token_ids: RepresentativeTensor::new(decoder.inputs.token_ids.id, query_capacity),
            positions: RepresentativeTensor::new(decoder.inputs.positions.id, query_capacity),
            query_indptr: RepresentativeTensor::new(
                decoder.inputs.query_indptr.id,
                indptr_capacity,
            ),
            classes: decoder
                .inputs
                .classes
                .iter()
                .map(|class| RepresentativeClassInputs {
                    write_slots: RepresentativeTensor::new(class.write_slots.id, query_capacity),
                    page_indices: RepresentativeTensor::new(
                        class.attention.page_indices.id,
                        compile.maximum_context_pages,
                    ),
                    page_indptr: RepresentativeTensor::new(
                        class.attention.page_indptr.id,
                        indptr_capacity,
                    ),
                    last_page_len: RepresentativeTensor::new(
                        class.attention.last_page_len.id,
                        batch_capacity,
                    ),
                })
                .zip(decoder.class_dimensions.iter().copied())
                .collect(),
            fixed_slots: decoder
                .outputs
                .fixed_states
                .iter()
                .map(|state| {
                    RepresentativeTensor::new(state.binding.destination_slots.id, batch_capacity)
                })
                .collect(),
            page_tokens,
        }
    }

    /// Keep maximum allocations stable while each candidate updates only its
    /// logical contents. The callback owns the fixture independently of setup.
    pub(in crate::model) fn install(self, runtime: &mut CudaRuntime, representative: &DynMap) {
        for (input, values) in self.tensor_values(representative) {
            runtime.set_data_with_capacity(input.id, values, input.capacity_bytes);
        }
        runtime.register_profile_input_generator(move |dims| self.values(dims));
    }

    pub(in crate::model) fn values(&self, dims: &DynMap) -> Vec<(NodeIndex, Vec<i32>)> {
        self.tensor_values(dims)
            .into_iter()
            .map(|(input, values)| (input.id, values))
            .collect()
    }

    fn tensor_values(&self, dims: &DynMap) -> Vec<(RepresentativeTensor, Vec<i32>)> {
        let s = dims[&Symbol::from('s')];
        let b = dims[&Symbol::from('b')];
        let tightest_pages = self
            .classes
            .iter()
            .map(|(_, class)| dims[&class.context_pages])
            .min()
            .expect("validated decoder has token-attention classes");
        let queries = query_rows(s, b, tightest_pages, self.page_tokens);
        let mut positions = Vec::with_capacity(s);
        let mut values = vec![
            // Zero is a valid vocabulary index; values seed profiling only.
            (self.token_ids, vec![0; s]),
            (self.query_indptr, indptr(&queries)),
        ];
        for (class_index, (inputs, class)) in self.classes.iter().enumerate() {
            let pages = dims[&class.context_pages];
            assert!(pages <= class.page_count as usize);
            let row_pages = (0..b)
                .map(|row| pages / b + usize::from(row < pages % b))
                .collect::<Vec<_>>();
            let page_indptr = indptr(&row_pages);
            let base_page = usize::try_from(class.backend_base_index)
                .expect("validated backend page index range");
            let mut slots = Vec::with_capacity(s);
            for row in 0..b {
                // Full last pages provide a deterministic legal context. This
                // is a fixture policy, not a serving attention assumption.
                let context_tokens = row_pages[row] * self.page_tokens;
                let start = context_tokens - queries[row];
                if class_index == 0 {
                    positions.extend((start..context_tokens).map(|value| {
                        i32::try_from(value).expect("validated backend token index range")
                    }));
                }
                let base = (base_page
                    + usize::try_from(page_indptr[row]).expect("validated profile index range"))
                    * self.page_tokens;
                slots.extend((start..context_tokens).map(|offset| {
                    i32::try_from(base + offset).expect("validated backend slot index range")
                }));
            }
            values.extend([
                (inputs.write_slots, slots),
                (
                    inputs.page_indices,
                    (base_page..base_page + pages)
                        .map(|page| {
                            i32::try_from(page).expect("validated backend page index range")
                        })
                        .collect(),
                ),
                (inputs.page_indptr, page_indptr),
                (
                    inputs.last_page_len,
                    vec![i32::try_from(self.page_tokens).expect("validated page token range"); b],
                ),
            ]);
        }
        values.push((self.positions, positions));
        values.extend(self.fixed_slots.iter().map(|&input| {
            (
                input,
                (0..b)
                    .map(|slot| {
                        i32::try_from(slot).expect("validated fixed-state slot index range")
                    })
                    .collect(),
            )
        }));
        values
    }
}
