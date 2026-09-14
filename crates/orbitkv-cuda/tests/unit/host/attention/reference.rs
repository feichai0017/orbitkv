//! Independent CPU softmax reference and page fixtures shared by providers.
use crate::host::DeviceBuffer;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::{bf16, f16};
use orbitkv_compiler::{
    dtype::DType,
    prelude::{DynMap, FxHashMap, NodeIndex},
};
use std::sync::Arc;

pub(crate) struct Case {
    pub head_dim: usize,
    pub page_size: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub query_tokens: usize,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub page_indices: Vec<i32>,
    pub query_indptr: Vec<i32>,
    pub page_indptr: Vec<i32>,
    pub last_page_len: Vec<i32>,
}

impl Case {
    pub fn new(head_dim: usize, page_size: usize, queries: &[usize], lengths: &[usize]) -> Self {
        assert_eq!(queries.len(), lengths.len());
        let query_heads = 6;
        let kv_heads = 2;
        let query_tokens: usize = queries.iter().sum();
        let mut query_indptr = vec![0];
        let mut page_indptr = vec![0];
        for (&q, &length) in queries.iter().zip(lengths) {
            assert!(q > 0 && q <= length);
            query_indptr.push(query_indptr.last().unwrap() + q as i32);
            page_indptr.push(page_indptr.last().unwrap() + length.div_ceil(page_size) as i32);
        }
        let pages = *page_indptr.last().unwrap() as usize;
        let values = |n: usize, period: usize, divisor: f32| {
            (0..n)
                .map(|i| ((i % period) as f32 - (period / 2) as f32) / divisor)
                .collect()
        };
        Self {
            head_dim,
            page_size,
            query_heads,
            kv_heads,
            query_tokens,
            query: values(query_tokens * query_heads * head_dim, 29, 32.0),
            key: values(pages * page_size * kv_heads * head_dim, 31, 64.0),
            value: values(pages * page_size * kv_heads * head_dim, 37, 16.0),
            page_indices: (0..pages as i32).rev().collect(),
            query_indptr,
            page_indptr,
            last_page_len: lengths
                .iter()
                .map(|n| ((n - 1) % page_size + 1) as i32)
                .collect(),
        }
    }
    pub fn scale(&self) -> f64 {
        1.0 / (self.head_dim as f64).sqrt()
    }
    pub fn dimensions(&self) -> DynMap {
        DynMap::from_iter([
            ('s'.into(), self.query_tokens),
            ('c'.into(), self.page_indices.len()),
            ('b'.into(), self.last_page_len.len()),
        ])
    }
    pub fn expected(&self, window: Option<usize>) -> Vec<f32> {
        compute_reference(
            &self.query,
            &self.key,
            &self.value,
            &self.page_indices,
            &self.query_indptr,
            &self.page_indptr,
            &self.last_page_len,
            self.head_dim,
            self.page_size,
            self.query_tokens,
            self.query_heads,
            self.kv_heads,
            self.scale() as f32,
            window,
        )
    }
    pub fn upload(&self, stream: &Arc<CudaStream>, dtype: DType) -> Buffers {
        let bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|&value| match dtype {
                    DType::Bf16 => bf16::from_f32(value).to_le_bytes(),
                    DType::F16 => f16::from_f32(value).to_le_bytes(),
                    _ => panic!("test requires 16-bit data"),
                })
                .collect::<Vec<_>>()
        };
        let mut storage = [&self.query, &self.key, &self.value]
            .into_iter()
            .map(|values| stream.clone_htod(&bytes(values)).unwrap())
            .collect::<Vec<_>>();
        for metadata in [
            &self.page_indices,
            &self.query_indptr,
            &self.page_indptr,
            &self.last_page_len,
        ] {
            let bytes = metadata
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            storage.push(stream.clone_htod(&bytes).unwrap());
        }
        storage.push(unsafe { stream.alloc::<u8>(self.query.len() * 2).unwrap() });
        let nodes = std::array::from_fn(NodeIndex::new);
        let map = nodes
            .iter()
            .zip(&storage)
            .map(|(&node, buffer)| {
                (
                    node,
                    DeviceBuffer::new(buffer.device_ptr(stream).0, buffer.len()),
                )
            })
            .collect();
        Buffers {
            storage,
            nodes,
            map,
        }
    }
}

pub(crate) struct Buffers {
    pub storage: Vec<CudaSlice<u8>>,
    pub nodes: [NodeIndex; 8],
    pub map: FxHashMap<NodeIndex, DeviceBuffer>,
}
impl Buffers {
    pub fn check(&self, stream: &Arc<CudaStream>, dtype: DType, expected: &[f32]) {
        let bytes = stream.clone_dtoh(self.storage.last().unwrap()).unwrap();
        assert_eq!(bytes.len(), expected.len() * size_of::<bf16>());
        let actual = bytes.as_chunks().0.iter().map(|&value| match dtype {
            DType::Bf16 => bf16::from_le_bytes(value).to_f32(),
            DType::F16 => f16::from_le_bytes(value).to_f32(),
            _ => unreachable!(),
        });
        for (actual, &expected) in actual.zip(expected) {
            assert!(
                actual.is_finite() && (actual - expected).abs() < 0.016,
                "actual={actual}, expected={expected}"
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    pages: &[i32],
    query_indptr: &[i32],
    page_indptr: &[i32],
    last_page_len: &[i32],
    head_dim: usize,
    page_size: usize,
    query_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    scale: f32,
    window_left: Option<usize>,
) -> Vec<f32> {
    let mut output = vec![0.0; query.len()];
    let group_size = query_heads / kv_heads;
    for request in 0..last_page_len.len() {
        let query_begin = query_indptr[request] as usize;
        let query_end = query_indptr[request + 1] as usize;
        let page_begin = page_indptr[request] as usize;
        let page_end = page_indptr[request + 1] as usize;
        let kv_tokens = (page_end - page_begin - 1) * page_size + last_page_len[request] as usize;
        for query_index in query_begin..query_end {
            let query_position = kv_tokens - (query_end - query_begin) + query_index - query_begin;
            for query_head in 0..query_heads {
                let kv_head = query_head / group_size;
                let mut scores = Vec::with_capacity(query_position + 1);
                let first_key =
                    window_left.map_or(0, |window| query_position.saturating_sub(window));
                for key_position in first_key..=query_position {
                    let logical_page = key_position / page_size;
                    let page_offset = key_position % page_size;
                    let page = pages[page_begin + logical_page] as usize;
                    let cache_base =
                        (page * page_size + page_offset) * kv_heads * head_dim + kv_head * head_dim;
                    let query_base = (query_index * query_heads + query_head) * head_dim;
                    scores.push(
                        query[query_base..query_base + head_dim]
                            .iter()
                            .zip(&key[cache_base..cache_base + head_dim])
                            .map(|(q, k)| q * k)
                            .sum::<f32>()
                            * scale,
                    );
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                for d in 0..head_dim {
                    let sum = weights
                        .iter()
                        .enumerate()
                        .map(|(index, weight)| {
                            let key_position = first_key + index;
                            let logical_page = key_position / page_size;
                            let page_offset = key_position % page_size;
                            let page = pages[page_begin + logical_page] as usize;
                            let cache = (page * page_size + page_offset) * kv_heads * head_dim
                                + kv_head * head_dim
                                + d;
                            weight * value[cache]
                        })
                        .sum::<f32>();
                    output[(query_head * query_tokens + query_index) * head_dim + d] =
                        sum / denominator;
                }
            }
        }
    }
    output
}
