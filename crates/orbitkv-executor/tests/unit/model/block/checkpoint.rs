//! Checkpoint-backed full-attention layer diagnostics across batch geometries.

use super::*;
use crate::cuda::PagedAttentionMetadata;
use crate::model::block::{TokenAttentionInputs, TokenAttentionLayer};
use crate::model::{
    DecoderClassDimensions, DecoderConfig, DecoderDimensions, DecoderWeightFeatures,
};
use crate::{AttentionBatch, AttentionClass, AttentionVisibility};
use half::bf16;
use orbitkv_compiler::{
    dtype::DType,
    prelude::{DimBucket, GraphTensor},
};
use orbitkv_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
};

#[test]
#[ignore = "requires CUDA, a checkpoint and independent full-attention layer fixtures"]
#[allow(clippy::too_many_lines)]
fn checkpoint_full_attention_layer_matches_repeated_requests() {
    let directory = PathBuf::from(std::env::var_os("ORBITKV_ATTENTION_PROBE_DIR").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    let model = PathBuf::from(manifest["model_directory"].as_str().unwrap());
    let config =
        DecoderConfig::from_json(&std::fs::read(model.join("config.json")).unwrap()).unwrap();
    let layer = std::env::var("ORBITKV_ATTENTION_PROBE_LAYER")
        .map_or(3, |value| value.parse::<usize>().unwrap());
    let requests = std::env::var("ORBITKV_ATTENTION_PROBE_REQUESTS")
        .map_or(1, |value| value.parse::<usize>().unwrap());
    assert!(requests > 0);
    assert!(matches!(
        config.layer_kind(layer),
        crate::model::DecoderLayerKind::Full
    ));
    let tokens = manifest["case"]["prompt_token_ids"]
        .as_array()
        .unwrap()
        .len();
    let query_tokens = tokens * requests;
    let page_tokens = 16;
    let pages = requests;
    let mut graph = Graph::new();
    let prefix = format!("{}.layers.{layer}", config.tensor_prefix);
    let repeat_rows = |tensor: GraphTensor| {
        if requests == 1 {
            tensor
        } else {
            let repeated = tensor.repeat((requests, 1));
            repeated.gather(repeated.graph().iota('z', repeated.dims()))
        }
    };
    let hidden = repeat_rows(
        graph
            .named_tensor(format!("{prefix}.input.0"), (tokens, config.hidden_size))
            .as_dtype(DType::Bf16),
    );
    let expected = repeat_rows(
        graph
            .named_tensor(format!("{prefix}.output"), (tokens, config.hidden_size))
            .as_dtype(DType::Bf16),
    );
    let positions = graph
        .named_tensor("probe.positions", query_tokens)
        .as_dtype(DType::Int);
    let write_slots = graph
        .named_tensor("probe.write_slots", query_tokens)
        .as_dtype(DType::Int);
    let context_pages = orbitkv_compiler::prelude::Symbol::from('c');
    let metadata =
        PagedAttentionMetadata::new(&mut graph, 0, requests.into(), context_pages.into());
    let key_cache = graph
        .named_tensor(
            "probe.key_cache",
            (pages, page_tokens, config.kv_heads, config.head_dim),
        )
        .as_dtype(DType::Bf16);
    let value_cache = graph
        .named_tensor(
            "probe.value_cache",
            (pages, page_tokens, config.kv_heads, config.head_dim),
        )
        .as_dtype(DType::Bf16);
    let attention = TokenAttentionLayer::new(
        &mut graph,
        &config,
        DecoderWeightFeatures {
            qkv_bias: false,
            qk_norm: true,
        },
        layer,
    );
    let class = AttentionClass {
        class_id: 0,
        name: "full".into(),
        layers: vec![u32::try_from(layer).unwrap()].into_boxed_slice(),
        page_tokens: u32::try_from(page_tokens).unwrap(),
        key_bytes_per_token_per_layer: u64::try_from(config.kv_heads * config.head_dim * 2)
            .unwrap(),
        value_bytes_per_token_per_layer: u64::try_from(config.kv_heads * config.head_dim * 2)
            .unwrap(),
        visibility: AttentionVisibility::Full,
    };
    let (actual, _, _) = attention
        .forward(
            &TokenAttentionInputs {
                hidden: &hidden,
                positions: &positions,
                write_slots: &write_slots,
                metadata: &metadata,
                k_cache: &key_cache,
                v_cache: &value_cache,
            },
            &class,
            &config,
            DecoderDimensions {
                query_tokens: query_tokens.into(),
                request_count: requests.into(),
                kv_width: config.kv_heads * config.head_dim,
            },
            DecoderClassDimensions {
                class_id: 0,
                context_pages,
                backend_base_index: 0,
                page_count: u32::try_from(pages).unwrap(),
                cache_slots: pages * page_tokens,
            },
        )
        .unwrap();
    let actual = actual.cast(DType::F32).output();
    let expected = expected.cast(DType::F32).output();
    graph.set_dim(context_pages, pages);
    let context = CudaContext::new(0).unwrap();
    let mut runtime = CudaRuntime::initialize(context.new_stream().unwrap());
    load_fixture(&mut runtime, &graph, &model, &directory);
    runtime.set_data(
        positions,
        (0..requests)
            .flat_map(|_| 0..i32::try_from(tokens).unwrap())
            .collect::<Vec<_>>(),
    );
    runtime.set_data(
        write_slots,
        (0..requests)
            .flat_map(|request| {
                (0..tokens).map(move |token| i32::try_from(request * page_tokens + token).unwrap())
            })
            .collect::<Vec<_>>(),
    );
    runtime.set_data(
        key_cache,
        vec![bf16::ZERO; pages * page_tokens * config.kv_heads * config.head_dim],
    );
    runtime.set_data(
        value_cache,
        vec![bf16::ZERO; pages * page_tokens * config.kv_heads * config.head_dim],
    );
    metadata
        .upload(
            &mut runtime,
            &AttentionBatch {
                class_id: 0,
                query_indptr: (0..=requests)
                    .map(|request| i32::try_from(request * tokens).unwrap())
                    .collect(),
                page_indptr: (0..=requests)
                    .map(|request| i32::try_from(request).unwrap())
                    .collect(),
                page_indices: (0..requests)
                    .map(|page| i32::try_from(page).unwrap())
                    .collect(),
                last_page_len: vec![i32::try_from(tokens).unwrap(); requests].into_boxed_slice(),
            },
        )
        .unwrap();
    let mut facts = runtime.compilation_facts();
    if let Some(provider) = std::env::var_os("ORBITKV_ATTENTION_PROBE_PROVIDER") {
        let provider = provider.to_str().unwrap();
        assert!(matches!(provider, "flashattention" | "flashinfer"));
        write!(facts, "\n(set (cuda-attention-policy) \"{provider}\")").unwrap();
    }
    let mut runtime = graph.compile(
        runtime,
        CompileOptions::default()
            .dim_buckets(
                context_pages,
                &[DimBucket::new(pages, pages).representative(pages)],
            )
            .search_graph_limit(8)
            .compiler_facts(facts),
    );
    runtime.execute(&graph.dyn_map);
    let actual = runtime.get_f32(actual);
    let expected = runtime.get_f32(expected);
    assert_eq!(actual.len(), expected.len());
    let maximum = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    let differing = actual
        .iter()
        .zip(&expected)
        .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
        .count();
    let rows_per_request = tokens * config.hidden_size;
    let replica_maximum = actual
        .chunks_exact(rows_per_request)
        .skip(1)
        .flat_map(|copy| copy.iter().zip(&actual[..rows_per_request]))
        .map(|(value, first)| (value - first).abs())
        .fold(0.0_f32, f32::max);
    if let Some(output) = std::env::var_os("ORBITKV_ATTENTION_PROBE_OUTPUT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        for value in &actual {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "layer": layer, "requests": requests, "maximum_absolute_error": maximum,
            "differing_elements": differing, "elements": actual.len(),
            "replica_maximum_absolute_error": replica_maximum,
        })
    );
}

fn load_fixture(runtime: &mut CudaRuntime, graph: &Graph, model: &Path, directory: &Path) {
    let mut weights = std::fs::read_dir(model)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    weights.sort();
    for path in weights {
        runtime.load_safetensors(graph, path).unwrap();
    }
    runtime
        .load_safetensors(graph, directory.join("step-0.safetensors"))
        .unwrap();
}
