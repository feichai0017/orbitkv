//! Isolate final normalization/projection from independently searched layer programs.

use super::*;
use crate::model::{DecoderConfig, block::DecoderNorm};
use orbitkv_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use std::path::PathBuf;

#[test]
#[ignore = "requires CUDA, a checkpoint and exported final-layer hidden states"]
fn checkpoint_projection_matches_selected_full_rows() {
    let directory = PathBuf::from(std::env::var_os("ORBITKV_LAYER_PROBE_DIR").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    let model = PathBuf::from(manifest["model_directory"].as_str().unwrap());
    let config =
        DecoderConfig::from_json(&std::fs::read(model.join("config.json")).unwrap()).unwrap();
    let tokens = manifest["case"]["prompt_token_ids"]
        .as_array()
        .unwrap()
        .len();
    assert!(
        tokens >= 2,
        "projection fixture needs at least two hidden rows"
    );
    let mut graph = Graph::new();
    let hidden = graph
        .named_tensor(
            format!(
                "{}.layers.{}.output",
                config.tensor_prefix,
                config.layers - 1
            ),
            (tokens, config.hidden_size),
        )
        .as_dtype(DType::Bf16);
    let indptr = graph.tensor(3).as_dtype(DType::Int);
    let norm = DecoderNorm::new(
        &mut graph,
        &config,
        &format!("{}.norm.weight", config.tensor_prefix),
    );
    let head_name = if config.tied_embeddings {
        format!("{}.embed_tokens.weight", config.tensor_prefix)
    } else {
        "lm_head.weight".to_owned()
    };
    let head = crate::model::weight(
        &mut graph,
        head_name,
        (config.vocabulary_size, config.hidden_size),
        DType::Bf16,
    );
    let full_normalized = norm.forward(&hidden);
    let selected_hidden = DecoderOutputRows::LastTokenPerRequest.select(&hidden, &indptr);
    let selected_normalized = norm.forward(&selected_hidden);
    let full = full_normalized.matmul(head.t()).cast(DType::F32).output();
    let selected = selected_normalized
        .matmul(head.t())
        .cast(DType::F32)
        .output();
    let full_norm = full_normalized.cast(DType::F32).output();
    let selected_norm = selected_normalized.cast(DType::F32).output();
    let context = CudaContext::new(0).unwrap();
    let mut runtime = CudaRuntime::initialize(context.new_stream().unwrap());
    load_fixture(&mut runtime, &graph, &model, &directory);
    runtime.set_data(indptr, vec![0_i32, 1, i32::try_from(tokens).unwrap()]);
    let facts = runtime.compilation_facts();
    let mut runtime = graph.compile(
        runtime,
        CompileOptions::default()
            .search_graph_limit(8)
            .compiler_facts(facts),
    );
    runtime.execute(&graph.dyn_map);
    let compare = |all, chosen, width| {
        let all = runtime.get_f32(all);
        let chosen = runtime.get_f32(chosen);
        assert_eq!(all.len(), tokens * width);
        assert_eq!(chosen.len(), 2 * width);
        assert!(all.iter().chain(&chosen).all(|value| value.is_finite()));
        [0, tokens - 1]
            .into_iter()
            .enumerate()
            .flat_map(|(row, index)| {
                chosen[row * width..(row + 1) * width]
                    .iter()
                    .zip(&all[index * width..(index + 1) * width])
            })
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    };
    let norm_error = compare(full_norm, selected_norm, config.hidden_size);
    let logits_error = compare(full, selected, config.vocabulary_size);
    let report = serde_json::json!({"query_rows":tokens,"selected_rows":2,"norm_maximum_absolute_error":norm_error,"logits_maximum_absolute_error":logits_error});
    std::fs::write(
        directory.join("projection-comparison.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{report}");
    assert!(
        logits_error <= 1.0,
        "projection exceeded the existing full-vocabulary error limit"
    );
}

fn load_fixture(
    runtime: &mut CudaRuntime,
    graph: &Graph,
    model: &std::path::Path,
    directory: &std::path::Path,
) {
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
