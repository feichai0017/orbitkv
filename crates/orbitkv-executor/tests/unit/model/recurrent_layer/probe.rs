//! Compare one checkpoint boundary with an independent, identical input.

use super::*;
use crate::model::block::{DecoderLayerEnvelope, DecoderNorm};
use orbitkv_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use std::path::PathBuf;

struct Boundary {
    actual: GraphTensor,
    expected: GraphTensor,
    state_inputs: Option<[GraphTensor; 3]>,
}

struct Fixture {
    config: DecoderConfig,
    layer: usize,
    tokens: usize,
    prefix: String,
}

impl Fixture {
    fn tensor(&self, graph: &mut Graph, suffix: &str, width: usize) -> GraphTensor {
        graph
            .named_tensor(format!("{}.{}", self.prefix, suffix), (self.tokens, width))
            .as_dtype(DType::Bf16)
    }

    fn build(&self, graph: &mut Graph, name: &str) -> Boundary {
        let config = &self.config;
        let geometry = config.gated_delta.unwrap();
        let channels = geometry.convolution_channels().unwrap();
        let (actual, expected) = match name {
            "input_norm" => {
                let input = self.tensor(graph, "input_layernorm.input.0", config.hidden_size);
                let norm = DecoderNorm::new(
                    graph,
                    config,
                    &format!("{}.input_layernorm.weight", self.prefix),
                );
                (
                    norm.forward(&input),
                    self.tensor(graph, "input_layernorm.output", config.hidden_size),
                )
            }
            "qkv" | "z" | "b" | "a" => {
                let core = GatedDeltaCore::new(graph, config, self.layer, DType::Bf16).unwrap();
                let input = self.tensor(graph, "input_layernorm.output", config.hidden_size);
                let (actual, suffix, width) = match name {
                    "qkv" => (core.input_qkv.forward(&input), "in_proj_qkv", channels),
                    "z" => (
                        core.input_z.forward(&input),
                        "in_proj_z",
                        geometry.value_elements().unwrap(),
                    ),
                    "b" => (
                        input.matmul(core.input_b.t()),
                        "in_proj_b",
                        geometry.value_heads,
                    ),
                    "a" => (
                        input.matmul(core.input_a.t()),
                        "in_proj_a",
                        geometry.value_heads,
                    ),
                    _ => unreachable!(),
                };
                (
                    actual,
                    self.tensor(graph, &format!("linear_attn.{suffix}.output"), width),
                )
            }
            "convolution" | "gdn_core" => return self.state_boundary(graph, name),
            "gated_norm" => {
                let rows = self.tokens * geometry.value_heads;
                let mut tensor = |suffix: &str| {
                    graph
                        .named_tensor(
                            format!("{}.linear_attn.norm.{suffix}", self.prefix),
                            (rows, geometry.value_width),
                        )
                        .as_dtype(DType::Bf16)
                };
                let input = tensor("input.0");
                let gate = tensor("input.1");
                let expected = tensor("output");
                let weight = crate::model::weight(
                    graph,
                    format!("{}.linear_attn.norm.weight", self.prefix),
                    geometry.value_width,
                    DType::F32,
                );
                let actual = (input.cast(DType::F32).std_norm(1, config.rms_epsilon)
                    * weight.expand_dim(0, rows)
                    * gate.cast(DType::F32).swish())
                .cast(DType::Bf16);
                (actual, expected)
            }
            "residual_mlp" => {
                let residual = self.tensor(graph, "input_layernorm.input.0", config.hidden_size);
                let core = self.tensor(graph, "linear_attn.output", config.hidden_size);
                let envelope = DecoderLayerEnvelope::new(graph, config, self.layer);
                (
                    envelope.finish(&residual, core),
                    self.tensor(graph, "output", config.hidden_size),
                )
            }
            _ => panic!("unknown diagnostic boundary: {name}"),
        };
        Boundary {
            actual: actual.cast(DType::F32).output(),
            expected: expected.cast(DType::F32).output(),
            state_inputs: None,
        }
    }
    fn state_boundary(&self, graph: &mut Graph, name: &str) -> Boundary {
        let config = &self.config;
        let geometry = config.gated_delta.unwrap();
        let channels = geometry.convolution_channels().unwrap();
        let core = GatedDeltaCore::new(graph, config, self.layer, DType::Bf16).unwrap();
        let state = graph.named_tensor(
            "probe.state",
            (
                1,
                geometry.value_heads,
                geometry.key_width,
                geometry.value_width,
            ),
        );
        let history = graph
            .named_tensor(
                "probe.history",
                (1, channels, geometry.convolution_kernel_width - 1),
            )
            .as_dtype(DType::Bf16);
        let indptr = graph.named_tensor("probe.indptr", 2).as_dtype(DType::Int);

        let (actual, expected) = if name == "convolution" {
            let qkv = self.tensor(graph, "linear_attn.in_proj_qkv.output", channels);
            let result = packed_causal_convolution(
                PackedConvolutionPlan {
                    input: qkv,
                    weights: core.convolution_weight.squeeze(1),
                    history,
                    query_indptr: indptr,
                },
                PackedConvolutionSpec {
                    channels,
                    kernel_width: geometry.convolution_kernel_width,
                },
            );
            let expected = graph
                .named_tensor(
                    format!("{}.linear_attn.silu.output", self.prefix),
                    (channels, self.tokens),
                )
                .as_dtype(DType::Bf16)
                .t();
            (result.values, expected)
        } else {
            let input = self.tensor(graph, "input_layernorm.output", config.hidden_size);
            let result = core
                .forward_packed(&input, &state, &history, indptr)
                .unwrap();
            (
                result.hidden,
                self.tensor(graph, "linear_attn.output", config.hidden_size),
            )
        };

        Boundary {
            actual: actual.cast(DType::F32).output(),
            expected: expected.cast(DType::F32).output(),
            state_inputs: Some([state, history, indptr]),
        }
    }
}

#[test]
#[ignore = "requires CUDA, a checkpoint and independent layer boundary fixtures"]
fn checkpoint_boundaries() {
    let directory = PathBuf::from(std::env::var_os("ORBITKV_LAYER_PROBE_DIR").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    let model = PathBuf::from(manifest["model_directory"].as_str().unwrap());
    let config =
        DecoderConfig::from_json(&std::fs::read(model.join("config.json")).unwrap()).unwrap();
    let layer = std::env::var("ORBITKV_LAYER_PROBE_LAYER")
        .map_or(0, |value| value.parse::<usize>().unwrap());
    let tokens = manifest["case"]["prompt_token_ids"]
        .as_array()
        .unwrap()
        .len();
    let fixture = Fixture {
        prefix: format!("{}.layers.{layer}", config.tensor_prefix),
        config,
        layer,
        tokens,
    };
    let name =
        std::env::var("ORBITKV_LAYER_PROBE_BOUNDARY").unwrap_or_else(|_| "input_norm".into());
    let mut graph = Graph::new();
    let boundary = fixture.build(&mut graph, &name);
    let context = CudaContext::new(0).unwrap();
    let mut runtime = CudaRuntime::initialize(context.default_stream());
    let mut weights = std::fs::read_dir(&model)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    weights.sort();
    for file in weights {
        runtime.load_safetensors(&graph, file).unwrap();
    }
    runtime
        .load_safetensors(&graph, directory.join("step-0.safetensors"))
        .unwrap();
    if let Some([state, history, indptr]) = boundary.state_inputs {
        let geometry = fixture.config.gated_delta.unwrap();
        runtime.set_data(
            state,
            vec![0.0_f32; geometry.value_heads * geometry.key_width * geometry.value_width],
        );
        runtime.set_data(
            history,
            vec![
                half::bf16::ZERO;
                geometry.convolution_channels().unwrap() * (geometry.convolution_kernel_width - 1)
            ],
        );
        runtime.set_data(indptr, vec![0_i32, i32::try_from(tokens).unwrap()]);
    }
    let facts = runtime.compilation_facts();
    let mut runtime = graph.compile(
        runtime,
        CompileOptions::default()
            .search_graph_limit(8)
            .compiler_facts(facts),
    );
    runtime.execute(&graph.dyn_map);
    let actual = runtime.get_f32(boundary.actual);
    let expected = runtime.get_f32(boundary.expected);
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite())
    );
    let maximum = actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f32, f32::max);
    let differing = actual
        .iter()
        .zip(&expected)
        .filter(|(a, e)| a.to_bits() != e.to_bits())
        .count();
    let report = serde_json::json!({"layer": layer, "boundary": name, "maximum_absolute_error": maximum, "differing_elements": differing, "elements": actual.len()});
    std::fs::write(
        directory.join(format!("layer-{layer}-{name}.json")),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("{report}");
}
