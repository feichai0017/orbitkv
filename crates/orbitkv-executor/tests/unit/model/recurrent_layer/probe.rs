//! Compare one checkpoint boundary with an independent, identical input.

use super::*;
use crate::model::block::{DecoderLayerEnvelope, DecoderNorm};
use crate::model::{DecoderActivation, linear_weight};
use orbitkv_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use std::path::PathBuf;

struct Boundary {
    actual: GraphTensor,
    expected: GraphTensor,
    state_inputs: Option<[GraphTensor; 3]>,
    state_outputs: Vec<(GraphTensor, GraphTensor)>,
}

struct Fixture {
    config: DecoderConfig,
    layer: usize,
    tokens: usize,
    requests: usize,
    step: usize,
    prefix: String,
}

impl Fixture {
    fn repeat_requests(&self, tensor: &GraphTensor) -> GraphTensor {
        if self.requests == 1 {
            *tensor
        } else {
            let mut repeats = vec![Expression::from(1); tensor.dims().len()];
            repeats[0] = self.requests.into();
            let repeated = tensor.repeat(repeats.as_slice());
            repeated.gather(repeated.graph().iota('z', repeated.dims()))
        }
    }

    fn tensor(&self, graph: &mut Graph, suffix: &str, width: usize) -> GraphTensor {
        let tensor = graph
            .named_tensor(format!("{}.{}", self.prefix, suffix), (self.tokens, width))
            .as_dtype(DType::Bf16);
        self.repeat_requests(&tensor)
    }

    #[allow(clippy::too_many_lines)]
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
            "convolution" | "delta_scan" | "gdn_core" => {
                return self.state_boundary(graph, name);
            }
            "gated_norm" => {
                let rows = self.tokens * geometry.value_heads;
                let mut tensor = |suffix: &str| {
                    let tensor = graph
                        .named_tensor(
                            format!("{}.linear_attn.norm.{suffix}", self.prefix),
                            (rows, geometry.value_width),
                        )
                        .as_dtype(DType::Bf16);
                    self.repeat_requests(&tensor)
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
                let normalized = input
                    .cast(DType::F32)
                    .std_norm(1, config.rms_epsilon)
                    .cast(DType::Bf16);
                let weighted =
                    normalized * weight.cast(DType::Bf16).expand_dim(0, rows * self.requests);
                let actual =
                    (weighted.cast(DType::F32) * gate.cast(DType::F32).swish()).cast(DType::Bf16);
                (actual, expected)
            }
            "state_residual" => {
                let residual = self.tensor(graph, "input_layernorm.input.0", config.hidden_size);
                let core = self.tensor(graph, "linear_attn.output", config.hidden_size);
                (
                    residual + core,
                    self.tensor(
                        graph,
                        "post_attention_layernorm.input.0",
                        config.hidden_size,
                    ),
                )
            }
            "post_attention_norm" => {
                let input = self.tensor(
                    graph,
                    "post_attention_layernorm.input.0",
                    config.hidden_size,
                );
                let norm = DecoderNorm::new(
                    graph,
                    config,
                    &format!("{}.post_attention_layernorm.weight", self.prefix),
                );
                (
                    norm.forward(&input),
                    self.tensor(graph, "post_attention_layernorm.output", config.hidden_size),
                )
            }
            "mlp_gate" | "mlp_up" => {
                let projection = name.strip_prefix("mlp_").unwrap();
                let input =
                    self.tensor(graph, "post_attention_layernorm.output", config.hidden_size);
                let linear = linear_weight(
                    graph,
                    config,
                    &format!("{}.mlp.{projection}_proj.weight", self.prefix),
                    config.intermediate_size,
                    config.hidden_size,
                    DType::Bf16,
                );
                (
                    linear.forward(&input),
                    self.tensor(
                        graph,
                        &format!("mlp.{projection}_proj.output"),
                        config.intermediate_size,
                    ),
                )
            }
            "mlp_activation" | "mlp_activation_product" => {
                let input = self.tensor(graph, "mlp.gate_proj.output", config.intermediate_size);
                let actual = match config.activation {
                    DecoderActivation::Silu => input.cast(DType::F32).silu().cast(DType::Bf16),
                    DecoderActivation::GeluTanh => unreachable!("Qwen GDN uses SiLU"),
                };
                if name == "mlp_activation_product" {
                    let up = self.tensor(graph, "mlp.up_proj.output", config.intermediate_size);
                    (
                        (actual.cast(DType::F32) * up.cast(DType::F32)).cast(DType::Bf16),
                        self.tensor(graph, "mlp.down_proj.input.0", config.intermediate_size),
                    )
                } else {
                    (
                        actual,
                        self.tensor(graph, "mlp.act_fn.output", config.intermediate_size),
                    )
                }
            }
            "mlp_product" => {
                let activation = self.tensor(graph, "mlp.act_fn.output", config.intermediate_size);
                let up = self.tensor(graph, "mlp.up_proj.output", config.intermediate_size);
                (
                    activation * up,
                    self.tensor(graph, "mlp.down_proj.input.0", config.intermediate_size),
                )
            }
            "mlp_down" => {
                let input = self.tensor(graph, "mlp.down_proj.input.0", config.intermediate_size);
                let linear = linear_weight(
                    graph,
                    config,
                    &format!("{}.mlp.down_proj.weight", self.prefix),
                    config.hidden_size,
                    config.intermediate_size,
                    DType::Bf16,
                );
                (
                    linear.forward(&input),
                    self.tensor(graph, "mlp.down_proj.output", config.hidden_size),
                )
            }
            "residual_mlp" => {
                let residual = self.tensor(graph, "input_layernorm.input.0", config.hidden_size);
                let core = self.tensor(graph, "linear_attn.output", config.hidden_size);
                let envelope = DecoderLayerEnvelope::new(graph, config, self.layer);
                (
                    envelope.finish(&residual, &core),
                    self.tensor(graph, "output", config.hidden_size),
                )
            }
            _ => panic!("unknown diagnostic boundary: {name}"),
        };
        Boundary {
            actual: actual.cast(DType::F32).output(),
            expected: expected.cast(DType::F32).output(),
            state_inputs: None,
            state_outputs: Vec::new(),
        }
    }
    #[allow(clippy::too_many_lines)]
    fn state_boundary(&self, graph: &mut Graph, name: &str) -> Boundary {
        let config = &self.config;
        let geometry = config.gated_delta.unwrap();
        let channels = geometry.convolution_channels().unwrap();
        let core = GatedDeltaCore::new(graph, config, self.layer, DType::Bf16).unwrap();
        let state_shape = (
            self.requests,
            geometry.value_heads,
            geometry.key_width,
            geometry.value_width,
        );
        let state = if self.step == 0 {
            graph.named_tensor("probe.state", state_shape)
        } else {
            let state = graph
                .named_tensor(
                    format!(
                        "{}.linear_attn.torch_recurrent_gated_delta_rule.input.initial_state",
                        self.prefix
                    ),
                    (
                        1,
                        geometry.value_heads,
                        geometry.key_width,
                        geometry.value_width,
                    ),
                )
                .as_dtype(DType::Bf16)
                .cast(DType::F32);
            if self.requests == 1 {
                state
            } else {
                state.repeat((self.requests, 1, 1, 1))
            }
        };
        let history = if self.step == 0 {
            graph
                .named_tensor(
                    "probe.history",
                    (
                        self.requests,
                        channels,
                        geometry.convolution_kernel_width - 1,
                    ),
                )
                .as_dtype(DType::Bf16)
        } else {
            let history = graph
                .named_tensor(
                    format!(
                        "{}.linear_attn.torch_causal_conv1d_update.input.conv_state",
                        self.prefix
                    ),
                    (1, channels, geometry.convolution_kernel_width),
                )
                .as_dtype(DType::Bf16)
                .slice((.., .., 1..));
            if self.requests == 1 {
                history
            } else {
                history.repeat((self.requests, 1, 1))
            }
        };
        let indptr = graph
            .named_tensor("probe.indptr", self.requests + 1)
            .as_dtype(DType::Int);

        let mut state_outputs = Vec::new();
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
            let expected = self.repeat_requests(
                &graph
                    .named_tensor(
                        format!("{}.linear_attn.silu.output", self.prefix),
                        (channels, self.tokens),
                    )
                    .as_dtype(DType::Bf16)
                    .t(),
            );
            let expected_history = graph
                .named_tensor(
                    format!("cache.layers.{}.conv_states", self.layer),
                    (1, channels, geometry.convolution_kernel_width),
                )
                .as_dtype(DType::Bf16)
                .slice((.., .., 1..));
            let expected_history = if self.requests == 1 {
                expected_history
            } else {
                expected_history.repeat((self.requests, 1, 1))
            };
            state_outputs.push((
                result.history.cast(DType::F32).output(),
                expected_history.cast(DType::F32).output(),
            ));
            (result.values, expected)
        } else if name == "delta_scan" {
            let operation = if self.step == 0 {
                "torch_chunk_gated_delta_rule"
            } else {
                "torch_recurrent_gated_delta_rule"
            };
            let tensor = |graph: &mut Graph, field: &str, shape| {
                let tensor = graph
                    .named_tensor(
                        format!("{}.linear_attn.{operation}.input.{field}", self.prefix),
                        shape,
                    )
                    .as_dtype(DType::Bf16)
                    .squeeze(0);
                self.repeat_requests(&tensor)
            };
            let query = tensor(
                graph,
                "query",
                (1, self.tokens, geometry.value_heads, geometry.key_width),
            )
            .cast(DType::F32);
            let key = tensor(
                graph,
                "key",
                (1, self.tokens, geometry.value_heads, geometry.key_width),
            )
            .cast(DType::F32);
            let value = tensor(
                graph,
                "value",
                (1, self.tokens, geometry.value_heads, geometry.value_width),
            )
            .cast(DType::F32);
            let log_decay = self.repeat_requests(
                &graph
                    .named_tensor(
                        format!("{}.linear_attn.{operation}.input.g", self.prefix),
                        (1, self.tokens, geometry.value_heads),
                    )
                    .squeeze(0),
            );
            let update_gate = self.repeat_requests(
                &graph
                    .named_tensor(
                        format!("{}.linear_attn.{operation}.input.beta", self.prefix),
                        (1, self.tokens, geometry.value_heads),
                    )
                    .as_dtype(DType::Bf16)
                    .squeeze(0)
                    .cast(DType::F32),
            );
            let result = packed_delta_scan(
                PackedDeltaScanPlan {
                    query,
                    key,
                    value,
                    log_decay,
                    update_gate,
                    state,
                    query_indptr: indptr,
                },
                PackedDeltaScanSpec {
                    key_heads: geometry.value_heads,
                    value_heads: geometry.value_heads,
                    key_width: geometry.key_width,
                    value_width: geometry.value_width,
                    normalization_epsilon: 1e-6,
                    round_normalized_qk_to_bf16: true,
                    round_final_state_to_bf16: false,
                },
            );
            let expected = self.repeat_requests(
                &graph
                    .named_tensor(
                        format!("{}.linear_attn.{operation}.output.0", self.prefix),
                        (1, self.tokens, geometry.value_heads, geometry.value_width),
                    )
                    .as_dtype(DType::Bf16)
                    .squeeze(0),
            );
            let expected_state = graph
                .named_tensor(
                    format!("cache.layers.{}.recurrent_states", self.layer),
                    (
                        1,
                        geometry.value_heads,
                        geometry.key_width,
                        geometry.value_width,
                    ),
                )
                .as_dtype(DType::Bf16)
                .cast(DType::F32);
            let expected_state = if self.requests == 1 {
                expected_state
            } else {
                expected_state.repeat((self.requests, 1, 1, 1))
            };
            state_outputs.push((result.state.output(), expected_state.output()));
            (result.values.cast(DType::Bf16), expected)
        } else {
            let input = self.tensor(graph, "input_layernorm.output", config.hidden_size);
            let result = core
                .forward_packed(&input, &state, &history, indptr)
                .unwrap();
            let expected_state = graph
                .named_tensor(
                    format!("cache.layers.{}.recurrent_states", self.layer),
                    (
                        1,
                        geometry.value_heads,
                        geometry.key_width,
                        geometry.value_width,
                    ),
                )
                .as_dtype(DType::Bf16)
                .cast(DType::F32);
            let expected_state = if self.requests == 1 {
                expected_state
            } else {
                expected_state.repeat((self.requests, 1, 1, 1))
            };
            state_outputs.push((
                result.next_recurrent_state.output(),
                expected_state.output(),
            ));
            (
                result.hidden,
                self.tensor(graph, "linear_attn.output", config.hidden_size),
            )
        };

        Boundary {
            actual: actual.cast(DType::F32).output(),
            expected: expected.cast(DType::F32).output(),
            state_inputs: Some([state, history, indptr]),
            state_outputs,
        }
    }
}

#[test]
#[ignore = "requires CUDA, a checkpoint and independent layer boundary fixtures"]
#[allow(clippy::too_many_lines)]
fn checkpoint_boundaries() {
    let directory = PathBuf::from(std::env::var_os("ORBITKV_LAYER_PROBE_DIR").unwrap());
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    let model = PathBuf::from(manifest["model_directory"].as_str().unwrap());
    let config =
        DecoderConfig::from_json(&std::fs::read(model.join("config.json")).unwrap()).unwrap();
    let layer = std::env::var("ORBITKV_LAYER_PROBE_LAYER")
        .map_or(0, |value| value.parse::<usize>().unwrap());
    let step = std::env::var("ORBITKV_LAYER_PROBE_STEP")
        .map_or(0, |value| value.parse::<usize>().unwrap());
    let requests = std::env::var("ORBITKV_LAYER_PROBE_REQUESTS")
        .map_or(1, |value| value.parse::<usize>().unwrap());
    assert!(requests > 0);
    assert!(step < usize::try_from(manifest["steps"].as_u64().unwrap()).unwrap());
    let tokens = if step == 0 {
        manifest["case"]["prompt_token_ids"]
            .as_array()
            .unwrap()
            .len()
    } else {
        1
    };
    let fixture = Fixture {
        prefix: format!("{}.layers.{layer}", config.tensor_prefix),
        config,
        layer,
        tokens,
        requests,
        step,
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
        .load_safetensors(&graph, directory.join(format!("step-{step}.safetensors")))
        .unwrap();
    if let Some([state, history, indptr]) = boundary.state_inputs {
        let geometry = fixture.config.gated_delta.unwrap();
        if step == 0 {
            runtime.set_data(
                state,
                vec![
                    0.0_f32;
                    requests * geometry.value_heads * geometry.key_width * geometry.value_width
                ],
            );
            runtime.set_data(
                history,
                vec![
                    half::bf16::ZERO;
                    requests
                        * geometry.convolution_channels().unwrap()
                        * (geometry.convolution_kernel_width - 1)
                ],
            );
        }
        runtime.set_data(
            indptr,
            (0..=requests)
                .map(|request| i32::try_from(request * tokens).unwrap())
                .collect::<Vec<_>>(),
        );
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
    let state_differences = boundary
        .state_outputs
        .iter()
        .map(|&(actual, expected)| {
            let actual = runtime.get_f32(actual);
            let expected = runtime.get_f32(expected);
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
            serde_json::json!({
                "maximum_absolute_error": maximum,
                "differing_elements": differing,
                "elements": actual.len(),
            })
        })
        .collect::<Vec<_>>();
    let report = serde_json::json!({"layer": layer, "step": step, "requests": requests, "boundary": name, "maximum_absolute_error": maximum, "differing_elements": differing, "elements": actual.len(), "state_differences": state_differences, "kernel_names": runtime.kernel_names()});
    if let Some(output) = std::env::var_os("ORBITKV_LAYER_PROBE_OUTPUT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .unwrap();
        file.write_all(&serde_json::to_vec_pretty(&report).unwrap())
            .unwrap();
    }
    println!("{report}");
}
