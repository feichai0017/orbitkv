# Decoder logits 的独立诊断

生成文本不同需要同时检查数值和选 token 的规则。`tools/logit_probe.py`
导出独立模型参考、比较完整词表；`model_execution/logit_probe.rs`
按 manifest 编译或严格重放 schedule、保存实际 logits/token，并提交和释放状态。
诊断代码不改编译候选、模型定义、sampling 规则或误差阈值。

## 输入与边界

Probe manifest 使用版本字段 `schema: 1`，包含：

- `model_directory`、`device_index`：本地 checkpoint 与执行设备。
- `page_tokens`、`kv_dtype_bytes`、`page_counts`：实际存储几何。
- `compile`：完整 `DecoderCompileConfig`，必须匹配已有 artifact；`output_rows`
  显式选择 `all_tokens` 或 `last_token_per_request`，旧配置需更新并重新编译。
- `cases`：唯一 `id`、`prompt_token_ids`、`continuation_token_ids`。

一个 case 的 continuation 长度就是要测量的输出步数。第零步执行完整 prompt，
之后每步输入前一个 continuation token；最后一个 continuation token 不再作为输入。
这使不同实现接收完全相同的历史，避免生成分岔后把不同输入的 logits 当作数值误差。
省略 `batches` 时逐个执行 case；显式提交计划可以包含不等长请求、prompt chunk、
重排及状态槽复用，见 [编译边界](compiler-boundaries.md)。它不替代 HTTP serving 测试。

Oracle 有两个 continuation 模式：`manifest` 原样使用输入中的 continuation；
`greedy` 保留其长度，重新生成参考 continuation，并输出实际使用的 `probe.json`。
执行器应读取这个输出文件。manifest 的 token、容量、页布局都属于实验数据，
工具没有内置模型名、prompt、GPU 型号或固定步数。

## 执行

使用已有的 Python 环境及本地模型，显式选择模型加载类和 dtype。例如：

```sh
python tools/logit_probe.py oracle \
  --manifest /absolute/path/inputs.json \
  --output-dir /absolute/path/oracle \
  --model-class AutoModelForCausalLM \
  --dtype bfloat16 \
  --continuation manifest
```

默认使用 Transformers 的算子实现。`--linear-backend deepgemm-fp8-block`
是显式的、受限的外部 provider 适配器：它仅接受动态 activation 量化、E4M3 权重、
float32 scale 和 DeepGEMM 定义的 128×128 权重 block ABI。
这个数值是 provider 格式约束，不是模型调优参数。其他格式必须使用对应实现，
适配器会拒绝不支持的输入。`--config-overrides` 接受单独的 JSON 覆盖文件，
用于记录有证据的 checkpoint 配置修正；工具不根据模型名称自动修改配置。

运行包含 `logit_probe::decoder_manifest_logits_and_drain` 的预构建测试二进制：

```sh
ORBITKV_LOGIT_PROBE_MANIFEST=/absolute/path/oracle/probe.json \
ORBITKV_LOGIT_PROBE_OUTPUT=/absolute/path/candidate \
ORBITKV_DECODER_ARTIFACT=/absolute/path/decoder.json \
ORBITKV_TUNING_PROFILE=/absolute/path/tuning.json \
/absolute/path/model_execution \
  --exact logit_probe::decoder_manifest_logits_and_drain --ignored --nocapture
```

输出目录必须不存在。测试要求已有 artifact，检查其 identity，并在完成后验证文件未变。
需要生成新 schedule 时，显式运行
`logit_probe::decoder_manifest_compile_logits_and_drain`；artifact 路径必须不存在，
测试不会覆盖旧 schedule。两种输出策略都导出每个逻辑步骤的最后一行完整词表，
trace 同时记录 `output_rows`，可用同一历史比较 LM head 行选择前后数值。
它保存每步输入、实际 selected token、little-endian f32 logits，以及每个 case 的 drain 结果。
产物保存 source/binary/environment 证明应由具体实验 runner 补充；仅给工具一个目录
不能证明测试二进制由该目录的源码构建。

比较使用相同 teacher-forced 历史的两份 trace：

```sh
python tools/logit_probe.py compare \
  --reference-dir /absolute/path/oracle \
  --candidate-dir /absolute/path/candidate \
  --top-count 5 \
  --output /absolute/path/comparison.json
```

`top-count` 控制报告展示数量，不改变任何计算或验收。比较会拒绝词表、case、
步数、输入 token 不一致，以及非有限 logits。它报告最大绝对误差、RMS 误差、
实际 selected token 的首次差异、两侧 top logits、最大值的全部并列 token，
以及双方在实际选中 token 处的分数；它不自动宣布性能或正确性达标。

## 并列值的语义

OrbitKV compiler 当前 `argmax` 和 `argmin` 在有限输入的并列极值中取**最大 index**；
CUDA 融合实现保留这个规则。PyTorch argmax 取最小 index。
报告中的 `argmax_equal` 是从导出的 logits 计算、统一采用最小 token ID 的比较，
必须结合 `first_selected_token_difference` 读取，不能用它代替运行时实际输出。
`highest_token_id_argmax_equal` 和 `candidate_selected_matches_highest_maximum_token_id`
则直接检查 OrbitKV compiler 的规则。

参考与候选之间的数值变化可能制造或打破并列值。只有实际保存的 logits
才能区分这种情况和选 token 实现错误；不能因文本差异较小就称为可接受误差。
改变默认 tie policy 会改变输出及 artifact 的语义，应作为独立兼容性变更处理。

## 逐层定位

`tools/export_layer_probe.py` 复用独立 oracle 的 metadata（加载类、dtype、
配置覆盖及量化 backend），用 hooks 保存原始模块输入/输出，不改参考计算。
`--module-pattern` 限定需要观察的模块，`--function module:function` 可补充
模块内部函数的边界。显式选择 case 和步数，输出目录必须不存在。

```sh
python tools/export_layer_probe.py \
  --manifest /absolute/path/oracle/probe.json \
  --oracle-metadata /absolute/path/oracle/metadata.json \
  --output-dir /absolute/path/layer-reference \
  --module-pattern '^model.language_model.layers.0(\.|$)' \
  --function torch.nn.functional:silu
```

执行器的 ignored unit test
`model::recurrent_layer::tests::probe::checkpoint_boundaries` 读取
`ORBITKV_LAYER_PROBE_DIR`、`ORBITKV_LAYER_PROBE_LAYER`（默认 0）及
`ORBITKV_LAYER_PROBE_BOUNDARY`（默认 `input_norm`）。当前边界包括
`qkv/z/b/a`、`convolution`、`gdn_core`、`gated_norm`、`residual_mlp`，使用
参考模块的同一输入隔离上游误差。当前测试仅比较零初始状态的首个 prefill；
最大误差和逐位差异数是诊断结果，不是通过数值门槛的声明。模型全词表及
不同历史的验收仍独立执行。CUDA 运行需先配置锁定的 provider 源码路径。
