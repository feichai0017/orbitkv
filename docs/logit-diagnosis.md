# Decoder logits 的独立诊断

生成文本不同需要同时检查数值和选 token 的规则。`tools/logit_probe.py`
导出独立模型参考、比较完整词表；`model_execution/logit_probe.rs`
只负责按 manifest 严格重放已有 schedule、保存实际 logits/token，并提交和释放状态。
诊断代码不改编译候选、模型定义、sampling 规则或误差阈值。

两处原 serving 分岔的实际 logits、冻结版本证明和检查结果见
[已审阅诊断报告](validation/fp8-logit-diagnosis-20260912/README.md)。

## 输入与边界

Probe manifest 使用版本字段 `schema: 1`，包含：

- `model_directory`、`device_index`：本地 checkpoint 与执行设备。
- `page_tokens`、`kv_dtype_bytes`、`page_counts`：实际存储几何。
- `compile`：完整 `DecoderCompileConfig`，必须匹配已有 artifact。
- `cases`：唯一 `id`、`prompt_token_ids`、`continuation_token_ids`。

一个 case 的 continuation 长度就是要测量的输出步数。第零步执行完整 prompt，
之后每步输入前一个 continuation token；最后一个 continuation token 不再作为输入。
这使不同实现接收完全相同的历史，避免生成分岔后把不同输入的 logits 当作数值误差。
当前 harness 每次执行一个请求，多个 case 顺序执行；batch capacity 仍来自 compile 配置。
它不替代并发、任意 chunk 划分或 serving 生命周期测试。

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
