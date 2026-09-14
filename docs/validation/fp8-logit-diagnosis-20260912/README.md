# 两个 serving 输出差异的独立 logits 诊断

诊断已完成并经独立审阅。两处差异都由**数值变化形成并列最大值，再由既有 argmax 规则选择较大 token ID**产生。
它们不构成 CUDA argmax 与 Luminal 图语义不一致的证据。默认 sampling 规则、共享 FP8 开关、数值阈值均未修改；整体性能仍未获验证。

原实验是 `fp8-serving-ab-20260912` 的 Qwen3.8-27B-FP8、H20、C1 serving 开关对照。
本实验从保存的客户端重建 receipt 取 request 0 和 3，使用新的通用 manifest harness，
严格重放原 off/on artifact。用于执行的 594 个冻结源文件全部与 v3 一致，只有两个明确的集成测试文件扩展；
未使用当前并行重构后的生产源码。两个 arm 均完成 2 个请求 × 8 步完整 logits 导出及状态 drain。

## 实际首次分岔

表中步数按人阅读习惯从 1 开始；JSON 的 `step` 从 0 开始。

| 请求 / 输出步 | 对比 token IDs | 参考 logits | OFF logits / 实际选择 | ON logits / 实际选择 |
|---|---|---|---|---|
| 0 / 第 8 步 | 550（`##`）、9010（`Art`） | 11.25 / 11.1875 | 11.25 / 11.25；9010 | 11.25 / 11.1875；550 |
| 3 / 第 4 步 | 44（`M`）、15208（`Track`） | 11.0625 / 10.625 | 11.0625 / 11.0625；15208 | 11.125 / 11.0；44 |

这些 token 对应原 serving 的首次文本分岔位置。分岔前的所有输入 token 相同。
第二个请求的参考 top2 margin 为 0.3125；表中 44 与 OFF 实际选择的 15208 的参考分数差为 0.4375，二者不是同一个指标。

Luminal frontend 的 `argmax` 使用 `(eq(max) * arange).max`，并列时取最大 index；
CUDA 融合实现明确保留这个规则。PyTorch 的 argmax 取最小 index。
两臂所有实际选择都确实是各自 logits 的最大值，且遵守 Luminal 的最高 index 规则。
对两臂全部 16 个导出词表按统一的最小 index 规则计算 argmax，则都与独立参考一致。
这个辅助观察不能代替实际的 selected token 一致性。

## 数值范围与证据边界

| 请求 | OFF 最大绝对误差 | ON 最大绝对误差 | OFF 首次 selected 差异 | ON 首次 selected 差异 |
|---|---:|---:|---|---|
| 0 | 0.7578125 | 0.78125 | 第 8 步 | 无 |
| 3 | 0.9609375 | 0.953125 | 第 4 步 | 无 |

参考是 Transformers 5.12.1 的独立模型实现及显式本地 DeepGEMM 2.6.1 block FP8 适配器。
checkpoint skip-list 的无效前缀由权重 scale tensor 名单推导，覆盖文件单独保存；未按模型名称写入工具逻辑。
外部 GEMM 库与 OrbitKV 共用，因此不能称为所有底层数值路径完全独立的 oracle。
权重 index/config/card 已记录，未逐字节哈希全部权重。

这是相同 teacher-forced 历史上的诊断。分岔后的 OFF logits 使用参考 continuation，
不能代表原 OFF 自由生成的后续路径。两个旧 whole-graph schedule 的 provider 选择有多处不同，
包括 decode LM head；本结果没有定位具体哪一处算子造成数值变化，也不能把变化归因于共享量化。
没有调整阈值或默认 tie policy 来令文本相同，也没有重新宣称 serving 性能达标。

## 记录

- `summary.json`：结论、逐请求数值、实际首次分岔、immutable artifact/tuning 检查。
- `oracle-vs-off-final.json`、`oracle-vs-on-final.json`、`off-vs-on-final.json`：逐步数值、实际 token、全部并列最大值和文件 hashes。
- `frozen-source-checks.json`：594 个原 v3 文件不变的证明；测试扩展路径明确列出。
- `execution.json`、`preparation.json`、`binary.json`：输入、运行源码、编译和二进制来源。
- `oracle/metadata.json`：oracle 环境和执行时 source identities。工具比较字段随后扩展，原 oracle 导出源码保存在下述完整原始目录的 `logit_probe.py`，最终比较源码为同一原始目录的 `logit_probe-final.py`；原路径后续变化不表示使用了新源码导出 oracle。
- `off.log`、`on.log`：严格 replay 与 drain PASS。两次进程分别约 41.46 / 43.63 秒，均为诊断初始化加 logits 执行时间，不是 serving 延迟。

通用接口与 tie 语义说明见仓库 `docs/logit-diagnosis.md`。后续要归因 provider，需保持其余选图及输入固定、保留逐候选身份，再做有独立数值参考的单项对照。

完整原始文件保存在 `/workspace/orbitkv/.qualification/fp8-logit-diagnosis-20260912`；紧凑包中的 `summary.json` 为它们记录绝对路径及 SHA256。

GPU 诊断完成后，当前通用测试另补 token-attention-only 执行分支及 GPU 前 manifest 校验；冻结 v3 的 hybrid 测量使用原测试副本，未把该新增分支记作设备验证。前端新增的有限值 tie 语义 ReferenceRuntime 回归已通过，未改 sampling 实现。
