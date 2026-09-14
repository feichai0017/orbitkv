# FP8 region tuning：验证与诊断结果

状态：**已审阅并保留；限定正确性验证通过，性能收益未获资格**。scratch 修复及四组 v3 全模型重放通过，serving 四次运行全部完成，但两臂输出不一致。保留 v2 失败历史，按二进制版本分别记录证据。

模型卡标识 **Qwen3.8-27B-FP8**，`base_model: Qwen/Qwen3.8-27B`；配置架构为 `Qwen3_5ForConditionalGeneration / qwen3_5_text`，64 层、hidden 5120。模型卡明确说明沿用 Qwen3.5 架构基础。发布名称与 HF 架构类分别记录，来源及 SHA 见 [environment.json](environment.json)。

## 已完成的 B1 诊断

| 产物 | cold / replay 秒 | replay decode 中位数 | LM head（profile均值） |
|---|---:|---:|---|
| 冻结 baseline | 378.64 / 39.54 | 24.286 ms | CuBlasLt / 0.673 ms |
| 冻结 control | 441.47 / 37.11 | 29.072 ms | GenericMatmul / 5.356 ms |
| 冻结 shared | 279.00 / 39.27 | 23.639 ms | GEMV / 0.6615 ms |

三份 B1 产物均通过 cold → strict replay → profile 的八步 oracle 与 drain。baseline 的最高 max_abs 为 0.5234375，control/shared 为 0.625。baseline 时间取 Rust test 总时长，control/shared 取子进程时长；都复用已有 JIT 缓存。decode 中位数取 decode-2..7，包含诊断 logits，**不是服务 TPOT**。

Shared 相比 control 的诊断中位数下降 18.69%，但 LM head 同时由 GenericMatmul 换成 GEMV（5.356 → 0.6615 ms）。所以该整体差值不能归因于共享量化；更不能套用到最终 v2 或 serving。两臂各自仅保留一个 direct winner 再进行 CUDA Graph 测量，搜索选择存在混杂。

实际 profile 与选中 artifact 的依赖边、循环展开计数一致：

- baseline/control：prefill 与 decode 均为 400 个 combined DeepGemm。
- shared decode：192 个量化节点供 208 个 prequantized GEMM 使用，另有 192 个 combined；176 个量化结果单消费者、16 个双消费者。
- shared prefill：176 个量化节点供 240 个 prequantized GEMM 使用，另有 160 个 combined；112 个单消费者、64 个双消费者。

这些是实际执行的 CompiledStep/HostOp 数量，不是 egraph 候选数，也不等于 CUDA kernel launch 数。详细源文件指纹与计数在 [summary.json](summary.json)。

## 已完成的算子验证

初始算子二进制通过独立 CPU FP8/scales/padding 字节对照、四个 DeepGEMM 变体 fanout 精确输出对照、语义搜索与 schedule replay。形状覆盖 M=1,3,4,5,8,16，N=128/256，K=256。

| 双消费者算子 M（N=17408，K=5120） | combined | shared | 延迟变化 |
|---|---:|---:|---:|
| 1 | 104.208 µs | 102.546 µs | -1.60% |
| 4 | 104.222 µs | 102.709 µs | -1.45% |
| 8 | 104.362 µs | 102.614 µs | -1.67% |
| 64 | 111.280 µs | 106.622 µs | -4.19% |

每个模式为预热后的 captured graph，31 轮 × 20 次；执行顺序固定，未交错随机化。这是同输入双投影微基准，不是整模型收益。固定 M 也不能证明动态 M、最大 batch 或峰值显存。后续 semantic replay 使用 v1（二进制已有 eligibility 与输入长度优化，尚无 ID alias 修复）；最终 v2 的 alias 修复仅有六个真实 egraph host 回归，该 GPU semantic fixture 未在 v2 重跑。版本分开记录。

## 最终 v2：独立版本

- model_execution SHA256：`17f8e394ceede4f6f53e49c035460829b77658b8be827886eb3d8f340d597dfe`。
- orbitkv-serve SHA256：`da5b2dfd35baa8effe58d6dbf8ce6e7e30190fc68bcf501b4d60eb0875fe7340`。
- 构建前后编译源指纹一致；两个 capacity host tests、真实六个 serialized egraph 的 eligibility 回归及记录中的 clippy 检查通过。冻结动作本身不代表设备执行通过。

| v2 资格任务 | 记录状态 |
|---|---|
| B8 aligned | cold / strict replay / profile 全通过；每阶段 64 次对照，max_abs=0.90625 |
| B4 aligned | 同源 artifact 的 strict replay / profile 全通过；每阶段 32 次对照，max_abs=0.625 |
| B4 ragged | strict replay 失败，exit 101；profile 未运行 |
| B8 ragged | 因前序失败而未运行 |
| serving off/on | 本版本未运行；后续 v3 结果见下文 |

B8 cold 的完整子进程为 955.551s，其中测试 `compile_or_load` 区段为 953.120s；strict replay 的对应数字为 44.422s / 42.524s。B4 aligned strict replay 为 45.625s / 44.015s。区段计时覆盖该调用内的 decoder 准备、加载与搜索，不能视为纯编译阶段或 Rust 构建时间；artifact JSON 读取/解析位于这段计时之外。两者 replay decode 诊断中位数分别为 79.287ms / 29.082ms。B8 decode 实际选择 GenericMatmul 词表投影，带事件 profile 平均 43.036ms，占整体 96.338ms 的约 44.7%；B4 相应桶选 CuBlasLt，0.678ms。二者形状不同，这定位了昂贵的实际执行步骤，不是同形状 provider 加速实验。

**历史 v2 B4 ragged 失败**：B2/s4 预热后，测试准备切换 B4/s12 混合长度 prefill；CUDA graph materialization 报 `CUDA_ERROR_INVALID_VALUE`（active_bucket=3，materialized_bucket_limit=Some(1)）。副本 artifact 未改变，尚未完成八步 oracle/drain，不能视为 ragged 通过。该轮后续 profile 与 B8 ragged 已停止；scratch 故障的定向复现与修复见下节，v3 全模型结果单独跟踪。

一个狭窄的重排实例：B1/s4/c8 代表桶中，direct 排名第 1/3 的候选，其 CUDA Graph 分别为 28.319999/26.283830ms；第 3 个候选按实际 graph 指标快 7.19%。这证明两种评分可出现次序反转，不证明服务收益，也未把各 finalist 与最终 artifact 指纹逐一关联。

已选 B8 head 的同一 eclass 确有两个 BF16 cuBLASLt alternative，其中一个与 GenericMatmul 的输入次序相同。现有记录只证明静态候选存在；没有同形状设备测量或各 finalist 的 head 映射，无法判定 CuBlasLt 是未采样、被拒绝，还是在哪一级评分中落选。详见 [head-choice-audit.json](head-choice-audit.json)。

B8 首次搜索使用 capacity=8、七个合法桶、keep_best=3；B4 aligned 已复用同一 artifact 完成独立 strict replay/profile，ragged 状态见上表。通过的 B4/B8 aligned 阶段分别有 32/64 次对照，即八步 × 请求数；所有请求使用相同 prompt 与 teacher-forced continuation，不代表 32/64 种提示、异构请求、任意交错或 free-running 文本一致性。ragged 通过预先推进一半请求构造不同 query 长度，v2 的完整设备资格未通过，后续 v3 结果单独跟踪。

后续 v3 serving 已完成；其输出差异与选中算子差异见下文，未获得同输出性能收益资格。

## Scratch 生命周期：独立修复证据

不含 FlashInfer 的 tiny combined DeepGEMM 图在修复前通过 s4，但扩到 s12 后，于 capture preparation 后立即报告 `CUDA_ERROR_INVALID_VALUE`。外部输入地址与容量固定；该冻结二进制 SHA 为 `3f04af030854e8988d6d12411f843befe08b1f6911558708f18309726d0156cf`。更早两次因 fixture 容量设置失败的尝试不作为故障证明。

代码追踪定位到捕获内 `device_ptr` 记录的 cudarc event 与 scratch 释放顺序。修复在捕获外取得 raw pointer，每个 child graph 保留捕获时的确切 `Arc` owner；resident 切换一起搬移 owner。新分配发布前同步分配流，重建、释放及 graph→direct 切换前同步执行流，按 executable → parent graph → child graph → owner 顺序退役。没有关闭全局 event tracking。

修复后二进制 `daa2857010cc6e1142417ed946fdf7972adaa7fb5ae43cb9e6a56b6589ae9ed5` 的回归通过：combined/reference × 普通重建/resident 四组，均执行 **4→12→4→24→1→4，共 24 次**，另有两个 freeze 后 replay。N=K=128，独立常量点积 oracle 精确检查 BF16；同时断言旧 scratch 只在 resident graph 需要时存活、切回缓存不重建、销毁后释放且无 deferred CUDA error。总测试 8.73s 是回归耗时，不是性能比较。

三个生产文件的最终 SHA 与 GPU 测试二进制记录一致；GPU 后仅测试代码做了 clippy 语法清理，测试源码前后 SHA 分别保留。host shared tests 5/5（3 个 GPU 测试被忽略）、capture tests 6/6、clippy 通过。内存预算按已分配高水位与计划容量的较大值保守估计，不能视为精确去重显存或峰值 VRAM。完整日志、物理行号及源码快照见 [scratch-lifetime-audit.json](scratch-lifetime-audit.json)。

v3 已独立重放原 B8 artifact，四组状态见下表；未重新 cold search。本节的 tiny 回归与全模型资格分开记录，不更改 v2 失败历史。

<!-- v3-status-start -->
## 最终 v3：修复后的独立重放

模型测试二进制 SHA256：`7b3b63cee9922369310f3654432e13f092af6e1c1dccae041824c2e761c84204`；server：`ac44ab7f9ee8bdd3c273e15e4218c4d010a5b18e27630bfcf5f1d2a3806fcbfd`。构建前后源码指纹一致，三个 scratch 修复生产文件与定向 GPU 回归记录逐一匹配。仍使用 v2 B8 cold 搜索产生的同一个 artifact；本轮没有 cold search。

| v3 资格任务 | 状态 | strict / profile 子进程秒 | strict decode 诊断中位数 |
|---|---|---:|---:|
| b4-ragged | strict / profile 通过；每阶段 32 次对照，max_abs=0.625 | 44.824 / 46.326 | 29.060 ms |
| b8-ragged | strict / profile 通过；每阶段 64 次对照，max_abs=0.78125 | 53.236 / 46.076 | 79.275 ms |
| b4-aligned | strict / profile 通过；每阶段 32 次对照，max_abs=0.625 | 49.429 / 43.472 | 29.053 ms |
| b8-aligned | strict / profile 通过；每阶段 64 次对照，max_abs=0.90625 | 43.982 / 44.623 | 79.281 ms |

首次形状使用另记：B8 ragged 的 s24 prefill 在 strict replay 中为 7.771256099s，随后独立 profile 进程为 0.299645327s；前者不包含在 43.445779354s 的 compile_or_load 区段内。当前日志没有阶段归因；源码允许按实际 M 调用 provider ensure_compiled，因此首次 setup/JIT 只是可能原因。相同 selected artifact 重放不代表所有动态形状已预备完毕或没有后续 JIT。

完成项均由独立 strict replay/profile 进程检查八步 per-request oracle、drain 与 artifact 不变。decode 指标含 logits 诊断，不能称为服务 TPOT；子进程时间包含加载、准备和完整验证。相同 prompt 与 teacher forcing 的覆盖限制仍然适用。Serving 四次运行已完成，结果见下节；本包保留限定正确性与已完成的服务诊断，不晋升性能收益声明。
<!-- v3-status-end -->

## v3 Serving：已完成的诊断结果

两个服务端均为冻结 v3 OrbitKV；vLLM 0.29.0 仅作为负载客户端。两臂都用 keep_best=3，off/on profile 仅共享量化开关不同。两轮顺序为 **off→on→on→off**；每次 C1、8 个随机请求、输入 4/output 8、seed=0、temperature=0。四次共 32 个正式请求，全部完成且无错误，长度一致；客户端 generation ready check 与 warmup 均关闭；initial-test banner 不表示额外生成请求。首 prompt 的 `/tokenize` 长度检查另计。实际线上 prompt/token-ID trace 未保存，不能把它视为 32 种独立提示。

| epoch / arm | 输出 tok/s | median TTFT | median TPOT | readiness |
|---|---:|---:|---:|---:|
| 1 / off | 20.012 | 126.365 ms | 40.570 ms | 287.423 s（stored） |
| 1 / on | 20.336 | 136.157 ms | 38.368 ms | 232.273 s（stored） |
| 2 / on | 20.658 | 131.866 ms | 37.934 ms | 38.042 s（loaded） |
| 2 / off | 20.249 | 121.229 ms | 40.448 ms | 38.042 s（loaded） |

配对比值中位数：输出吞吐 **1.0182×**、TPOT **0.9418×**、TTFT **1.0826×**；p95 TTFT 也为 1.0767×。只有两个 epoch，且两个 epoch 都在输出索引 **0、3** 出现两臂文本差异；各臂自身跨 epoch 的输出摘要完全重复。没有这些提示的独立 oracle，不能把差异解释成已知 near ties。**`performance_qualified=false`，共享量化继续默认关闭。**这是已完成的诊断实验，不是未完成 benchmark，也不是 vLLM/SGLang 服务端性能对比。

保存/加载证据明确：epoch 1 两臂日志分别记录 stored schedule，epoch 2 记录 loaded，后者没有搜索记录。两份 artifact、运行输入及冻结 binary 的 SHA 均已核验。readiness 是到服务就绪的完整等待，保留已有 JIT/provider 缓存，不能当成纯 JIT 或 Rust 编译时间。

选中 artifact 同时揭示明显混杂：decode 的 off LM head 是 GenericMatmul，on 是 cuBLASLt；on 的 decode 80 个独立 quantizer 全为单消费者，量化总数仍为 400，没有多消费者共享节省。prefill 的 on 为 176 combined +224 prequantized GEMM，176 个独立 quantizer，将量化数由 400 降至 352，减少 48 次。decode 13/13、prefill 8/13 个对应 rolled FP8 投影组的 variant 也不同。以上按实际选中 IR 的循环次数展开，是算子数而非融合后的 kernel launch 数；整体时延差异不能归因于共享量化本身。

客户端还记录较大的首个流式间隔：epoch 2 的 off/on 中位数为 **113.647/120.233 ms**，随后六段为 **28.247/24.205 ms**。原始两段尾部近零间隔保留。冻结代码将 materialized bucket 上限设为 1，prefill→decode 重建可能贡献首次间隔，但缺少阶段 trace，不能直接归因。客户端 `max_concurrent_requests=4` 是整秒分桶累计的派生值，不表示实际 C4；本轮配置为 C1。

事后重建 receipt 使用本地固定版本客户端的实际 sampler，对四份保存的 Namespace 均重建出相同八个提示，并核验 33 个相关文件指纹。差异索引 0 的本地 token IDs 为 `[210992,210993,210994,210995]`，索引 3 为 `[66921,66922,66923,66924]`，可用于下一步独立 logits probe。它是本地重建，不是线上请求体或服务端 token-ID 捕获；源码指纹记录于重建时，尚无这些提示的独立数值 oracle。receipt、helper 与日志指纹也保存在 serving audit 中。

完整四份原始指标、摘要复算、逐请求长度、artifact marker 行号、分布及选中 provider 审计见 [serving-audit.json](serving-audit.json)。

## 文件

- [summary.json](summary.json)：分版本结果、证据路径与 SHA、后续问题、适用范围。
- [environment.json](environment.json)：本地模型卡与架构、构建工具、二进制和输入指纹、缓存与测量约定。
- 外部原始日志与 artifact 路径保持指向 `.qualification`；本包是已审阅索引，没有替代原始记录。
