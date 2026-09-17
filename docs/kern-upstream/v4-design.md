# Schema v4 设计：serving 协议进 manifest（2026-09-03 定稿）

目标：用户写自己的 manifest JSON，起任何名字，`kern verify` 在不碰 GPU
的情况下告诉他离 `kern-serve` 还缺什么；接上之后 kern-serve 里不再有任何
buffer 名或 program 名的字符串字面量。

背景讨论见 [manifest.md](manifest.md)（v3 契约）、
[serve.md](serve.md)（scheduler 现状）、[spec-decode.md](spec-decode.md)。

## 1. 问题：契约有三层，schema 只写了最底下那层

今天一份 manifest 和 kern-serve 之间的契约分三层，只有第一层在 schema 里：

| 层 | 在哪 | 内容 |
|---|---|---|
| schema | `kern-manifest` | `programs` 是 `BTreeMap<String, Vec<Call>>`，名字无意义。唯一例外是 `spec` 块：一个"runtime 不解释、driver 读"的 caller 契约 |
| runtime | `kern-runtime` | **按结构推导，不按名字**：`page_tables()` / `seq_tables()` 由 `domain.index_into` 指向 paged state 还是 per-seq state 决定。这是做对了的样板 |
| caller | `kern-run` + `kern-serve` | 全部硬编码 |

caller 层硬编码的清单（scheduler.rs、kern-run/lib.rs、test.rs）：

- program 名 9 个：`prefill` `decode` `decode_batch` `decode_spec` `draft`
  `verify` `draft_precompute` `advance` `round`；
- input 名 8 个：`token_ids` `positions` `slot_mapping` `seq_lens`
  `cu_seqlens_q` `block_table` `anchor_token` `num_accepted_tokens`；
- output 名 3 个：`next_token` `draft_tokens` `verify_tokens`；
- var 名 2 个：`tokens` `seqs`（`Manifest::seq_slots()` 还认 `rows`，
  一处名字依赖已经漏进 schema crate）；
- 嗅探 3 处：`prefill_emits` = prefill 有没有写 `next_token`；`advance`
  存在 ⇔ `num_accepted_tokens` 存在；`round` 存在 ⇒ fused；
- 写死的填充约定：positions / slot_mapping 是 i64、seq_lens / cu_seqlens
  是 i32、`cu_seqlens_q = [0, c]`、decode 的 `seq_lens = pos + 1`、draft
  行是 `[anchor, mask × (block−1)]`、verify 行是 `[anchor, drafts]`、
  host 做最长前缀接受。

证据是它已经裂了：k3 的 manifest（`examples/k3-ep4.json`）只有
`decode`，var 多了 `rows`，没有 `positions` 和 `cu_seqlens_q`。今天
`Contract::check` 直接拒绝它。同一个仓库里两个生成器已经说两种方言。

## 2. "program 不是随便的字符串"其实是两件事

一个 program 名今天混着两个正交的东西：

1. **填充约定**：调用前 caller 往哪个 input 写什么。token id、position、
   slot、seq_len、cu_seqlens、line 下标。这是**逐 buffer** 的语义，不是
   逐 program 的。
2. **形状**：这个 program 接受一次多大的调用。每序列几行、最多几个序列。

第二件事 runtime 已经用 `index_into` 解决了一半：input buffer 声明"我是
什么"，caller 按声明填。把这个模式推到底，就是 `fill`。

第一件事经过消融只剩形状。曾考虑过 `role`（prefill / decode / draft /
verify / precompute / advance / round），逐个去掉后发现：

- verify 是 target 在每序列 n+1 行上的 forward，output 每行一个 token；
- draft 是 draft 模型在每序列 block 行上的 forward，权重和 state 不同是
  manifest 内部数据流，scheduler 不碰；它的页表已经是通用机制填的；唯一
  让 host 知道"这是 draft"的是往 token_ids 填 `[anchor, mask…]`；
- `round` 已经证明 splice 和 accept 可以在设备上做（`splice_verify`、
  `spec_accept` 两个手写核）；把 mask 也 splice 到设备上、把接受数作为
  output 暴露，round 就是"每组 8 行、每组出 count 个 token"的普通 forward；
- prefill 是"1 组 c 行"的 forward，decode 是"b 组 1 行"的 forward。
  admission 和 step 用同一个选择函数，prefill 只是 policy 里"新 prompt 先
  以一个宽组进状态"那一步，不是 manifest 概念。

于是角色只剩一个值 `forward`，一个只有一个值的枚举是零信息量：它唯一
能表达的"这个 program 是不是被 serve 驱动的"，已经被形状声明的有无表达
了。**删掉 role，形状声明的存在就是角色。**

## 3. v4 的 schema 变化

三样东西，没有第四样：

### 3.1 buffer 上的 `fill`（闭集）

```json
"token_ids":    {"dtype": "i64", "shape": ["tokens"], "kind": "input",  "fill": "token"},
"positions":    {"dtype": "i64", "shape": ["tokens"], "kind": "input",  "fill": "position"},
"slot_mapping": {"dtype": "i64", "shape": ["tokens"], "kind": "input",  "fill": "slot"},
"seq_lens":     {"dtype": "i32", "shape": ["seqs"],   "kind": "input",  "fill": "seq_len"},
"cu_seqlens_q": {"dtype": "i32", "shape": [257],      "kind": "input",  "fill": "cu_seqlens"},
"next_token":   {"dtype": "i64", "shape": ["seqs"],   "kind": "output", "fill": "tokens"},
"verify_tokens":{"dtype": "i64", "shape": ["seqs", 8],"kind": "output", "fill": "tokens"},
"nacc":         {"dtype": "i32", "shape": ["seqs"],   "kind": "output", "fill": "count"}
```

- input fill：`token`（每行一个，或形如 `[seqs]` 时每组一个即 anchor）、
  `position`、`slot`、`seq_len`、`cu_seqlens`。line 下标不加 fill，
  `index_into` 一个 per-seq state 本来就是它；页表同理。
- output fill：`tokens`（形如 `[seqs]` 每组一个，`[seqs, r]` 每行一个）、
  `count`（每组接受几个；缺省恒 1）。
- 和 `domain` 平级、可共存。caller 按 buffer 声明的 dtype 编码，不再写死
  i64 / i32；domain 校验照常。
- 没有 buffer 要 `position`，caller 就不写。k3 没有 positions 不再是契约
  不符。
- 闭集，和表达式集合一个原则：填空模板，不是语言。加一个 fill 就是 bump
  schema，这是有意的。

### 3.2 program 上的 `batch`（可选）

`programs.<name>` 从 `Vec<Call>` 变成 `{"batch"?: {...}, "calls": [...]}`：

```json
"prefill":      {"batch": {"groups": 1,   "rows": "tokens"}, "calls": [...]},
"decode":       {"batch": {"groups": 1,   "rows": 1},        "calls": [...]},
"decode_batch": {"batch": {"groups": 256, "rows": 1},        "calls": [...]},
"round":        {"batch": {"groups": 128, "rows": 8},        "calls": [...]}
```

- `groups`：上界，总 ≤ `seqs.max`。bs=1 的 decode 靠它比 `decode_batch`
  紧。
- `rows`：每组行数，**精确值**（round 的 8 行是 mask 布局，调 3 行就是
  错的），或字面 `"tokens"`：一组、行数就是这次调用的 tokens（prefill 的
  chunk 随末块变短）。
- 没有 `batch` 的 program 不被 serve 驱动：kern test 按段切的材料、ep0 的
  barrier、k3 的单层 MoE 测试。
- 不能从 output 反推：纯 KV 的 prefill 没有 output。

### 3.3 删掉的

- `spec` 块整个删除。`block` 就是 `rows`；`mask_token` 进 manifest 做常量
  （`splice_draft` 的字面量实参）。
- `Manifest::seq_slots()` 不再认 `rows` / `seqs` 名字，读所有 `batch` 的
  `groups` 上界。
- `schema_version` → 4，golden 重生成。无兼容层。

## 4. 类型与分层：谁拿到什么

今天 kern-serve 拿原始字符串递给 `Runtime::load`，runtime 内部
`from_json + verify`，scheduler 再回头读 `rt.manifest`；serve.rs 为了取
`model` 又用 `serde_json::Value` 解析第二遍，run.rs 为算 capacity 也解析
第二遍。三个 caller 都在重复 runtime 的活，说明边界画错了。

```
JSON ──from_json──▶ Manifest ──verify──▶ Verified ──Protocol::check──▶ Protocol
                                            │                              │
                                            ▼                              ▼
                                   Runtime::load(&Verified, …)     KernScheduler::new(rt, protocol, policy)
```

| 类型 | 证明了什么 | 带什么 | 谁消费 |
|---|---|---|---|
| `Verified` | 声明自洽：引用解析、dtype、读写序、grid 在界内。"runtime 能执行它" | `Manifest` 的 newtype，不加信息 | runtime、kern test |
| `Protocol` | manifest 与 serving 循环之间的契约成立。"一个循环能驱动它" | 派生事实：每个 batch program 的形状、每个 fill 对应的 buffer 和 dtype、line/page table 的行列、`seqs` / `tokens` 界 | kern-run、kern-serve |

- `verify` 返回 `Result<Verified, VerifyErrors>`；`Verified` 只能这样构造。
- `Runtime::load` 收 `&Verified`，内部那行 `verify(&manifest)?` 删掉，
  类型系统替它保证。runtime 继续只认 `index_into`，不认 fill 和 batch。
- `Protocol::check(&Verified) -> Result<Protocol>` 纯函数，不碰 GPU。
  kern-serve 在起 scheduler 线程之前、装权重之前就跑它；今天
  `Contract::check` 在权重绑定后才跑，一个 fill 拼错要等两分钟。
- 顺序由类型强制：Protocol 只收 `Verified`。
- `Protocol` 是只读投影。scheduler 和 kern-run 不再翻
  `rt.manifest.buffers[..].shape`；`stage_lines` / `Caller::new` 里那些
  pattern match 全是 Protocol 构造时该算好的。做完后 `rt.manifest` 在
  caller 侧只剩 kern test 在用，因为它真的要按 call 切 program。

Protocol 的检查项（从 `SpecPlan::check` / `Contract::check` 搬来，
放进 verifier 之后的第二遍）：

- 每个 fill 至多一个 buffer；`token` / `position` / `slot` 形如
  `[tokens]`；`seq_len` 形如 `[seqs]`；`cu_seqlens` 一维、长度 ≥
  `seqs.max + 1`、`monotone`；
- `tokens` output 形如 `[seqs]` 或 `[seqs, r]`，后者 r 必须等于某个 batch
  program 的 `rows`；`count` 形如 `[seqs]`，且只在存在 `[seqs, r]` 的
  `tokens` 时合法；
- batch program：`groups ≤ seqs.max`，`groups × rows ≤ tokens.max`
  （`rows = "tokens"` 时 `groups = 1`）；两个 program 同形状是错误；
- 至少一个 batch program（否则不是 serve 的 manifest，报的是这句话）。

## 5. scheduler 的一步

plain decode 和投机轮是**同一段代码**：

```
r = policy 选的每组行数            # 来自 manifest 声明了的形状
for seq: stage (next, pos, r 个 slot)
run 接受 (b, r) 的 program
读 tokens [groups, r]，读 count [groups]     # 没 count 视为恒 1
每组 emit tokens[g][..count[g]]，pos += count[g]
```

- **选 program**：形状包含 `(b, r)` 的 program 里 `groups` 上界最紧的
  那个。`if b == 1 { "decode" } else { "decode_batch" }` 消失。provider
  只给 `decode_batch`，bs=1 自动走它；多给一个 `groups: 1` 的自动被选中；
  serve.md 提过的"bs 2–16 补一个 split-KV"就是再加一个 `groups: 16` 的
  program，不改代码。
- **admission**：要一次 `(1, c)` 的调用，走同一个选择函数，命中
  `rows: "tokens"` 的 program。跑完有 `tokens` output 就读第一个 token
  （GDN 模型的 prefill），没有就不读（纯 KV）。`prefill_emits` 嗅探消失。
- **staging** 已经是统一的：`spec_round` 里的 `Group::push(ids, pos, pages)`
  往五个 fill 追加一组行，`decode()` 和 `prefill()` 各自手写的填充只是它
  的特例。
- **headroom**：租约 = `prompt + max_tokens + (r − 1)`。任何 r 行的
  forward 都写 `pos .. pos + r`，被拒的行下一轮覆写；这是通用推论，与投机
  无关。
- **`--spec` 变成 policy 旋钮**："每步取几行"，作用于 manifest 声明了的
  形状（`--rows 8`，或默认取最宽）。
- **接受率指标**：n_drafts = r − 1，accepted = count − 1，pegainfer 的
  `SpecDecodeCounters` 照旧。

## 6. 投机解码在 v4 下的样子

`round`（qwen3.8-27b-dflash2）今天是 1249 个 call 的一条 program：draft
99 → `splice_verify` → verify 939 → precompute 17 → `spec_accept` →
advance 192，一张图一次 sync。host 还做的三件事全是重复劳动，消掉：

1. 往 token_ids 填 `[anchor, mask × 7]` → 加 `splice_draft`，从
   `anchor_token`（`[seqs]` 的 `token` fill）和字面量 mask 拼进 carry，
   和 `splice_verify` 一个模子。
2. 读 `draft_tokens` / `verify_tokens` 在 host 重算前缀匹配 →
   `nacc_adv` 已经是设备算好的，暴露为 `count` output，host 直接信。
3. 租约多留 `n_drafts` → 见 §5 的 headroom。

之后 round 对 scheduler 就是一个 `(128, 8)` 的 forward。

**分相路径退役**。`draft` / `verify` / `advance` / `draft_precompute`
作为独立 program 被 host 轮流调（qwen3-4b-dspark 走的）不再被 serve
驱动：每轮四次 sync，serve.md 已经写了它不如 fused。它们可以留在
manifest 里给 kern test 按段切，只是没有 `batch`。要做的活：

- dspark 的生成器补一个 round。`splice_verify`、`spec_accept` 两个核现成；
  dspark 的 draft 7 行、verify 8 行不同宽，round 要求同宽，draft 补一行
  pad 或 verify 的 tail 改一下，二选一在生成器里定。
- `decode_spec` 收编：dflash2 的 prefill 已自己出 token、写 tap；dspark
  的 prefill 加最后一行 argmax 和 precompute 那 17 个 call，positions /
  slot_mapping 本来就是同一批，spec-decode.md 里写过这合法。收编后第一
  个 token 从 prefill 来，`first_token()` 消失。

**为什么 round 不能和 1 行的 decode 合成一个 program**：program 是写死的
launch 列表，没有"只喂 1 行就别跑 draft"这种分支。round 比 decode_batch
多出三百来个 launch，在 1 行输入下没有意义但照样会跑。工作量不同就是两个
program，这是 manifest.md 拒绝 `repeat` / `if` 的直接推论，不是投机解码
的特殊情况。两个 program 里 ~900 个 target forward 的 call 是重复的，和
64 层展开一样是故意的冗余。

**投机 manifest 里同时留 1 行和 8 行的 forward 是合理的**，两个理由都是
"要两个形状，什么时候用哪个我来选"，不是"要一个 mode 开关"：

1. oracle：投机无损，门禁是 conc1 输出逐字等于 plain。同一份 manifest、
   同一套 state 布局上跑 plain 才能把差异归因到 spec 接线，而不是 GDN 的
   spec 布局本身。这个 `decode` 跑在 spec 布局上，比 plain manifest 的
   略贵，它是测试夹具。
2. 大 batch：verify 每步 8·b 行，b 小时是访存瓶颈 8 倍行几乎免费，b 到
   128 变算力瓶颈 plain 反而快。policy 按 b 选 rows：b ≤ 32 取 8，再大
   取 1。

不要 in-manifest oracle 也不按 b 切换的，manifest 就只剩 prefill 和
round 两个 program。

## 7. 明确不做的

- `rows: "any"`（每组行数各不相同的混批）。unified 2D causal 实例本来就
  支持，vLLM 就这么用它。但这是一个新形态，现在加就是为不存在的 caller 留
  门。K5 落地的是它的最小特例：`batch.span`——每组 1 行的 decode 步里**一组**
  喂一段 run，长度进 `span` var、首行进 `span_at` fill（manifest.md
  「Serving 协议」）；混批仍不做。
- tree draft / EAGLE：只要设备端把接受的路径按行序线性化输出，还是
  `tokens + count`。需要 host 在一轮中间做决策的算法才需要角色，目前
  没有。
- policy 不进 schema：bucket 表、chunk、prefill 预算、前缀缓存策略是
  kern-serve 的；tokenizer 和 stop token 是 HF 目录的。协议说"能调什么、
  怎么填"，不说"什么时候调"。
- 跨 rank 的 batch（k3 的 `rows` var）：E 线的事，`batch.groups` 先按
  本 rank 的序列数解释。

## 8. 迁移与门禁

顺序，每步一个 commit：

1. `manifest`: `Verified` newtype，`verify` 返回它，`Runtime::load` 收
   `&Verified`。run.rs / serve.rs 的二次解析删掉。（不改 wire format）
2. `manifest`: v4 类型：`fill`、`batch`、删 `spec`、`schema_version = 4`、
   golden 重生成、`seq_slots` 改读 batch。三个生成器
   （`gen_qwen3_decode.py` / `gen_qwen35.py` / `gen_k3_decode.py`）
   输出 v4，`examples/` 重生成。
3. `manifest`: `Protocol::check`，§4 的检查项，单元测试用 `plain()` /
   `speculative()` 夹具逐条拒绝。
4. `k3` / `qwen3`: dspark 补 round、prefill 收编 precompute、
   `splice_draft`、`count` output。
5. `kern-serve` + `kern-run`: 按 Protocol 驱动，§5 的一步；删
   `SpecPlan` / `Contract` / `first_token` / `prefill_emits` / `DRIVEN`
   / `DECODE_LIKE`；`--spec` → `--rows`。

门禁：

- `kern test`：qwen3-4b / qwen3.8-27b / k3-4l 三份 manifest v3 → v4 前后
  logits 位一致（wire format 变了，运行的东西没变）。
- kern-serve：conc1 输出逐字等于 `kern run`；dflash2 接受率不塌
  （serve.md 现有数字为基线）；dspark 走 round 后 conc1 逐字等于分相
  路径的输出。
- `kern verify examples/k3-ep4.json` 报的是"没有 program 接受 (1, tokens)
  的调用"，不是"缺 `prefill`"。
- CI：`crates/kern-serve` 和 `crates/kern-run` 里 grep 不到
  `"token_ids"` / `"prefill"` / `"decode_batch"` 等任何 fill 或 program
  名字面量。这条是机器能查的，进 CI 不进 CLAUDE.md。

## 9. 落地记录（2026-09-03，tray03）

按 §8 的顺序落地。与设计稿不同的决定：

- **`once`**：k3 的 `tp_init`（allreduce 的 poison 预填）不是 forward
  也不是 kern test 材料，装载后跑一次即可。program 多一个 `once: true`，
  与 `batch` 互斥；runtime 不解释它，`Protocol.once` 列出来由 caller
  在 peers 导入之后跑。
- **`error` fill**：tray 的 `tp_err` 输出以前靠名字认，现在是
  `fill: "error"`，非零即该步失败。
- **`Protocol::check(&Manifest)`**：`Verified` 只是 `Deref` 到
  `Manifest` 的 newtype，Protocol 在类型上不要求它——verify 与 Protocol
  各证各的，单元测试用小夹具直接构造 Protocol。`Runtime::load` 仍只收
  `&Verified`。
- **轴从 fill 派生**：不认 `seqs` / `tokens` 名字。行轴 = `slot` fill
  的 var，组轴 = `seq_len` fill 的 var，fill 或 line 表跨过的第三个 var
  是 tray 轴（k3 的 `rows`）。§4 写的"`seqs` / `tokens` 界"就是这两个。
- **dspark 的 round 是 7 行**（`[anchor, d0..d5]`，6 个 draft），不是
  设计稿说的"draft 补一行"：draft 与 verify 同宽最省事，头一行是
  anchor，verify 少看一个 draft，每轮最多取 7 个而非 8 个 token。
- **prefill 收编 precompute 与 head**：dspark 的 prefill 尾部加
  `last_row`（把末行 hidden 拷成 `[1, H]`）→ lm_head GEMM（m=1）→
  `argmax_row`，再跑 precompute 把整个 chunk 的 tap 写进 draft KV；第一
  个 token 从 prefill 来（`Forward.emits`），`decode_spec` 与
  `first_token()` 一起消失。
- **GDN 的常量 nacc**：qwen3.8 的 `decode` 与 round 里的 draft 用同一批
  "按接受数推进"的核，1 行时推进数恒为 1——`nacc_one` carry 由
  `ones_i32` 核在 program 开头写 1，不要 host 写常量的 input。
- **`kern verify <manifest>`** 子命令：打印 Protocol 的全部事实（行 /
  组 / tray 界、每个 fill 的 buffer 与轴、页表、line 表、每个 forward
  的形状与 emit / count、once 列表），协议不成立退出 1。
- **`--spec` → `--rows`**（kern run、kern-serve 同名）：每组行数，缺省
  取 manifest 声明的最宽，`--rows 1` 就是 plain decode。kern-serve 的
  `Plan::check`（rows ≤ 页、rows − 1 ≤ 前端计数器上限、rows > 1 要有
  chunk program、`max_seqs` 按 `tokens.max / rows` 与 `groups` 上界收
  紧）在装权重之前算好；`step()` 一段代码，headroom = rows − 1。

门禁（tray03 GB300，2026-09-03）：

- `kern test qwen3-4b` PASS，logits 位一致；`kern run` qwen3-4b /
  qwen3.8-27b 输出与 v3 binary 逐字节同（2.6 / 11.7 ms/step）。
- kern-serve conc1 与 `kern run` 逐字同：qwen3-4b、qwen3-4b-dspark
  （rows 7）、qwen3.8-27b-dflash2（rows 8）。
- dflash2 rows 8 的输出与 v3 `--spec` 逐字节同（1.75 tok/step，接受
  10.4%，114 tok/s vs 110）。
- dspark rows 7 与 v3 `--spec`（8 行 verify）在第 19 个 token 分叉
  （"who had" / "who was"）：plain decode 在该位 top-1/top-2 的 bf16
  logit 差 0.125，verify 的 GEMM 从 m=8 变 m=7 翻了个近平局；`--rows 1`
  与 plain 逐字同，"The capital of France is" 32 token 与 plain 逐字同
  （3.56 tok/step、44% 接受、1087 tok/s；v3 分相 3.44 / 38% / 948）。
  按 CLAUDE.md 的门禁判为核噪声。
- 接受率不塌：serve.md 投机一节的表（conc32 34% / 24%）。
- `kern verify examples/k3-ep4.json` 报的是协议事实，不是"缺 prefill"。
- CI：kern-run / kern-serve 非测试代码 grep 不到任何 fill / program /
  var / buffer 名字面量。
