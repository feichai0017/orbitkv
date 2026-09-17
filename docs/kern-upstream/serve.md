# `kern-serve`：continuous batching + OpenAI 兼容 endpoint

```bash
# workspace 成员但不在 default-members 里（serving 栈不进裸 cargo build / CI 的编译）；需要 protoc、libssl-dev
cargo build --release -p kern-serve
target/release/kern-serve --manifest examples/qwen3-4b.json --kernels kernels --weights weights/Qwen3-4B --gpus 3 --port 8000   # --gpus 0,1,2,3 drives a tray
# state 池默认按显存自动定：权重/激活/scratch 分完后，剩余显存减 1 GiB 全给 state，
# KV 页与 state slot 共用这份预算、按需互换（runtime.md）。`--capacity <tokens>` 显式给则照旧。
# /v1/completions、/v1/chat/completions（流式 + chat template）、/v1/models、/metrics
```

manifest / kernels / weights 走 `--manifest` `--kernels` `--weights`（与 `kern run`
同一套 flag）。kern-serve **不读 kern.toml**：那是改 kernel 的开发循环的文件
（reference、dumps、test seed），服务进程的输入全在命令行上，和 vLLM 一样。
API 里的模型名缺省是 manifest 的 `model`，`--served-model-name` 覆盖。
**前端**要的 HF 目录（config.json、tokenizer、chat template、`generation_config.json`
的 eos）就是 `--weights` 第一项所在的目录：目录本身，或去掉 `.safetensors` 文件名、
截到第一个 `{ep}`/`*` 分片段之前（`dense-tp4/r{tp}/l*.safetensors` → `dense-tp4`）。前端整个来自 pegainfer（`pegainfer-frontend`，底下是 vLLM 官方的
Rust server crates，git dep 钉 pegainfer main 的一个 rev），kern 只贡献引擎：`crates/kern-serve`。

当前钉在 pegainfer `139d925e` / vLLM `89dbb264`（2026-09-11）。包含上游 #56260：
DSV4/V4.1 历史 tool call 的非法 JSON 或非对象参数按原文包在 `arguments` 中，不再拒绝请求。
双重编码的对象也保留原文，不按 checkpoint 的旧 `encoding.py` 二次解码；renderer 回归测试
保留旧 oracle fixture，只对这一处已知差异调整期望，仍比较完整对话文本。


## 分工

- **`kern-serve::scheduler::KernScheduler`** 实现 pegainfer 的 `Scheduler`
  契约（`submit` / `step` / `metrics`），跑在 pegainfer 的 `drive` 轮询线程
  上，独占一个 `Runtime`。策略刻意简单：
  - prefill 优先、不混批：每步先把 waiting 里能坐下的请求按序逐个（bs=1、
    按 `--chunk` 切块、逐 launch 不走图）整条 prefill 完，再对全部 running
    序列做一步 decode；最后一个 prompt token 作为首个 decode 步的输入。
    `--chunk` 缺省是 manifest 的 `tokens` 上界，只能改小。没有预算、没有分时：
    一条长 prompt 进来，正在 decode 的序列等它整条喂完（roadmap D2）。
    **2026-09-09 tray07 GB300 × 1，Qwen3.8-27B（registry cubins）**：`kern run` 35.9k token
    prompt，prefill 2048 块逐 launch 22.3k tok/s（走图 22.3k，512 块 18.2k），48 步 decode
    图 11.14 ms/step、`--eager` 11.13 ms/step，三种设置文本逐字相同；`--chunk 4096` 报
    "the manifest's `tokens` bound is 2048"（当时的 manifest）。kern-serve 同一 manifest：conc1
    completion 与之前逐字同，29.9k token 的 prompt 走 15 块 prefill 21.6k tok/s，日志里只有
    `decode` 捕图一次。同日 tray08，上界提到 8192 的 manifest（bdd2805）：5 块 23.5k tok/s
    对 18 块 22.4k，文本逐字节同；kern-serve chunk=8192，29.9k token 24.0k tok/s。
  - 准入即预留：请求在准入时向 runtime 租下最坏情况 `prompt + max_tokens`
    的全部 KV 页（`Runtime::lease` → `Lease`，序列结束即 drop 归还），
    decode 永远不缺页、不抢占。超过单序列上限（最窄页表行长 × 页）→
    `ContextLength`，超过整池 → `KvBudget`。`slot_mapping` / `block_table`
    的值只能从 `Lease` 算出来，scheduler 不碰裸页号。
  - decode 按 bucket（1,2,4,8,16,24,32,48,64,96,128,192,256）pad，声明了
    `graph` 的 program 每个 bucket 首次使用时 capture 一张图；pad 行写进
    scheduler 自己租的一页。
  - greedy：采样就是 manifest 里的 `argmax`。非 greedy 参数 warn 一次后按
    greedy 服务。EOS 本身不发出（pegainfer 约定），仍计入 `max_tokens`。
- **manifest**（`tools/gen_qwen3_decode.py`）多了一个 var `seqs`（≤256）和
  一个 program：
  - `decode`：原样，bs=1 契约（3D split-KV unified + reduce_segments——挖到
    的 reduce 是 Triton 在 `num_seqs=1` 下特化出的实例，ABI 里没有
    num_seqs，只能 bs=1）；
  - `decode_batch`：`seqs` 个序列各一行（tokens = seqs），attention 用
    prefill 的那份 2D causal 实例（vLLM 自己在 num_seqs 超过 3D 阈值时 decode
    就走它），grid.x = `ceil(5·tokens/4)`——盖住 vLLM 的 q-block 索引空间
    `tokens//4 + num_seqs`（seqs ≤ tokens 下恒成立），多出的 block 核内提前
    返回；表达式集合因此不用加"两个 var 相加"。
  - 元数据按序列：`block_table [seqs, 256]`、`seq_lens [seqs]`、
    `cu_seqlens_q [257]`（shape 不能是表达式，按上界声明）、`logits [seqs,
    V]`、`next_token [seqs]`；lm_head 的 m = `seqs`。
  - 哪个 bs 走哪个核是 manifest 的选择（两个 program），caller 按 bucket
    选，runtime 不知情。以后补 bs 2–16 的 split-KV = 再捕一次 bs=2 的
    decode 拿未特化的 reduce，多加一个 program。
- **runtime** 只改一处：CUDA graph 按 `(program, var 值)` 键控（原来一个
  program 一张图），加 `is_captured`。

## 实测（GB300 单卡，Qwen3-4B，2026-09-01）

- bs=1 `kern run` 不变：2.6 ms/step，输出逐字一致。
- `vllm bench serve --backend openai --dataset-name random --random-input-len
  1024 --random-output-len 128 --num-prompts 256 --max-concurrency 64
  --ignore-eos`：256/256 成功，**5138 tok/s 输出吞吐**（总 46k tok/s），TPOT
  中位 11.7 ms / P99 11.9 ms，TTFT 中位 62 ms / P99 1.0 s，E2E 中位 1.55 s。
  引擎侧：decode 5.5 ms/step @ ~60 seq，prefill 79.5k tok/s（chunk 512 走图）。
- 一致性：8 个不同 prompt 并发两轮逐字相同（确定性）；同 batch 内 16 份
  相同 prompt 里同 cohort 的 14 份逐字相同（行没有错位）。并发 vs 串行有
  2/8 在 ~25 token 处的近平局分叉，两边都连贯——batch 大小改变 GEMM 选核
  和 attention 归约顺序，和 vLLM 一样不 batch-invariant。

## 投机解码（`--rows`，2026-09-03）

```bash
target/release/kern-serve --manifest examples/qwen3-4b-dspark.json --kernels kernels \
    --weights weights/Qwen3-4B --weights weights/dspark-qwen3-4b-b7                  # 缺省取最宽：7 行的 round
target/release/kern-serve --manifest examples/qwen3.8-27b-dflash2.json --kernels kernels-qwen38-dflash2 \
    --weights weights/Qwen3.8-27B --weights weights/Qwen3.8-27B-DFlash2 --rows 8     # 显式给；--rows 1 是 plain
```

投机不是模式开关，是 manifest 声明的一个形状：`round` program 的
`batch: {groups, rows}`（dspark 7 行、dflash2 8 行）。scheduler 装权重前
先算 `Plan`：`--rows`（缺省 manifest 声明的最宽）选出 `step` forward，
`rows − 1 ≤ MAX_SPEC_TOKENS`（前端计数器）、rows ≤ 页（pad 页要装下一组）、
rows > 1 时必须有 chunk program（prompt 逐 token 走 step 的模型只能 1 行），
`max_seqs` 按 `tokens.max / rows` 与 `groups` 上界收紧。之后 plain decode
与投机轮是同一段 `step()`：

- 每序列 stage `rows` 行（行轴的 token fill 里 `[next; rows]`，组轴的
  token fill 装 anchor），跑 `forward(b, rows)` 选中的 program，读
  `tokens [seqs, rows]` 与 `count [seqs]`（没 count 视为恒 1），每序列取
  `count` 个、`pos += count`。租约 = `prompt + max_tokens + rows − 1`：被
  拒的行落在新 pos 之后，下一轮覆写，target KV 与 draft KV 一样免费回滚。
- round 内的活全在设备上：`splice_draft`（anchor + mask 拼 draft 的
  ids）→ draft → `splice_verify`（anchor + draft 拼 verify 的 ids）→
  verify → precompute（verify 全部行的 tap 进 draft KV）→ `spec_count`
  （前缀匹配写 `count` 输出）→ GDN 模型再加 `spec_lines` + advance。一张
  图一次 sync；host 不再读 draft 再拼 verify，`nacc` 直接信。
- admission：chunk program 自己出第一个 token（`Forward.emits`：dspark
  的 prefill 尾部带 head + precompute，dflash2 本来如此）。
- greedy only；每轮 verify 是 `rows·b` 行的 target 前向，b 大到算力瓶颈
  后一轮比一步贵，`--rows 1` 关掉。分段的 draft / verify program 不再存
  在，逐轮 dump 用 `kern run --probe-dir`（按 label 后缀取 call 的输出）。

**验证**（tray03 GB300，2026-09-03，docs 段落做 prompt、512 token、
`ignore_eos`；stats 是 5 s 窗口的均值，取 running 最多的一条）：

| | conc1 == `kern run` | conc32 吞吐 | conc32 接受率 | 2026-09-01 基线 |
|---|---|---|---|---|
| qwen3-4b-dspark（rows 7，`--capacity 65536`） | 逐字同（683 tok/s，2.29 tok/step） | 5930 tok/s（32×512，含 prefill） | 3.02 tok/step、34%（running=22 的窗口） | conc32 5850 tok/s、19.2%（128 token） |
| qwen3.8-27b-dflash2（rows 8） | 逐字同（114 tok/s，1.75 tok/step） | 1709 tok/s | 2.65 tok/step、24%（running=32） | conc32 1236 tok/s |

512 token 的中文散文有几条掉进重复循环（一条 dspark conc1 接受 71%），
接受率偏高是它们抬的；不塌就是门禁。conc1 与 `kern run` 的对照 prompt
是英文 96 token（"In the summer of 1848…"），tok/s 是 `kern run` 的。
v3 的分相路径（host 轮流调 draft / verify / precompute，dspark 8 行
verify）：conc1 20.8% / conc32 19.2%，conc 1 / 8 / 32 ≈ 600 / 2560 / 5850
tok/s，普通模式 353 / 2048 / 6800——交叉点在 bs 16–32，v4 没有重测。

## 前缀缓存（K1，2026-09-02）

调度器持有一张 `Prefix` 表（kern-runtime，纯 host）：结束的序列留成 checkpoint，
新 prompt 从覆盖其真前缀的最长 checkpoint 起步（`Runtime::lease_from`），prefill
只补剩下的。两种模型一套机制，差别只在"何时留"：

- 纯 KV（qwen3-4b，页 16 token）：序列每填满一页就 `Runtime::checkpoint` 一次——页进
  共享链，不拷字节，所以任何早先的 prompt 或输出都按页粒度可复用；
- 带循环状态（qwen3.8-27b，页 784 token，GDN state 154 MB/序列）：只在请求结束时
  `Runtime::retire`——结束序列的 state slot 原样成为 checkpoint 的，不拷；因此只有
  "续着上一轮整段上下文"的 prompt 命中，同一 prompt 重发不命中（checkpoint 比它长）。
  slot 从 manifest 的 `seqs.max + 2` 个起，与 KV 页共用一份显存预算按需互换（K1b）：
  睡着的 session 的 checkpoint 拿着 slot，活跃请求再要 slot 就从空闲页拆，页不够
  再从空闲 slot 拆回来；租约 `Busy` 时按最久未命中淘汰，`Remapping` 时等它落地
  （stats 行的 `slots_used`/`slots`/`remaps`）。

命中数逐请求报给调用方：`usage.prompt_tokens_details.cached_tokens` 就是这条请求
没有付钱的 prompt token 数（`Row::prefix()`，与 stats 行的 `prefix_hit_tokens`
同源）。DSv4.1 一次两轮（tray05，prompt 6159 token、生成 96）：第二轮 prompt
6263 token，报 6254 = 6159 + 96 − 1，快照不复用最后一个 token。未命中时上游的
序列化器不发 `prompt_tokens_details`，字段是 `null`。

门禁（本机 GB300，warm 服务与 cold 服务各一，greedy，`max_tokens 64`，prompt 为 46 KB
工程日志）：

| 模型 | prompt | cold prefill | 命中 | 命中后 prefill | 多轮 R3 warm vs cold |
|---|---:|---:|---:|---:|---|
| qwen3-4b | 13 983 tok | 888 ms（请求 1.23 s） | 13 968 tok | 27 ms（请求 0.36 s） | 64 token 逐字一致 |
| qwen3.8-27b | 14 256 tok | 请求 1.90 s | 14 319 tok（上一轮全文） | 请求 0.85 s | 64 token 逐字一致 |

qwen3.8 重发同一 prompt 不命中（唯一的 checkpoint 比它长），输出与首发一致。qwen3-4b
重发同一 prompt 命中 13 968，输出在第 10 个 token 后与首发分叉；两次命中之间完全一致。
分叉不是缓存的：同一 prompt 在 cold 服务上用 `--chunk 144`（末块同样是 15 个 token）
全量 prefill，得到第三种续写——三种切块三种 attention 归约顺序，这个位置是 bf16
近平局，与 `decode` / `decode_batch` 之间的分叉同类。
host 侧回放见 roadmap K1 / K1b 行（`crates/kern-run/examples/agentx_replay.rs`）。

K1b（页与 slot 共用预算）后的门禁（2026-09-02）：上表两行的输出逐字不变；qwen3.8-27b 单卡
预算 223.9 GiB = 9551 块 × 24 MiB，起步 4287 页 + 130 slot；200 个不同的短请求依次结束后
slot 长到 191（每个 checkpoint 拿一个，活跃请求再要就从空闲页拆，61 次 remap），此后重发
早先的 prompt 与首次输出一致、46 KB prompt 的 64 个 greedy token 与 cold 服务一致。
这条门禁第一次跑就抓到一个 manifest 的 bug：`gen_qwen35.py` 把 `(seqs.max + 2) × 48 = 6240`
当 vLLM conv kernel 的 `num_cache_lines` 写成字面量，kernel 用它做越界掩码，slot ≥ 130 的
conv state 静默丢掉（短 prompt 单块 prefill 从零态起步看不出来，14k 的 prompt 第二块起就错）。
按 slot 编号二分定位：页编号高没事、`kern run` 单序列没事，只有 kern-serve 里 slot ≥ 130
的多块 prefill 出错。manifest 里不许再出现 slot 数：conv kernel 现在拿 i32 最大值，掩码永不生效。

带状态模型的部分命中在算力上没有意义：state 快照之后的 token 必须整段重跑 forward
才能把状态推过去，attention 层的投影和注意力都省不掉，所以有效命中 = 最深的带
state 的 checkpoint，KV 页只是顺带共享（vLLM v1 也是这样：hit 取各层组最小值，
attention 的命中被截到 mamba 块边界）。快照放哪由调用方定——请求结束（agent 多轮）
和显式断点（roadmap K1b），不在每个块边界都存。

## 睡到 DRAM 的 session（K3，2026-09-03）

`--host-gib N` 在 tray DRAM 里 pin 一块 N GiB（`Runtime::reserve_host`，落在这张卡的
NUMA 节点上）。租约 `Busy` 时最久未命中的 resident checkpoint 不再直接丢，而是 park
到这块 host 内存（`Runtime::park`，页和 slot 都拷；host 放不下就先丢最冷的 parked），
runtime 攥着它直到拷贝落地才还页；命中 parked checkpoint 的 prompt 走 `Runtime::wake`：
按请求的最坏长度一次取够页、把前缀拷进去，请求在 `waking` 队列里等 `Runtime::awake` 交出
这份 `Lease`，落地即开跑（2026-09-15 起，`pool.md`；之前先醒成 `Checkpoint` 插回索引、再
`lease_from`，两次分配在 DSv4.1 EP4 的第二轮上 wake ↔ park 活锁，见 lessons.md），其间
decode 步照常走——compute stream 从不等 transfer stream。stats 行多了
`parked / host_gib / parks / host_evictions / wakes / wake_tokens`，以及按 tier 分的命中
`resident_hits / resident_hit_tokens / host_hits / host_hit_tokens`（host 层有没有被打到
一眼可见）。

门禁（tray08 GB300 单卡，qwen3-4b，`--capacity 32768 --host-gib 32`，greedy 64 token，
prompt 8189 token）：第一次请求 cold prefill；14 个不同的 12–13k 长 filler 把它挤到 host
（16 个 checkpoint 中 14 个 parked，24 GiB）；重发同一 prompt 命中 8176 token 的 parked
条目，wake 后 prefill 19 ms、请求 0.31 s（首发 0.67 s），64 个 token 与首发、与 cold 服务
逐字一致。

decode 抖动（同卡，`--capacity 65536 --host-gib 48`，8 路 stream 各 256 token 的 decode
负载，另一路 churn 轮流重发 3 个 25k-token 的 prompt、`max_tokens 1`、按 token id 发，
每个请求 wake 3.5 GiB；40 s 一组，`target/gate-logs/jitter.py`）：

| 负载 | token 间隔 mean / p50 / p90 / p99 (ms) | churn 请求 |
|---|---|---|
| 只有 decode（无 host 层） | 2.92 / 2.92 / 3.10 / 3.22 | — |
| decode + churn，命中 resident（无 host 层，池够大） | 31.3 / 3.03 / 85.4 / 85.6 | 89 ms |
| decode + churn，命中 parked（每个请求 wake 3.5 GiB） | 26.1 / 3.08 / 83.6 / 85.5 | 104 ms |
| 只有 decode（host 层开着） | 2.95 / 2.94 / 3.11 / 3.25 | — |

拷贝本身给 decode 添的抖动在 p50 上是 3.03 → 3.08 ms（+1.7%），p90/p99 不变，wake 的
成本（+15 ms）全落在被唤醒的那个请求自己身上。p90 的 85 ms 与 host 层无关：命中之后
剩下的几个 token 走 `prefill`，这一块在 25k 上下文上要 40 ms（挖来的 prefill attention
核只按 head 并行，长上下文下带宽吃不满），prefill-first 的每步又把它排在 decode 前面——
resident 命中同样如此。这是 K5（prefill 作为 decode 步的 filler）要解的，不是 K3 的。

同一 session 醒来后再睡，醒进 lease 的整页节点带着 host twin，只拷新增的页（`pool.md`）。

## DSv4.1 Flash 的 host 层命中（2026-09-11，tray05）

四卡 EP4、`--capacity 8192 --host-gib 4`：2013 token 的开场白 + 312 token 的回答（EOS
结束）留下 checkpoint；第二轮立刻问命中 resident（`prefix_hit=2325`，prefill 14 ms）；
32 条 2.5–3.3k 的 filler 把每张卡的池子打满（`parks=8`，日志 `parked tokens=2325`）后再问
同一轮，`woken=true`、新租 19 页、prefill 12 ms，**64 token 与 resident 命中逐字相同**。
新起的 server 上冷 prefill 同一 prompt 在第 35 个 token 处同义替换后重新汇合（回答的 312
token 一边走 decode 步一边走 prefill chunk，核和累加顺序不同，近平局翻转）。

有 per-seq state 的模型只在快照的整长上可用，所以命中还要求 chat template 重新渲染的
历史 re-tokenize 出模型当初生成的那串 id：被 max_tokens 截断在词中间的回答（64、400 都
试过）再问一轮全是 `prefix_hit=0`，resident 也不命中。记录与脚本在
`bench_results/2026-09-11-dsv41-host-tier/`。

## 每步的 host 空档（2026-09-11，tray05）

1M context / 每卡 256 seq 的 manifest 上，`stats` 报 `step_ms=7.73`，但 5 秒只跑完
534 步（每步 9.36 ms）——每步丢了 1.6 ms，32k 的 manifest 上没有。pegainfer 的
driver 每步调一次 `Scheduler::metrics()`，`KernScheduler::metrics` 读
`pages_used()` / `pages_total()`，而这两个数当时是把整条页状态数组扫一遍数出来的：
扫描量正比于池子，池子正比于 context（`--capacity 32768` 1132 页/卡空档 0.06 ms，
host 表 146k 页 0.13 ms，HBM 表 1.62M 页 0.91 ms，1M/256 seq 3.37M 页 1.64 ms）。

`Pool` 改成写状态时维护计数（`Inner::set_page` / `set_slot` 是唯一的写入口，
`Tally { live, held }` 随之移动），`total` / `used` / `slots` / `slots_used` /
`anything_held` 全部只读计数，不再扫描。同一 manifest 的门禁：

| | 步/5s | `step_ms` | 每步实际 | 空档 | tok/s |
|---|---|---|---|---|---|
| rows 1，扫描 | 534 | 7.73 | 9.36 | 1.63 | 107 |
| rows 1，计数 | 669 | 7.41 | 7.47 | 0.07 | 134 |
| rows 6，扫描 | 439 | 9.68 | 11.39 | 1.71 | 227 |
| rows 6，计数 | 518 | 9.42 | 9.65 | 0.23 | 274 |

16 条 greedy conc1 的输出两种 rows 都与改前逐字节相同。记录在
`bench_results/2026-09-11-kern-step-host-gap/`。

## 每步的拆分（2026-09-14，tray18）

打点版 kern-serve（每 rank 三个 CUDA event：写输入前、graph 前、graph 后，加各 host 段的
`Instant`），1M/256-seq manifest，16 条 greedy conc1，每步均值（µs）：

| | rows 1 | rows 6 | 32k manifest rows 1 |
|---|---:|---:|---:|
| 整步 | 7360 | 9340 | 7220 |
| GPU graph | 7070 | 9033 | 7063 |
| graph 前的 H2D | 245 | 242 | 88 |
| host 超出 GPU 的部分（stage 31、launch、wake、读回 18） | ~100 | ~110 | ~80 |
| 两步之间（driver、metrics、ledger） | 5 | 4 | 8 |

graph 占 96%，步间已无空档。H2D 里 155 µs 是 `write_input_at` 把整块 staging 拷上去：
page_table 是 256 × 8192 i32 = 8 MiB，每步每卡整块 DMA。改成只拷写入的字节后 H2D
245 → 109 µs，步时 7.36 → 7.22 / 9.34 → 9.21 ms，输出 16/16 逐字节相同；剩下的 ~100 µs
是 12 次 DMA 的串行发射延迟。记录在 `bench_results/2026-09-14-dsv41-step-host-split/`。

## tray 级（E5 第四块，2026-09-03；t=4 门禁 2026-09-04 过）

`kern-serve --gpus 0,1,2,3` 一个进程驱一个 tray：`tray.rs` 持 n 个 `Runtime`，单线程
按序驱动（graph launch 1–3 µs，一线程驱四卡与四线程等价，实测见 multi-gpu.md），每步
所有 rank 跑同一个 program、同一组 var 值、同样的行数；manifest 的 `topology` 里 `tp`
组是"一个 batch 的组"（连续的 rank），其余组（`ep`）跨全部 rank；跨 tray 的 world
不在这里（rendezvous 是 harness 的事）。scheduler 只看得到 `Row` / `Snapshot` /
`Sleeping` / `Rising` 和 `Cell`，看不到 `Runtime`、`Lease` 或某个 rank 的输入。

- **先全部 launch 再逐个 sync**：一步里 n 个 rank 的图先全部 launch（`Runtime::enqueue` /
  `enqueue_captured`），再逐个 `synchronize`——EP 的 dispatch 与 tray collective 都在核里等
  peer，先 sync rank 0 就等到核的超时（`CUDA_ERROR_LAUNCH_FAILED`，2026-09-03 E3 第一次
  跑 K3 EP4 就撞上；qwen 单卡 smoke 看不出来）。
- **一行 = tp 组每个成员一份**：owner 卡持 MLA 页 + KDA slot（`Runtime::lease`），
  peer 卡只持 slot（`Runtime::lease_slot`，runtime 新增 slot-only 租约，见 runtime.md）。
  lease / lease_from / fork / checkpoint / retire / park / wake / awake 各在组里每个成员
  做同样的事；**全成或全不成靠构造**：`Row` 只在每个 rank 都给了 `Lease` 之后才存在，
  哪个 rank 说 `Denied`，`?` 提前返回把前面拿到的 drop 掉即回滚（各 rank 的 pool
  独立记账、slot 编号不跨卡比对，所以没有一致性要重建）。park 是唯一不能半途撤销
  的动作（拷贝已入队），runtime 拆成 `room`（找地方，`Room` drop 即退）+ `park`（拷），
  tray 先在每个成员上找齐再拷。`Waking` 提前 drop 会等拷贝落地，所以 wake 的回滚就是 drop。
- **owner**：新行落到"还开着（行数 < `--max-seqs`，per rank）且行数最少、再比页占用最少"的 rank
  （2026-09-04 之前只比页：页数含留着的快照，持有 12k 快照的卡在 conc8 里一行都分不到——`[3, 2, 0, 3]`），
  终身不变，后代（checkpoint、parked、wake 回来的）都跟着它——页在那张卡上，pinned
  块绑在那张卡的 NUMA 节点上。
- **`Prefix` 按 tray 键**：`Prefix<Snapshot, Sleeping>`，键还是 token 哈希链、与卡无关。
- **staging 按 fill 的轴**：fill 或 line 表跨过的第三个 var 就是 tray 轴（k3 的
  `rows`）；tray 轴上的 token fill / line 表 / tokens 输出跨组（本卡的行先、再按组序轮到
  其它成员的块，collective 假定的布局），行轴 / 组轴上的 slot / seq_len / 页表是本卡自己
  的行；qwen 的契约（没有 tray 轴）是 t=1 的特例，同一条代码。**块不等长**（2026-09-04）：
  每个 rank 的块是它自己的行数（至少 1 行 pad），组内块长之和垫到阶梯就是 tray 的行数
  （`rows` var；多组时各组垫到同一个数，垫在最小的块上；t=1 退化成每卡等块），块的前缀和写进
  `blocks` fill，collective 核按它换算行号（peer_collective.cu）；own 轴的 buffer 仍按最大块的
  bucket b 铺、pad 页每卡一页。一条序列 256 行的 run 于是让 tray 是 256 + 3 行而不是 4 × 256——
  之前 768 行 pad 走完整套核和每层 allreduce，是 TP4 span 步 142 对 t=1 87 ms 的全部差额。
  256 之上的阶梯加了 264 / 272 / 288 / 304 / 320，一条 run 加几张卡的一两行只垫几行（曾试过把 run 缩到
  阶梯值上一行不垫：6 个 token 的 prompt 被切成两步、1k 的 4 步变 5 步，省几行 pad 换整整一步，撤了）。`Staged` 借住 `&mut Tray` 直到输出读完，中间不能 lease / fork /
  再 stage。manifest 有 `error` fill 的输出时每步读一次，非零即该步失败。
- **K3 没有 prefill program，prompt 走 span**：`prefill` 可选；没有时 prompt 在 prefix
  命中之外的部分走 decode 步，该行的输出在最后一个 prompt token 进去之前丢掉。manifest 有
  带 `batch.span` 的 program（K5：`"batch": {"groups": …, "rows": 1, "span": "span"}` +
  `span_at` fill 的 `[1]` i32 输入，`Protocol::spanned(b)` 选它）时一步里**每个 tp 组一个** cell 可以是
  一段 run——同一序列的 c 个连续 token 各占一行（`Layout` 里每 cell 有行数，run 排在 owner 块
  的最前面，`span_at` = 本组 run 的 owner 块的偏移，每个 rank 各自算），位置 / slot / seq_len
  逐行递增，输出取末行；**组里没有 run 的 rank 在自己块最前面垫 c 行 pad**（`Layout.lead`）
  ——var 是全 tray 一份，每个 rank 都跑 span program、都跳过 `[span_at, span_at + c)`，
  EP4 下 run 落在 rank 1 时 rank 0 若不垫，它自己的第 0 行就被当 span 跳过、又被 span
  核当 span 算进那条序列的 state（2026-09-03 E3 第一版：8 条相同 prompt 出 8 种答案）。scheduler
  每步在每个 tp 组里挑最老的还在喂 prompt 的序列各喂一段 run（`runs`，纯函数；2026-09-04 之前整个 tray
  每步只放一条，t=1 的另外三张卡在 span 步里跑的是 pad），所有 run 一样长：
  c = min(`--chunk`, manifest 的 `span.max`, 各 run 待喂 token 数的最小值, 每个 rank 的余量)——
  余量按 rank：放 run 的 rank 是 `seqs.max + 1 − 本卡行数`，垫 lead 的 rank 是 `seqs.max − 本卡行数`；
  一条短 prompt 会把同步别人的 run 拉短一步，换它自己一步内出首 token，不排在别人后面。
  其余序列各一行。每 rank 的行数：k ≤ `--max-seqs` 时 bucket 再钉到 max_seqs（原规则），run 超过它时
  走阶梯（`rows_per_rank`）——同一个 run 不论同步还有什么都落同一个 bucket；cuBLAS 按 m 选核，2k prompt
  的尾块 223 行不垫到 256、12.9k 的尾块 160 不垫到 192，输出就在近平局处与 K5 线分道（2026-09-04 移植第
  二轮踩过，t=1 conc1 的 sha 全部对不上，runtime 用 93 层 12.9k oracle 证明是同的）。没有 span program 时
  逐 token（12.9k 的 prompt 要 12.9k 步）。b>1 走形状包含
  `(b, 1)` 的 program 里 `groups` 上界最紧的那个；没有 chunk program 时 `--rows` 只能是 1。
- 权重按 rank：`--weights` 里 `{ep}` / `{tp}` 换成该 rank 在组里的下标，文件
  名里的 `*` 按名字序展开（`dense-tp4/r{tp}/l*.safetensors`），mmap 不读入。
  `--capacity` / `--host-gib` / `--max-seqs` 都是 per rank。

**t=1 smoke（2026-09-03，tray03 GPU 1，qwen3-4b，`--capacity 65536`）**：conc1 输出与 `kern run`
逐字同；4 路并发两次跑结果一致、与单跑一致（prompt 0 的分歧是 `decode` / `decode_batch`
两个核的近似平局，见 lessons）；prefix 命中三条路径——resident（8972 token 的 prompt 命中
4576）、host 全量 wake（命中 8960 后 12 token prefill）、host 半量 wake（4589 的 prompt 命中
4576）——warm 与 cold 24 token 逐字同；wake 回来的 937 页与 park 时逐页 digest 相等（运行时
`park_wake` 例子加了 `--wake` / `--every-page`，串接每页 checkpoint 的链与部分 wake 都逐字节回）。
`--host-gib 8` 下 c4 的填充就把 host 层打穿（一条 15k 的快照 2.06 GiB），测 wake 用 24。

**K3 93 层 EP4 span（E3，2026-09-03，tray07 4×GB300，`--max-seqs 16`，`--chunk 256` 对逐 token 的
`--chunk 1`；prompt 是 docs/*.md 语料按 K3 tokenizer 切的 12.9k / 2k / 8 × 1k token，脚本与
逐请求数据在 `~/bench_results/2026-09-03-k5-span-kernels/`）**：

| 场景 | span（chunk 256） | 逐 token（chunk 1） |
|---|---|---|
| conc1 短 prompt 64 token | 逐字同，ITL 32 ms | ITL 32 ms |
| conc1 2k prompt TTFT | **0.83 s** | 63.9 s |
| conc1 冷 12.9k prompt TTFT | **4.6 s**（51 个 span 步 ≈ 90 ms/步） | 416.8 s |
| conc8 × 1k prompt TTFT | 0.56–3.2 s（span 一步一条，排队） | 38 s（8 条同时逐 token） |
| conc8 稳态 decode ITL p50 | 38 ms（B=8） | 38 ms |
| conc7 decode + 3 s 后冷 12.9k 到达 | 12.9k 的 TTFT 4.8 s；其余 7 条 256 步的 ITL p50 38 / mean 52 / p90 91 ms | — |
| 同上，`--chunk 64` | 12.9k 的 TTFT 16.4 s；其余 7 条 ITL 61 ms 整段 | — |
| 多轮 prefix 命中（1k + 答案 + 新一轮） | 第二轮 TTFT 112 ms（第一轮 463 ms） | — |

- **span 步的税**：其余序列在有 256 行 span 的那一步 ITL ≈ 90 ms（+52 ms），64 行 ≈ 61 ms（+23 ms）。
  12.9k 按 256 切是 51 步、按 64 切是 202 步，所以 chunk 256 两头都好（TTFT 4.8 对 16.4 s，
  并发者 mean 52 对 61 ms）。K5 的门"ITL ≤ +25%"按 mean 算是 **+37%**（52 对 38，p50 不变）——
  未达；再往下要 D2 的预算 policy（span 长按稠密 / attention 预算定，或 span 步只带一部分
  decode 行），不是 span 实现的事。
- **MegaMoE BLOCK_M 96（2026-09-04）**：`k3_moe_bench` 在真实 router 分布下扫 ladder，BLOCK_M 96 在 8–512
  token/rank 全段最快（真实路由 −12～15%，输出逐位同），`pinned_config` 改钉它后复测：decode 步 ITL p50
  38.0 → 36.5 ms，12.9k 末尾 200 行 span 步 76.2 → 73.2 ms，冷 12.9k TTFT 4.62 → 4.48 s，conc1 输出 sha 同；
  门仍未过。数据在 bench_results 2026-09-04-k5-span-profile。
- **输出**：conc1 短 prompt 64 token 与逐 token 逐字同；2k / 12.9k 的输出与逐 token 在第 8 / 第 1
  个近平局后分道（两条数值路径，cuBLAS 按 m 选核，同 t=1 smoke 的注）；同一路径自己是确定的
  ——冷 12.9k 的 256 token 在 5 次不同并发环境下 sha 全同（ad2eea8cd2f0），2k conc1 重启服务后同。
- **相同 prompt 并发不是逐字相等的门**：8 条"The capital of France is"锁步，3 条答 " Paris."、5 条答
  " the capital of France is…"——`k3_golden` 把两种 batch 形状的 logits 倒出来比，两种形状的
  hidden / KDA state / KV 都只差 bf16 噪声，" the" 对 " Paris" 的 top-2 差 **0.2 logit**（13.53 对
  13.32；另一形状 13.63 对 13.90），是近平局，不是串扰。串扰（第一版没垫 lead pad）的样子是
  8 条 8 种、互不成句的答案。见 lessons。

**K3 EP4×TP4（t=4，2026-09-04，tray07 4×GB300，二进制与 t=1 同一份；等块版的脚本与逐请求数据在
`~/bench_results/2026-09-04-k5-v4-port-tp4/`，块不等长版在 `~/bench_results/2026-09-04-tp4-blocks/`）**：

- **4 层逐 token 对 `k3_golden`**（`k3-4l-ep4-tp4.json`，`--max-seqs 6 --chunk 8`，前 i 个 token 作 prompt、取 1 个，
  按 k3_golden 的近平局规则判）：EP4×TP4 35/40 + 4 excused + 1 近平局（step 13，3.0 ulp，与 `k3_golden` TP4
  同一个 token）；EP4 t=1 37/40 + 3，与 `k3_golden` 同。块不等长之后两者不变（tray 4 rank 等块时 `blocks` 表
  退化成 q × b，走的是新核）。
- **93 层 E3 场景在 EP4×TP4 上**（`k3-93l-tp4-span.json`，`--max-seqs 16 --chunk 256`；中列是块不等长之前同日的
  等块版，右列是同日同二进制的 EP4 t=1）：

| 场景 | EP4×TP4（块不等长） | EP4×TP4（等块，之前） | EP4 t=1（同日） |
|---|---|---|---|
| conc1 短 prompt 64 token | 逐字同（sha 与 t=1、K5 线同），ITL **22.7 ms** | 同 | 31.2 ms |
| conc1 2k prompt TTFT | 0.86 s | 1.26 s | 0.80 s |
| conc1 冷 12.9k prompt TTFT | **4.59 s**（51 个 span 步 ≈ 90 ms/步） | 7.39 s（≈ 145 ms/步） | 4.47 s（≈ 88 ms/步） |
| conc8 × 1k prompt TTFT | 0.64–3.32 s | 0.89–5.18 s | 0.67–3.29 s |
| conc8 稳态 decode ITL p50 | **28.5 ms** | 29.3 ms | 36.3 ms |
| conc7 decode + 3 s 后冷 12.9k 到达 | 12.9k 的 TTFT 4.87 s；其余 7 条 ITL p50 28.8 / mean 42–48 / p90 92–93 ms | 9.31 s；28.7 / 52–64 / 146 ms | 4.64 s；36.4 / 47–52 / 88–89 ms |
| 多轮 prefix 命中第二轮 TTFT（接在 E 之后，命中的是 D 里 conc8 步算的快照） | 487 ms | 722 ms | 462 ms |

  - decode 步 TP4 比 t=1 快 8–9 ms（E5 的账）。**span 步曾反过来慢 55 ms**（142 对 87，nsys 拆解在
    `~/bench_results/2026-09-04-tp4-prefill/`）：全部是 pad 行的账——tray batch 每 rank 行数相同，一条序列 256 行的
    run 让 tray 变成 1024 行、768 行是 pad，它们被 `kda_core` / `conv_silu` 当 decode 行算（+25 ms，t=1 的 span 行
    是跳过的）、进每层的 allreduce（+19）、land / rms（+8）、allgather（+4）；真实计算 TP4 66 ms 对 t=1 81（GEMM 切列
    快 7）。**块不等长之后（上表左列）span 步 ≈ 90 ms，与 t=1 持平**：conc7 里并发者的 ITL p90 就是 span 步
    （93 对 t=1 的 88），12.9k 冷 TTFT 4.59 对 4.47 s。持平而不是更快，因为 MoE 扫描、MLA、KDA 递推都不按 rank
    分摊；要更快得把 run 的 token 切到各 rank（K4/K5 重定义）。conc8 里 TP4 的 ITL p90 有四条是 84–88 ms：
    它们先出首 token，之后跟着别人的 span 步走（一步一个 span，四卡一个 tray），等块版 TTFT 慢到大家几乎一起
    出首 token，反而没这一段；总时长 8.9 → 6.6 s。`k3_golden` 的 tray batch 不接 span，TP4 的 span 路径没有
    oracle 门禁。
  - 输出：t=1 的 conc1 sha（短 / 2k / 12k 冷 / 12k 迟到）与 K5 线 2026-09-03 与 BLOCK_M 96 复测全同
    （ec2071a1bd49 / eed14fa316ea / 607802289f15 / ad2eea8cd2f0），块不等长之后 t=1 全部不变；TP4 短 prompt 同，
    2k / 12k 在第一个近平局后分道（TP 归约是另一条数值路径；块不等长之后 TP4 的 12k 冷 sha 反而与 t=1 同，
    2k 仍不同——近平局两边落哪边）。并发的 1k 条目每次跑的 sha 都不同（到达顺序决定 bucket 组合），K5 线自己
    两次跑也如此，不作门。无 panic、无 `tp_err`。

**每个 tp 组每步各一条 run（2026-09-04，tray07，同一天的 E3 场景；脚本与逐请求数据在
`~/bench_results/2026-09-04-t1-runs-per-rank/`）**：之前整个 tray 每步只放一条 run，t=1 的另外三张卡在 span 步里
跑的是 c 行 pad——两种配置的 decode 都是四卡并行，prefill 都只用一张卡的算力。改成每个 tp 组各挑最老的还在喂
prompt 的序列各喂一段 run（`scheduler::runs`），所有 run 一样长；同时 owner 从"页最少"改成"行最少、再页最少"
（页数含留着的快照，之前持有 12k 快照的卡在 conc8 里一行都分不到，`[3, 2, 0, 3]`，改后 `[2, 2, 2, 2]`）。
对照左列是块不等长那天的同二进制 t=1：

| 场景（EP4 t=1） | 之前 | 现在 |
|---|---|---|
| A 短 prompt TTFT / ITL p50 | 221 ms / 31.0 | 233 / 31.0（sha 同） |
| B 2k TTFT | 801 ms | 816（sha 同） |
| C 冷 12.9k TTFT | 4466 ms | 4476（sha 同） |
| D conc8×1k TTFT 均值 [min–max] / ITL p50 / p90 | 1962 [669–3294] / 36.4 / 65（四条 83） | **917 [612–1135]** / 34.5 / **34.5** |
| E 迟到 12.9k TTFT；其余 7 条 TTFT 均值 / p90 | 4644 ms；2075 / 88 | 4543；**1199** / 87 |
| F turn2 TTFT | 462 ms | 474（sha 同） |

  - conc8：8 条 1k 各 4 个 chunk，两条一卡，第二条等第一条的 4 步——最慢一条 1.13 s 就是 8 个 span 步；之前
    32 个 span 步串在一张卡上，最慢 3.3 s。ITL p90 从 65 回到 34：没有人再跟着别人的 span 步走。
  - 只对 span 步分摊，decode 步没变（ITL 34–36 同以前的抖动范围）。同 prompt 的 conc1 sha 四个场景全同；
    并发条目的 sha 变了，一是同以前"不作门"，二是 room 之前按整个 tray 的行数算（`seqs.max + 1 − n`），conc8
    下把 run 切成 249、247、…（旧日志里 span 249/247/246/244 的 capture 就是它），现在按 rank 算、整 256。
    TP4（一个组，每步仍一条 run）conc1 sha 同；conc8 按 TTFT 排序后相邻两条的间隔（一条 1k prompt 的 4 个
    span 步）三轮都是 ≈ 400 ms，首条的 TTFT 638 / 688 / 809 ms 在轮与轮之间漂，不是 span 步变了。
  - 一条短 prompt 会把同步别人的 run 拉短到它的长度（c 取最小）：换它一步出首 token。conc8 里没触发
    （8 条一样长）；混合长短的公平性是 K5 D2 预算策略的事。

**没测**（按 CLAUDE.md 的门禁排队）：
1. ~~t=1 qwen3.8-27b（有 slot 的路径）conc1 与 `kern run` 同，K1/K3 门禁数字不变~~（e2e 门禁一节，2026-09-14）；
2. owner-only 页在 t>1 下：mixed 行、prefix 命中（retire → lease_from）、park / wake 之后 warm == cold；
3. 全成或全不成：某一卡 `--host-gib` 故意给小，park 整体退回、四卡 host 占用回到原值；
4. 93 层短 prompt 的 conc1 / conc8 步时对 k3_golden 的 20.8 / 25.5 ms（E3 量到 ITL 32 / 38 ms，含 tray 的 staging 与 HTTP，没拆）。

## e2e 门禁（`tools/e2e`，2026-09-14，tray06 / tray07 / tray09，各 4×GB300）

`python3 tools/e2e/e2e.py` 把一份 kern.toml 的每个 target 过同一组场景（表在 `tools/e2e/README.md`）：
`kern test`、conc1 对 `kern run`、重复命中、turn2（prompt + 答案 + 追问，发 id）warm 对 `kern run` 与对冷
server、12 条并发、流式中途挂断、小池子 + `--host-gib 2` 的 park / wake、刚好装一条 turn2 的池子上的 wake
（`wake_room`）、`--max-seqs 2` 起的 slot 增长、
投机 manifest 的 `--rows 1`。字节一致是门；单 rank target 的分歧再拿 `kern run --prompt-ids <分歧前的
上下文> --rows 1 --probe-dir` 倒那一步的 logits，两个 token 都在 top-1 的 4 个 bf16 ULP 内才放过。
每次 64 个 greedy token（K3 4 层 16 个）。结果与日志：`~/bench_results/2026-09-14-e2e-gate/`。

| target（tray） | rank / rows / checkpoint | conc1 = `kern run` | 命中：重复 / turn2 | warm = `kern run` / = 冷 | 12 并发 · 接受率 | park / wake（小池子） | slot 增长（`--max-seqs 2`） | `--rows 1` = `kern run` |
|---|---|---|---|---|---|---|---|---|
| qwen3-4b（tray06） | 1 / 1 / 每页 | 12/12（`kern test` 位一致） | 16,16,16,0 / 80,80,80,64 | 2/4 + 2 近平局 / 同 | 12/12 · — | parks 16、wakes 5、host_hits 5，答案 11/12 + 1 近平局 | — | — |
| qwen3-4b-dspark（tray06） | 1 / 7 / 每页 | 12/12 | 同上 | 1/4 + 3 近平局 / 同 | 12/12 · 24%（2.47 tok/步） | parks 14、wakes 5 | — | 4/4 |
| qwen3.8-27b（tray06） | 1 / 1 / 请求结束 | 12/12 | 0×4 / 87,83,87,79 | 4/4 / 4/4 | 12/12 · — | parks 19、wakes 4、host_hits 4，12/12 与 4/4 | remaps 15，slot 5 → 20，12/12 与 turn2 4/4 | — |
| qwen3.8-27b-dflash2（tray06） | 1 / 8 / 请求结束 | 12/12 | 0×4 / 87,84,88,0（6 个 `not kept`） | 4/4 / 4/4 | 12/12 · 20%（2.37） | parks 15、wakes 4、host_hits 3 | remaps 9，5 → 14 | 4/4 |
| dsv41-h152（tray09，EP4） | 4 / 6 / 请求结束 | 无 oracle（冷 12 条作基准） | 0×4 / 86,84,0,0（7 个 `not kept`） | — / 4/4（只报） | 12/12（12 条与 conc1 同）· 27%（2.33） | 26 条填满：parks 11、wakes 2、host_hits 2，12/12 与 4/4 | remaps 24，20 → 44（4 卡合计） | — |
| k3-4l-ep4（tray07，EP4，16 token） | 4 / 1 / 请求结束 | 无 oracle | 0×4 / 38,35,37,31 | — / 0/4（只报，见下） | 12/12（2 条与 conc1 同）· — | 17 条填满：parks 13、wakes 4、host_hits 4，12/12 | remaps 36，20 → 56 | — |

2026-09-15 合入前 DSv4.1 EP4 在 tray06 重跑（`--chunk 128 --max-seqs 16`）：第一遍 host 会话在 park_wake 的
turn2 上挂死——wake ↔ park 活锁（`server-host.log` 236 万行 `parked tokens=86`，见上面 toy 那节和 lessons.md）；
wake 改成一次分配、行上限减掉 pad 之后 8 门 3 报全过，新加的 `wake_room` parks=1、命中 86、醒进一条 2 页的租约
（`~/bench_results/2026-09-14-e2e-gate/results/tray06-r7-dsv41-final` → `tray06-r9-dsv41-wake`）。

近平局都有 logits 证据（`results/<tray>/<target>/probe-*/`），例如 qwen3-4b turn2 第 56 个 token：top-4
29.25 / 29.125 / 29.125 / 28.75，server 的在第 2、差 1 ULP。加载：qwen3-4b 10 s、qwen3.8 10 s、DSv4.1 37 s、
K3 4 层 4 s（`--capacity 262144`）。

过程中改了三处，都是 e2e 写不顺才发现的：

- **kern run 只吃文本**：turn2 按文本发，DSv4.1 上答案文本切回去的 id 与生成的 id 不同（4 条里 3 条命中
  0）——不是 cache 的错，是 client 造的 prompt 变了。turn2 改发 id，`kern run --prompt-ids` 让 oracle 跑
  server 跑过的那串。
- **投机轮多算的 checkpoint**：一轮接受的 token 可以越过 `max_tokens` 或 stop（`emit` 截断，state 里已经
  有了），请求结束的快照就以这串多出来的 token 为键——下一轮永远发不出这个前缀，快照白占一个 slot 到被
  淘汰（DSv4.1 4 条 turn2 命中 2 条、dflash2 3 条）。scheduler 现在不留这种快照（`not kept` debug 行），
  只多一个 stop token 的留着（chat template 下一轮就带着它）。e2e 的 turn2 门：命中要么 ≥ 第一轮 prompt
  要么 0，0 的条数 ≤ `not kept` 的条数。
- **scheduler 线程 panic 后端口还开着**：请求全挂到 client 超时（driver 干等了 15 分钟）；现在进程跟着退 101。

近平局的判法也改了一版：投机 manifest 的第 k 步是另一串（`--rows 1` 的 run 在更早的平局上已经分岔），
拿它的 logits 判分歧是错的；要用"prompt + 分歧前双方一致的 token"作 prompt 单独跑一步。

没有 oracle 的 tray target（EP4 没有 `kern run`），换了数值路径的一致性只报不门：K3 4 层 pruned
checkpoint 近乎平局遍地（12 条并发只有 1 条与 conc1 同、warm 对 cold 0/4 在第 0–1 个 token 就分），
同一条路的（重复、abort 后）4/4、1/1 一致照门。K3 4 层用默认预算：256 GiB 切成 131 079 个 2 MiB 块
（页 64 token × 4 层太小，块取最小对象的一半），每 rank map 40–60 s，一次在 rank 3 的 `cuMemSetAccess`
OOM——块大小该随预算长，先记着，e2e 里 K3 显式 `--capacity 262144`。

"没测"清单的第 1 条（t=1 qwen3.8-27b conc1 对 `kern run`、K1/K3 门禁）由这一节覆盖；2、3（t>1 的
owner-only 页、park 的全成或全不成）和 4 仍没测。

## toy 门禁（`tools/toy`，2026-09-15，tray03 / tray06 GB300）

真模型的 e2e 一轮 20 分钟、要三台 tray，还有近平局要争。`tools/toy` 是一组不是模型的
manifest：整数 kernel，每个 token 槽开头存 64 个位置相关的 mark、其余到槽尾每个字都是
头部的函数，下一个 token 由序列所有位置的头字按位置旋转后的和决定（有 line 的再加上
line 的折叠，line 的尾部同样是折叠值的函数）；尾部哪个字对不上就把和毒掉，页序换了和
也变。Python 参考（`tools/toy/model.py`）逐 token 精确，只算头字。kern-serve、kern-pool、
runtime 走的是同一条路——它们本来就不认识模型。

```
python3 tools/e2e/e2e.py --config target/toy/kern.toml --reference tools/toy/model.py --gpus 0,1
```

把同一组场景过一遍，每个门都精确，包括真模型上只能"报不门"的：并发 12 条对 conc1、warm 对
cold、醒来的对 cold；第二张卡上 `kern run` 自己也被 reference 门住（`run_equals_reference`），
五种形状的每个 kernel 都过 `kern test` 的 A/B（256 对 128 线程块的两个 cubin）。

| target | 形状 | 命中：重复 / turn2 | park / wake / host_hit | slot 增长 | 投机 |
|---|---|---|---|---|---|
| toy-paged | 4 KiB/token、页 16 | 96,80,96,64 / 144,160,112,176 | 13 / 4 / 4 | — | — |
| toy-stateful | + 1 MiB/seq line | 0×4 / 117,219,160,107 | 14 / 4 / 4 | 5→20（15 remap） | — |
| toy-spec | 4 行 round | 同 paged / 144,160,112,160 | 13 / 4 / 4 | — | 50% 串行、50% 并发（2.49/轮）、rows1 4/4 |
| toy-stateful-spec | line + round | 0×4 / 117,220,0,0（req-2/3/14/15 `not kept`） | 15 / 2 / 2 | 5→15（10 remap） | 50% / 50%（2.5/轮） |
| toy-big | 64 KiB/token、页 64、281 GiB state | 64×4 / 128,128,64,128 | 12 / 4 / 4 | — | — |

2026-09-15 tray06，每条 256 token、12 条 prompt，五个 target 全部通过，整轮 3 分 09 秒，两张卡。
结果与日志：`~/bench_results/2026-09-15-toy-e2e/`（`results/tray06-r8`；qwen3-4b 同日同机 9 门全过，`tray06-r9-qwen3-4b`）。
同日 tray03 加上 `wake_room` 再跑一轮（`results/tray03-r14-wake-green`）：五个 target 全过，带 state 的两个
parks=1、命中 117、醒进租约，纯 KV 的三个 parks=0 原地续上；qwen3-4b / -dspark 10 门与 11 门全过（`tray03-r14-qwen`）。

第一版（tray03，`results/tray03-r4`）的表里五个 target 也"全过"，但并发 12 条那时只报不门，
toy-stateful 的 12 条并发答案其实**全错**（0/12 与 conc1 同）：`gen.py` 给 prefill 和 decode
的 `fold` 用的是同一个调用，rows 绑在 `tokens` 变量上，decode 的 tokens 是整批的行数，每个 group
把别人的行折进了自己的 line。设计评审（Opus）读出来的，把并发在精确 oracle 下改成门就复现
（`results/tray06-r5-red`，`FAIL: concurrent`），decode 改绑字面量 1 就过。同一轮 TDD 补的门：
近平局只放过一个 token、之后从 server 选的 token 续算继续比（qwen3-4b 上 repeat 第 46 个 token
两个 logit 相等 21.5，放过后其余 18 个逐字同）；`turn2_hit` 的 0 命中要配自己第一轮请求的
`not kept`，不再数条数；投机接受率对串行自己（同样 12 条 prompt，按 steps 加权）的 0.9 倍而不是固定 20%，toy 还要落在 40–60；
单 rank target 没 oracle 是 FAIL 不是只报；一个 target 都没跑退出码非零。这些裁决是纯函数，
`tools/e2e/test_e2e.py` 在 CI 里跑。driver 自己的两处竞态也是这轮撞出来的：小池子的填充停在
"日志里有 4 条 `parked`"，读日志的时机决定 turn2 要的 checkpoint park 没 park（一轮 wakes=0），
改成填到 turn2 命中的那几个长度都 `parked` 为止；`serving` 在 pegainfer 绑端口之前打出，看到就连
偶发 connection refused，ready 改为端口应答。

toy 抓到的几条：

- master 把 state 块的上限提到 64 MiB（bb50189，DSv4.1 的 836k 页从 13 s 降到 0.5 s）之后，
  `--capacity 2832`（177 页 × 64 KiB）变成了整整一个块的 1023 页——`--capacity <tokens>` 的契约破了，
  小池子什么都不 park。真模型的页都比 64 MiB 大得多，e2e 看不见；toy 的 4 KiB/token 一跑就翻。
  `Pool::new` 现在拿到要的 token 数，页数以它为上限，块的尾巴空着。
- `kern run` 与 kern-serve 对"权重是文件时 tokenizer / eos 在哪"的规则不一致，master 同日已统一
  （`kern_run::checkpoint_dir`）。
- toy-big 加上 `-ref` 之后 `kern test` 炸在 `position past the lease`：perf 的 prefill 扫描点取到
  manifest 的 `tokens.max`（8192），租约只有 `--capacity` 的 4096 个位置；真模型的 `tokens.max`
  从没超过 4096。扫描点现在以租约为界。
- wake ↔ park 活锁（DSv4.1 EP4 的 host 会话先撞到，toy-stateful 上 `wake_room` 场景 30 秒复现，
  608 万行 `parked`）：命中 parked 条目的请求先醒成快照再租，两次分配；现在 `Host::restore`
  一次醒进请求自己的租约（`pool.md`、lessons.md）。同一场景顺手抓到第二条：`max_request_tokens`
  没减 pad 那一页，恰好占满池子的请求既不被拒也永远坐不下；`Tray::max_seq_tokens` 现在以 pad
  之外的页为界，超过的在租之前就按 ContextLength 拒。
- kern-pool 两处（评审读出、集成测试复现）：`Prefix::evict` park 成功那条路把"为腾地方丢掉的
  parked 条目数"扔了，scheduler 的 `host_evictions` 少计；`lookup` 给一条一页都没共享到的旁路
  候选盖时间戳，一次什么都没命中的查询把最冷的条目变热、淘汰错人。

toy 不测 kernel 数值和性能，也没有 tray（EP4 的 toy 要一个走 peer 指针的 collective，还没写）。

## 没做（按需要加）

span 长的预算 policy（K5 D2：按稠密 / attention 预算定 c，现在是 `--chunk` 上限）、抢占 / 动态页分配、
真采样（temperature/top-p 作为 manifest 内的 `sample` op；投机下是 rejection sampling）、logprobs / echo、
bs 2–16 的 split-KV decode、步间 host 空转（token 反馈进图）。
