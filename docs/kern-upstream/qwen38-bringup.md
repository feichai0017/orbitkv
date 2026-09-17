# Qwen3.8-27B（+ DFlash2）带上 kern：时间线与证据

> 这是 2026-08-31 的时间线，用的是当时的 schema v2 词汇。对照 v3
> （见 [manifest.md](manifest.md)）：`dispatch`→call、`step`→launch、
> `symbol`→var / entry、`State.bytes_fixed`→`bytes`、`meta.spec`→顶层 `spec`、
> `manifest-v2.schema.json`→`manifest-v3.schema.json`。

任务书：[qwen38-bringup-prompt.md](qwen38-bringup-prompt.md)。全程一个 agent
（Claude Code）在一台 GB300 节点上做，人工介入次数在末尾统计。
时间戳均为 UTC。

## 时间线

### 2026-08-31T12:02Z 开始：读文档 + 环境

- `nvidia-smi`：4×GB300 全空。选 GPU 0 做 vLLM 挖矿、GPU 1 做 kern-run。
- 读 README / docs（design / manifest / kernel-mining / runtime / spec-decode /
  test / roadmap）/ tools/README、`gen_qwen3_decode.py`（934 行）、
  `mine_capture.py`、`export_weights.py`、`extract_kernels.sh`、
  `kern-manifest/types.rs`、`kern-runtime/lib.rs`、`kern-run/lib.rs+main.rs`。
- 读 vLLM 0.28 的 `qwen_gdn_linear_attn.py`（2055 行）、`qwen3_5.py`：
  Qwen3.5 的 GDN 层在 `gdn_prefill_backend=triton` 下走
  `causal_conv1d_fn`（Triton）→ `fused_post_conv_prep`（l2norm + gating）→
  `chunk_gated_delta_rule`（FLA 的一串 Triton 核）→ `RMSNormGated`
  （`layer_norm_fwd`，Triton）→ `out_proj`；decode 走
  `causal_conv1d_update` + `fused_recurrent_gated_delta_rule_packed_decode`
  （`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`）或
  `fused_sigmoid_gating_delta_rule_update`；spec 走
  `fused_gdn_decode_post_conv_mtp`（vLLM CUDA op，`num_accepted_tokens`
  语义：从 state 重放）。
- 运行时事实：`Runtime::load` 的 state 用 `alloc_zeros` 分配 → 一个进程一条
  prompt 的前提下 GDN 递归 state 天然零初始化，不需要 reset 原语。

### 2026-08-31T12:09Z 挖矿 capture（GPU 0）→ 12:10Z 完成

- `tools/capture_qwen38.sh`：TRITON_ATTN + `gdn_prefill_backend=triton`，5 条
  长度 1/34/85/174/250 的散文 prompt 各 `max_tokens=4`，一次 bs=1。
  92 个 module cubin、71487 次 launch，输出正常（"hi" → ", i want to"）。
- `mine_capture.py` 的自动切分对这个模型失效：eager 下 launch 太密，5 ms
  空隙切不开 forward，"core 核"选到了 ATen 的 copy。没有修它——改用
  ArgMax（采样）当 forward 边界、`_triton_mrope_forward` 的 grid.x 当
  tokens，写了个一次性的逐 launch ABI 打印（scratchpad），够用。

### 12:12Z–12:40Z 逐核 ABI 分析（不动 GPU）

结论直接决定生成器怎么写，全部记下：

- **Triton 核的真实 ABI 不能从 launch 猜**：Triton 把 `==1` 的整型参数和
  `None` 指针从签名里删掉、`%16` 的整型加 `tt.divisibility` 假设。用
  `~/.triton/cache/*/<kernel>.ttir` 的 `tt.func` 签名（按 cubin sha256 与
  dump 的 module 对上）当 ground truth。同名核有多份实例：
  `_fused_post_conv_kernel` 三份（L==1 特化 / L%16 特化 / 通用），
  `layer_norm_fwd_kernel` 两份（rows_per_block 4 / 1，REG 62 / 28），
  `_triton_mrope_forward` 两份（num_tokens==1 特化 / 通用）。生成器必须
  选"通用"实例并按 sha256 钉死（L%16 与通用实例参数布局相同，runtime 的
  `cuFuncGetParamInfo` 消歧不了）。
- **GDN state 的页布局被烧进了 kernel**：`_causal_conv1d_fwd/update` 的
  state 行距 1605632 elem（= 3211264 B，vLLM "align" 模式的 KV 页大小），
  `fused_recurrent_..._packed_decode` 的 h0 = 页基址 + 0xf000（= conv state
  10240×3 bf16）、行距 802816 f32。即 vLLM 把 conv state（61440 B）和
  SSM state（48×128×128 f32 = 3145728 B）放在同一 3211264 B 页里，尾部
  4096 B 空。kern 的 GDN state 照抄这个页格式；索引 0 是 null block
  （kernel 里 `state_idx <= 0` 跳过），所以第 0 行留空，GDN 第 i 层用第
  i+1 行，索引通过一张常量表 + buffer offset 传。
- attention 的 KV 也被 `BLOCK_SIZE=784`（constexpr）绑死：vLLM 为了和
  mamba 页对齐把 block_size 选成 784。kern 的 KV state 用 `[page][16 层]
  [784][4 head][K|V][256]`，bytes_per_token = 16×4096；block_table 的
  domain unit = 784。
- **norm 是 ATen 一串（pow/mean/rsqrt/mul），不可挖**：`GemmaRMSNorm` 走
  `ir.ops.rms_norm` 原生实现，`weight.float()+1`，残差在 f32 上相加、
  归一化用未舍入的 f32 和。要逐字节一致就得复现 ATen `reduce_kernel` 的
  归约顺序（4 路向量累加 → block_x 归约：宽度 >32 时先 shared 折半、再
  shfl_down 1/2/4/8/16；宽度由行数决定：1 行 512、2–3 行 256、4–7 行
  128、8–15 行 64、≥16 行 32）、`mean = sum * (1/N)`、`rsqrtf`。决定手写
  一个核（`tools/kernels-src/gemma_rms_norm.cu`）逐位复现，并用 vLLM 自己
  的 op 做 oracle 测到 bit-exact。attention 输出门 `attn * sigmoid(gate)`
  同理（ATen，两次 bf16 舍入），手写 `sigmoid_mul`。
- prefill 里 z 门要做一次 strided 拷贝（`layer_norm_fwd_kernel` 的 M 有
  `%16` 假设，不能用 ngroups 技巧绕），手写 `copy_rows`；prefill 要出
  logits 就得取最后一行（`tokens-1` 不在表达式集合里），手写 `last_row`。
- kern-run 的 4B 契约是 "prefill 不出 logits，最后一个 prompt token 走
  decode"。对 GDN 这会让最后一个 token 走递归核而不是 chunk 核，与 vLLM
  的数值路径不同。改契约：prefill 也出 `next_token`（多 3 个 dispatch），
  driver 全部 prompt token 走 prefill。这是 driver（kern-run）改动，不是
  runtime。
- schema 扩展（唯一预期的）：`State` 加 `bytes_fixed`（per-seq 定长），
  runtime 分配 `bytes_per_token×capacity + bytes_fixed`。

### 12:31Z–12:54Z  权重导出 / 参考输出 / 手写核 / 生成器（无人工干预）

- 12:32Z 并行起两件后台事：`tools/export_qwen35.py`（GPU 1，合并导出
  50 GiB safetensors + rope 表 + FLA/conv 常量表）和 `tools/qwen38_ref.py`
  （GPU 0，vLLM eager TRITON_ATTN + triton GDN，5 条散文 prompt × 400 token
  greedy → `docs/qwen38/ref.json`）。两个都首跑失败：ref 是 flashinfer
  JIT 找不到 `ninja`（venv 的 bin 没进 PATH），export 是 `MRotaryEmbedding`
  作为 CustomOp 要 `set_current_vllm_config` 上下文。各改一行 12:36Z 重跑，
  12:39Z 都完成（ref 5×400 token，全部 length 截止，文本连贯）。
- 12:37Z–12:44Z 手写核对 vLLM 自家 op 做 bit-exact 测试
  （`tools/test_kernels_qwen35.py`，cuda-python 直接 launch cubin）。
  `sigmoid_mul` / `copy_rows` 一次过；`gemma_rms_norm` 两处偏差，各花几分钟
  定位：
  1. torch 2.13 把 warp 内 shfl_down 的 offset 顺序改成了递减（16→1，
     `Reduce.cuh` 注释说是为了和 Triton 数值对齐），我凭记忆写的 1→16 在
     ~30% 的行上差 1 ulp。看安装的头文件而不是靠记忆。
  2. nvcc 默认把 `sum * factor + eps` 缩成一条 FFMA，少一次舍入，rsqrt 差
     1 ulp，最终 bf16 在千万分之几的元素上翻转。用 `__fmul_rn/__fadd_rn`
     禁掉缩并。教训：复现 ATen 数值时，编译器的 FMA 缩并也是"顺序"的一
     部分。
  之后所有形状（行数 1–2048、q/k 每头 norm、fused 残差）全部逐位一致。
- 12:44Z–12:54Z 写 `tools/gen_qwen35.py`（新生成器）。要点：
  - 直接解析 launches.jsonl，按 ArgMax 切 forward，对 T=34/85/174/250 四个
    prefill forward 和 decode forward 做拓扑断言（相邻核之间的 buffer 同址、
    stride 字面量、grid 公式），一次通过。
  - Triton 同名实例钉定：按 launch 的寄存器数筛 module，再把 dump module
    的 sha 对到 `~/.triton/cache` 里的条目、读 `.ttir` 签名，选"运行时
    int 参数最多、divisibility 属性最少"的最泛化实例（`_fused_post_conv_
    kernel` 三个实例寄存器数相同，只能这么分）。decode 的 mrope 也用泛化
    实例（vLLM 用的是 num_tokens==1 特化版，算术相同）。
  - mrope 的 `num_tokens` 只用来算 t/h/w 三个平面的步长，传 0 让三个平面
    重合，cos/sin 各只需一次按 position 的 gather（复用 embedding 核）。
  - post_conv 的 a/b 直接用 `ba` 的 strided 视图（vLLM 拷成连续，核吃
    stride），u 不再和 A 共用存储（vLLM 为省显存别名）。
  - h0/ht 直接指向 state 行（vLLM 用临时张量拷进拷出）：每个 program 先读
    自己的 h0 tile、最后写 ht tile，原地无竞争。
  - final norm 在全部 T 行上做（ATen 归约宽度随行数变），再取最后一行喂
    lm_head（M=1），argmax 出 next_token——prefill 出 token。
  产物：`examples/qwen3.8-27b.json`（706 buffer，27 kernel，prefill 1079 /
  decode 742 dispatch）。`tools/extract_kernels.sh` 改成带参数（manifest、
  dump、输出目录），按钉定的 cubin 名 + sha 抽取；输出到 `kernels-qwen38/`
  （不能和 4B 的 `kernels/` 混：同名同 ABI 的 reshape_and_cache 实例
  block_size 不同）。

### 12:55Z–13:20Z  首次端到端 + 逐 token 对比（无人工干预）

- 12:56Z 容器里 `cargo build --release` 8 s，schema 重新生成（`bytes_fixed`
  一项）。kern-run 加 `--stop-tokens`（Qwen3.8 的 eos 是 248046/248044）和
  "prefill 出 next_token 就全量 prompt 走 prefill" 的契约分支。
- 12:57Z 第一次跑：manifest 校验通过、27 个 kernel 全部解析到钉定的 module、
  668 个权重按名绑定（50 GiB，共享盘冷读一次要 10 min，之后 10 s），
  prefill 17 token + 20 步 decode 一次跑通，输出连贯，eager 82.8 tok/s。
  一次都没改 manifest。
- 13:08Z 5 条 prompt × 400 token 与 vLLM 逐 token 对比
  （`tools/qwen38_compare.py`）：分别在第 66/93/62/255/178 个 token 处分叉，
  之前逐字节一致。整条链路是对的，差一点 ulp 级的东西在近似平局的 argmax
  上翻转。
- 13:10Z 用同一个 capture 注入库抓 kern-run 自己的 launch，和 vLLM 的
  decode forward 逐条比 GEMM：算法名、grid 全部相同（只有 dynamic smem 差
  16 B——kern 链的是系统 cuBLAS，torch 用自带的 wheel，版本不同）。GEMM
  算法选择不是原因。
- 13:12Z 给 kern-run 加 `--probe-dir`：prefill/decode 按 dispatch 区段分段
  执行（`run_range`），在每层 `down_proj` 后 dump `y`，最后 dump logits——
  不重复执行任何 dispatch，所以 state 不会被重复更新。vLLM 侧
  `tools/qwen38_probe_vllm.py` 用 forward hook 抓同样的东西。
- 13:14Z **外部干扰**：本节点四张卡被其它作业占满（各 182 GiB）。
  权重/venv/binary/triton cache 全在共享盘，GPU 工作迁到另一台空闲节点（ssh）。
- 13:18Z 在新节点上重跑探针，逐层比较（prompt 0，prefill + 2 步 decode）：
  embedding 完全一致；**第一处差异在 layer 0（GDN 层）的 prefill**，
  y 有 11% 元素差 ≤1 ulp（max |d| 4.9e-4）；decode1 的 layer 0 差 0 个元素
  （state 精确时 decode 路径 bit-exact），差异随深度累积到 layer 63 的
  max |d| ≈1–2，三步的 argmax 仍然相同。范围收窄到 GDN 层的 prefill 路径。

### 13:20Z–13:37Z  定位到最后一个 op；验收标准放宽（1 次人工干预）

- 13:22Z 怀疑 cuBLAS 版本：kern-run dlopen 的是系统 `/usr/local/cuda`
  （13.4），torch 用 wheel 自带的 13.0。`LD_LIBRARY_PATH` 指到 torch 的
  wheel 重跑 prompt 0：仍在第 66 个 token 分叉——**不是库版本**。
- 13:25Z kern-run 探针加 `KERN_PROBE_LAYER=<i>`：该层每个 dispatch 之后
  dump 它第一个 out 参数的 buffer（通用，靠 manifest 的 param dir，不需要
  知道模型）；vLLM 侧 `PROBE_LAYER=<i>` hook 该层的子模块
  （in_proj_qkvz/in_proj_ba/out_proj/post_attention_layernorm/mlp.*）。
  `tools/qwen38_probe_fine.py` 按 dispatch 顺序对比。
- 13:26Z **外部干扰 ×2**：迁去的节点又被其它作业占掉（各 175 GiB），
  vLLM 探针 OOM；再迁到第三台空闲节点（之后的 GPU 工作都在这台上）。
- 13:36Z 结果（prefill，M=43）：`in_proj_qkvz`（N=16384）**bit-exact**；
  `in_proj_ba`（N=96，K=5120）4/4128 个元素差 1 ulp（max |d| 7.8e-3，
  rel 0.8%）——它是 layer 0 里第一个不同的 op，后面 out_proj/post-norm/
  gate_up/silu/down 的差异全是继承的。decode0 的 in_proj_* 都 exact，
  只有 out_proj 起差（继承 prefill 写进 state 的差）；decode1 每个 op 都
  exact。结论：**整条 GDN/attention/norm 链路的算术都和 vLLM 对上了，
  残差是 cuBLAS 在 N=96 这个瘦形状上的算法选择**（同一库版本也如此——
  extern GEMM 是 manifest 里唯一没被 sha 钉住的东西，见 Stage 3 的设计
  发现）。
- 13:37Z **人工干预（用户）**："大概精度吻合就行，不需要 bit wise。"
  Stage 1 验收改为：(a) 对 vLLM 逐 token 一致的前缀长度 + 逐层激活
  ulp 级一致的探针证据；(b) kern 自身 chunk=1 / chunk=512 / chunk=2048 /
  eager / graph 之间**逐字节一致**（这个不放宽——同一份算术不能因分块或
  图捕获而变）。四种配置同时在四张卡上跑
  （`tools/qwen38_compare.py` + `tools/qwen38_consistency.py`）。

### 13:38Z–13:45Z  Stage 1 验收（无人工干预）

- 5 条短 prompt（41–48 token）× 400 token，四种配置并行跑在四张卡上：
  - eager/chunk=512、graph/chunk=512、graph/chunk=2048 三者**逐字节一致**
    （graph 捕获不改算术）；对 vLLM 分别一致到第 66/93/62/255/178 个 token，
    之后分叉（`docs/qwen38/compare-*.json`）。
  - chunk=1 与 chunk=512 不一致（分别在 66/93/62/91/178 处）。**这是 GDN 的
    固有性质，不是 kern 的缺陷**：chunked FLA 核在 T=1 时对 state 的结合顺序
    和整段处理不同。验证：vLLM 自己把 `max_num_batched_tokens` 设成 16 再跑
    同样 5 条 prompt（`docs/qwen38/ref-vllm-chunk16.json`），和它自己的
    单块结果在第 66/11/341/2/339 个 token 处分叉——和 kern-vs-vLLM 是同一
    个量级的差异带。任务书里 "chunk=1 / chunk=512 / eager 三路一致" 对这类
    模型只能在 eager/graph 和 64 的倍数分块之间成立。
- 长 prompt（1787 token 散文，`docs/qwen38/long-prompt.json`，vLLM 参考单块
  prefill）× 200 token：kern chunk=512（4 块）eager 与 graph **200/200 逐字节
  一致**，prefill 10.0k tok/s（eager）/ 9.1k（graph）；chunk=2048（单块）在第
  85 个 token 分叉（`docs/qwen38/compare-long-*.json`）。没有再追这条
  （M=1787 与 M=512 的 GEMM 算法选择、或分块边界，二者之一）。
- `kern test --a --b` 自己对自己：DIFF 段遍历了 prefill 1079 / decode 742
  个 dispatch，报 "nothing to test: the programs are identical"
  （`docs/qwen38/test-self.txt`）——harness 能读这份带 `bytes_fixed` state
  的 manifest。
- 数值残差的最终定性：prefill 路径上 in_proj_qkvz（N=16384）exact、in_proj_ba
  （N=96）1 ulp；decode 路径在 state 精确时每个 op exact；kern 在 4×512 分块
  下与 vLLM 单块 1787 token 完全一致。**kern 的算术就是 vLLM 的算术**，剩下的
  是 cuBLAS 在小 M / 瘦 N 形状上的算法选择——这也是 manifest 里唯一没被
  sha256 钉住的 extern。

## Stage 2 — DFlash2 投机

### 13:45Z–14:00Z  准备（无人工干预）

- 13:45Z Stage 1 提交 `f273c9a`。
- 13:46Z `tools/capture_qwen38_spec.sh`：同一套注入库抓 vLLM 的 dflash 投机路径
  （`dumped-kernels/pid1576524`，173 cubin）。与 Stage 1 capture 对
  symbol 集合：新增 94 个 symbol。要点——
  - verify 路径的 GDN 层走的是**递推核**（`fused_sigmoid_gating_delta_rule_update`
    ×1249 ≈ 26 轮 × 48 层，`causal_conv1d_update`），不是 chunked FLA；每轮还有
    `precopy/preprocess_mamba_align_fused` + `postprocess_mamba_fused` 三个
    state 对齐核（被拒 token 的 state 回滚）。
  - draft 的 attention 走了 `flash_fwd_sm100`（cute FA，packed-struct ABI，
    不可钉）——draft 用 Stage 1 已钉的 Triton unified attention 实例代替。
    draft 只影响接受率不影响正确性（greedy 投机无损），这条替换不需要 oracle。
  - selector：`_selector_walk_kernel`、`_prepare_dflash_inputs_kernel` 等
    Triton 核 + ATen topk/sort。
- 13:48Z vLLM bs=1 CUDA-graph 参考（Stage 3 性能表用；5 条 prompt ×
  400 token 均值）：普通 decode **95.0 tok/s**（默认后端）/ **96.0 tok/s**
  （TRITON_ATTN + triton GDN）；DFlash2 投机 **149–199 tok/s**
  （`docs/qwen38/vllm-perf-*.json`）。对照 kern Stage 1 graph decode 80.8 tok/s
  （5120-hidden、64 层、bs=1，每步 742 个 dispatch）。
- 13:50Z runtime 改动一处：`Runtime::load_weights` 接受多份 safetensors
  （target 50 GiB + draft 单独一份，不重复导出 50 GiB）；kern-run / kern test
  的 `--weights` 可重复。+10 行。
- 13:52Z vLLM DFlash2 参考（开 stats 重跑）：5 条 prompt 均值 **175.6 tok/s**
  （149–208），817 轮 / 5719 draft token / 1191 接受 → **2.46 token/轮**，
  接受率 20.8%，逐位接受 [521, 307, 164, 93, 46, 35, 25]（第 1 位 64%，第 7 位
  3%）。普通 decode 95.0 → 投机 1.85×。

### 14:00Z–14:16Z  生成器 + driver + 抽核 + 首次端到端（无人工干预）

- 14:00Z–14:10Z `tools/gen_qwen35.py --spec <spec dump>`：同一个生成器长出六个
  program（`prefill`/`decode`/`decode_spec`/`verify`/`draft`/
  `draft_precompute`）。设计要点——
  - **target 在投机下的 GDN 换成 vLLM 自己的投机核**：`causal_conv1d_update`
    （seqlen 8 / state_len 10 constexpr，按 `num_accepted_tokens-1` 取历史）+
    `fused_sigmoid_gating_delta_rule_update`（T=8 逐行 checkpoint SSM state 到
    `ssm_state_indices[i]`，初始 state 取 `[num_accepted-1]`）。state 布局改为
    每层 8 页（conv 204800 B + SSM 3 MiB，385 页 = 1.31 GB `bytes_fixed`），
    prefill 的 FLA 链只是把 `chunk_h` 的 h0/ht 指到新页、`conv_fwd` 换成 page
    stride 重烤的实例（`pin_nearest`：同符号同 REG 的实例里取 .ttir diff 最小
    的）。**回滚免费**：被拒 token 的 checkpoint 页下一轮直接覆盖，vLLM 那三个
    `*_mamba_align/rollback` 核一个都不要。
  - `decode`/`decode_spec` = 同一套投机核在 tokens=1 下跑（`gdn.one` 当
    num_accepted）；Stage 1 的 packed decode 核在这份 manifest 里不出现。
  - DFlash2 的 5 个 tap（第 5/19/33/47/61 层之后的 hidden+residual）：fc 的
    [5120, 25600] 拆成 5 个列块，在 tap 处做 β=0/β=1 的 GEMM 累加进 `fc_out`，
    不拼接（新 extern `cublaslt_bf16_tn_acc`）。
  - draft：5 层非因果 Qwen3 层 + 每层两对 grouped conv（prepare/finish），
    KV 用 DSpark 的布局和它借来的 `unified_noncausal.cubin`；draft 的 norm /
    rope / cache write 直接钉 vLLM 的 CUDA 核（`rms_norm<8,2>`、`<8,3>`、
    `fused_add_rms_norm`、`rotary_embedding`、`reshape_and_cache_flash`，用
    参数个数 + REG 从 spec dump 里钉）。K-only rope = 同一核，q 指向 k、
    kv_heads=0。
  - 三个新手写核（`tools/kernels-src/`，共 159 行）：`dflash_conv`（两 tap
    grouped conv，bf16 链顺序照 ATen）、`topk_row`（top-16）、`dflash_select`
    （rank-256 双线性打分 + 7 步贪心游走）。为什么手写：vLLM 里这三段是
    ATen 算子链 / 带 packed-struct ABI 的 Triton 核，不可钉；它们只影响接受
    率不影响输出。`tools/test_kernels_dflash2.py` 对 ATen 参考全过。
  - 跨 dump 钉核：Stage 1 dump 里有 spec dump 没有的实例（conv_fwd、
    reshape_and_cache、unified 2D/3D），生成器按 sha 前缀命名 cubin，
    `extract_kernels.sh` 改成多 dump 目录按 sha 查找。
  - Stage 1 manifest 由同一生成器重生成，与提交的 `examples/qwen3.8-27b.json`
    逐字节相同（非 spec 路径零改动）。
- 14:10Z–14:14Z runtime/driver：`kern-manifest` 加 `meta.spec {block,
  mask_token}`（+15 行，schema +31 行，runtime 不解释）；kern-run 的
  `spec_decode` 泛化（draft 行数 / mask token 来自 meta，`num_accepted_tokens`
  = 1 + 上轮接受数，首 token 来自 prefill）；`Caller::new` 填所有
  `*block_table`。kern-runtime 本阶段只有多 blob 那 10 行。
- 14:14Z 首次 verify 失败：manifest 里留着 Stage 1 的 packed decode 核和
  `gdn.line_index`，runtime 拒绝死核/死 buffer——生成器加了按引用裁剪。
- 14:16Z **首次端到端投机跑通**（eager）：10-token prompt → 63 token，
  25 轮，2.52 token/轮，接受率 24.0%，逐位 [16, 11, 8, 4, 2, 1, 0]，
  15.5 ms/轮 = **162 tok/s**，文本连贯（Aqua Appia, 312 BC …）。
  对照 vLLM 的 2.46 token/轮 / 20.8%。

### 14:16Z–14:20Z  Stage 2 验收（无人工干预）

5 条参考 prompt × 400 token，`tools/qwen38_compare.py --spec`
（oracle 换成 Stage 1 的 kern decode 输出）。结果文件
`docs/qwen38/spec-{eager,graph}.json`、`docs/qwen38/spec-manifest-plain-graph.json`。

- **eager 与 graph（draft + verify 各一张 CUDA graph）5/5 逐字节一致**——
  投机路径本身是确定的。
- 与 Stage 1 plain decode 的一致长度 **[66, 132, 62, 185, 193]/400**。不是
  逐字节：verify 走的是 vLLM 的投机核（sigmoid-gating 递推核 + T=8 的 2D
  attention），和 Stage 1 的 packed decode 核算术不同。但分叉点 66、62 正是
  Stage 1 与 vLLM 分叉的位置——同一批 near-tie。全矩阵（一致长度）：

  | vs | kern spec | kern plain (S1) | vLLM spec | vLLM plain graph | vLLM eager |
  |---|---|---|---|---|---|
  | kern spec | — | 66/132/62/185/193 | 66/196/196/82/112 | 124/196/196/82/112 | 124/93/264/185/178 |
  | vLLM spec | | | — | 66/262/222/400/165 | 66/93/196/82/112 |
  | vLLM plain graph | | | | — | 130/93/196/82/112 |

  vLLM 自己的投机 vs 自己的 plain graph 也只有 66/262/222/400/165；kern spec
  对 vLLM eager 参考的一致长度（124/93/264/185/178）反而比 Stage 1 plain
  （66/93/62/255/178）还长。所有配置两两都落在同一个"near-tie 翻转"区间里。
- 接受统计（5 prompt 合计 765 轮 / 1995 token）：**2.61 token/轮**，接受率
  24.6%，逐位 [518, 301, 183, 108, 63, 38, 26]；vLLM 自己是 817 轮、
  2.46 token/轮、20.8%、[521, 307, 164, 93, 46, 35, 25]——draft（手写
  selector + 借来的 attention 实例）和 vLLM 的 draft 行为几乎重合。
- 吞吐（graph）：**177.9 tok/s** 均值（133.6–220.8），15.1 ms/轮；eager
  169.5。vLLM DFlash2 graph 175.6（149–208）。同一 manifest 的 `decode`
  program（投机核在 tokens=1）80.8–83.1 tok/s = Stage 1 的数字；kern
  plain→spec **2.2×**（vLLM 1.85×）。
- 附带发现：spec manifest 的 `decode`（投机核 T=1）对 Stage 1 decode 的一致长度
  89/93/62/185/**12**——prompt 4 在第 12 个 token 多了一个逗号。这个位置在
  其它 9 个配置里全都一致。用 vLLM 的 top-2 logprob 间距核实
  （`tools/qwen38_margins.py` → `docs/qwen38/margins.json`）：该位置间距
  **0.125**——正好是 bf16 logit 在这个量级上的一个量子；见 Stage 3。

## Stage 3 — 统计与结论

### 时间

| 阶段 | 起止（UTC） | 用时 | 人工干预 |
|---|---|---|---|
| 读文档 + 环境 + capture | 12:02Z–12:10Z | 8 min | 0 |
| 逐核 ABI 分析 | 12:12Z–12:40Z | 28 min | 0 |
| 权重导出 / 参考 / 手写核 / 生成器 | 12:31Z–12:54Z | 23 min（部分并行） | 0 |
| 首次端到端 + 逐 token 对比 | 12:55Z–13:20Z | 25 min | 0 |
| 精度定位到最后一个 op；验收放宽 | 13:20Z–13:37Z | 17 min | **1**（放宽为"合理一致"） |
| Stage 1 验收 + 提交 `f273c9a` | 13:38Z–13:45Z | 7 min | 0 |
| Stage 2 准备（spec capture、vLLM 参考、draft 导出） | 13:45Z–14:00Z | 15 min | 0 |
| Stage 2 生成器 / driver / 抽核 / 首次跑通 | 14:00Z–14:16Z | 16 min | 0 |
| Stage 2 验收 + 提交 `9ece6a4` | 14:16Z–14:20Z | 4 min | 0 |

从零到 Stage 1 验收 **1 h 43 min**，Stage 2（投机）再 **35 min**；全程 1 次人工
干预（验收标准），另有 1 次中途询问（用户问"是不是在调精度"，未改变工作）。
没有超过 30 分钟的卡点；最长的一段（精度定位，12:55Z–13:37Z 共 42 min）
分两个阶段各有产出（先定位到 layer，再定位到 op）。

### 代码量（`git diff bf0e944..9ece6a4`）

| 位置 | 改动 | 说明 |
|---|---|---|
| `crates/kern-manifest` + `crates/kern-runtime`（3646 行） | **+49 / −12** | schema：`State.bytes_fixed`（+11）、verify 放行固定字节 state（+12）、`meta.spec`（+15）；runtime：state 分配按 `bytes_fixed`（+3）、多 blob 权重（+10） |
| `crates/kern-run`（示例 caller，不是 runtime） | +284 / −81 | `prefill_emits_next_token` 契约、`--stop-tokens`、`--weights` 可重复、`KERN_PROBE_LAYER`、投机 driver 泛化（meta.spec / num_accepted / 首 token 来自 prefill / 所有 `*block_table`） |
| `schema/manifest-v2.schema.json` | +31（生成） | |
| `tools/gen_qwen35.py` | **1417 行**（新） | 两份 manifest 的全部模型知识：layer 结构、ABI、钉核、投机 state 布局 |
| `tools/kernels-src/*.cu` | 6 个文件 360 行 | Stage 1：Gemma norm 146、sigmoid_mul 26、copy_rows 29；Stage 2：dflash_conv 41、topk_row 62、dflash_select 56 |
| 其余 `tools/`（导出、参考、对比、探针、测试） | ~1000 行 | 一次性脚本，模型无关的部分可复用 |

新模型的成本落点：**runtime 49 行 vs 生成器 1417 行 + 手写核 360 行**。
runtime 那 49 行里没有一行提到 GDN、DFlash 或 Qwen——`bytes_fixed` 是"固定
大小的 state"，`meta.spec` 是 caller 契约，多 blob 是权重装载。

### 手写核为什么存在（六个，全部 < 150 行）

manifest 的目标是零手写。六个例外各有一个不可钉的理由：
- `gemma_rms_norm`（norm + fused-add + per-head 三个入口）：vLLM 的 Gemma norm
  是 `forward_native`，纯 ATen 算子链，没有单个核可钉。
- `sigmoid_mul`：attention 输出门控 `attn * sigmoid(gate)`，ATen。
- `copy_rows`/`last_row`：把 strided 视图变连续、取最后一行——vLLM 里是
  `.contiguous()` / 索引。
- `dflash_conv`：DFlash 的 grouped conv，ATen 链。
- `topk_row`：`torch.topk`（ATen，cub 内部核不可钉）。
- `dflash_select`：Triton `_selector_walk_kernel` 带 packed-struct ABI，不可按
  cuFuncGetParamInfo 消歧。
每个都对 ATen 参考做了逐 bit 测试（`tools/test_kernels_*.py`）。

### 性能（GB300 单卡，bs=1，CUDA graph 两边都开；vLLM 0.28）

| | kern | vLLM | 备注 |
|---|---|---|---|
| plain decode | **80.8 tok/s**（12.4 ms/step，742 dispatch） | 95.0（默认后端）/ 96.0（TRITON_ATTN + triton GDN） | kern 85%；差距来自 742 次独立 kernel launch 没有融合（vLLM 的 decode 也是同一批核，但 graph 内 launch 更密）。5 条 prompt × 400 token 均值。 |
| chunked prefill | 10.0k tok/s（1787 token，4×512 chunk，eager） | — | Stage 1 |
| DFlash2 投机 | **177.9 tok/s**（133.6–220.8） | 175.6（148.8–207.6） | 2.61 vs 2.46 token/轮；kern plain→spec 2.2×，vLLM 1.85× |

投机的速度差距为零而 plain decode 差 15%——投机每轮的 verify 是 T=8 的一次
forward，launch 开销被摊薄。

### runtime 改动清单（本次 bring-up 全部）

1. `State.bytes_fixed`：state 可以是固定字节而不是每 token 字节（GDN 的
   conv/SSM state 按层不按 token）。
2. verify：`bytes_fixed` state 的 offset 上界按固定字节检查。
3. `Runtime::load_weights(&[&[u8]])`：多份 safetensors，每个权重恰在一份里。
4. `meta.spec {block, mask_token}`：投机 caller 契约，runtime 不解释。

### 设计发现（精度调试 → kern 的机制）

- **manifest 里唯一没被 sha256 钉住的东西就是残差的来源。** Stage 1 的逐 op
  对比把 kern 与 vLLM 的差异收敛到 cuBLAS 在 M=43、N=96 GEMM 上的算法选择
  （1 ulp / 4 个元素）——`extern:cublaslt_bf16_tn` 是 manifest 里唯一由
  runtime 自行挑算法的 dispatch。下一步自然是把 cublasLt 的 algo id 也写进
  manifest（`extern` 带 `algo` 字段），让 GEMM 和 Triton 核一样可钉、可 diff。
- **`kern test` 需要"外部参考"这一侧。** 现在它 diff 的是两份 manifest；
  这次真正有用的是"manifest vs vLLM 的逐 op 中间量"（`KERN_PROBE_LAYER` +
  `qwen38_probe_vllm.py`）。把 vLLM 的 forward hook 输出当作一份"参考 tap"
  喂给 kern test，就能在一次运行里回答"第一个不一致的 dispatch 是哪个"。
- **near-tie 翻转不是 bug 信号。** 5 条 prompt 上，vLLM 自己 graph vs eager、
  spec vs plain、chunk 16 vs 单块，两两一致长度都在 66–400 之间；kern 的每个
  配置也落在同一区间。要区分"算术不同"和"算术错误"，需要看的是 top-2 logit
  间距，而不是一致长度（`tools/qwen38_margins.py`）。

### 14:20Z–14:27Z  near-tie 核实 + 文档 / README / 网站（无人工干预）

- `tools/qwen38_margins.py`：vLLM eager 参考轨迹上每一步的 top-2 logprob
  间距（5 prompt × 200 步）。**所有落在参考轨迹上的分叉点，间距全部
  ≤ 0.125**（0.0 / 0.0625 / 0.125 三个值，即 bf16 logit 的量子），而各 prompt
  的中位间距是 1.1–2.0：

  | prompt | 分叉位置 → 间距 | 中位间距 |
  |---|---|---|
  | 0 | 66 → 0.125，124 → 0.0 | 1.12 |
  | 1 | 93 → 0.0 | 1.38 |
  | 2 | 62 → 0.0625，147 → 0.0，196 → 0.125 | 1.5 |
  | 3 | 82 → 0.0，185 → 0.125 | 2.0 |
  | 4 | 12 → 0.125，112 → 0.0，178 → 0.0 | 1.25 |

  （分叉后不在参考轨迹上的位置——如 prompt 1 的 132、prompt 4 的 193——
  没有可比的间距。）prompt 4 第 12 个 token 那个"多出来的逗号"也是 0.125。
  结论：kern 的每一次分叉都是 bf16 量子级的 near-tie 翻转，与 vLLM 自己
  graph/eager、spec/plain 之间的翻转同类；**没有算术错误的证据**。
- 两次 vLLM 启动失败都是环境问题（脚本缺 `__main__` 守卫 → spawn 重入；
  非登录 ssh shell 的 PATH 没有 venv 的 `ninja`），各 1 分钟。
- README Proof 加一行、网站 06/MEASURED 加一张卡（`website/dist` 是容器里
  root 建的删不掉，改用 `vite build --outDir` 到 scratchpad 验证构建通过）。
