# Kernel 来源：从 vLLM 挖（kernel mining）

MVP 的 cubin 不自己写，从 vLLM 生产路径里挖——挖出来的是 GB300 上真正在
跑的一流核。工具 `tools/kernel-capture/`（vendored from pegainfer PR #982，
Apache-2.0）是一个 CUPTI 注入库：`CUDA_INJECTION64_PATH` 挂进任意 CUDA 进程，
把每个 module 的 cubin、每次 launch 的 grid/block/attrs 和完整 staged 参数
（含指针→device/host + 所属 allocation range 归类）dump 到
`dumped-kernels/pid<N>/`（`module_*.cubin` + `launches.jsonl`）。host `cc`
直编（CUPTI 在 `/usr/local/cuda/targets/sbsa-linux`）。

`tools/capture_qwen3.sh`：uv venv 装 vLLM 0.28.0，`enforce_eager=True`（每个
launch 都是真实 `cuLaunchKernel`、staged 参数可读），4 个长度递增的 prompt
逐个发（bs=1）、各出 4 token。两个必须的环境设置：
`VLLM_ENABLE_V1_MULTIPROCESSING=0`（engine 跑主进程，注入才作用到真正 launch
的 pid，stderr 也不被子进程吞），PATH 里要有 ninja/nvcc（FlashInfer 运行时
JIT attention 核）。Qwen3-4B 多 prompt 实测挖到 **79 cubin / 13042 launch**
（`dumped-kernels/pid5992`，带 `t_ns`；旧的 pid4185422 是单 prompt 无时间戳版）。

## dump 的结构：绝大多数 launch 是 dummy pass——而这是特性

单 prompt 实测里，rms_norm 的 grid 分布显示总共 ~8 个 forward pass，真实请求
只占 prefill(grid=2) + decode(grid=1)×2 三个；其余全是 vLLM 的
memory-profiling（16384 token 满配）和 warmup dummy pass。两个推论：

- 生成器第一步必须**切 pass**。capture 每条 launch 记 `t_ns`（CLOCK_MONOTONIC）。
  多 prompt dump 实测：**>5ms 空隙切到请求级**（4 个请求干净分离，final-norm
  的 grid.x 与 prompt token 数 1/34/85/175 逐一对上）；但请求内 prefill 与各
  decode step 之间最大空隙只有 ~1.7ms 且边界有偏移，**请求内的 forward pass
  要按锚点核（final-norm/采样簇，或按 launch 序列的周期性）切**，不能靠时间。
- dummy pass 不是噪声，是**表达式拟合的采样点**：真实 pass 的 shape 太退化
  （tokens=1/2 时 `ceil_div(tokens,c)` 和 `tokens` 无法区分），多个不同 token
  数的 pass 才能把 grid 表达式定下来。capture 脚本发多长度 prompt 就是为了
  加采样点。

另一个粒度陷阱：**allocation range ≠ 逻辑 buffer**。K/V 两个页池是同一个
3.2GB range 内的不同偏移（PyTorch allocator 一次 malloc 内部切分）；
slot_mapping/positions 等小 buffer 全挤在同一个 2MB arena。buffer 身份要用
`(range_start, offset)` + 跨 pass 指针稳定性（跨 pass 不变 = weight/持久
state，随 pass 变 = activation workspace）来推，不能只看 range。

## 关键发现：两个主力核挖得到、却 replay 不了

capture 按 kernelParams 逐参数拆分，对 flat ABI 的核完美——但恰好最重要的两个
核不是 flat ABI：

- **GEMM**（`nvjet_sm103_tst_*`，11 个变体）走 `cuLaunchKernel` 的 packed
  `extra` 缓冲区传参，不走 kernelParams → capture 只能记 `params: null`，
  参数指向抓不到。（cublasLt `splitKreduce` 反而是 flat ABI、21 参数全抓到，
  但它有两个 host 指针参数（pointer-mode alpha/beta）被当成普通 8 字节标量
  记录、pointee 没 dump——**host 指针不解引用是 capture 的系统性盲点**，值形如
  `0xffff...` 才能肉眼认出；且它只跟在 nvjet splitK 后面跑，孤立 replay 无意义。）
- **Attention** 是两个核：prefill 用 `fmhaSm103aKernel_...PersistentContext`，
  decode 用 `fmhaSm100fKernel_...SwapsAbForGen`（sm100f binary 跑在 sm103 上）。
  都只有 **1 个 1280 字节 packed struct 参数**，指针全埋在 struct 内部。字段
  布局在 flashinfer（trtllm-gen `KernelParams`）源码里是公开的，理论可 rebind，
  但版本锁死、随 flashinfer 升级漂移，不依赖。

flat ABI 的核全部可直接 replay：`rms_norm`(12 param)、`rotary_embedding`(13)、
`act_and_mul`/silu(6)、`reshape_and_cache_flash`(16)。注意模板实例化按
call-site 的**静态维度**选择（rms_norm 的 hidden=2560 与 head_dim=128 是两个
不同 symbol），不随 tokens 变——per-call pin entry 是安全的；但 vec-width
路径假设指针 16B 对齐，runtime 分配 buffer 对齐别低于 vLLM（保守按 256B）。

**结论**：mining 用作 ABI 参考 + 收编 flat 核；**GEMM 特判**——runtime 以内置
extern op 形式直接调 cublasLt（不啃 nvjet ABI，也是将来 collective extern op
的先例）；**attention 收编 vLLM TRITON_ATTN backend 的 Triton 核**（flat
kernelParams，可同法挖、可校验、可 rebind）。

## attention backend ABI 普查（探针实测，`tools/capture_abi_probe.sh`）

vLLM 0.28 起选 backend 只能用 `AttentionConfig(backend=...)`（
`VLLM_ATTENTION_BACKEND` 环境变量已删除，设了静默无效）。GB300 上三家的
launch ABI：

- **FLASHINFER（默认）**：trtllm-gen fmha，1 个 1280B struct → 不可 rebind。
- **FLASH_ATTN = FA4**（FA3 仅 sm90，FA2 无 Blackwell 核）：CuTe-DSL 核，
  19 参数全是 12–128B 的 packed struct/TMA descriptor，**0 个裸指针**——
  rebind 需要 host 侧按真实地址重编 TMA descriptor，比 trtllm-gen 更不可收编。
- **TRITON_ATTN**：`kernel_unified_attention`（prefill/decode 统一）+
  `reduce_segments`（decode split-KV 归并）+ Triton 版 reshape_and_cache，
  **全部 flat ABI**，KV 布局的 stride 全是普通标量参数。bs=1 decode 探针
  吞吐三家几乎持平（15.4/15.2/13.9 tok/s）——decode 是带宽活。

**Triton 同名 ≠ 同 ABI**：unified 的 2D(prefill) 实例 28 参数、3D(decode)
实例 31 参数；`reduce_segments` 的 `num_seqs` 无类型注解、值为 1 时被
Triton 特化进 binary（不再是运行时参数）。收编必须 pin 具体实例，且
capture 目前不记 launch→module 映射，同名多实例分不清出自哪个 cubin
（capture 待补 module id）。

## 跨框架实测：dump SGLang（capture 不挑框架）

同一注入库对 SGLang 直接可用（`tools/capture_sglang.sh`，docker 镜像里跑
——pip 装不通：pypi 的 sgl-kernel 轮子链 CUDA 12 库，与 aarch64 必需的
cu130 torch 冲突）。GB300 上 Qwen3-4B 抓到 55 cubin / 10079 launch，
`mine_capture.py` 无改动切出 3 个 forward、tokens 参照 34/85/175 全对。

但 SGLang 的 ABI 面貌比 vLLM 难挖得多——**几乎全员 struct 化**：

- attention：trtllm-gen fmha（`fmhaSm100f` decode / `fmhaSm103a` prefill），
  1 个 1152B packed struct，同 vLLM FLASHINFER 后端，不可 rebind；
- GEMM：全 nvjet（packed `extra` 缓冲区，`params:null`），与 vLLM 同款；
- 自家 JIT elementwise 核（fused_qknorm/fused_rope/store_kvcache/
  act_and_mul）：**单个 40–80B struct-by-value 参数**，指针全埋在 struct 里；
- 连 flashinfer 的 norm 核参数也是 16B tensorptr 结构（指针+元数据）。

结论：vLLM 至少有 TRITON_ATTN 全 flat 的收编路径，SGLang 只有 norm/
splitKreduce 是多参数 flat；struct 布局都在源码里公开，但逐版本漂移，
rebind 即锁版本。挖矿目标选 vLLM 是对的。

## KV 布局：vLLM 是 paged（实锤）

`reshape_and_cache_flash` 的 ABI 暴露了 vLLM 的 KV 模型：新 token 的 K/V 写进
两个 paged 池，靠 `slot_mapping` buffer 定位；`block_size=16`、`kv_heads=8`、
`head_dim=128`。attention 再吃页池 + block_table。写 KV 与读 KV 是两个独立核。

对我们的 schema：`state<KV>` 在边界处解构出来的不是单指针，而是「页池 ptr +
runtime 每 step 填的 `block_table`/`slot_mapping`/`seq_lens` 小 buffer」。state
声明从「`bytes_per_token` 一个数」演进为「runtime 要替我维护哪几样东西」的清单
——runtime 依旧不知道池子里是 K 还是 V，边界不破，只是更具体。

## 生成器分析前端：`tools/mine_capture.py`

三步全自动，无模型知识输入：`launches.jsonl` → 人读报告 + `--json` 全量结果。

1. **切 pass**：时间空隙切请求窗口；窗口内按 core 核（全局最高频核，每
   forward 一个密集 burst）的 burst 空档中点切 forward。边界有 ±1 launch
   抖动，call-site 对齐按**众数出现次数**容错。
2. **tokens 参照自验证**：tokens = 某个每 forward 恰一次、grid.x 变化的核
   的 grid.x。选哪个不靠先验——对每个候选试拟合全部 site，取「拟合失败+
   欠定」最少者（选错参照会大面积拟不出/欠定，数据自己会说话）。实测自动
   选中 final-norm。
3. **对齐 + 分类 + 拟合**：call site = (symbol, forward 内出现序号)
   （enforce_eager 下 launch 序列确定）。指针按 (range_start, offset) 跨
   forward 稳定性分三类；grid 各轴与标量参数按封闭表达式集合拟合。

pid5992 实测结果：5 个 flat 核全部拟出（grid `tokens`/`tokens*32`/`tokens*8`，
7 个 token 采样点），指针分类与人工判读一致（逐层 KV 池、slot_mapping 全局
persistent、逐层权重、workspace）。仅有的拟不出标量是 **stride 且 prefill/
decode 取值不同**（reshape_and_cache 的 value stride：vLLM decode 1024=
连续、prefill 6144=qkv 融合视图）——在我们的布局里 v 恒在融合 qkv 中，
取 6144 即两个 program 共用的常量（decode tokens=1 时 stride 无效，当初
照抄 1024 也"对"，prefill 把真值逼了出来）。

## 生成器后端：`tools/gen_qwen3_decode.py`

产出 `examples/qwen3-4b.json`（310 buffer；`prefill` 433 call +
`decode` 436 call），**过 verifier**（`qwen3_decode_mined_verifies`
测试）。数据源是 TRITON_ATTN capture（`dumped-kernels/pid3977275`）。
bs=1，两个 program：

- 七个真核：rms/rms_head/rope/fused（vLLM CUDA cubin）+ Triton 版
  reshape_and_cache + `kernel_unified_attention`（decode 用 3D split-KV
  实例 31 参数 + `reduce_segments` 12 参数；prefill 用 2D 实例 28 参数）。
  真实 entry、逐参数类型/方向表、标量字面量逐位取自代表性 decode/prefill
  forward。
- 连线是手写的（provider 知道模型），**挖矿数据负责证伪**：发射前断言
  q/k/v 在 qkv 融合 buffer 中的视图偏移（+0/+8192B/+10240B，q/k norm
  out-of-place）、residual 全程同址、6×36 个逐层权重指针互异、KV 池
  k/v 相距 256B、cache 与 attention 共享同一 KV 池与 k/v scale 指针、
  unified 与 reduce 共享 segm 部分和缓冲、eps 一致。
- KV state 从 vLLM 逐层池改为层交织 `[page][layer][16][8][2][128]`——
  同一批 kernel 靠 page-stride 参数 ×36 和 state offset 字面量
  （layer×65536）适配，bytes_per_token=147456，runtime 仍只知一个数。
- decode attention 是一个两 launch impl：unified 3D 写 f32 segm 部分和
  （impl 私有 scratch），reduce_segments 归并到 attn_out——调用方只见
  `attn` 接口（normalize 把 ABI 常量折进 impl 后只剩 buffer/state 参数），一次 call/层。**prefill attention
  （`attn_prefill`）是同一接口的另一份实现**：2D 实例单步无 scratch，
  其 28 参 launch ABI 恰好就是接口本身（接口切分正确的实证）；grid =
  `[ceil_div(tokens,4), 8, 1]`（两个真实 prompt 长度拟合证实）。
  seq_lens/cu_seqlens_q 是 caller 每次调用前填的 input buffer。
  rms_norm_head 因 q/k 调用点 grid 不同（×32 vs ×8，grid 归
  impl）拆成 qhead/khead 两个 op。
- **prefill program = decode 去掉 final_norm/lm_head/sample 尾巴**（只落
  KV），chunked prefill = caller 连调 prefill + 最后一个 prompt token 走
  decode 出首个 logits——"prefill_last"就是 decode，免掉 var 依赖的
  offset。三个 decode 时被 tokens=1 掩盖的字面量由 prefill capture 现形
  并修正：head-norm 输入行距=6144（融合 qkv）、head-norm 总 head 数 =
  tokens×heads（表达式标量）、cache value 行距=6144。logits/next_token
  定常形状 `[1,·]`、decode 专属 kernel（attn 3D/argmax）grid 与 scratch
  定常——否则按 CHUNK_MAX=2048 上界要多付 ~1GB。
- GEMM launch 用 `extern:cublaslt_bf16_tn` entry（runtime 按前缀特判为
  cublasLt matmul）；embedding 是唯一手写核（`tools/kernels-src/embedding.cu`，
  `kern_embedding_i64_bf16`，20 行 gather）。

这一步倒逼出的 schema 演进（都已进 verifier）：buffer/state 实参字节
offset、`u8` 标量（rope 的 1 字节 bool 参数）、`shared_mem` 上限检查。
