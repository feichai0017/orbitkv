# Manifest 格式（schema_version 4）与 Verifier

Model provider 交付 `manifest.json + kernels/ + weights`，runtime 负责：
加载时像 rustc 一样苛刻地校验 manifest，运行时按声明闭眼执行。runtime
不感知任何模型语义——它只会调度不透明的 op 调用、按字节数供应不透明
的 state、以及对一个封闭的标量表达式集合求值算 launch 几何。

## 词汇表：每一层一个词，互不撞

```
programs.<name>[]            call     调用一个 op        {"op": "attn", "args": [...]}
ops.<name>                   op       接口 + 实现        {"params": [...], "impl": {...}}
ops.<name>.impl.launches[]   launch   起一次 module 入口 {"module": "argmax", "entry": "kern_argmax_partial"}
modules.<name>               module   launch 钉住的工件  {"source": "argmax.cubin", "sha256": "…"}
vars.<name>                  var      caller 每次调用供应的标量，有上界
states.<name>                state    不透明持久内存，runtime 按字节供应
buffers.<name>               buffer   有类型的张量：input / output / weight / workspace / carry / peer
topology.groups.<name>       group    多卡 SPMD 的 rank 组：只有名字和大小
```

`call` 调 `op`，`op` 由若干 `launch` 实现，`launch` 起 `module` 里的
`entry`。"kernel"这个词在 manifest 里不出现——它留给 CUDA 意义上的
`__global__`（`kernels/` 目录、`kern kernels` 子命令说的都是 cubin）。

## 顶层结构

顶层平铺，全部名字唯一、引用必须解析、不允许未知字段：

```json
{ "schema_version": 5, "model": "qwen3.8-27b",
  "topology": {"groups": {"ep": 4}},
  "vars": …, "states": …, "buffers": …, "modules": …, "ops": …, "programs": … }
```

- `schema_version`：wire format 版本，只认 4。`model`：标签，runtime
  不赋予含义（kern-serve 用它做对外的模型名）。
- `vars`：caller 每次调用时提供的标量（如 `tokens`），声明 `max`，下界
  恒为 1；所有静态校验在界上进行，运行时拒绝越界值。**会改变内存尺寸或
  launch 几何的标量才是 var**；别的标量（temperature 之类）是数据，走
  `[1]` 形状的 input buffer。
- `states`：不透明持久内存。**runtime 只知道字节数**——`bytes_per_token`
  （按 token 容量伸缩：paged KV）、`bytes_per_seq`（每个活跃序列一个
  slot：GDN 的 conv + SSM 递归状态；runtime 供应 `seqs.max + 2` 个 slot，
  slot 0 永不出租，kernel 可把 line 下标 0 当 null）或 `bytes`（定长），
  三选一。内部布局是 provider 生成器里的算式，以字面量 offset
  传给 provider 自己的 kernel。state 一律走 VMM 分配（`cuMemCreate` +
  reserve + map），别的 rank 可映射（设备支持时经 fabric handle，否则
  同进程内经 allocation handle，见 runtime.md 多卡一节），所以 `peer`
  buffer 可以 `of` 一个 state（P/D push、跨 rank 读 KV 的入口）。
- `buffers`：`dtype + shape + kind`，可选 `domain`（内容的先验，见下）与
  `fill`（在 serving 循环里的角色，见「Serving 协议」）。shape 维度是常量或 var 名；kind 说
  的是"谁供应、活多久"：`input`（runtime 写入）/ `output`（runtime 读回）
  / `weight`（load 时从 checkpoint 的张量拼出来，`bind` 说是哪些，见「权重」）/ `workspace`（runtime 规划，跨次执行
  不保留）/ `carry`（一个 program 写、另一个 program 读的交接棒，跨次
  执行保留；谁先跑是 caller 契约，verifier 只要求它被某个 program 写到
  ——投机解码的 aux 隐状态逼出来的）/ `peer`（runtime 填的地址数组，见
  下）。任何非 peer buffer 可加 `"export": true`：分配走 VMM，别的 rank
  可映射。
- `topology`（可选）：`{"groups": {"ep": 4}}`，只声明组名和大小。有它的
  manifest 是 SPMD 的：每个 rank 装同一份，装载时给出自己在每个组里的下标
  （`Runtime::load(.., Some(&Topology))`）。成员是谁、handle 怎么交换是
  caller 的事；runtime 不知道什么是 all-reduce。两样东西用到组：
  - `peer` buffer：`{"dtype": "u64", "shape": [4], "kind": "peer", "of":
    "flags", "group": "ep"}`——`u64[组大小]`，第 i 项是组内 rank i 的
    `of`（一个 `export` 的 buffer 或一个 state）的设备基址，本 rank 也在
    里面。`import_peers` 之后由 runtime 填好；对 op 只读；ABI 上就是
    `in buffer<u64>`，kernel 拿到指针数组自己寻址（vLLM custom_allreduce
    / TRT-LLM / DeepEP intranode 都是这个形状）。
  - `{"rank": "ep"}`：call 或 launch 的标量实参来源，本 rank 在组里的
    下标，load 时烧成常量，只能接 `i32`/`i64` 参。
- `modules`：manifest 的依赖清单（必填）——每个 kernel launch 钉住的代码工件：
  `source`（本地文件名 `argmax.cubin`，或 registry ref
  `hf:<org>/<repo>/<path>[@revision]`）+ `sha256`。**身份是 sha256，
  source 只是标签**（registry ref 时兼做 URL）。
- `ops`：**接口 + 实现分离**（为 kernel 可插拔）。
  - **接口**是调用点契约：类型化参数列表——`"in buffer<bf16>"` /
    `"out buffer<fp8e4m3>"` / `"inout state"` / `"i32"`/`"u8"` 等；
    buffer/state 必须声明方向，方向驱动数据流校验。
  - **实现（`impl`）是可整体替换的微程序**：`scratch`（impl 私有工作区，
    dtype+shape 声明，调用方看不见）+ `launches` 顺序 launch 列表。每个
    launch 二选一——**kernel launch**：`module`（`modules` 表里的名字，
    必填：manifest 是完整的依赖清单，runtime 只装载它点名的工件；同一
    module 里同名的 Triton constexpr 实例靠 `params` 布局区分）+ `entry`
    （module 里的入口点）+ 几何；**extern launch**：`entry` 写
    `extern:<name>` 表示 runtime 内置（`cublaslt_bf16_tn` / `_acc` /
    `cublas_bf16_tn_f32`，见 runtime.md），没有 module 也没有几何。其余字段：
    `params`（该 launch 自己的 ABI；**不写 = 同接口**）、
    `block`/`grid`（grid 用下述表达式集合；可选 `shared_mem`，上限 227KB
    opt-in；可选 `cluster: [x, y, z]` 线程块簇，grid 每轴必须是它的倍数，
    runtime 走 `cuLaunchKernelEx`；可选 `pdl: true` 程序化依赖启动：runtime
    给 launch 加 `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`，
    核可以在前一个 launch 收尾时就上 SM，**manifest 承诺**该核在读写前一个
    launch 产出或消费的任何东西之前执行 `griddepcontrol.wait`，wait 之前只许
    读 weight；verifier 不检查也检查不了，extern 不接受）、`args` 连线：`{"param": i}` 转发接口第 i 参 /
    `{"scratch": name}` 接私有工作区 / 字面量标量（impl 私有常量）/
    `{"rank": group}` / `{"pack": {...}}`；**不写 = 按序转发接口参数**。
    **bytes<n> / pack** 是 launch 私有的参数类型：核的 ABI 收 struct
    （CUTLASS、CuTe DSL 编出来的核都是），manifest 就声明 `"bytes<48>"`，
    实参 `{"pack": {"size": 48, "fields": [{"at": 0, "param": 3},
    {"at": 8, "i32": 512}, {"at": 12, "var": "tokens"}, {"at": 16,
    "i64": 4608}]}}`：每个字段一个字节偏移和来源（接口参——指针带 call 的
    offset、标量带值——/ scratch / 字面量 / var / expr / rank / tensormap），
    没写到的字节为 0；宽度默认随来源（指针与 i64 8、i32/f32/var/expr 4、
    u8 1、tensormap 128），`"width"` 可改。指针和字面量在装载时定死，var
    字段每次 run 重算，与标量参一样。
    **tensormap** 字段是 128 字节的 `CUtensorMap`，偏移必须 64 对齐：
    `{"at": 0, "tensormap": {"param": i, "dtype":
    "u8"|"u4"|"i32"|"bf16"|..., "dims": [内层在前, 元素数], "strides":
    [第 2 维起的字节步长], "box": [smem tile 元素数], "swizzle": 0|32|64|128,
    "l2_promotion": 0|64|128|256, "oob_nan": bool}}`——对接口第 i 个 buffer
    / state 参（call 的 offset 照算）在装载时 `cuTensorMapEncodeTiled`；
    dtype 是 TMA 眼里的元素类型，与 buffer 的 dtype 无关（一个 `u8` slab
    上可以同时挂 fp8 activation 和 i32 scale 的描述符）；`dims` 最外层可以
    写 0，意思是"铺满这个 buffer / state"——装载时按 call 的 offset 之后
    剩下的字节数算出该维（分页 cache 的页数是 runtime 定的，manifest 不
    知道）。核收裸描述符（CuTe DSL）就是 `bytes<128>` 里一个 at 0 的字段；
    核收 cute `TiledCopy`（CUTLASS）就是 `bytes<256>`：描述符在 0，动态
    stride 的 int 在 128（`k3-kernel-abi.md` K8）。DeepGEMM 这类 TMA kernel
    的 18 个描述符就这样从 manifest 里长出来，host 侧不再有 launch 代码。
    FlashInfer 的 MLA decode 核 28 个参数（5 个描述符 + 张量 struct +
    FastDivmod）也是这么铺平的（`k3-kernel-abi.md` K5）。所以一个"ABI 即接口"的单 launch op 只
    需要 `module`、`entry`、`block`、`grid` 四个字段；两段式 argmax、
    vLLM attention（unified + reduce_segments）这类"一个逻辑算子 = 多次
    launch + 私有中间缓冲"整体折叠成一个 impl，不向调用方泄漏。
- `programs`：每个 program（如 `prefill`/`decode`）是
  `{"batch"?, "once"?, "calls": [...]}`——`batch` / `once` 说它怎么被
  serving 循环调（见「Serving 协议」），`calls` 是一段顺序 call 列表：
  `op` 名 + 接口实参（buffer/state 实参可带字节 `offset`，默认 0：kernel
  收到 base+offset——provider 用它寻址融合 buffer 里的视图如 qkv 的
  q/k/v 切片、state 里的逐层区域，offset 是 provider 布局算术的字面量，
  runtime 只做加法；`label` 只给人看）。launch 几何在 impl 里，不在 call
  里。

**表达式**是封闭集合：常量、var 名（裸字符串，和 shape 一个写法）、
`{"ceil_div": [e, c]}`、`{"mul": [e, c]}`——这不是语言，是填空模板，永远
不会加控制流。grid、`shared_mem`、domain 的界都用它；call 的标量实参
除 `{"var": "tokens"}` 和字面量外还可以是 `{"expr": {"mul": ["tokens", 32]}}`
（prefill 逼出来的：head-norm 的"总 head 数"参数 = tokens×heads）或
`{"rank": "ep"}`。call 实参里 var 要带 `var` 标签，因为它和 buffer 名混在
一起。

```json
"ops": {
  "gemm":       { "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                  "impl": { "launches": [{ "entry": "extern:cublaslt_bf16_tn" }] } },
  "argmax_row": { "params": ["in buffer<bf16>", "out buffer<i64>"],
                  "impl": { "scratch": { "pmax": {"dtype": "f32", "shape": [1, 64]}, "pidx": {"dtype": "i32", "shape": [1, 64]} },
                            "launches": [
                              { "module": "argmax", "entry": "kern_argmax_partial_bf16",
                                "params": ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                                "block": [1024, 1, 1], "grid": [1, 64, 1],
                                "args": [{"param": 0}, {"scratch": "pmax"}, {"scratch": "pidx"}, {"i32": 248320}] },
                              { "module": "argmax", "entry": "kern_argmax_final_i64",
                                "params": ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                                "block": [64, 1, 1], "grid": [1, 1, 1],
                                "args": [{"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 1}, {"i32": 64}] } ] } }
},
"programs": {
  "decode": { "batch": {"groups": 1, "rows": 1}, "calls": [
    { "label": "embed", "op": "embedding", "args": [{"buf": "token_ids"}, {"buf": "model.embed_tokens.weight"}, {"buf": "residual"}, {"var": "tokens"}, {"i32": 5120}] },
    …
    { "label": "sample", "op": "argmax_row", "args": [{"buf": "logits"}, {"buf": "next_token"}] } ] } }
```

**ABI 常量属于 impl，不属于接口。** 挖矿来的 kernel 带着一堆
stride/flag/eps 参数；它们对一个模型是常量。生成器的 `normalize` 后处理
（`tools/kern_manifest.py`）把"每一次 call 都传同一个字面量"的接口标量
折进 impl 的 launch 字面量，接口只剩会变的东西：qwen3.8 的 `attn` 从 28
参降到 10 参（全是 buffer/state），`silu_mul` 只剩 `(out, in)`。这让接口
真正是契约——换 impl 的人自带自己的常量，调用点一字不动。同一个后处理
也把 launch 里的内联 `cubin/sha256` 提升到 `modules` 表、抹掉恒等连线和
重复的 `params`。

**Domain（buffer 内容的先验，可选）**：`buffers.<name>.domain` 声明"这个
buffer 里的值合法长什么样"。挂在 buffer 上而不是 op 接口上——知道
`buffer<i32>` 是页表而不是激活的是接模型的人，不是写 kernel 的人；换
impl 不必重写先验，kernel package 零改动。两种形式互斥：

- `{"min": lo, "max": hi}`：闭区间，端点可以是整数、浮点或同一封闭表达式
  集合（`"tokens"`），任一端可省略；
- `{"index_into": "<buffer|state>", "stride": n}`：每个元素是目标 buffer
  的**行**下标或目标 state 的 **token 槽**下标，下标 i 指向第 `i×stride`
  行/槽（默认 1；paged KV 的 block_table 一个下标覆盖 16 个 token）。
  指向 `bytes_per_seq` state 时元素是 **line** 下标，一条 line 是
  `stride` 字节（GDN 的一层一页）；这样的 buffer 形如 `[lines, seqs]`，
  runtime 按租约填 `slot × (bytes_per_seq / stride) + line`。宽表
  `[lines, seqs, w]` 给每个 (line, seq) 格 w 个项，供按序列取 line 列表的
  kernel（vLLM 投机核的 `ssm_state_indices[seq, 8]`）：caller 把 line 填在
  其中一项（哪一项由 program 的契约定，如 verify 填 0、advance 填接受数），
  其余填 0 = null line。

附加 `"monotone": true` 要求非递减序列（`cu_seqlens` 这类前缀和）。

```json
"token_ids":    {"dtype": "i64", "shape": ["tokens"], "kind": "input",
                 "domain": {"index_into": "model.embed_tokens.weight"}},
"block_table":  {"dtype": "i32", "shape": [256], "kind": "input",
                 "domain": {"index_into": "kv", "stride": 16}},
"cu_seqlens_q": {"dtype": "i32", "shape": [2], "kind": "input",
                 "domain": {"min": 0, "max": "tokens", "monotone": true}}
```

它是一元的（只说一个 buffer，不表达 buffer 之间的关系）、是先验不是行为
描述：verifier 只证明它对着自己的声明自洽（界的类型 vs dtype、
`index_into` 可解析、min ≤ max），不证明任何 kernel 需要或维护它。**不填
完全合法**——e2e 一模一样地跑；填了以后 runtime 在 `write_input` 时校验
host 写入（O(n)，免费），`kern test` 据此为整数 buffer 合成合法随机值、
并检查 kernel 产出的值落在声明域内（后置条件）。浮点 buffer 不写 = 任意
有限值。整数 buffer 不写 = kern test 不检查它的产出。

**可插拔**：换一个 op 的实现 = 只改它的 `impl` 块（可能在 `modules` 里
多一行），接口、程序连线、其余 manifest 一字不动；verifier 静态把关新
impl 与接口的自洽（方向、dtype、scratch 数据流），runtime 加载时用
`cuFuncGetParamInfo` 比对每个 launch 声明的 ABI，`sha256` 钉住工件。这
就是"kernel 市场"的交换单元。

**module 的身份是 sha256，source 只是标签。** runtime 把 kernel 目录里的
每个工件算哈希，**只装载 manifest 点名的那些**，按哈希给 launch 找模块，
**从不按文件名找**。所以
一个目录可以同时放一个核的每一个版本（`gemm8-3f9a1c2d4e5b.cubin`、
`gemm8-9b0c….cubin`，`tools/extract_kernels.sh` 就这么落地：module 名 +
sha 前 12 位），上一个 commit 的 manifest 和这一个都能从同一个目录解析
——`kern test` 的 A/B 正是靠这个。推论：**换编译器、换 flag、改一行源码
= 换核**，哈希不同就是另一个工件，manifest 钉的是生成它时在场的那一次
build，两个 build 数值上是否等价由 `kern test` 说了算，不由名字说了算。

**Registry ref**：`source` 写 `hf:<org>/<repo>/<path>[@revision]`
（revision 默认 `main`）时 runtime 加载时把它物化进内容寻址缓存
（`$KERN_CACHE_DIR` 或 `~/.cache/kern` 下 `blobs/<sha256>`，命中免网
络），下载后先验哈希再落盘，**传输通道零信任**：名字只是 URL，身份是
哈希。工件可以是裸 cubin，也可以是 host 共享库（如 HF kernel hub 的
torch 扩展 .so）：runtime 剖开 ELF 取 `.nv_fatbin` 里的设备代码逐容器
装载，torch/python 绑定整个丢弃，entry + ABI 逐位核对照旧。实证：
`examples/qwen3-4b.json` 的 `silu_mul` impl 指向 module `activation` =
`hf:kernels-community/activation`（PyTorch 生态在用的原装包），输出与
挖矿基线逐字节一致。

**Wire format 的 ground truth 是 `kern-manifest` 的 Rust 类型**（parser
即法律）；`schema/manifest-v5.schema.json` 是导入时的上游 v5 快照，
OrbitKV 当前扩展由 `schema/manifest-v6.schema.json` 描述。runtime 仍接受
v5，只有 scratch-backed TMA 需要 v6。两份 schema 都给生成器/agent 当
形状契约用。

Manifest 是**生成产物**（类比 `Cargo.lock`）：provider 手写的是生成器，
不是 manifest。生成器把一切写长（每个 launch 带完整 ABI 和连线、每次
call 带 kernel 要的全部标量、工件内联），最后过 `normalize`
（`tools/kern_manifest.py`）得到最小的 wire form——像 linker 一样，不改变
运行的东西。最小完整样例 `examples/minimal.json`（六段、一个 op、一次 call，网站
schema 页开头就是它，测试保证它永远过 verifier）；真实样例见
`tools/gen_qwen3_decode.py` → `examples/qwen3-4b.json`
（Qwen3-4B，两个 program：`prefill` 433 call / `decode` 436 call，真实
挖矿 ABI）与 `examples/qwen3-4b-dspark.json`（同上 + DSpark 投机解码：
target+draft 权重同处一份 manifest，多一个 7 行的 `round` program，见
[spec-decode.md](spec-decode.md)）。

**故意留下的重复**：64 层展开成 64×26 个 call（decode 742 个 call 里
737 个是逐层模板）——加 `repeat` 就是加控制流，kern test 按 call 切、
verifier 按 call 查都靠展开；weight buffer 的 dtype/shape 与 checkpoint 的
safetensors header 重复——没有权重文件也要能 verify，load 时再对账。冗余在 manifest 不在 schema，
"源码"是生成器。

## 权重：`bind`

权重的 ground truth 是模型自己的 HF checkpoint（一堆 safetensors shard），
kern 没有自己的权重格式。每个 `weight` buffer 声明 `bind`：一列 checkpoint
张量（或张量的一个矩形），按顺序首尾相接铺满这个 buffer：

```json
"model.layers.0.self_attn.qkv_proj.weight": {
  "dtype": "bf16", "shape": [6144, 2560], "kind": "weight",
  "bind": [{"tensor": "model.layers.0.self_attn.q_proj.weight"},
           {"tensor": "model.layers.0.self_attn.k_proj.weight"},
           {"tensor": "model.layers.0.self_attn.v_proj.weight"}]},
"draft.fc.1.weight": {
  "dtype": "bf16", "shape": [2560, 2560], "kind": "weight",
  "bind": [{"tensor": "draft.fc.weight", "cols": [2560, 5120]}]},
"lm_head.weight": {
  "dtype": "bf16", "shape": [151936, 2560], "kind": "weight",
  "bind": [{"tensor": "model.embed_tokens.weight"}]}
```

张量按 `[dim0, 其余维之积]` 看成矩阵，`rows` / `cols` 是 `[from, to)`，缺省
全取；连续段一次 memcpy，列块一次 2D copy。`tensor` 和 `rows` 都可以按
rank 选：`{"group": "ep", "tensors": [...]}` 每个 rank 装不同的张量（EP 的
专家），`"rows": {"group": "ep", "ranges": [[0, R], [R, 2R], ...]}` 每个
rank 装同一张量的一片（一张跨组分片的大表，配 `export: true` + `peer`
buffer 让每个 rank 读到整张；末片可以回退重叠，让各片等长，buffer 的
shape 只有一个）。表长必须等于组大小；host placement 的权重是 rank 间
共享的，不能按 rank 选。张量从哪来是 runtime 的 `Tensors` 接口（runtime.md
「权重来源」）：本地 safetensors shard（只读 header，mmap 后按段拷进显存），或另一个进程
映射进本 context 的设备内存（按段 device-to-device 拷）；
张量必须在恰好一份 shard 里，dtype 要等于
buffer 的，各段字节数之和要正好等于 buffer——差一个字节都是 artifact 错误。
verifier 只查 `bind` 非空、名字非空、区间非空；形状对账要有 checkpoint 才能
做，留给 load。

checkpoint 里没有的东西——Gemma norm 的 `weight + 1`、A_log 的 f32、rope 的
cos / sin 表、全 1 的 kv scale、chunk 索引表——不是权重，是 `carry` buffer，
由 manifest 自己的 `once` program（`load`）在 load 之后算出来（
`tools/kernels-src/weight_prep.cu` 的几个 elementwise 核）。拼接方式与派生
表都是模型知识，住在生成器（`tools/qwen_weights.py`），不住在 runtime。

## Serving 协议：`fill`、`batch`、`once`

runtime 之上还有一层契约：一个 serving 循环（`kern run`、kern-serve 的
scheduler）怎么调这份 manifest——哪个 buffer 装 token、哪个装 slot、哪个
program 接受几组几行、跑完从哪读 token。v3 把这层写在 caller 的代码里
（按 `token_ids` / `prefill` 这些名字找），v4 把它搬进 manifest，三个声明，
闭集、填空模板、不是语言：

- **buffer 上的 `fill`**：这个 buffer 在循环里的角色。input：`token`
  （每行一个；形如 `[seqs]` 时每组一个，即投机轮的 anchor）、`position`、
  `slot`、`seq_len`、`cu_seqlens`、`span_at`、`blocks`（tray batch 各成员的块：
  按成员序的行数前缀和，`[t + 1]` 个 i32，末项是 tray 的行数；块不必等长，
  collective 核按它把本卡行号换算成 tray 行号）；output：`tokens`（`[seqs]`
  每组一个，`[seqs, r]` 每行一个）、`count`（每组取几个，缺省恒 1）、`error`
  （非零即该步失败，tray 的 allreduce 用）。fill 只许落在 input / output
  的 i32 / i64 buffer 上，与 `domain` 平级可共存；caller 按 buffer 的
  dtype 编码，不写死 i64 / i32。页表和 line 表不加 fill：`index_into`
  一个 state 本来就是它。没有 buffer 要 `position`，caller 就不写。
- **program 上的 `batch`**：`{"groups": 上界, "rows": 精确行数 | 一个 var
  名}`——这个 program 接受"几组、每组几行"的调用。`rows` 是精确值（round
  的 7 行是 mask 布局，喂 3 行就是错的）；是 var 名（`"tokens"`）时只能
  一组，行数就是这次调用的 var 值（prefill 的 chunk 随末块变短）。可选的
  `"span": var 名`（K5，`{"groups": 256, "rows": 1, "span": "span"}`）：
  这是一个每组 1 行的 decode 步，其中**一组**可以喂一段 run——同一序列的
  连续 token 各占一行，run 的长度是这次调用的 `span` 值、首行写进 `span_at`
  fill 的 `[1]` i32 输入（没有 run 时写 0）；只有带 run 的调用走它。没有
  `batch` 的 program 不被循环驱动（kern test 按段切的材料、k3 的单层 MoE
  测试）。
- **program 上的 `once`**：装载后跑一次、不再驱动（k3 的 `tp_init` 预填
  allreduce 的 poison 值）。与 `batch` 互斥。
- **program 上的 `graph`**：runtime 按调用形状（这次调用的 var 值）首次
  capture 成一张 CUDA graph、之后 replay，一步一次 launch。给形状固定、
  launch 多而短的 program（decode 步、投机 round）；行数随调用变的
  program（prefill chunk）不写，逐 launch 跑——实测 Qwen3.8-27B 的 prefill
  走图与否吞吐相同（2026-09-09，chunk 2048 都是 22k tok/s，512 时走图
  反而慢）；decode 的收益看模型：Qwen3-4B 一步 2.7 对 3.0 ms，Qwen3.8-27B
  单序列 646 个 launch 走图和 eager 都是 11 ms（GPU 才是瓶颈）。哪个 program
  值一张图是 manifest 作者量过才知道、runtime 猜不到的事，所以写在这里；
  `--eager` 只是调试开关，全部按 launch 跑。要求有 `batch`。

轴全部从 fill 派生：行轴的 var 是 `slot` fill 的维（`tokens`），组轴的
var 是 `seq_len` fill 的维（`seqs`），fill 或 line 表跨过的第三个 var 是
tray 轴（k3 的 `rows`）；manifest 里没有一个名字是 caller 认得的。派生
这层的是 `kern_manifest::Protocol::check(&Manifest)`（纯函数，不碰 GPU，
verify 之后的第二遍）：每个 fill 至多一个 buffer、形状与角色相符、
`[seqs, r]` 的 `tokens` 的 r 是某个 `batch` 的 `rows`、`count` 只在有它时
合法、两个 program 不得同形状（span 算形状的一部分）、`groups × rows ≤
tokens.max`、`span` 只落在 `rows: 1` 的 program 上、全 manifest 一个 span
var 且 `span.max ≤ tokens.max`、有 span 就得有 `span_at` fill、至少一个
`batch` program。得到的 `Protocol` 是只读投影：`forward(groups, rows)`
选形状包含 `(b, r)` 的 program 里 `groups` 上界最紧的那个，`chunk()` 是
接受 `(1, var)` 的那个，`spanned(b)` 是接受 b 组、其中一组带 run 的那个，
`vars(b, per, t)` 是一次调用的 var 表（run 的 `span` 由 caller 加），每个
`Filled` 带 dtype 与轴，每个 `Forward` 知道自己 emit 不 emit token、按哪
个 `count` 取。`kern verify <manifest>` 打印这层事实；kern-run 与
kern-serve 只拿 `Protocol` 驱动 runtime，不读 JSON（CI 用 grep 保证它们
的代码里没有 fill / program / var / buffer 的名字面量）。

一步的形状因此统一：prefill 是"1 组 c 行"，decode 是"b 组 1 行"，投机轮
是"b 组 r 行、每组取 `count` 个"，span 是"b 组 1 行、其中一组 c 行"——同一
段代码，`--rows` 选每组行数（缺省取 manifest 声明的最宽）。设计与取舍见
[v4-design.md](v4-design.md)。

## Verifier（`kern-manifest`）

`verify()` 收集全部错误一次报告（`VerifyErrors`）：

1. `schema_version`；
2. var `max ≥ 1`；
3. state 恰有 `bytes_per_token` / `bytes_per_seq` / `bytes` 之一非零；
4. buffer shape 解析、字节数在 var 上界下不溢出；`fill`（若有）落在
   input / output 的 i32 / i64 buffer 上，input 角色对 input、output 角色
   对 output；domain（若有）自洽：
   界的类型对得上 dtype、`index_into` 指向存在的 buffer/state、
   `index_into` 与 min/max 互斥、`monotone` 只许一维、min ≤ max 在
   var 两端成立；
5. module：`sha256` 64 位 hex、`source` 非空、registry ref 语法；
6. op impl 逐 launch：kernel launch 的 `module` 可解析（没有 module 的
   launch 必须是 `extern:` 内置，带 module 的 entry 不得是 `extern:`）；
   block 不超 CUDA 限制、grid 在 var 上界不超 CUDA 限制/下界不为零；launch args 与 launch params（缺省 = 接口）
   数量/逐位类型匹配；
7. impl 与接口的自洽：launch 不得写穿接口 `in` 参、接口 `out` 参必须被
   某个 launch 写到、scratch dtype + 跨 launch 数据流（禁止读未写的
   scratch）、未使用的 scratch 拒绝；
8. call：op 引用、实参与**接口**参数逐位匹配（dtype 精确匹配、state 只
   能接 `state`、var/表达式的取值范围必须装进标量参数类型、offset 对齐
   且在界内）；
9. 逐 program 数据流：禁止读未写（read-before-write）、禁止写 input/
   weight；output / carry 必须被**某个** program 写到（prefill 这类只落
   state 的 program 合法地不写任何 output；carry 在每个 program 内视为
   已写——它的生产者是另一个 program）；program 的 `once` 与 `batch`
   互斥，`batch.groups ≥ 1`、`rows` 是 ≥ 1 的常量或已声明的 var（fill 之间
   是否凑成一份 serving 契约是 `Protocol::check` 的事，不在这里）；
10. 拒绝一切未使用的声明（buffer / op / module / state / var / topology
    group）；
11. 多卡：组大小 > 0；`peer` buffer 必须 `dtype: u64`、shape 恰为
    `[组大小]`、`of` 指向一个 `export` 的 buffer 或一个 state（不能是
    自己、不能是另一个 peer）、`group` 已声明、不带 domain、自身不可
    `export`；`of`/`group` 只许出现在 peer 上；peer 对 op 只读且视为初始
    已写；`{"rank": g}` 只接 `i32`/`i64` 参且 g 已声明；**带 extern launch
    的 op 不得收到 peer buffer**——runtime 内置（cublasLt）永远不碰 peer
    内存；
12. pack / tensormap / cluster：`bytes<n>` 只许作 launch 参（不进 extern）、
    只接 `{"pack"}` 实参且 `size == n`，字段都在 image 内、互不重叠，引用的
    接口参 / scratch / var / group 都存在；引用 `out` 接口参的指针字段算
    该 launch 写了它。tensormap 字段宽 128、偏移 64 对齐、只能描述接口上
    的 buffer 或 state 参；维数 1–5、`strides` 比 `dims` 少一且为 16 的正
    倍数、`box` 每维 1..=256、内层 box 字节数是 16 的倍数且不超过 swizzle
    跨度、`dims[0]` 字节数是 16 的倍数、只有最外层维可为 0（铺满）；描述
    符寻址的字节数（末元素之后）在每个 call 上不得超过 `buffer 字节数 −
    offset`（var 上界；铺满的维不算），`out` 参上的描述符算写了它。`cluster`
    无零维、不超 16 块，grid 在 var 上下界都被它整除。

反序列化层已拒绝：未知字段（包括实参对象里的未知键）、重复名字、非法
参数类型串。

**信任边界**：verifier 证明的是"声明自洽"，不是 kernel 行为。谎报自己
读写范围的 cubin 在边界之内被信任（debug 路径可用 compute-sanitizer 兜
底）。加载 cubin 后用 `cuFuncGetParamInfo` 比对参数个数/字节布局属于
runtime crate 的 phase-2 校验。唯一一条 runtime 替 verifier 查的 kernel
行为：**收到 peer buffer 的 launch 在装载时被 `cuobjdump -sass` 反汇编，
任何搬内存的 multicast 指令（`UTMALDG/UTMASTG/UTMAREDG.*.MULTICAST`、
`UBLKCP/UBLKRED.*.MULTICAST`）都拒绝**——multicast TMA 打到 peer/fabric
地址会把发起的 GPU 卡死到整机重启（GB300 实测），而 verifier 看不见
SASS。簇内 barrier 的 `UTCBAR.2CTA.MULTICAST` 不碰全局内存，放行
（MegaMoE 用它）。没有 cuobjdump 就拒绝装载。

## 从 v3 到 v4

serving 协议进 manifest：buffer 加 `fill`（闭集，见上），program 从
`[call…]` 变成 `{"batch"?, "once"?, "calls": [...]}`，顶层 `spec` 块删除
（`block` 就是 round 的 `rows`，`mask_token` 是 `splice_draft` 的字面量
实参）。caller 侧不再认 `token_ids` / `prefill` / `decode_batch` /
`seqs` / `tokens` 这些名字：`Verified`（verify 过的 manifest 的 newtype，
`Runtime::load` 只收它）与 `Protocol`（fill / batch 派生的投影）取代了
kern-run 的 `Contract`、kern-serve 的 `SpecPlan`、`first_token` /
`prefill_emits` 嗅探和三处重复的 JSON 解析。投机解码从"host 轮流调
draft / verify / precompute"变成一个 `round` program（dspark 也补了：
splice、接受计数、line 表推进都在设备上），`--spec` → `--rows`。golden
重生成，无兼容层。落地记录见 [v4-design.md](v4-design.md) §9。

## 从 v2 到 v3

命名：每一层一个词（上表）；`symbols`→`vars`（它每次调用都变，不是
const，也不是 shape 意义上的 dim）、`kernels`→`ops`、`steps`→`launches`、
`symbol`→`entry`、`cubin`→`module`（可以是 .so）、`dispatches`→`calls`、
`{"arg"}`→`{"param"}`、`{"sym"}`→`{"var"}`、`class`→`kind`、
`inout ptr`→`inout state`、`unit`→`stride`、`bytes_fixed`→`bytes`、`meta`
拆平到顶层、`version`→`schema_version`。

结构：`modules` 表取代逐 launch 的 `cubin+sha256`，且每个 kernel launch
都必须钉 module（v2 允许只写符号、由 runtime 在全部已装载模块里猜——
kern test 的 c10 假阴性就是它咬的）；`Program` 单字段包装拆掉；launch 的 `params`/`args` 有缺省；`extern:` 不写几何；ABI 常量折进
impl。删掉从没被读过的字段：`states.*.align`、`symbols.*.min`、scratch
实参的 `offset`、`u32` 标量。
