# Runtime：`kern-runtime` + `kern run`

## MVP 范围（已定）

- **只做 `decode` 一个 program**：prefill 视为特殊的 decode——prompt 逐 token
  过 decode 路径（tokens=1）。慢但正确，先让端到端闭环；真 prefill program
  之后加。
- **bs=1**，不考虑 batching。（已扩展：`kern-serve` 的 continuous batching 见 [serve.md](serve.md)——decode 多序列，prefill 仍 bs=1。）
- GEMM 走 runtime 特判（cublasLt extern op）；attention 与 reshape_and_cache
  收编 vLLM TRITON_ATTN backend 的 Triton 核；norm/rope/silu 收编 vLLM CUDA
  cubin。自己写的核只有两个（`tools/kernels-src/`）：embedding（trivial
  gather）和 argmax（greedy 采样）。

## 执行器（`crates/kern-runtime`）

`kern-runtime` 是模型无关的执行器（依赖只有 crates.io：cudarc/half/
safetensors/thiserror，可开源）：verify manifest → `kernels/` 下每个
cubin 算 sha256，**只装载 manifest `modules` 表点名的哈希**（其余文件不碰；
`tools/extract_kernels.sh` 按这些哈希从 dump / 手写核 build 里凑齐，落地名
`<module>-<sha12>.cubin`）→ 逐 op 逐 launch 在自己钉的模块里解析 entry
（文件名不参与），**同一模块里同名的 Triton 多 constexpr 实例靠
`cuFuncGetParamInfo` 参数布局与 manifest params 比对来消歧**（phase-2
ABI 校验兼做实例选择，绕开了 capture 缺 launch→module 映射的坑）→ 按
var max 分配全部 buffer / 分配 state（分页与 per-seq 的走下面的块池）→
按每个 weight buffer 的 `bind` 从 checkpoint 里拷张量段拼出 buffer
（张量来自一个 `Tensors`，见下「权重来源」；scratch 按 impl 声明另行私有分配）
→ 跑 `once` program 算派生表 → 顺序重放
call 表：接口实参解析一次，逐 launch 按 `args` 连线转发/接 scratch/
填字面量后 raw `cuLaunchKernel`（实参 staging 成小端 u64 slot；>48KB
动态 shmem 自动 `cuFuncSetAttribute`）。

**state 内存（K1b）**：定长 state 按声明分配；分页 state（`bytes_per_token`）与
per-seq state（`bytes_per_seq`）**共用一份物理块预算**（`kern-pool`：`chunks.rs` 记块，
`pages.rs` 的 `Pool` 在其上记页与 slot）。每个这样的 state 保留一段虚拟地址
（`cuMemAddressReserve`，一次保留永不搬，`DeviceBuf::Reserved`），页与 slot 各是
地址上的一段块区间；物理块 `cuMemCreate` 一次建齐，块大小是 2 MiB 粒度的整数倍、
不超过最小对象的一半、封顶 64 MiB（qwen3.8 24 MiB）；最小对象连两个粒度都不到时
「对象跨两块」本就不成立，直接取封顶 64 MiB（qwen3-4b / K3 / DSV4.1）——driver 对池子
的每个操作按块数计费且全 tray 串行，DSV4.1 的 102 GiB 从 52245 块 13–16 s 降到 1632
块 0.5 s（tray18，2026-09-14）。谁用谁 map。块留在上次用它的地方：还回的页还是页、还回的 slot 还是 slot；只有一类用光时
才从**另一类的空闲对象**上拆块（`Remap`：先 unmap 再 map 再 `cuMemSetAccess`；
拆最高编号的、补最低编号的空位；跨对象边界的块按使用计数共享，最后一个用户走了
才 unmap）。计划由 runtime 的后台线程执行：先等 stream 上记的事件（此前入队的
kernel 都过了才 unmap），完成后主线程在下一次 `lease` / `checkpoint` /
`lease_from` 里收下、把新 map 的块在 stream 上清零，新页 / 新 slot 才可租。同一
时刻最多一个计划在飞。租不到时三种拒绝分工明确：`Remapping`（计划已定，落地后再
问，别淘汰）、`Busy`（有被持有的对象挡路，淘汰点什么）、`ExceedsPool`（怎么摆都
放不下）。`seqs.max` 只限一步 batch 的行数；slot 从 `seqs.max + 2` 个起按需长
（session 睡着时它的 checkpoint 拿着 slot，活跃请求再要就从空闲页拆），
`index_into` per-seq state 的域上界是运行时的 slot 上限（`Provision`）。预算：
调用方以 `Capacity { tokens, seqs }` 报自己的数——`seqs` 是它要同时活着的序列数
（kern-serve：每 rank `(max_seqs + 1) × t`，pad 也算一条），**不是 manifest 的
`seqs.max`**（那是一步能寻址的行数；93 层 K3 的 slot 449 MB，按 `rows.max` 256 预留
就是 115 GB，EP4 直接 OOM）；`tokens` 给则 = tokens × Σbytes_per_token +
(seqs+2) × Σbytes_per_seq，tokens 向下对齐到 manifest 里 `index_into` 该 state 的
最大页单位（block table 的 `stride`），不会出现半页；`tokens: None`（kern-serve
不给 `--capacity` 时的默认）则在 buffer、scratch 和定长 state 都分完之后
`cuMemGetInfo`，剩余显存减
`HEADROOM`（1 GiB）全给；每个 pooled state 自成一个整块的 arena，预算按 state
逐个向上取整（按 Σ 取整一次会让每个 state 的最后一页 / 最后一个 slot 差一块）。
整块读写 state（`read_state` / `write_state_at` / `zero_states`，kern test 用）
只在第一次 remap 之前有效；program 里 state 指针和铺满维（`dims` 末维 0）的
tensormap 按整段预留编码，remap 新做出的页和 slot 在装载时编好的图里就可寻址。
`Runtime::lease(tokens)` 一次租下 KV 页和每个 per-seq state 的一个 slot
（租时在 stream 上清零），`Lease::seq_line(table, r)` 给出 line 表的项（宽表
`[lines, seqs, w]` 的格宽由 `seq_width` 给出，caller 决定 line 落在哪一项）；
租约 drop 时一起归还。`Runtime::lease_slot()` 只租 slot 不租页：TP 下一行的
KDA state 按头分在组里每张卡上、MLA 页只在 owner 卡，peer 卡上这一行就是一个
slot-only 的 `Lease`（`paged() == false`，不能 `slot(pos)`），它的 checkpoint /
retire / fork / restore / park / wake 只搬 slot、长度由调用方说（E5 tray 级）。

**checkpoint（K1）**：`Runtime::checkpoint(&mut lease, len)` 把序列前 `len` 个
token 留成 `Checkpoint`——页进共享链（一页一个引用计数节点，一条序列每页一个
checkpoint 共用一条链，所以 checkpoint 深度再大也只是一个节点），有 per-seq
state 时把 state 拷进一个新 slot（没有空 slot 报 `Denied::Busy`）；
`Runtime::retire(lease, len)` 是请求结束时的零拷贝形式：`len` 之外的页归还，
slot 原样移交。`Runtime::lease_from(&checkpoint, tokens)` 从 checkpoint 起一条
新序列：整页共享，`len` 落在页中间时把那一页拷一份（新序列往里追加，
checkpoint 自己那页不动），state 拷进新 slot；租约的 `prefix()` = `len`，
`slot(pos)` 拒绝 `pos < prefix`，也拒绝落在已封进链的页里的位置（`checkpoint` 之后
那些整页有了 `Node`，封它们的租约也不能再写；harness 读字节走 `page_ids()`）。谁拿着句柄谁持有：页在最后一个 lease /
checkpoint drop 时回池，checkpoint 本身不会被 runtime 淘汰。
纯 host 的 `Prefix` 索引（`kern-pool/prefix.rs`，设计见 `pool.md`）是 token 键的 radix
tree：`insert(&tokens, cp)` 把 checkpoint 挂在它的 token 路径上（同样的 token 是同一个
条目），`lookup(tokens)` 给出覆盖 prompt 真前缀（不含最后一个 token）的最长可用条目，
`Hit` 持有条目的一个 clone；`evict(park)` 拿最久未命中的 resident 条目 park 进 host
或丢掉（同一条路径叶子先走，drop 叶子才真正还页）；逻辑时钟计数，不读钟，同样的
token 序列给同样的判定。决策（共享哪些页、拷哪一页、
拷哪个 slot）由 `Pool` 在 host 上算成 `Copies`，runtime 只在 stream 上执行拷贝。
kern run / kern test 仍默认 4096（test 的 workload 抽样以 capacity 为界）。

**fork（K2）**：`Runtime::fork(&mut parent, len, tokens)` 从一条活着的序列的前 `len`
个 token 分出一条新序列：整页共享（引用计数 +1），`len` 落在页中间时把那一页拷一份
给孩子（父序列继续往自己那页写，孩子往拷贝写，各写各的），有 per-seq state 时把父
的 slot 拷进新 slot——所以带循环状态的模型只能在父的当前位置分叉（state 就是此刻的
state），纯 KV 模型可以分在任何位置。和 `lease_from` 是同一套 `Pool` 决策（`Copies`），
区别只在源头是 `Lease` 还是 `Checkpoint`，父序列不用先 checkpoint。`lease_from`
现在也接受 `len < checkpoint.len`（只对纯 KV 的 checkpoint，整页处起步）。

**host 层（K3）**：`Runtime::reserve_host(bytes)` 一次 `cuMemHostAlloc` 一块 pinned
DRAM（GB300 上约 100 ms/GiB，只做一次），分配前把线程的 NUMA 策略绑到这张卡所在的
节点（sysfs 的 `numa_node` + `set_mempolicy`，分完恢复默认）：一个 GB300 tray 两颗
Grace 各挂两张卡，本地 DRAM 拷贝 180 / 197 GiB/s（park / wake），跨到另一颗 CPU 的
只有 115 / 117——同一个 6 GB 唤醒 31 ms 与 52 ms 的差别就在这里；park 分两步：`Runtime::room(checkpoint) -> Result<Room, Checkpoint>`
只在 host 块上找地方（放不下时把 checkpoint 原样退回，调用方淘汰点什么再试；`Room`
drop 即退地），`Runtime::park(room) -> Parked` 才拷页和 slot——分开是为了 tray 级的
park 能先在四张卡上都找到地方再动一个字节（半途失败的 park 无法撤销）；runtime 攥着
这个 checkpoint 直到拷贝落地，页和 slot 才回池，前缀一直可查；`Runtime::wake(&parked, len, tokens) -> Waking`
按 `tokens` 一次取够页（`Host::restore`，与 `lease_from` 同形）并把前 `len` 个 token 的页（和 slot）
拷进去，`Runtime::awake(waking) -> Result<Lease, Waking>` 不阻塞地问拷贝落地没有，落地了才交出
这份 `Lease`（`Runtime::landed(&waking)` 只问不拿，tray 用它先看齐四张卡再一起 awake）——没有
`Lease` 就没有程序能读到还在路上的页，这是类型保证的，不靠 compute stream 等事件（`Waking`
提前 drop 会等拷贝完再还页）；一个请求的房间只问一次（2026-09-15，lessons.md）。host 上
的页也是链（`kern-pool/host.rs`，一页一个节点，device 节点记着它 host 副本的 `Weak`），同一 session
下一轮再 park 只拷新增的页，醒来的再睡一个字节不拷；一页在 host 上是所有分页 state 的该页首尾相接，slot 同理，
按 64 KiB 粒度 first-fit（页从低端长、slot 从高端长）。拷贝走单独的 transfer stream
（`cuMemcpy2DAsync`，连续页折成一次），transfer stream 在 compute stream 已入队的一切
之后开始，compute stream 从不等它，decode 步不排在拷贝后面。`Prefix` 表按
`Tier::{Resident, Parked}` 分层：`lookup` 先挑 resident；纯 KV 的 checkpoint 链在表里
是一个随页增长的条目（每个深度都登记），所以 parked 的条目部分命中时只醒需要的页；
`evict(Some(|cp| ...))` 是调用方腾地方的一个动作（park 最冷的 resident，host 满了先丢
最冷的 parked，没有 host 就丢）；表对它存的东西是泛型的（`Prefix<R, P>`，`R: Kept`、`P: Kept` 只要求 `tokens()` / `has_slot()`），
单卡存 `Checkpoint` / `Parked`，kern-serve 的 tray 存四张卡的元组。实测
（tray08 GB300 单卡，2026-09-03，`crates/kern-runtime/examples/park_wake.rs`，写入
按位置的 pattern、park、清零、wake、逐字节比对）：qwen3.8-27b 形状 98k token =
125 页 × 49 MiB + 147 MiB slot = 6.12 GiB，park 34 ms（180 GiB/s）、wake 31 ms
（197 GiB/s），发起各 0.1 ms，pinned 块落在另一颗 CPU 上时 53 / 52 ms；qwen3-4b 形状 40k token = 2500 页 × 2.25 MiB = 5.49 GiB，
park 31 ms、wake 28 ms，发起 0.9 / 0.5 ms（2500 页折成 ~50 次拷贝）。C2C 单向 memcpy
实测 116–119 GiB/s（D2H）/ 85–117（H2D），多 stream 不涨、双向叠加 208 GiB/s；2-D
拷贝比逐页 memcpy 快得多，每次 memcpy 发起约 12 µs。
`extern:cublaslt_bf16_tn` 特判：行主序 `C[m,n]=A[m,k]@W[n,k]^T` 映射成列
主序 `C'=W_cm^T×A_cm`（transa=T、lda=ldb=k、m'=n、ldc=n）；
`extern:cublaslt_bf16_tn_acc` 是同一条路径 β=1（`C += A@W^T`，c 参
`inout`），投机解码的 fc 分块累加与 markov 偏置都靠它，省掉 concat
缓冲和拷贝核。
这两个 BF16 输出入口接受 `[a,w,c,m,n,k]`，可追加第 7 参 C 行步长
（默认 n），再追加第 8 参 A 行步长（默认 k），单位都是元素。
配合 buffer 字节 offset，可直接读取交错存储的组并写入输出列带，
无需先转置或复制输入。
`extern:cublas_bf16_tn_f32` 同一映射但结果落 **f32**（cublasGemmEx，
`CUBLAS_COMPUTE_32F` / `DEFAULT_TENSOR_OP`，独立 cuBLAS handle + 32 MiB
workspace，可捕获）：K3 的每条稠密投影都是 f32 partial 再由认证的 `k3_land`
核落成 bf16，所以 landing 的舍入链与 pegainfer 一致；权重行带 / 输出列带
用 arg 的字节 `offset` 加第 7 参 `ldc` 表达。

**多卡（E0/K0）**：state 一律 VMM 分配（`cuMemCreate` + `cuMemAddressReserve`
+ `cuMemMap` + `cuMemSetAccess`），`export: true` 的 buffer 同样。设备报
`HANDLE_TYPE_FABRIC_SUPPORTED` 就带 fabric handle（GB300/NVL72；driver
拒绝则退回并 warn），否则（HGX B300、没有 IMEX channel 的 tray）不带：
`PeerHandle` 是 `Fabric`（64 B handle）或 `Local`（allocation handle 本
身，只在导出进程内有意义，`cuMemMap` 直接收），两条路对 kernel 完全一样
——peer buffer 里都是本 rank 可寻址的 VA。`KERN_NO_FABRIC=1` 强制走
`Local`，是在有 fabric 的 tray 上跑 B300 路径门禁的钩子。manifest 带 `topology` 时
`Runtime::load(.., Some(&Topology))` 给出本 rank 在每个组的下标（大小
必须与 manifest 一致），`{"rank": g}` 实参在 compile 时烧成常量。
`export_handles()` 返回每个 export buffer / 每个 state 的 `PeerHandle`
（handle + 映射字节数；`Fabric` 有 `to_bytes()` 72 B 一行的线格式，传输
是 caller 的事：共享盘、TCP 都行；`Local` 没有线格式，只在同一进程里传）；`import_peers(group, members)`
收全组每个成员的 handle 表（自己那份随意，用本地地址），
`Fabric` 经 `cuMemImportFromShareableHandle`、`Local` 直接 reserve + map
后把 `u64[组大小]` 写进
该组的每个 `peer` buffer（同一目标只映射一次），映射与 runtime 同寿命。
所有 peer buffer 填满之前 `run`/`run_range`/`capture`/`time_*` 一律
`Api` 错误。装载时收到 peer buffer 的每个 kernel launch 都过
`cuobjdump -sass` 扫描，见 manifest.md 信任边界一节。门禁
`crates/kern-runtime/examples/peer_barrier.rs`：同进程 4 个 Runtime 各占一
卡，一份 SPMD manifest（`tools/kernels-src/peer_barrier.cu`：release 存到
每个成员的 flags、acquire 自旋 + globaltimer 超时、错误码落 output），
tray03 4×GB300 实测 captured burst **3.75 µs/barrier**、eager run+sync
15.8 µs；`--drop r` 让 r 缺席，其余 rank 2 s 内报"等 r 超时"而不是挂住；
换成 multicast bulk copy 的同名 kernel 装载即被拒。

**TMA 描述符与簇 launch**：pack 里的 `tensormap` 字段在装载时对
finished 指针（buffer 基址 + call offset）`cuTensorMapEncodeTiled` 成
128 字节镜像，拷进 pack 镜像的字段偏移处，launch 时整个 pack 按值塞进参数
槽（ABI 校验里它就是一个 `bytes<n>` 参数）；`cluster` 走 `cuLaunchKernelEx` +
`CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION`（无 cluster 的 launch 也走同一
条路，attrs 为空）。E1 门禁 `crates/kern-runtime/examples/k3_moe_ep.rs`：
K3 pruned 第 1 层 MoE（224 expert，top-16，mxfp4 权重）作为三个 op 的
program（`tools/kernels-src/k3_mega_stage.cu` 的 quant_x / write_routing
往 slab 里写，然后 DeepGEMM MegaMoE 一个 launch：dispatch → L1 → situ →
L2 → combine），slab 是一个 `export` 的 carry buffer，peer 数组由 kernel
在设备上读（`tools/k3-mega/`：fork 只改签名，`SymBuffer` 的偏移表从
`__grid_constant__` 换成 `const int64_t*`）。权重与测试向量由
`tools/export_k3_moe.py` 导出（含 host 参考），manifest 由
`tools/gen_k3_moe.py` 生成（几何与 slab 偏移来自 `k3_mega_layout_dump`）。
tray04 4×GB300 实测（2026-09-02）：EP4 每 rank 64 token **227 µs/层**
（captured），四个 rank 的输出与 EP1（256 token 单卡，733 µs/层）对应行
**逐字节一致**；EP1 对 host 参考 max |err| 0.015、相对 RMS 1.7e-3，
917504 个元素无一超 5%+0.05。

**权重来源（`Tensors`）**：`load_weights` 只认一个 trait——按名字给出张量的
dtype、shape 和字节（`Blob`：本进程内存的一个切片、一个文件的一段，或本 context
能读的设备地址，比如另一个进程的分配经 `Runtime::map` 映射进来）。`plan` 是纯
函数，把一个 buffer 的 `bind` 段变成一列拷贝并核对铺满；shell 按 `Blob` 的形态选
拷贝：设备地址 device-to-device，host / 文件字节经 pinned staging 上卡（下），
host placement 的 buffer 反过来，host / 文件字节 CPU 读进去、设备字节 DtoH。
runtime 内的实现是 `Safetensors`（shard 只读 header，`open` 拿文件、`parse` 拿
内存里的字节，同名张量出现在两份 shard 里就拒绝）；权重缓存、对象存储是 caller
侧的实现，自己实现 `Tensors`。

文件不 mmap 而是 `pread`：mmap 后 memcpy 每页付一次缺页，缺页在 mm 的锁上串行，
读者再多也没用；`pread` 是 page cache 自己的拷贝，随读者数线性涨（GB300 读 Ceph
上的 shard，warm：mmap+memcpy 1/8/32 读者 7/25/28 GiB/s，pread 11/76/109 GiB/s；
`bench_results/2026-09-15-dsv41-e2e-swap/scripts/readbench.py`）。上卡走 pinned
双缓冲：16 条线程各持两块 16 MiB 的 page-locked 块和一条 stream，把一段拷贝
`pread` 进一块的同时另一块在 DMA，段按计数器分给线程；一个连续拷贝不论多少行
一次读完（按行 `pread` 在 DSV4.1 是每 rank 六千万次 syscall，300 s）。以前是
mmap 后 pageable `cuMemcpyHtoD`，驱动在调用线程上做同样的 staging，缺页也在那
条线程上。DSV4.1 EP4 每 rank 126 GiB、4 rank 同时装、page cache warm
（2026-09-15）：Ceph 上 pageable 5.8–16.3 s/rank，pread staging 8 线程 7.2–8.5 s、
16 线程 4.4–5.4 s（24–29 GiB/s，四 rank 合计 ~105 GiB/s，就是 readbench 的上限）；
本地 ext4（tray07）pageable 15.4–16.7 s，pread 8 线程 4.8–6.0 s、16 线程 4.4–5.8 s。进程也不再挂着
500 GB 的映射（RSS 490 GB → 3 GB）。
连续的设备侧拷贝走 `cuMemcpyAsync`，按 8 条 stream 轮流发：`cuMemcpy2DAsync` 在
fabric 映射的源上每次约 90 µs 固定开销，DSV4.1 每 rank 24923 次 78.8 GiB 单 stream
2.3 s、8 条 2D 1.2–2.2 s、8 条 1D 0.5–0.7 s（111–148 GiB/s）；先发完设备拷贝再填
host placement 的 buffer（第一个到的 rank 填，其余等），engram 一张 94 GiB 表 DtoH
约 1 s，整段 2.2 s（tray18，2026-09-14）。
同一张卡上接连装两份只差 kernel 的 manifest（`kern test` 的 A 与 B）时权重不必
装两次：`Runtime::into_resident` 把已绑定的设备权重 buffer 从要丢弃的 runtime
里拿出来（`Resident`，其余 state、workspace、program 随 runtime 释放），
`Runtime::load_over` 在分配时按名字认领声明相同的 buffer——dtype、shape、
placement、export、`bind` 的张量都相同——直接当作已填，`load_weights` 只拷没
认领的；没被认领的随 `Resident` 释放。调用方保证两边绑的是同一份 checkpoint 的
同一个 rank（gpu 和 rank 不同时 `load_over` 拒绝）。host placement 的权重不走
这条路，它们本来就经 `HostWeights` 在同一 scope 内共享。

**错误分类**（`kern_runtime::Error`，按"谁需要行动"分变体）：
`ManifestParse`/`ManifestVerify`/`Manifest`（provider 修生成器）、
`KernelArtifact`（cubin 缺失/哈希不符/ABI 不匹配/peer launch 含 multicast
指令，重新抽核）、
`WeightArtifact`（权重与 manifest 不符，重新导出）、`Api`（caller 用法
错误：未知 buffer/program、kind 不符、var 越界、graph vars 不一致）、
`Call`（定位 call 表位置并包住底层错误）、`Cuda`/`Driver`/`Blas`
（执行期 CUDA 失败）。

## Caller 契约（`crates/kern-run`）

`kern run` 是 qwen3-4b 的 caller 契约（CLI 用 clap，日志走 tracing 到
stderr，`RUST_LOG` 控制级别，stdout 只出生成文本）：**chunked prefill**——
前 n-1 个 prompt token 按 `--chunk`（默认 512，clamp 到 tokens 上界）切块
连调 `prefill`（每块填 token_ids/positions/slot_mapping 前缀 + seq_lens=
已见数 + cu_seqlens_q=[0,块长]；`write_input` 支持前缀写，尾部 stale 字节
grid 界内永不被读），最后一个 prompt token 走 decode 出首个 logits；此后
每 step 填四个小 input（=pos），block_table 恒等；greedy argmax。权重就是
HF checkpoint（`--weights` 给 snapshot 目录或若干 .safetensors）：qkv /
gate_up 的拼接写在 buffer 的 `bind` 里，rope 表 / kv_scales / tied lm_head
这类派生物由 manifest 的 `load` once program 在设备上算（manifest.md「权重」）。

**CUDA graph（manifest 的 program 声明 `graph`，`--eager` 是调试回退）**：tokens=1 下 436 个 call 的
grid/标量实参全是常量，每步只有 4 个小 input buffer 的**内容**变、指针不变
→ 整个 call 表 stream-capture 成一张静态图，H2D 写留在图外，每步一次
`cuGraphLaunch`。`Runtime::issue` 按 program 的 `graph` 决定走图还是逐 launch；
graph 按 (program, vars) 键控——vars 只需给这个 program 的 launch 真正读到的 var（grid、
shared_mem、标量实参、pack 字段），其余 var 不属于它，caller 给不给、给多少都归一成最小值（K3 的 `decode` 不读
`span`，`decode_span` 读）：decode 捕在 tokens=1；prefill 不声明 `graph`，
每块按实际长度逐 launch 跑（走图不快，见 manifest.md）。要点：capture
不能用 legacy NULL stream（runtime 已改 `new_stream()`）；cublasLt 可被
捕获（workspace 预分配，算法启发式在捕获时定死，顺带省了每步的 CPU
开销）；`run_captured` 校验 vars 与捕获时一致（var 值烧死在图里）。

**greedy 采样已下沉 GPU**：`tools/kernels-src/argmax.cu` 两段式行 argmax（64 block
分部归约 + 1 block 收尾；单 block 版 nsys 实测 55.7µs/步——单 SM 读 300KB
只有 5.5GB/s，两段式 5.5µs），平局取最小下标与 CPU 扫描语义一致，折叠成
一个两 launch 的 `argmax` op impl（partial 缓冲是私有 scratch）进
manifest 进 graph；`logits` 降级为 workspace，新增 output `next_token`
i64["tokens"]，每步回读从 300KB 变 8B。input 侧 H2D 走常驻 pinned
staging（pageable 会退化成驱动同步拷贝）。

## 实测（GB300）

输出连贯（"The capital of France is Paris. The capital of Germany is
Berlin. ..."；150 token 长文不劣化，KV 跨页正常），完整 step（含采样回读）
graph 2.7 ms ≈ 377 tok/s、eager ~3.0 ms，两路输出逐 token 一致。对照：
vLLM 0.28 本尊同卡 bs=1（TRITON_ATTN、graph 默认开）2.44 ms/token ≈
409 tok/s——kern ~92%。
**chunked prefill 实测**：709 token prompt 两块（512+197）58–60ms ≈
**12k tok/s**，vs 逐 token 假 prefill 2.18s——TTFT 提升 ~37×；三路交叉
验证（chunk=512 走图 / chunk=1 逐 token / eager）生成逐字节一致，2D 与
3D attention 实例在重叠输入上数值互证。

**nsys 定位（别猜，profile；纯 decode 窗口 + CUPTI 区间求并对账）**：
- GEMM 虽是唯一没从 vLLM 挖的核（nvjet ABI 挖不动，runtime 自己调
  cublasLt），但 heuristic 选出的 nvjet 内核和 vLLM 完全一致、逐核耗时
  持平（128x8 17.1 vs 17.7µs、splitK 7.9 vs 8.6、lm_head GEMV 124 vs
  126µs 已近带宽极限）——GEMM 不是差距。
- 每步 GPU busy：kern 2.25ms < vLLM 2.58ms（我们的 kernel 时间反而短，
  vLLM torch.compile 的 triton 小核并不更快）；差距全在每步边界 GPU
  空转：kern ~174µs vs vLLM ~71µs——我们 sync→8B 回读→4×H2D→graph
  launch 纯串行，vLLM async scheduling 把 host 活藏进 GPU 时间。

## 端到端流程（dump → manifest → run，宿主机裸跑即可，不需要 docker）

```bash
# 1) 挖：CUPTI 注入抓 vLLM（TRITON_ATTN）的全部 cubin + launch ABI 流水
#    （自动建 .venv 装 vLLM；挑张空卡跑，~几分钟）
CUDA_VISIBLE_DEVICES=0 tools/capture_qwen3.sh        # -> dumped-kernels/pid<N>/

# 2) 分析（可选，看切 pass/指针分类/表达式拟合报告）
.venv/bin/python tools/mine_capture.py dumped-kernels/pid<N>/launches.jsonl

# 3) 生成 manifest：真实 ABI + 手写连线，挖矿地址逐项断言证伪
.venv/bin/python tools/gen_qwen3_decode.py dumped-kernels/pid<N>/launches.jsonl
                                                     # -> examples/qwen3-4b.json

# 4) 抽核：按 manifest 钉的 sha256 从 dump 里拷 module、从 target/cubins 拷
#    手写核（tools/build_kernels.sh 编的），落地 <module>-<sha12>.cubin；目录只增不减
tools/extract_kernels.sh examples/qwen3-4b.json dumped-kernels/pid<N>   # -> kernels/
# 或 `kern kernels`：按 kern.toml 的 [kernels].dumps/.sources 给每个 target 的 manifest 与 reference 落 cubin

# 5) 权重：就是 HF checkpoint 目录，不导出（拼接在 `bind`、派生表在 `load`
#    once program，都由生成器写进 manifest；见 manifest.md「权重」）
ln -s /path/to/hf/snapshots/<rev> weights/Qwen3-4B

# 6) 跑（构建在 kernel-lab 容器里做；binary 宿主机 dlopen CUDA 直接跑）
./target/release/kern run            # 一切来自 kern.toml 的 target；flag 覆盖：
./target/release/kern run \
  --manifest examples/qwen3-4b.json --kernels kernels \
  --weights weights/Qwen3-4B \
  --gpu 3 --capacity 4096 --chunk 512 --prompt "The capital of France is" --steps 320
```

启动输出即配置声明的展示面：manifest 元信息/var/state/buffer 分类统计、
逐 op 逐 launch 的 entry+参数布局+解析到哪个 module（gemm 显示 runtime
built-in）、权重绑定、graph 捕获（436 call → 每步 1 次 graph launch）。
