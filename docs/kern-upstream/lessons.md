# 教训（按时间倒序，每条：现象 → 原因 → 以后怎么做）

设计写在 runtime.md / serve.md / multi-gpu.md 里；这里只记那些"不写下来下次还会
再踩一遍"的事，以及它们落到了哪条规则上。

## 2026-09-15，DSv4.1 EP4 第二轮 wake ↔ park 活锁

**一个请求的房间只问一次。** 命中 parked 条目的请求先 `wake` 成 `Checkpoint`、插回索引、
回到队首再 `lease_from`：两次分配、两次"够不够"。DSv4.1 EP4（`--chunk 128 --max-seqs 16`）
的第二轮，醒来的快照加半页拷贝加续写比池子多一页，第二次 `Busy` 时 `make_room` 能 park
的 resident 只有刚醒的那条，于是醒 → park → 醒，一次 host session 打出 236 万行
`parked tokens=86`。四卡真模型跑一次才撞到；toy-stateful 单卡上 e2e 的 `wake_room` 场景
（池子刚好装一条 turn2 的最坏情况）30 秒复现，608 万行。规则：**分配和"够不够"的判断在
一个动作里做完**——`Host::restore` 与 `Pool::restore` 同形，按最坏长度一次取够、前缀拷进去；
腾地方的循环里不能有"腾出来的正是自己刚要的"这条边，一旦要靠"记住刚 park 了谁"来打补丁
（当时试过一版），就是接口在两处做了一件事。另一条：host 场景的 e2e 要有一个刚好差一页的
池子，宽裕的池子什么都撞不到。

## 2026-09-14，半页 checkpoint 之后 host 层复用过期副本（kern-pool 重写）

**共享的页谁都不能再写。** `Pool::checkpoint` 在页中间留快照时把那半页原样挂进链，
lease 接着往同一页写；host 层按页节点去重，第二次 park 认出"已经有一份"就跳过，那份
是过期的字节（review P1）。追下去这不是一处笔误：接口没把"链上的页是冻结的"放进类型
里，`Checkpoint::nodes()` 交出的 `(id, page)` 让别处按 id 相等去推内容相等。规则：**一个
`Node` 存在的那一刻它的页就不再被写**——整页共享、半页拷给新持有者、写者只有 lease
（`pool.md`）；字节级 property test 在 `tests/pool.rs`，随机序列对着"位置 → 戳"的模型，
它直接复现 P1。跨模块认"同一份字节"用节点自己持有的 `Arc` / `Weak`，不用 id 表。

## 2026-09-11，DSv4.1 tray 偶发 NVLink barrier 超时（host 表被 NUMA balancing 搬走）

**注册过的主机内存在 GB300 上不是 pinned 的，NUMA balancing 会在 GPU 读的时候搬它。**
`cuMemHostRegister` 在 Grace 上走 ATS（`VmPin 0`），2×92 GB 的 engram 表每一页都是
balancing 的候选：扫描把 PMD 标成 PROT_NONE，CPU 装完权重再也不碰，hinting fault 全部
来自 GPU 读，由 IOMMU page-request 路径在任意 CPU 的 kworker 里处理，那个 CPU 的 node
就成了迁移目标；一次 512 MiB 拷贝期间读这页的 warp 全停，下一个 kworker 再搬回去。
lookup 一次取几百个随机行，冷启动第一条 prefill 要 10–15 s，某个 rank 的链超过 60 s，
DeepGEMM 的 NVLink barrier 就 trap。规则：**host 表 mmap 后 `mbind(MPOL_LOCAL)`**，VMA
自带策略没有 migrate-on-fault 标志，`task_numa_work` 会跳过；验证看 `/proc/vmstat` 的
`thp_migration_success` 在服务期间不动，冷启动第一条 prefill 和后面的一样快。

**卡死的进程别上 gdb。** gdb 读目标内存时持有目标的 mmap_lock，目标里任何新线程
（Rust 线程启动要 mprotect sigaltstack 保护页）和 munmap 都排在它后面变成 D 态，看起来
像是"锁被内核占了"，其实是 gdb 自己造成的；yama ptrace_scope=1 之下 gdb 只能做启动器，
卡住后 kill -9 让 tray05 整台失联到重启。规则：看内核栈用 **privileged 容器
（`docker run --privileged --pid=host`）读 `/proc/<tid>/stack`**，`sysrq w` 把全系统 D 态
任务连内核线程倒进 dmesg，`/sys/kernel/tracing` 里 migrate / iommu / kprobe 都能开；这些
都不碰目标的 mmap_lock，进程到 60 s 自己 trap 退出。`pgrep -f` / `ps` / `cmdline` /
`numa_maps` 会要 mmap_lock，对着卡死的进程一样挂。

**有 state 的模型，截断的回答再问一轮不命中。** 快照只在自己的整长上可用，命中要求
chat template 重新渲染的历史 re-tokenize 出当初生成的那串 id；`max_tokens` 把回答截在词
中间，重编码就差一个 token，resident 和 host 都不命中，日志里只看得到 `prefix_hit=0`。
规则：测命中的脚本要断言第一轮 `finish_reason == "stop"`；生产上想让截断的回答也命中，
得让前端保留生成的 token id 而不是拼文本（router 那条老规矩）。

**先算再抓。** barrier.cuh 打印的 counter 是它自己加过之后的值，每次 MegaMoE 三个
barrier，prefill 的 MoE 有自己的 workspace，所以 `counter=883` 和 `403` 都换算成"prefill
chunk 第 15 个 MoE 的第一个 barrier"，缺席的 rank 一定卡在 layer 14 这一段（engram
lookup 读主机表）。两个数字就把范围从 673 个 call 缩到 40 个，比再跑十轮统计有用。

## 2026-09-10，DSv4.1 window ring（per-seq state 第一次走 TMA）

**铺满维要按预留算，不按装载时做出的对象算。** window cache 改成每序列一个
`bytes_per_seq` ring 后，tray05 上 57 个 checkpoint 拿满了 slot，remap 拆页补出
第 20 个 slot，落在那个 slot 上的 10k prompt 整段输出全是"的"。tensormap 的
铺满维是装载时按 state 的 `bytes`（初始布局：19 个 slot）编的，新 slot 在图外，
TMA 越界读回零。之前的 per-seq state（compressor 4 KB）都是普通指针读，从没走过
TMA；per-token state 装载时就把预算填满，remap 只加 slot 不加页，所以没暴露。
规则：state 指针的可寻址长度是整段预留（`DeviceBuf::span`），不是当前做出的
对象数；同一条规则也适用于任何"整块"语义。同一次还发现预算按 Σbytes 取整一次，
43 个 ring state 各自的 arena 一共差 33 块，装载直接拒绝——预算按 state 逐个取整。
门禁：hits.py 的多轮 / 并发场景把 slot 用满触发 remap，再跑 10k prompt。

## 2026-09-08，权重改绑 checkpoint（`bind` + `load` once program）

**A/B 共用的 bug，`kern test` 看不见。** rope 表改成设备上由 once program 算，核用
`blockIdx.y` 取行而 grid 把行放在 x 上，表只有第 0 行对；`kern test qwen3-4b` 报
PASS——A、B 两份 manifest 读的是同一张错表，逐位一致。`kern run` 出乱码才露馅，
定位靠 `program_io --dump` 把新路径的 buffer 与旧导出逐字节比。规则：换一条生产
同一份数据的新路径（导出 → 设备上算、拷贝 → bind），门禁是与旧工件逐字节比，
`kern test` 只证明 A/B 一致，不证明它们对。

**钉住的 sha 只有一台机器能复现。** master 的手写核 sha 是 kernel-lab 容器的
nvcc 13.0 编的，host 的 13.1 / 13.4 出来的 cubin 字节不同，改一个核就要把全套
手写核重钉。规则：手写核在 kernel-lab 里编（`tools/build_kernels.sh`），换 nvcc
就是换核，换核就是一次门禁。

## 2026-09-03，K5 span（FlashKDA 接入、k3_golden 多行 span、runtime 容量）

**capture 里的分配字母不是参数名。** FlashKDA recurrence 的 16 个参数按模板签名猜成
q/k/v…，kern 的 out 全零、probe 却对；把 probe 里每个 buffer 的地址打出来对 capture 的
参数表，才看到 p0 是 v、p10 是 out 的 TMA **store** 描述符、p11 才是 out 指针。规则：
vendored 核的 ABI 用 `kernel-capture lift.py --names`（地址→名字表）把参数直接落成
名字，再拿 `program_io` 喂 probe 的 dump 与 probe 输出逐位比对，不看模板。

**表在 run 之前重铺一次就把 mixed 行毁了。** `k3_golden --span` 第一版在每步 run 之前
把 line table 按"每行一 cell"的老布局再铺一遍，行 0 的 span 于是读到行 1 的 slot，
行 1..3 从第 1 步起就离开 fixture；改成表每步只按本步的 cell 布局铺一次、run 之后
不动。多行 staging 的规则：一步之内"铺表 → run → 读输出"三件事之间不许有第二次铺。

**运行时不许拿 manifest 的 var 上界当活跃序列数。** 池预算按 `seqs.max + 2` 个 slot
预留，93 层 K3 的 slot 449 MB × 258 = 115 GB，EP4 每卡 190 GB 权重之后 `cuMemCreate`
在第 39 878 块失败。序列数是调用方知道的（它的 batch、它的 `--max-seqs`），
`Runtime::load` 现在收 `Capacity { tokens, seqs }`；var 上界只说一步能寻址多少行
（见 serve.md 的"manifest 里不许出现 slot 数"——同一条规则的另一半）。

**一个线程驱多张互相等待的卡，launch 与 wait 必须分开。** kern-serve 的 tray 逐 rank 调
`Runtime::run`（launch + sync），K3 EP4 第一步 rank 0 的 MoE dispatch 在核里等 rank 1
——它还没被 launch——55 s 后核超时、`CUDA_ERROR_LAUNCH_FAILED`。qwen 单卡的 smoke 与
k3_golden（每 rank 一个线程）都看不到。规则：runtime 的 `run` = `enqueue` + `synchronize`，
锁步驱动的调用方先把每个 rank 都 enqueue 再挨个 synchronize；凡是核里有跨卡等待的
manifest，单线程多卡的门禁必须跑过一次。

**var 是一步一份、批是一 rank 一份：批不同的 rank 也得把 var 说的东西摆出来。** EP4 t=1
下每个 rank 各有自己的行，但 `span` / `span_at` 全 tray 同值；span 在 rank 1 时 rank 0
的第 0 行（别的序列的 decode 行）被 K2/K3 当 span 跳过、又被 K9/FlashKDA 当 span 拿去更新
那条序列的 state——输出是"只看见 span 那几个 token"的续写。k3_golden 看不到：它每个 rank
跑同一份行。规则：没有 span cell 的组在块前面垫 c 行 pad（`Layout.lead`），让 `span_at`
指向 pad；门禁用**多条相同 prompt 并发**——锁步的相同序列必须逐字相同，不同即串扰
（比 fixture 便宜且直接）。

**"相同 prompt 不同答案"先读 margin，再找串扰。** lead pad 修好之后 8 条"The capital of France
is"还是 3 条 " Paris."、5 条 " the capital of France is…"，又追了几小时的核（K2/K3 harness 加
`--span`、K9 行界、pad 行）——最后 `k3_golden` 用 5 个 token 的 fixture 复现（span 5 单行错、
span 5 + 一行对），把两种形状每层的 KDA state / KV / hidden / logits 倒出来比：全部只差 bf16
噪声，top-2 差 0.2 logit。自然语言 prompt 也有近平局（E5 那条"随机 token 必翻"是它的特例），
一条通顺但走偏的续写正是近平局的样子；真串扰是 8 条 8 种、互不成句。规则：**相等门禁的分歧，
第一步是倒两边的 logits 看 top-2 差**（`K3_GOLDEN_DUMP` + `K3_GOLDEN_DUMP_BUFS=logit_partial`），
差在一个 logit 内就换 prompt 或换成有 margin 的 fixture，不读核。

## 2026-09-04，K5 span 移植到 v4 master 与 t=4 门禁

**行数一变，输出就变；相等门禁要连 bucket 规则一起搬。** 移植时把 `step()` 的 bucket 写成
`bucket(k).min(max_seqs)`，256 行的 run 被截成 16 行——93 层输出全是乱码加 `position past the
lease` panic，这个好抓。改成 `.min(max_seqs.max(k))` 之后一切"正常"：P2 逐 token 门禁过、E3
文本通顺，只是 conc1 长 prompt 的 sha 全部对不上 K5 线。这个写法对 run 给的是 k 本身而不是阶梯
值：2k 的尾块 223 行不再垫到 256、12.9k 的尾块 160 不垫到 192，cuBLAS 按 m 选核，近平局一翻
就是另一条续写。定位靠的是 `k3_golden` 在 v4 线重跑 93 层 12.9k oracle（115/128、13 步逐 token
与 K5 同）把 runtime 排除，再对两条线 `stage` 的行数。规则：**同一条输入的 sha 是门禁时，
每一步的 m（bucket）也是契约的一部分**，改行数规则先跑 conc1 的 sha 对照；行数规则写成
可单测的纯函数（`rows_per_rank`），不写成 closure 里的算术。

**make_room 之后旧的命中不能再用。** `admit` 拿到 prefix 命中后 lease 报 `Busy`，`make_room`
把最冷的 resident 快照 park 掉——只有一条时正是那个命中——重试还拿着旧的 `Resident` 去取，
`expect("hit")` panic，之后每个请求 500。master 上就有，4 层 fixture 一步一请求、`--capacity 2048`
才撞上。规则：凡是"腾地方"之后的重试，重新查一次；`expect` 只留给类型已经排除的状态。

## 2026-09-03，v4 投机轮

**"旧路径与 plain 逐字同、新路径不同"不等于新路径有 bug。** dspark 的 verify 从 8 行改
7 行后，96 token 的散文在第 19 个 token 处分叉；先用 `kern run --probe-dir` dump plain 在
该位的 logits：top-1/top-2 差 0.125（bf16 一个 ulp）。cuBLAS 按 m 选核，m=8 与 m=7 各翻
一边，旧路径与 plain 的"逐字同"本来就是运气。规则：**分叉先读 margin 再定性**
（CLAUDE.md 门禁那句），margin ≤ 几个 ulp 就记进 docs 当核噪声，不追。

**压测 prompt 的散文要够长、够多样，报接受率要看窗口。** 512 token 的中文 docs 段落有
几条掉进重复循环（"用 `--show-raw` 显示原始数据。"×N），那条的接受率 71%；wall-clock 的
tok/s 按 `usage.completion_tokens` 算，不按 `max_tokens` 算。规则：
**报数字注明 prompt 集与窗口（running 多少）**，接受率异常高先看输出是不是循环。

## 2026-09-03，M1 MLA decode 核换成 CuTe DSL 预编译核

**LD_PRELOAD 钩 `cuLaunchKernelEx` / `cuTensorMapEncodeTiled` 抓不到 CuTe DSL 的启动。**
DSL 的 JIT host 代码调 `libcute_dsl_runtime.so` 自己导出的 `_cudaLaunchKernelEx` /
`_cuTensorMapEncodeTiled`，里面静态链接 cudart，驱动函数走 export table，没有一次按名字的
符号解析。规则：**抓一个闭源/JIT 栈的 ABI，先 `nm -D` 看它自己导出了哪层包装，钩最靠近
调用方的那层**；参数字节用 `cuFuncGetParamInfo` 按核的真实布局切，不猜。

**描述符字节对不上不等于参数错。** DSL 自己 encode 的 CUtensorMap 比驱动 encode 多两个 bit、
box 字段错一字节，穷举 dtype/swizzle/L2/OOB 也复现不出。不追字节：用驱动 encode 的描述符
独立起同一 cubin，输出逐位一致就是证明。规则：**反解 ABI 的验收是"独立 launcher 复现输出"，
不是字节相等。**

**split-KV 切多了反而慢。** B=16 × 13k：每行 4 个 split 55 µs，6 个 67 µs，32 个 87 µs——
cluster 数一超过一波（nsm/2）就排第二波，归约还按 (行 × split) 读 256 KB。规则：**split 规划
按"一波 cluster"摊 tile，B=1 长上下文才要 >32 的 split**；工作区大小跟 split 上限走，别默认开大。

**生成器的常量折叠只认顶层 `{"param": i}`。** `pack` 字段和 `tensormap` 里的 `param`
不改写，折掉一个标量参之后引用全错位，verifier 报"param #12 out of range"才发现。规则：
**manifest 里凡是能引用接口参的位置，normalize 的每个 pass 都得覆盖**（`fold_constants` 现在
遍历 pack 字段与 tensormap）。

## 2026-09-03，E5 tray 级 checkpoint 的 smoke

**随机 token 的 prompt 不能做 warm == cold 的相等测试。** 15k 个均匀随机 token id 的 prompt，
prefix 命中后 8 token 的短 prefill 与 30 块整 prefill 给出不同的首 token（' 1000…' 对
'Okay, I need…'），resident 命中和 host wake 一样，HEAD 一样，逐页 digest 证明 wake 回来的字节
与 park 时相等——花了两小时在 room / park / wake 里找一个不存在的 bug。随机上下文下每个
logit 都是近似平局，两条数值路径（chunk 形状不同）必翻。规则：**相等测试用自然语言 prompt
（docs/*.md 拼起来就够），随机 id 只用来占页**；见到 warm ≠ cold 先换 prompt 再读代码。

**字节问题用字节证明。** 这次真正切开问题的是给每页算 digest 对比 park 与 wake，五分钟；
读代码找了两小时。规则：**数据搬运路径上的疑点，先在两端 hash，再谈逻辑。**

## 2026-09-03，E5 tray 内 TP

**`asm volatile` 上的 `"memory"` clobber 把整条链路串行化。** LL collective 的 16 B
槽用内联 PTX 收发，加了 memory clobber 之后每个 pack 都是 load → 3 store → load，
B=16 的 allreduce 只有链路的三分之一。去掉 clobber（flag 随数据走，顺序由 volatile
访问自身保证）、一次轮询所有源，25 µs。规则：**collective 先测只发不收**，发的时间
就是链路上限，其余都是回读协议的代价，别在错的一半上优化。

**弱 load 看不到 peer 的写。** `ld.global.cg` 轮询 peer 写的槽永远"没到"，超时；
`ld.volatile`（sys scope）才是内存模型承诺的路径。gpu-scope 在 GB300 上也能看到且
更快，但没有承诺，不用。

**tp 组的每个 rank 必须跑一模一样的 launch 序列。** k3_golden 的参考跑（每个 distinct
feed 的复制批、stray 的 from-scratch 批）和 fork 都要在组里每个 rank 上做同样的次数，
不然 collective 在某个 rank 上等一个不会来的 epoch，超时报 err。规则：harness 里
"这个 rank 要不要跑"永远由组共有的量决定（feeds 数、fork 步），不看 rank 号。

**每卡持有 tray 批每一行的 state，slot 数就得跟行界走。** KDA 按头切后一张卡的 state
是 4B 行 × 24 头，pool 按 `seqs` 界开 slot，四卡各租 4B 行时第一步就 `lease denied:
remapping`。`Manifest::seq_slots` 改为跟 `rows` 界（没有 `rows` 时才是 `seqs`）。
规则：slot 数的来源是 line table 的宽度界，不是 batch 的序列数。

## 2026-09-03，K2 fork + K3 host 层

**拷贝流不能让计算流等。** 第一版 wake 让 compute stream 的下一次 launch 等 transfer
stream 上的拷贝，结果每次 6 GB 唤醒卡 decode 30 ms，8 路 decode 的 token 间隔 p50 从
3 ms 变成 20 ms。原因是"等事件"把两条流又串成了一条。改成 runtime 攥着 park 中的
checkpoint 直到落地才还页、wake 只给 `Waking`、`awake` 落地了才交 `Lease`：没有
`Lease` 就没有程序能读到路上的页，不用任何 stream 等待。规则：**异步资源的"可用"
用类型表达，不用事件等待表达**——一个句柄能被拿到，就意味着它背后的检查已经做完
（CLAUDE.md 里"a type exists because something was checked"的又一个实例）。

**pinned 内存要落在卡所在的 NUMA 节点。** 同一个 6 GB 唤醒在 GPU 0 上 31 ms、在
GPU 2 上 52 ms，一度以为是卡的差异；用 `numactl --membind` 一试，GPU 2 也是 31 ms。
一个 GB300 tray 两颗 Grace 各挂两张卡，`cuMemHostAlloc` 跟着调用线程的内存策略走，
跨到另一颗 CPU 的 C2C 只有 115 GiB/s，本地 180 / 197。规则：**runtime 自己绑**
（sysfs `numa_node` + `set_mempolicy` 包住分配），不指望调用方记得 numactl；任何
host 侧带宽数字都要写明块在哪个节点上。

**参照 batch 要和被测 batch 走同样的 bucket。** fork 门禁第一轮"分叉后各行都分歧"，
查了半天 fork 的页拷贝和 slot 拷贝，其实是 batch 从 4 行长到 6 行、bucket 从 4 变 8，
随机行的近平局翻了；同样 5 行不 fork 的对照只错 1 步（3 ULP）。改成参照 batch 在同一
步同样分叉，全部一致。规则：**行与行的对照只在同一 bucket 内有意义**，任何会改变
batch 大小的动作（fork、准入、结束）都要在参照里原样重演；比对失败先问"两边跑的是
不是同一组 kernel"，再问数值。

**基线要把变量隔开，先量再猜。** decode 抖动门禁一开始 p90 100 ms，第一反应是拷贝
在抢带宽；给 admitted 日志加了 lookup / lease / prefill 三段计时才看到：lookup 0.2 ms、
lease 0.3 ms、prefill 40 ms——命中之后剩下几个 token 的 prefill 尾块在 25k 上下文上
就要 40 ms（挖来的 prefill attention 只按 head 并行），而且 resident 命中一样慢，跟
host 层无关；wake 只多 15 ms，还全落在被唤醒的请求自己身上。规则：**门禁的对照组
必须是"同样的负载、只差被测的那一件事"**（这里是 resident 命中 vs parked 命中），
churn 请求用 token id 发，把前端 tokenizer 摘出去；先加计时再改设计。

**结论从数据里读，别从叙事里读。** 第一版 jitter 报告把 tokenization、prefill 尾块、
拷贝三件事混在一个 p90 里，看起来 K3 门禁差得远；分开之后 K3 本身只贡献 +1.7%，
另外两件是既有问题（K5 的事）。写 serve.md 时把三件事分行写清，谁的账归谁。

**`std::mem::zeroed()` 不能用在含 Rust enum 字段的 FFI 结构上。** cudarc 的
`CUDA_MEMCPY2D_st` 的 `srcMemoryType` 是 `CUmemorytype` 枚举，0 不是合法值，
release 构建直接 panic（"attempted to zero-initialize type … which is invalid"）。
规则：FFI 结构逐字段写全，宁可啰嗦。

**kern-serve 曾是独立 workspace，runtime 改了要单独重建。** 修了 device.rs 之后只重建
了 kern-runtime 和 examples，kern-serve 还是旧的 runtime，门禁跑到一半 panic。规则：
**运行门禁前先看 binary 的时间戳**。2026-09-14 起 kern-serve 是同一个 workspace 的
成员，`cargo build -p kern-run -p kern-serve` 一条命令、一个 target 目录，这条坑不再有；
留着是因为"两个 binary 一起跑门禁先对时间戳"仍然成立。

**四层 K3 的 fixture 不在这个仓库。** `k3_golden` 默认的 `tests/fixtures/k3_4l_greedy.json`
在 pegainfer-k3 里；tray08 的 K3 权重只有 EP8 shard，EP4 在 tray07。规则：门禁命令
连同它依赖的文件位置一起写进 roadmap 的门禁列，别只写结果。

**醒来的 session 再睡会再拷一份。** host 上的页链按 device 页节点去重，wake 出来的是
新页，所以同一 session 第二次 park 认不出上一次的副本，旧的那份只是变成最冷的先走。
没改，记在 serve.md 里；等真有 session 反复睡醒的负载数据再决定要不要让 wake 带着
host 节点的身份。（2026-09-14 起 wake 出来的节点带 twin，2026-09-15 起醒进 lease 的整页
节点也带，闭合。）

**没能复现的事也要记。** kern-serve 一致性门禁的某一次运行里一个 12k 的 filler 请求
花了 15 s（stats 里 prefill_tok_s 只有 1651），之后 4 次重跑都是 0.7 s。当时的日志是
INFO 级，看不到分段计时；以后门禁默认开 `kern_serve=debug`。

## 2026-09-02 之前

散在各处：qwen38-bringup.md（复现 ATen 数值时编译器的 FMA 缩并也是"顺序"的一部分）、
serve.md（manifest 里不许出现 slot 数：conv kernel 的 `num_cache_lines` 字面量让
slot ≥ 130 静默丢 state）、runtime.md（`cuMemAddressReserve` 的对齐必须是 2 的幂；
whole-state 读写只在第一次 remap 之前有效）、multi-gpu.md（peer / fabric 映射的显存
禁止 multicast TMA，会把发起的卡卡死到只能重启）。
