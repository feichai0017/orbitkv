# kern test：一次 kernel 替换的证据

`kern test` 拿两份 manifest——A（参考，**默认正确**）和 B（候选）——
产出一份 test report：换掉的东西在哪、每个 span 数值上等不等价、随机输入
下是否一致、快了多少。老 program 就是 oracle；manifest 里**没有阈值**。

```bash
./target/release/kern test qwen3-4b --out test-report.json   # A/B/kernels/weights 来自 kern.toml 的 target
./target/release/kern test \                                 # 或全用 flag（没有 kern.toml 时）
  --reference examples/qwen3-4b.json --manifest examples/qwen3-4b-silu-mined.json \
  --kernels kernels --weights weights/Qwen3-4B --out test-report.json
                                  # --diff-only 只看静态 diff；--no-perf 跳过计时
                                  # --no-graph-step / --no-sweep 关掉 TPOT graph 计时 / prefill 扫描
```

`kern.toml`（`crates/kern-run/src/config.rs` 有完整说明）里一个 target 就是
`manifest`（B）+ `reference`（A，用户自己拷一份信得过的）+ `kernels` +
`weights`。`kern test` 一次只测一个 target：只有一个就不用点名，有多个必须
点名；target 没有 `reference` 直接报错——没有 A 就没有测试。要跑全部，
shell 里 for 一下。target 的名字用户随便起，kern 不解释。

报告走 stdout，一行一个事实：行首是段名（`diff` / `tap` / `local` /
`logits` / `noise` / `perf` / `sweep` / `roofline`），事实之间用
` · `，两边用 `a → b`，段耗时在行尾；**最后一行以 verdict 开头**（`PASS` /
`FAIL` / `INCONCLUSIVE`，同退出码 0 / 1 / 2），`tail -1 | cut -d' ' -f1`
就是结论。相同的只给计数（`900/900 bit-identical`），不同的才点名，最坏的
在前、最多 8 条，其余进 `--out`。`--json` 改成在末尾打一个 JSON 对象（同样
的段、同样的字段，`jq .verdict.pass`、`jq .perf.steps[0].graph_ms` 直接
取）。`--out` 写存档：`summary` 就是那个对象，`detail` 是每一条有差异的
比较（span、buffer、差异元素数、ulp）——相同的不存，PASS 的存档只有几 KB。
stderr 默认安静（`RUST_LOG=debug` 看 runtime 装载）。

`examples/qwen3-4b-silu-mined.json` 是自带的 A/B fixture：和
`qwen3-4b.json` 唯一的区别是 `silu_mul` 的 impl 从 HF hub 的
`kernels-community/activation` 包换回挖矿得到的 vLLM cubin，接口与全部
call 一字不动——纯 impl 替换。两份 manifest 共用一个 `--kernels`
目录：module 按 sha256 解析，目录里放着两边各自钉的版本即可（`tools/
extract_kernels.sh` 对 A、B 各跑一次，只增不减）。`--capacity` 会向下
对齐到 manifest 的页单位。

## 设计：先录 A，再放 B，之后全是 span 级

流水线里跑完整 program 的只有两趟：**record**（A 跑一遍 seeded
workload，每个 program run 起点的 state 镜像、每个 span 的 frontier 输入
与参考输出都留在设备上）和 **replay**（B 从 A 的镜像起跑同一 workload，
每个 span 先写进 A 的输入再跑，之后再从零 state 自由跑一遍给端到端
logits）。噪声地板、计时都在各自那一侧按 span 重放：把 frontier
输入写回去、`run_range` 跑那几个 call、读写出的 buffer。成本随 span 大小
走，不随模型走。

**A、B 永不同时在卡上。** 录完 A 就卸掉，再装 B：把卡装满的模型（DSv4.1
Flash EP4 每 rank 119 GB 权重）也能测，代价是装载两次。录下来的东西
（`Side::Buf`）是设备侧句柄，活得比分配它的那一侧长：两次装载共用 primary
context，B 直接 D2D 读 A 的镜像；只有要下判断的字节走 host。A 的噪声地板
在 A 还在时测（每个保留的 span 重放对自己）；A 的计时同理。

**快照存的是"这次和上次的差"，不是 buffer 有多大。** 一个 buffer 第一次
被快照时整块留（活跃前缀），之后每次只留与上一次不同的 64 B 块——差异在
设备上对着一份影子副本用 `changed` 找，块间隙 < 4 KB 的并起来少切几段；
内容没变就是同一个快照，不占字节；差异过半就重新整块留，链不会比写它的
次数长。装回去沿链走到整块再逐段叠。这样 DSv4.1 Mega kernel 那块 1 GB
的 `moe.e384.slab`（carry，活跃前缀恒为整块）每层只花它真写的几百 KB，
整套 sm148 换核的快照从 360 GB 变成几 GB；而 span 对每个 buffer 的
**写集**——它前后两个快照的差——恰好是这个 span 自己写的字节，比较只在
这上面做，另计 B 在写集之外写了多少（`wrote N outside A's write-set`，与
state 同一规则）。整块比会把 kernel 工作区里的陈旧内容当成差异：gate 那
次 noise 报 A 自己在 slab 上 60 万个元素不确定，其实 gate 只写了几 KB。

**端到端只看 logits，不生成。** 每个 span bit 相同 ⇒ 整体必相同，不需要
再证；span 有差异时，oracle 是 B 自由跑同一 workload 后每一步的 `logits`
（manifest 里的 buffer，读出来就行）：量的是分布不是存储——每行算
KL(A‖B)，argmax 翻转按这一行的 KL 分成"限内的平局"与真翻转。裁不了的（没有
logits 的 program、A 自己不确定）报 INCONCLUSIVE（退出码 2）。

## 多 rank：一侧可以是几张卡

带 `topology` 的 SPMD manifest（`groups.ep = 4`）一侧就是 4 个 rank：
`kern test --gpu 0,1,2,3`（或 `--gpu 0`，从它起连续取），每个 rank 各自
一个 runtime、各自的 `{ep}` 权重分片，`export_handles / import_peers` 连
上 peer buffer，`once` program 跑过，之后每个 rank 是一个 `Caller`，**全部
喂同一条序列**（EP 下每个 rank 看到的是同一批 token，这正是 serve 里的
情形）。`Side` 的方法里，动字节的都点名 rank（`read(q, ..)`、
`load_state(q, ..)`），跑东西的（`run / time / capture`）一律全 rank 同时
发——span 里的集合通信要等 peer 一起发射才会返回——由 kern-run 里的
`Ranks` 一 rank 一线程扇出，最慢的 rank 回来才算回来；哪个 rank 报错就
点它的名，600 s 没回来的 rank 视为挂死（peer 死了它在等），进程退出 2。
计时取最慢 rank 的每 call 时间：这是一步真正等的。

比较是 **rank-local** 的：B 的 rank q 只对 A 的 rank q，镜像、frontier
输入、参考输出、write-set 都按 rank 录，报告里的行带 `rank q` 前缀
（`rank 1 step 0 A[3..4) B[3..4) act: 12/12 differ`），端到端的 output /
state 也逐 rank 列。A、B 的 rank 数必须相同，否则 replay 直接报错。

静态 diff 里 **peer buffer 算成它 `of` 的那个 buffer / state**：拿到全组
地址的 kernel 读写的是每个 rank 上那份目标（含自己的），所以目标进 span
的 frontier 读与写集合，地址数组本身（装载期常量）不进。B 若不再用
peer（比如换成 fused 核），目标就成了单侧写，报告点名不比较。

## 结构：kern-test 是独立 crate

harness 在 `crates/kern-test`，只依赖 kern-manifest，不碰 CUDA，在
`default-members` 里，测试在 host 上跑。它对两个 `Side` 说话——一个
trait，方法是"跑这段 call、读这个 buffer、把这块 state 存进你那边的一个
句柄"——`kern` binary（kern-run）用 `Runtime` 实现真的那一侧，并保留
`kern test` 子命令；`crates/kern-test/tests/common` 里有一个 host 上的
假 Side：一个 2 层 × 4 维 × 16 词表的解释器族，`embed / scale / mix /
head` 四个 op，每个 entry 名对应一种行为（`scale_same` 无操作替换、
`scale_negzero` 只差符号零、`scale_round` 差一次舍入、`scale_drift` 漂
5%、`scale_nan` 写 NaN、`scale_noisy` 同一输入回来一次翻一次符号、
`scale_crash` 遇到网格外的输入就崩、`head_swap` 翻 argmax、`head_bad`
越域、`mix_leak` 写到 write-set 外……）。`tests/harness.rs` 每条判定档
一个 fixture，`tests/diff.rs / compare.rs / workload.rs` 是对齐、比较、
扰动、抽样的性质测试（小字母表上穷举、每个 dtype 的关键值两两配对、多
seed）。这些不需要 GPU，CI 跑。

**数据不下设备。** 早先 tap 每个 program run 前把 B 的 state 整体
D2H 再 H2D（qwen3-4b 604 MB × 每 run），大模型上 tap 以秒计。现在
`Side` 里的 `Buf` 是不透明的设备侧暂存句柄：state 同步、run 前的镜像、
快照的 frontier 输入都是 D2D（`Runtime::scratch / save_state /
load_state / save_buffer / load_buffer`）；两个 runtime 共用设备的
primary context，指针互相可见。**比较也在设备上做**：`Side` 有三个比较
原语——`compare`（两段字节按 dtype 逐元素比，回 5 个计数）、`changed`
（两段字节按 64 B 块出差异 bitmap，回区间；state 的 write-set 就这么找，
不再整块读回 host 逐字节 diff）、`logits`（每行 argmax / margin / KL /
top-20 重合 / A 的 token 在 B 里的名次，一 block 一行）——kern-runtime 里
是三个 kernel（`compare.cu`，PTX 随源码入库、driver JIT，和 `profile.ptx`
一个做法）。A 的参考输出、logits 行全留在设备侧 scratch，
CPU 只拿事实：计数、ulp、区间、KL。host 的 `compare` / `changed_blocks` /
`logit_stats` 仍是定义，假 Side 用它们，kernel 对它们做性质测试
（`cargo test -p kern-run --test device_compare -- --ignored`，要一张 GPU；
每种 dtype 随机字节、每种操作数、非 64 整倍数长度、7 到 129280 列的行）。
真正过 host 的字节只剩：write-set 里的几 KB、有 domain 的整数输出、e2e
的 output buffer。

## 四段

1. **DIFF（静态）**：逐 kernel 比接口（`params`）和实现（`impl`），分
   interface / impl / added / removed；逐 program 用 LCS 对齐两边的
   call 列表（变了的 op 两边永不对齐），切成 Same / Changed 段，
   Changed 段就是 **span**（两侧各一个 call 区间 `A[lo..hi) B[lo..hi)`，
   span 之间的 gap 一一对应、两边等长）。每个 span 由数据流图给出 frontier：读了哪些
   外部 buffer、写了哪些——`a+b → c` 的中间 buffer 不在 frontier 上，自动
   略过。两边 frontier 不等会标 ⚠（表示不是 span 内替换）。
2. **TAP（seeded workload）**：workload 完全由 `(seed, manifest, 选项)`
   决定，谁跑都一样、也不依赖 A 的数值：token 随机（vocab 从 `token_ids`
   的 `index_into` 域来），prefill 长度一半概率在 `[1, capacity − steps]`
   均匀抽、一半概率抽**结构边界**（一页 ±1、一个 chunk ±1、`tokens` max、
   上限——kernel 出错的地方），chunk 从 {`tokens` max, 512, 一页, 随机}
   抽，decode 步数在 `[steps/2, steps]` 抽（默认 32），decode 喂的 token
   也随机（不是 A 的 argmax，否则换个参考序列就变）。`--prompt` 可以换成
   真文本 prefill，`--prefill/--chunk/--decode-steps` 可以钉死。默认 seed
   固定是为了两次运行可比；覆盖靠换 seed（`--seed`），抓到问题的 seed
   写进报告钉成回归。
   **record**：A 一个 run 一个 run 地跑，每个 run 之前把共有 state 全量
   D2D 存成镜像，每个 span 之前把 frontier 输入和它要写的 buffer（按当前
   var 值取活跃前缀）快照下来，跑完再快照写出的 buffer：前后两个快照的差
   就是写集（快照是差分链，见上）。weight 和 `once` program 写出的 buffer
   （打包好的权重、rope 表：装载期常量，B 有自己的一份，可能是别的布局）
   不算 frontier 输入——DSv4.1 每个 attention span 都读 268 MB 的 rope
   表，120 个 span × 9 个 run 会把卡塞满；**replay**：B 每个 run 先装 A 的镜像，
   每个 span 先装 A 的输入（和它要写的 buffer 在 A 那里跑前的样子）再
   跑，在 A 的写集上比 B 写出的 buffer、另计 B 写到写集外的字节。所以每一行 span 结果
   都是 **span-local**：B 拿 A 的输入、A 的 state 跑这一刀，差多少就是这
   一刀自己的事，不混前面层漂移下来的误差（早先不注入时，c7 那种 state
   差会让下游每个 buffer 都显示 47/48 spans differ、几十万 ulp，看不出哪
   一刀是根）。
   replay 结束后 **B 从零 state 自由跑一遍同样的 workload**，什么都不
   注入——表末的 `end-to-end` 行（output 类 buffer + 每个 state 全量字节
   差）就是调用方真正会拿到的东西。**state 按 span 的 write-set 比**：A 跑前后各读一次 state，差异的字节区间就是这个 span 的
   write-set（pre-image）；先把 A 的 pre-image 写进 B 再跑 B，然后只在
   write-set 上比 A、B 的 post-image，另报 B 在 write-set 之外写了多少
   字节。整个 state 不能拿来比——其余字节是别的层的历史，B 的历史又是
   B 自己的。第一个 prefill chunk 和 decode step 的每个 span 存 write-set
   （前后像），后面的段全靠它。最后比 output 类 buffer。
2b. **LOGITS（端到端判据）**：record 里每个 run 之后读 A 的 `logits*`
   buffer（manifest 里本来就有，`logits`、spec 的 `logits_blk`），自由跑
   里读 B 的，逐 run（`logits_blk` 逐行）比：KL(A‖B)、argmax 是否一致、
   A 的 top-1 − top-2 margin、A 的 argmax 在 B 里排第几、A 的 top-20 有
   几个还在 B 的 top-20 里。**尺度是 KL，与存储 dtype 无关**——早先按
   "logits 自身尺度的 ulp"（Δ / ulp(max |logit|)）算，在 bf16 logits 上
   还说得通（c9 的求和序改写 Δ 0.156 = 1.25 ulp），到 DSv4.1 的 f32 logits
   上一次换核动 2.6 就是 1.7e7 ulp，阈值失去意义；而 logits 存成什么只是
   head 核的输出格式，流水线本身是 fp8 / bf16 的。KL 直接量"概率质量挪了
   多少"，对整体平移不敏感。翻转不单独设 margin 规则：一行 KL 很小却翻了
   argmax，只能是两个近乎并列的 token 换位（0.55/0.45 互换就要 KL 5e-3），
   算 **限内的平局**；KL 超过 `--logit-kl` 的翻转才是 **FAIL**（第一个真正
   的 FAIL 档）。早先"A 的 margin ≤ Δ 就算 near-tie"是拿 B 造成的偏差解释
   B 造成的翻转，B 错得越大越多翻转被算成平局，已删。top-20 重合只进报告
   不进判决：带 margin 排除它是 KL 的子集，不带就被尾部的并列刷屏；给人看
   漂移发生在头部还是尾部很直观。noise 里另跑一遍 A 自己的 workload，A 对
   A 的 KL / 翻转数是任何端到端结论所在的带子（fp8 参考核常常自己就不
   确定）。
3. **NOISE FLOOR**：每个快照写回 A，重跑 A 自己的 span，和参考输出比。
   带 inout state 的 span 不幂等（重放一次 conv 窗口再移一位、SSM 再递推
   一步），所以**每次重放先把 pre-image 写回，跑完把 A 的 post-image 写
   回**——A、B 皆然。只有 write-set 还不够：span **读**的 slice 里有它没改
   的字节（step 0 恰好没变的 SSM 项），后面的 step 会改它们；所以每个被
   快照的 run 留一份 run 之前的全量 state 镜像（prefill chunk 0 = 全零，
   decode step 0 = A prefill 之后的 state，各一份），重放序列切换 run 时
   两边整体写回——各层 write-set 不相交，这份镜像就是该 run 每一刀的
   pre-state。这样参考自己
   逐位可复现，band 为零；仍不 clean 才是 A 真的不确定（atomics 之类），
   此时 B 按这条带子判（`--no-noise` 跳过）。早先没有复位时 A 自比差
   上百 MB，band 无限宽，B 在 state 上的真错误全躲在带内。
4. **PERF**：每个变了的 program 整步 eager 跑 N 次，逐 call event 计时
   取最小——同一份数据既给**整步**（Σ 全部）又给 **Σ spans**（换掉的那
   块）。表里 `B measured` 旁边就是 **`B derived`** = `A − Σspan_A +
   Σspan_B`（只看 span 就能推出的整步预估），实测与推导的差就是换 kernel
   带来的 launch 间隙 / L2 交互效应。decode 再两边各捕获 graph 跑 100 次
   取中位 = **TPOT**，同样给实测和推导两列（`--no-graph-step` 关）。prefill 扫 `tokens ∈ {1, 16, 128,
   512, 2048, 4096, max} ∩ [1, max]`（`--no-sweep` 只跑 tap 的 chunk 长），
   token id 按 `token_ids` 的 domain 随机、结构输入由 driver 填。对变了
   的 kernel 用 manifest 声明的读写 buffer 字节数算 roofline 下界——不需
   要任何模型知识（state 不透明，标为 "+ opaque state"）。
   整步计时不是 span 级的，但成本只是"program 跑 N 次"，和 tap 同量级。

**没有 fuzz。** 早先有第五段：每个快照的浮点 frontier 输入按六种分布扰动
（jitter / noise / scale / shuffle / resample / outliers），两边各跑一次比
写出的 buffer。2026-09-15 删了。它回答的问题没人问：manifest 不给浮点
domain（定过，不加），扰动后 B 出 NaN 而 A 没出，只能记进报告没法判；
shuffle / resample 之后 attention 的 q 与 KV 已不对应，比出来的差什么都
不说明；jitter 说的事 noise floor 已经说了。换核会出的 bug 各有别的段在
覆盖（布局、边界行：tap；形状：perf 的 token 扫描；确定性：noise；
时间：多步 workload），两个真机 case 和几十次 fixture 它从未改变过一次
判定，却占 DSv4.1 EP4 一次 `kern test` 的 5 分钟（A 侧单线程 CPU 逐元素
扰动 21 G 个 bf16）和几百行代码。要"B 在 workload 之外也像 A"这条证据，
换一个 `--seed` 再跑一遍：端到端、走 KL、每一行可解释。fuzz 唯一留下的
是它的后置条件：B 端到端写出的 output 若声明了 domain，每个元素必须落
在域内，违反即 FAIL。

## driver

manifest 不规定 program 叫什么；能把真实 workload 喂进去的是 **driver**
（`crates/kern-run/src/lib.rs` 的 `Caller`：知道 `token_ids` /
`positions` / `slot_mapping` 怎么填、prefill 按 chunk 推位置、`tokens`
是 prefill 的尺寸符号）。它是模型家族契约，`kern run` 与 `kern test`
共用，目前只有 qwen3 一份。kern test 遍历 manifest 的 programs；变了但
driver 不会 stage 的 program 在 TAP 里标红、判 INCONCLUSIVE。

## 判定

按顺序取第一条命中的：

- `FAIL`（退出码 1）：B 端到端写出的 output 越出声明域（后置条件）。
- `FAIL`：端到端某一步 B 换了 argmax，且这一行的 KL 超过 `--logit-kl`
  （不是限内的平局；B 一侧出 NaN 也在此，KL 无限）。A 对 A 的端到端
  自己就超限或翻转时不下这个结论（span 吵不算：`o_lowrank` 不确定的参考
  端到端照样可能复现）。
- `INCONCLUSIVE`（退出码 2）：变了的 program driver 喂不了（覆盖缺口，
  再好的 logits 也只说明被喂到的那些）。
- `PASS: bit-identical`：每个 span 逐 bit 相同。
- `PASS: value-identical`：只差 ±0 符号位（silu 类 kernel 常见）。
- `PASS: logits bit-identical`：span 有差，端到端每一步 logits 逐 bit 相同。
- `PASS: logit evidence`：端到端每一行 KL(A‖B) ≤ `--logit-kl`（默认
  0.01 nat），argmax 一致或只有限内的翻转（报告逐条列出 margin、KL、A 的
  token 在 B 里的名次）。合法的舍入序改写（c9 的 SSM 求和序：state 差
  280 KB、logits 差 1 ulp）进这一档。阈值按流水线定：bf16 的舍入序噪声
  在 1e-3 量级，fp8 attention 换核到 1e-1 也不奇怪，看 noise 里 A 对 A
  的那一行再定。
- `PASS: within noise floor`：span 差异不超过 A 自己重跑的差异。
- `INCONCLUSIVE`：其余——logits 动得超过阈值但没翻 argmax、或没有 logits
  可比。报告里有最坏一行的 KL、翻转数、top-20 重合、A 对 A 的带子，交给
  上层判断。

`--out` 写完整 JSON（每个 span 每个 buffer 的 n_diff / max ulp / max |Δ| /
nan / signed-zero、端到端每步 logits 的 KL / argmax / margin / top-20、noise
带、逐 kernel roofline、sweep 曲线）。

## fixture 实测（GB300，2026-08-31）

HF hub `activation` 包 → 挖矿 vLLM cubin（packed 变体）：tap 72 个 span
（prefill 36 + decode 36）全部 bit 相同、next_token 相同，快照 35.7 MB；
noise floor clean；fuzz 六种分布 bit 相同、`edge` 下只差 signed zero；
perf：decode eager 整步 4.231 → 4.167 ms，Σ36 spans 306 → 232 µs
（−24%），推导 4.157 vs 实测 4.167（差 10 µs），graph TPOT 2.606 →
2.549 ms（384 → 392 tok/s）；prefill eager 整步从 tokens=1 的 −1.9% 到
2048 的 −3.4%（27.4 → 26.5 ms），Σ spans −26% … −44%。roofline 列直观
展示 bs=1 下 58 KB 的 silu 只到峰值带宽的 0.1%——launch 主导，这正是
bs>1 才轮到 kernel 本身说话的原因。全程 7 s，其中 2.4 s 是装两个
runtime，PERF 1.8 s。

**2026-09-15（crate 拆分 + D2D）**：同一 fixture、同一 seed，报告除计时
字段与 cut → span 改名外逐字段相同：108 spans、local 900/900、logits
20/20 bit-identical、noise 108/108、fuzz 648/648、PASS 6.7 s；tap
1.3 s → 486 ms（state 同步不再经 host），noise 34.7 → 20.5 ms。

## DSv4.1 Flash EP4 实测（GB300 tray18，2026-09-15）

第一次多 rank 运行：A = `a-paged`（paged attention），B = `b-fused`（fused
decode + fp8 `attention_raw`），2026-09-10 的 A/B 对，4 rank × 1 GPU，
`--capacity 512 --prefill 64 --chunk 64 --decode-steps 8 --no-sweep`（存档
`~/bench_results/2026-09-15-kern-test-multirank/`）。20 个 op 换掉，三个
program 各 120–129 个 span，`load`（once）也有 span 但不算未驱动。

- 时间：装 A 34 s → record 364 s（workload 本身 3 s，其余是每个 span
  之后把整个 state 读回 host 找写集）→ 装 B 34 s → replay ~400 s（tap
  54 s、noise 13 s、fuzz 288 s、perf 3 s：都是 host 上逐元素比较），全程
  797 s。设备上留了 13.5 GB 的 span 快照和 84 GB 的 state 镜像（后者是
  整块分配的尺寸，见 roadmap）。
- local：0/2312 bit-identical——B 的 `q` 是 fused prep 重排过的布局，同名
  不同义；`attention_raw`（bf16 → fp8）、`attention_sf`、`o`、`q_rotated`
  按"声明不同 / 单侧写"不比。端到端 `next_token` 四个 rank 全部 bit 相同，
  `verify_tokens` 5/6 不同（draft 走岔了），`nacc` 差 1。
- logits：4036 行里 148 行有差，76 行 argmax 翻转、全是 near-tie（A 的
  margin < Δ）。target 行（prefill + decode_batch）只有 16 行有差，max
  |Δ| 5.6，翻的是 prefill 那一行（margin 0.19，Δ 2.6）；draft 行 60 行有
  差，max 4.1；verify 行的 Δ 到 16——它们跟在已经不同的 draft token 后面，
  算的不是同一个输入。"17162863 ulp at scale" 是 f32 logits 的 ulp，在
  fp8/bf16 流水线上没有意义——这次之后判据改成了 KL（见 2b）；按存档的
  行重算：分叉前的行 KL 0.09–0.33，step 5 verify 行 4–12，margin 2.69 的
  那次翻转 KL 4.4，新规则下不再算平局。
- noise：A 自己 492/984 个 span 不确定（`o_lowrank` 50% 元素、16k ulp）：
  paged 参考核本身不是确定性的，B 按这条带子判。
- perf：decode_batch eager 17.7 → 15.0 ms（−15.6%），graph TPOT 11.41 →
  10.57 ms（88 → 95 tok/s），120 span 5.75 → 3.07 ms（−46.6%）；round
  TPOT 12.97 → 11.93 ms（77 → 84 tok/s）；prefill 64 行 22.4 → 18.9 ms。
  fused attention 281 MB/call，37.6 µs = 7.5 TB/s（93% 峰值）。
- 判定 INCONCLUSIVE：logits 动得超过阈值但没有 wide flip，A 自己不确定。
  按 lessons 的规则看 margin：翻转都在 near-tie 上，decode 的 token 一致，
  是"合理一致"还是 fp8 attention 的精度代价，交给人判。

**同日复跑（比较下设备 + KL 判据，`results/ab2.log`）**：同一对 manifest、
同一 seed、同一 tray，比较改成设备 kernel 之后全程 797 → **402 s**：装 A
34 s、record 323 s（workload 1.3 s）、装 B 34 s、tap 54 s → **0.9 s**、
logits 0.2 s、noise 13 s → 0.55 s、fuzz 288 s → **7 s**、perf 3.3 s。数字
逐项与首跑一致（local 0/2312、noise 492/984、fuzz 1045/5904、perf
decode_batch TPOT 11.41 → 10.57 ms、round 12.97 → 11.93、prefill 22.5 →
19.0 ms），只有判定变了：A 对自己端到端 4036 行 max KL 0、0 次翻转
（noise 底线成立），B 在 3960/4036 行 argmax 一致、76 次翻转全部超过
`logit_kl = 0.01`（prefill chunk 0：A 271 → B 201，KL 0.127，margin 0.19，
A 的 token 在 B 里排第 2；step 1/3/5 的 verify 行 KL 4–12 是 draft 分叉后
的不同输入，见 roadmap 的 teacher forcing），所以 **FAIL**：A 自己能复现，
差异就是 B 的。0.01 nat 对 fp8 attention 是否过严，由跑的人按
`--logit-kl` 定，harness 只负责把 A 的底线和 B 的差摆在一行里。剩下的
323 s 归因：workload（含 7 次 run 和 84 GB 的 state 镜像 D2D）只占
1.3 s，`cuMemAllocAsync` 实测 1 MB 5 µs、3 GB 3 µs、几千个存活也不涨，
noise 0.55 s，perf 约 3 s——余下全是 fuzz 的 A 侧：每轮把 kept span
的输入读回 host，单线程逐元素解码、扰动（jitter / noise 每元素 20 ns）、
编码、写回，6 轮 21 G 个 bf16 约 330 s。fuzz 随后整段删除（见"四段"），
删后同 tray 第三跑（`results/ab3.log`）：**全程 74.5 s**——装 A 33.6 s、
record 5.3 s（workload 1.1 s）、装 B 31.3 s、tap 0.8 s、logits 0.2 s、
noise 0.55 s、perf 3.3 s；判定、计数、perf 与前两跑逐项相同。state 镜像
从 84 GB 变成 339.5 MB（12 MB/rank/run，即活跃部分）：master 的 pool 改
按 capacity 的 token 数分配 state，与 harness 无关。一次四 rank 的
DSv4.1 换核验证现在是两次装载加 10 秒。

**2026-09-15 master b325e82 之上，tray06 / tray09，`hbm-sm152`（1M context，
Engram 表在 HBM）对 B300 的 148-SM 实例**：整块快照时，换 `dsv41_mhc` /
`dsv41_mega_moe` 在 record 就 OOM——这份 manifest 权重 126 GiB、workspace/carry
另要 ~89 GiB，每卡剩 ~70 GiB，而这两个 op 的 span 把 1 GB 的 `moe.e384.slab`
（carry，活跃前缀恒为整块）整块当输入/输出快照，40 层 × 3 program 装不下；
只换 draft 的 `dsv41_gate_e128` 能跑（47 s），但整块比把 slab 里的陈旧工作区
当差异（local 4/72、noise "A 不确定" 60 万元素）。改成差分快照 + 写集比较后：

| B | 换的 op / span | 快照 | 装 A + B | record | replay（tap · noise） | 判定 | 全程 |
|---|---|---|---|---|---|---|---|
| gate（draft 的 `dsv41_gate_e128`） | 1 op / 3 span | 2.6 GB、3949 段 | 19.3 + 14.4 s | 4.4 s | 0.5 · 2.5 s | **PASS，72/72 bit 相同，noise 24/24 clean** | **42 s** |
| 全套 sm148（mhc ×4、mega_moe ×2、gate ×1） | 7 op / 262 span | 26.4 GB、162k 段 | 20.0 + 17.6 s | 40.3 s（workload 22.7 s） | 30.6 · 15.2 s | PASS：span 上 slab 写集 4M/12M 字节不同，A 对自己同样不同（Mega kernel 的工作区不确定），端到端 64516 行 logits bit 相同 | **113 s** |

装 B 的 14.4 / 17.6 s 里权重占 6–16 s/rank：A 和 B 在同一组卡上，换的是 kernel，
权重 buffer 的声明一样。现在 record 完把 A 的权重从 runtime 里拿出来留在卡上
（`Runtime::into_resident`），B 装载时按名字认领声明相同的 buffer
（`Runtime::load_over`，见 runtime.md「权重来源」），只有声明变了的才走文件：
两个 target 都是 126 GiB/rank 全部认领，装 B 2.8 s（gate 全程 42 → 33 s，判定不变：
72/72 bit 相同、noise 24/24 clean；tray06，2026-09-15，`results/*-resident.log`）。
装 A 那一次的权重再走 pread staging（runtime.md「权重来源」）：Ceph 上 5.8–16.3 s/rank
→ 4.4–5.4 s，装 A 15.7–20 s → 11.0 s，gate 全程 **26.5 s**（`results/*-pread16.log`）。
一次四 rank 的 DSv4.1 换核验证：装 A 11 s、record 5 s、装 B 3 s、replay 3 s。

同一份全套换核三跑的路：整块链 479 s（tap 302 s：链长几百、逐段同步拷）→
扁平 piece 181 s → `changed` 的 host 端按字跳零 113 s（1 GB 的 bitmap 逐块扫
是 20–40 ms 一次，三千多次就是 record 那 65 s）。剩下的 record 22.7 s 主要是
16 万段各一次同步 D2D（`save` 每段一次分配加拷贝）。gen 的 manifest 之前没给
`input_ids` 声明 vocab 域（workload 抽不到 token），`tools/dsv41/gen.py` 现在给
`input_ids` / `anchor_token` 加 `index_into embed.weight`。存档
`~/bench_results/2026-09-15-dsv41-e2e-swap/`。

## 位置

- 静态 diff、frontier、录制、重放、比较、报告全在 `crates/kern-test`
  （`diff.rs` / `compare.rs` / `workload.rs` / `report.rs` / `harness.rs`
  录 A、`replay.rs` 放 B 并判定，`lib.rs` 的 `Side` trait 是与设备之间唯
  一的边界）；`kern test` 子命令、`Side` 的真实现（`Ranks`：每 rank 一个
  `Caller`，`Buf = Scratch`）和 kern.toml 解析在
  `crates/kern-run/src/test.rs`（caller 契约在 `crates/kern-run/src/lib.rs`，
  和 `kern run` 共用）。
- runtime 只加了不在服务路径上的原语：`run_range`（按 call 区间
  eager 执行）、`read_buffer_prefix` / `write_buffer` / `read_state_at` /
  `write_state_at`、`save_* / load_*`（设备侧暂存的 D2D
  存取）、`compare` / `changed` / `logits`（`compare.rs` + `compare.cu`
  的三个比较 kernel，操作数 `At::{Buffer, State, Scratch}`）、`time_range`
  （区间内逐 call event 计时）、`time_captured`（graph 中位数）、
  `check_domain`。元素编解码在 `kern_manifest::values`
  （bf16/f16/f32/fp8e4m3/整数 ↔ f64，ulp 距离），kernel 里逐条复刻。
