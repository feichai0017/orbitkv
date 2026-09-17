# kern-pool：tokens → store

状态的记账层重设计。起因是一个 review 找到的 bug（半页 checkpoint 之后 host 层按
node id 复用旧副本），追下去发现它不是一处笔误，而是几个接口没把不变量放进类型里。
这份文档定下 kern-pool 的模型和接口（2026-09-14 一个 PR 落地，第 7 节是门禁结果）。
设计对照了三个开源实现：
NVIDIA Dynamo 的 KVBM（`lib/llm/src/block_manager`）与 kv_router 的 radix indexer
（`lib/kv-router/src/indexer`），以及 SGLang 的 RadixCache / HiRadixCache /
MambaRadixCache 和它的 Rust 内核（`rust/sglang-radix-tree`）。

## 1. 结论

- **索引是 tokens → store。** 调用方拿一串 token 来，得到一个持有前缀字节的 store；
  page 不出现在索引的契约里，也不出现在 kern-pool 的公开词汇里。
- **共享单位仍然是 page，节点自创建起冻结。** 这是三个参考实现的共同选择，也是
  kernel 按 page table 寻址（一页一个写者）决定的。索引按 token 键，字节按页共享，
  两者不冲突：索引条目持有一个 store，store 持有一条页链。
- **所有权，不是 id。** 每个句柄（`Lease`、`Store<T>`、`Hit`）持有它依用的东西，drop
  即归还；裸 `i32` / `u64` 不穿过任何模块边界。同 tier 与跨 tier 的"已经有一份"都
  由节点自己的 `Arc` / `Weak` 表达，不再有按 id 查的镜像表。
- **两个 tier 一套类型。** device 和 host 是 `Storage` 的两个实现（`Tier` 这个名字留给
  索引里的 `Resident` / `Parked` 枚举），`Store<Pool>` 是 `Checkpoint`，`Store<Host>` 是
  `Parked`，链、引用计数、拷贝计划都只写一次。
- **单线程。** 索引与池由 scheduler 线程独占，`&mut` 修改，没有并发；`Arc` 和一把从不
  争用的 `Mutex` 只为 `Send`（句柄可能在别的线程 drop）。

## 2. 名词

### Storage

```rust
pub trait Storage: sealed::Sealed + Send + Sync + 'static {
    type Page: Copy + Ord;                 // Pool: 页号 i32；Host: 字节偏移 u64
    type Slot: Copy + Ord;
    type Twin: Default + Send + Sync;      // 节点对另一 tier 副本的了解
    fn give_page(&self, page: Self::Page);
    fn give_slot(&self, slot: Self::Slot);
}
```

trait 只管归还：分配留在各自的类型里，因为 `Pool::take` 的三态拒绝（`Busy` /
`Remapping` / `ExceedsPool`）和 `Host` 的按字节 first-fit 没有共同的签名可抽。`Pool`
（chunk、remap、manifest 的表）和 `Host`（pinned 块，页从低端、slot 从高端）实现它，
trait 是 sealed 的，别处不能。页号和偏移是裸 `i32` / `u64`，但只出现在两处：交给
runtime 执行的 `Copies`，和只读的观测（`Lease::page_ids()`、`Checkpoint::page_ids()`、
`Parked::offsets()` / `slot_offset()`，harness 与测试读字节用）——没有任何接口按它们
找东西。

### Node<T>：一页，冻结

```rust
pub struct Node<T: Storage> {              // 字段 crate 私有；pub 只因为它是 Twin 里的类型
    page: T::Page,
    parent: Option<Arc<Node<T>>>,          // 持有一个节点就持有它到根的每一页
    tier: Arc<T>,                          // drop 时把 page 还给它
    twin: T::Twin,                         // Pool: Mutex<Weak<Node<Host>>>；Host: ()
}
```

不变量：**一个 `Node` 存在的那一刻它的页就不再被写。** 节点只从三处产生：lease 写满
的整页（`checkpoint` / `retire` / `fork` 把它移进链）、为一个新持有者拷出来的半页、
`retire` 时 lease 消失后原地封存的半页。运行中的 lease 永远不写任何 `Node` 的页。
这一条之前在 `Pool::checkpoint` 上被打破（半页 checkpoint 后 lease 接着写同一页），
host 层按 node 去重就读到过期字节；这是 review 的 P1。

`twin` 是 device 节点对它 host 副本的弱引用。park 时逐节点看 twin：活着就复用，否则
分配、拷贝、记下。用节点自己的字段而不是按 id 的表，两个原因：id 表的 key 可以在内容
变了之后仍然相等（这次的 bug），用指针当 key 又有 ABA；节点持有自己的链接，两者都没有。
wake（`Host::restore`）进 lease 的整页节点在创建时就带上 twin，所以同一 session 醒来再睡
不再拷第二份（lessons.md 2026-09-03 记的那条缺口）。

### Store<T>：一个前缀在一个 tier 里的字节

```rust
pub struct Store<T: Storage> {
    len: usize,                            // 持有的 token 数，≥ 1
    pages: usize,                          // ceil(len / unit)；slot-only 时 0
    chain: Option<Arc<Node<T>>>,           // 页链；slot-only 时 None
    slot: Option<Arc<SlotOwn<T>>>,         // 恰在 len 之后的 recurrent state；Drop 归还
}
pub type Checkpoint = Store<Pool>;
pub type Parked = Store<Host>;
```

`Store<T>: Clone`：clone 是同一份字节的又一个持有者（`Arc` 链、`Arc` slot），最后一个
持有者 drop 时归还。索引条目、`Hit`、runtime 手里在飞的 park 拿的是同一份，不需要
id 也不需要类型擦除。带 slot 的 store 只在精确 `len` 上可用（state 是"恰好这些 token
之后"的状态）；不带 slot 的可在它任何整页上可用。

### Lease：唯一能写的东西

```rust
pub struct Lease {
    chain: Option<Arc<Node<Pool>>>,        // 共享的整页
    shared: usize,
    pages: Vec<i32>,                       // 全部页号；shared 之后的是自己的，Drop 归还
    slot: Option<Arc<SlotOwn<Pool>>>,
    prefix: usize,                         // 已填的位置，不再命名
}
```

`Lease` 不泛化：它是唯一被 kernel 寻址的对象（page table 行、slot mapping、line
index），host 单元从不进表。行由 `extend_row` 铺，位置由 `slot(pos)` 给：`pos` 必须
过了 `prefix`，也必须过了 `shared` 页——一个 `Node` 存在的那一刻它的页就不再被写，
封页的租约自己也不例外（`slot` 断言，`tests/pool.rs` 的
`a_lease_refuses_the_pages_it_sealed`）；`page_ids()` 留给 harness 读字节，不是句柄。

`Pool::new(manifest, chunk, chunks, first_slots, tokens)` 的 `tokens` 是容量的上限：
页数不超过 `tokens / unit`，块的尾巴空着。块的上限是 64 MiB（`chunk_for`），小页的
manifest 一个块就装几百页，`--capacity` 若按块取整就没有小池子可测——toy 门禁
第一天撞上的就是这条。

### Copies：借用来源的拷贝计划

```rust
pub struct Copies<S = i32, D = i32> {
    pub pages: Vec<(S, D)>,
    pub slot: Option<(S, D)>,
}
```

之前的 `Copies`（device→device）、`Park`（device→host）和 wake 里手拼的 `Vec<(i32, u64)>`
合成一个类型。plan 由要它的 shell 当场执行、不存（runtime 的 `Room` 把 plan 和它
引用的 `Checkpoint` 放在一个结构里，`Waking` 同理），所以没有绑生命周期。

### Prefix：token 索引

```rust
pub struct Prefix<R: Kept = Checkpoint, P: Kept = Parked> { unit, root: Node, lru, clock, .. }

pub struct Hit<R, P> {
    pub len: usize,                        // 可用的前缀长度
    pub found: Found<R, P>,                // Resident(R) | Parked(P)：条目的一个 clone
}
```

条目持有 `R` / `P`，`Hit` 持有它们的 clone（`Kept: Clone`，`Store<T>` 和 tray 的
`Group<X>` 都是）。`make_room` 把条目 park 掉或删掉都不影响手里的 `Hit`（lessons.md
2026-09-04"make_room 之后旧的命中不能再用"这类错误从类型上消失）。条目按 token 键
操作：`insert(&key, r)`、`lookup(&tokens)`，腾地方只有一个动作 `evict(park)`：最久未命中
的 resident 条目经调用方的拷贝 `park` 进 host（host 满了先丢最冷的 parked 条目再试，
丢光了还放不下、或没有 host 层，就丢它），返回 `Evicted::{Parked(key), Dropped {key,
parked}}` 说明动了谁——scheduler 与 `agentx_replay` 之前各写一份同样的循环，现在
是表的。哪个条目最冷是表自己的事，外面看不到 LRU。TP 下 `R` / `P`
是 tray 的 `Group<Checkpoint>` / `Group<Parked>`，索引仍然是 tray 一棵，不变。

## 3. 操作

| 操作 | 签名 | 页 | slot |
|---|---|---|---|
| `Pool::lease(tokens)` | `-> Lease` | 全新 | 全新，runtime 清零 |
| `Pool::checkpoint(&mut Lease, len)` | `-> (Checkpoint, Copies)` | 整页入链共享；**`len` 在页中间时那半页拷给 checkpoint**，lease 继续写自己的 | 拷进新 slot |
| `Pool::retire(Lease, len)` | `-> Checkpoint` | 整页入链，半页原地封存，多余的还 | 易主，不拷 |
| `Pool::restore(&Checkpoint, len, tokens)` | `-> (Lease, Copies)` | 整页共享，半页拷给 lease | 拷进新 slot |
| `Pool::fork(&mut Lease, len, tokens)` | `-> (Lease, Copies)` | 整页共享，半页拷给孩子 | 拷进新 slot |
| `Host::park(&Checkpoint)` | `-> (Parked, Copies<i32, u64>)` | 沿链走，twin 活着的复用，其余分配并拷 | 拷 |
| `Host::restore(&Parked, &Pool, len, tokens)` | `-> (Lease, Copies<u64, i32>)` | 按 `tokens` 一次取够页，前 ceil(len/unit) 页从 host 拷进去，整页封成带 twin 的节点，半页归 lease | `len` 是整长时拷进新 slot |

三条规则贯穿全部操作：**整页共享、半页拷给新持有者、写者只有 lease。** `restore` 和
`fork` 本来就这样，`checkpoint` 是唯一的例外，改成一致。半页拷贝只在页中间
checkpoint 时发生；scheduler 只在页边界（纯 KV）或请求结束（`retire`）留快照，所以生
产路径零拷贝不变。K1c 的显式断点（system prompt 末尾，带 state 的模型）正是需要页中
间 checkpoint 的地方，付一次页拷贝。

**wake 直接醒进请求的租约。** 2026-09-14 的版本让 wake 得到一个 `Checkpoint`，scheduler
把它插回索引、请求回到队首再按 resident 命中 `restore`：两次分配、两次"够不够"的判断，
没有谁把它们合起来看。DSv4.1 EP4（`--chunk 128 --max-seqs 16`）第二轮命中 parked 条目就
活锁——醒来的快照加半页拷贝加续写比池子多一页，`Busy` → `make_room` 能 park 的只有
刚醒的那条 → 再醒 → 再 park（一次 host session 打出 236 万行 `parked tokens=86`；toy-stateful
上 e2e 的 `wake_room` 场景 608 万行）。现在 `Host::restore(&Parked, &Pool, len, tokens)` 与
`Pool::restore` 同形：按 `tokens` 一次取够页，前 ceil(len/unit) 页从 host 拷进这些页里，
整页封成带 twin 的节点（再睡零拷贝），半页是 lease 自己的；runtime 的 `Waking` / `awake`
交出的是 `Lease`，tray 的 `Rising` 落地即 `Row`，scheduler 落地即 admit。parked 条目原样
留在索引里，请求结束时更长的上下文照常成 resident 条目。没了的东西：`Host::wake`、醒来
的快照进索引、scheduler 的 `woken` 名单、`Got::Rising` 里带的 key。

## 4. 索引：token radix tree

树的边是 token 段，节点上挂条目；**树只索引，不管字节**。字节的共享关系全在 store
的链上，两个条目共享哪些页由它们从哪条 lease 派生决定，树不知道也不需要知道。

这一点和 SGLang 不同（它的树节点直接持有 KV 索引，所以 split 必须在页边界，state
只能挂在页边界的节点上，页中间的 state 靠 tombstone 处理），原因是 kern 的带 state
快照落在请求结束的任意长度上，半页是快照私有的：同一 session 的下一轮从 L₁ restore
（拷半页）再 retire 到 L₂，两个快照在 token 上是前缀关系、在页上不是。条目持有
store 让树可以在任意 token 位置分叉而不碰任何一页。

命中规则：沿 prompt（不含最后一个 token）走树，
- 路径上的条目（prompt 完整覆盖它）在自己的 `len` 上可用，带不带 slot 都是；
- 路径之外不带 slot 的条目在它与 prompt 共有的整页上可用：从深度 d 的节点分出去的
  兄弟子树共有 d 个 token，prompt 走进一条边的中间时那棵子树共有到分叉处；每个节点
  的 `paged_below` / `resident_below` 子树计数说这样的条目有没有，不用扫；
- 取最长，同长 resident 优先。

命中触碰路径上的每个条目，根最新、叶最旧，同一条链一起老化、叶先走；`evict` 取的
最冷条目来自各 tier 一张按 tick 的 `BTreeMap`。不带 slot 的 resident 条目比同路径上一个
不带 slot、只在 device 的条目深一页时替换它：纯 KV 序列每页 checkpoint 一次，
索引里只留一条会长的。

**这个索引不提高命中长度。** 纯 KV 模型命中仍对齐到页，带 state 模型仍只命中精确
长度（serve.md 记过：state 快照之后的 token 必须整段重跑，KV 部分命中没有算力意义）。
换掉 hash chain 换来的是：键是 token 本身，没有碰撞；`Chain` 不再由每条序列携带、
每 token 折叠；`(depth, hash)` 桶、tail 重哈希、"条目按页生长"的特殊路径都没了；
两个 session 打出同样的 token 在树上就是同一条路径。付出的是树里存 token
（每 token 8 字节，相对 KV 每 token 几十 KB 可忽略；AgentX 393 个 session、ctx p50
219k 全存也不到 1 GB）和 split 逻辑。插入从根走一遍 memcmp，每页一次，摊到每步
微秒级。

树的实现：`children: BTreeMap<i64, Node>` 按边的第一个 token 键，确定性；条目删空
且只剩一个孩子的节点合并回去（dynamo 的 router 因为并发读不能合并，我们可以）；LRU
是单调计数器（SGLang 的 tick，不是时钟）。

## 5. 从参考实现借了什么

| 来源 | 借 | 不借 |
|---|---|---|
| Dynamo KVBM | 冻结 = `Arc::new(mutable)`，只读性从 `Arc` 加只有 `Deref` 的包装里长出来（kern 的 `Node` 就是这个）；Drop 里迭代解链防栈溢出；registry 的值是 `Weak`，删除由 Drop 驱动 | 三个泛型参数传遍全部类型再用 `dyn Any` 擦回去；`BlockState` 是运行时枚举、每个操作都是可失败的 match；无界 channel；跨 tier 用相等的 hash 链接（我们用节点上的 twin） |
| Dynamo kv_router | 单线程 actor；删除时合并节点；LRU 侵入节点 | `Rc<RefCell>` 加旁路 lookup 表当真索引、split 要重写 O(后缀) 条 lookup；arena + `u32` id（所有权丢了） |
| SGLang | 页对齐是键类型的性质不是散落的算术；单调 tick；淘汰只看可淘汰的叶子，策略只打分不能 pin；lock 回执（我们用 RAII，回执就是 `Arc`）；state 只在节点精确末端有效这一条做成类型 | 节点是 (component × tier) 值矩阵按下标算；panic 当错误处理；split 深拷值；`cache_protected_len` 把半页挂在请求上；mamba tombstone + 每次匹配重扫；为 FFI 加的 `Mutex` |

三家都没有页中间共享。KVBM 半块直接丢；SGLang 匹配向下对齐到页、split 断言页边界；
dynamo router 按 block hash。kern 的 `restore` 会拷半页，所以在带 state 的精确长度命中
上比它们多拿到那不足一页的部分，这是保留 exact-length 快照的理由。

## 6. 与之前的差异

删掉的公开项：`Checkpoint::nodes()`、`Parked::pages(n)` / `slot()`、
`Host::park(&[(u64, i32)], page_bytes, slot, len)` 的四元组签名、`Park` 类型、
`Pool::wake` / `wake_slot`、`Prefix::resident(id)` / `parked(id)` 的 id 查询、`Chain`。
留下的：`Kept`（tray 的 `Group<X>` 要靠它回答 `tokens()` / `has_slot()`）、
`Lease::page_ids()` / `Checkpoint::page_ids()` / `Parked::offsets()`（只读观测）。

修掉的已知问题：review 的 P1（半页 checkpoint 后 host 副本过期）由不变量消失；
"醒来再睡拷第二份"由 wake 带 twin 消失；"make_room 后旧 hit 失效"由 `Hit` 持有 clone
消失；`Host` 层 `page_bytes` / `slot_bytes` 改在 `Host::new` 给一次，不再每次 park 传；
分叉在条目中间时同长的 resident 条目优先于 parked 的（之前按 id 取第一个）。

不动的部分：`Pool` 内部的 chunk / remap / `Denied` 三态、`Lease` 的表接口、runtime 的
两条流与 `Waking` / `Room` 模式、tray 级"全成或全不成"、scheduler 的快照策略。
review 的 P2（`rebalance` 对共享 chunk 重复计费导致误报 `ExceedsPool`）在 `Pool` 内部，
与本设计正交，单独修。

## 7. 门禁

一个 PR 落地（2026-09-14）。测试都在 `crates/kern-pool/tests/`，不用 GPU：

- **字节级 property test**（`pool.rs`）：随机的 lease / 填写 / checkpoint / restore /
  fork / retire / drop 序列，每个位置由它的 lease 写上唯一戳，对着"位置 → 戳"的
  参考模型比对每个句柄能看到的每个位置和 slot，池的占用等于句柄命名的页与 slot。
  纯 KV 与带 state 各跑一遍。旧的 `checkpoint` 在这个测试上直接复现 P1。
- **park / wake 字节 property test**（`host.rs`）：加上 park / wake，host 上的字节按
  偏移比对，parked 句柄命名的偏移两两不交、恰好是 `Host::used()`。
- **索引 property test**（`prefix.rs`）：随机 token 序列的插入 / 查询 / park / 淘汰
  对着"所有条目线性扫描取最长、同长 resident 优先"的参考模型；同一操作序列跑两遍
  trace 相同。

AgentX 回放（`agentx_replay`，qwen3.8-27b 形状：页 784、64 KiB/token、147 MiB slot、
130 slot 起、单卡 250 GiB、并发 32；`~/bench_results/2026-09-14-pool-radix-replay/`）：
不带 host 命中 93.1%（与 K1b 相同）；带 host 512 GiB 见 roadmap K3b 行。回放现在
按 tier 分开报命中（resident / host 各多少请求、多少 token），kern-serve 的 stats
行也加了 `resident_hits` / `resident_hit_tokens` / `host_hits` / `host_hit_tokens`，
host 层有没有被打到一眼可见。

## 8. 未决

- 带 state 的模型要不要在页边界也留 state 快照（SGLang 的 `mamba_checkpoint_grid`），
  换取页对齐的部分命中。代价是每页一个 slot 拷贝和 slot 占用，收益要用 AgentX
  trace 回放算，不先做。
- `Device` 的 Drop 归还走 `Mutex` 还是 KVBM 那样的 channel。单线程下 `Mutex` 永不争用，
  先不改。
- 树是否需要按 `KeyNamespace`（SGLang 的 extra_key / cache_salt）分片：LoRA 或多模型
  共用一个池时才需要，现在没有。
