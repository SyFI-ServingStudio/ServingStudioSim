# L5 Worker 组合空间与兼容性矩阵

- 状态：设计输入（用于决定 FSM 是否应成为独立组合轴）
- 关联设计：[L5_redesign.md](L5_redesign.md)（最终权威文档；原 refactor/interfaces/module_apis 等草案已归档至 repo 外 `../../../L5_old_docs/`）
- 原则：只排除**语义上自相矛盾**的组合；尚未实现、当前部署未使用、实现成本高，都不能成为排除理由。

## 1. 为什么需要这份矩阵

不能从当前六个 worker 反推抽象，因为那只会重新得到当前六种硬编码组合。这里从完整组合空间出发，回答三个问题：

1. 哪些 `Model × Kv × Admission × Fsm` 组合在语义上成立；
2. 哪些组合需要新的 capability，但仍必须由接口表达；
3. FSM 是否真的与 Model、KV、Admission 正交到值得抽成类型参数。

本文使用**归一化穷举**，而不是手写一万多行 tuple。若一个 tuple 在下列全部关系表中都不是 `×`，它就被保留；这条规则能唯一判定每一个 tuple，不允许实现者再凭“目前没有这种 worker”删组合。

矩阵不是封闭的 feature enum。最终抽象还必须通过 solution 文档 §15 的 unknown-variant test：一个设计时未枚举的新 layout/lifecycle/policy/model input/cadence，应只新增对应 leaf、必要的薄 adapter、declarative registration 与测试；若必须修改现有 behavior body 或复制已有机制，即使本矩阵全部通过也仍算失败。

## 2. 判定符号

| 符号 | 含义 | 是否保留 |
|---|---|---|
| `●` | 该组合的自然职责或首要实现目标 | 是 |
| `○` | 语义完整，虽然未必已有实现 | 是 |
| `△` | 语义成立，但必须补上表中写明的 capability、外部状态或协议变体 | 是 |
| `×` | 两个契约互相矛盾；实现后会改变其中至少一个名字的含义 | 否 |

合成规则：

1. tuple 任一关系为 `×`，整个 tuple 才被排除；
2. 没有 `×`、但至少一个关系为 `△`，tuple 为“需要显式扩展”；
3. 其余 tuple 全部保留；`○` 不代表低优先级，更不代表“不支持”；
4. capability 缺失只能产生 `△`，不能把基础组合偷偷改成 `×`。

## 3. 轴的定义

### 3.1 Model execution contract

这里枚举的是 **worker 所执行的模型切片**，而不只是模型商品名。否则“MoE 模型的 FFN worker 没有 KV”会与“同一 MoE 模型的 attention worker 有 KV”混为一谈。

| ID | 执行契约 | 说明 |
|---|---|---|
| `M0` | `TextDenseFull` | 文本、dense FFN、单一 token-growing attention/state，执行完整 decoder |
| `M1` | `TextMoeFull` | 文本、MoE FFN、单一 token-growing attention/state，执行完整 decoder |
| `M2` | `TextDenseHybrid` | 文本、dense FFN、full/SWA/recurrent/MLA/DSA 等混合 state，执行完整 decoder |
| `M3` | `TextMoeHybrid` | 文本、MoE FFN、混合 state，执行完整 decoder |
| `M4` | `MultimodalDenseFull` | multimodal encoder/cross-attention + dense decoder + 单一 decoder KV |
| `M5` | `MultimodalMoeFull` | multimodal encoder/cross-attention + MoE decoder + 单一 decoder KV |
| `M6` | `MultimodalDenseHybrid` | multimodal + dense decoder + 混合 decoder state |
| `M7` | `MultimodalMoeHybrid` | multimodal + MoE decoder + 混合 decoder state |
| `M8` | `AttentionShardFull` | AFD/分层 attention 侧，只执行单一 token-growing attention；FFN 类型与它无关 |
| `M9` | `AttentionShardHybrid` | AFD/分层 attention 侧，执行 mixed-attention/state |
| `M10` | `FfnShardDense` | 纯 dense FFN task consumer，不执行 attention |
| `M11` | `FfnShardMoe` | 纯 MoE/EP FFN task consumer，不执行 attention |
| `M12` | `EncoderMediaOnly` | 只执行 image/audio/video encoder、projector 或 media preprocessing，不执行 autoregressive decoder |

下列是 model 的正交修饰项，不另复制 13 个基础契约：

| 轴 | 取值 | 说明 |
|---|---|---|
| 模型驻留 | `SingleModel` / `MultiArch` | 一个 worker 只驻留一个模型，或在迭代边界选择多个 resident arch 之一 |
| 生成协议 | `Standard` / `Speculative` | 普通生成，或 draft → target verify → accept 可变 token 数 |
| 并行布局 | local / TP / DP-attn / EP-FFN / HP / CP | 改变 grouping、通信和成本，不改变上述模型语义 |

因此 `MoE + Hybrid + Multimodal + MultiArch + Speculative` 是明确保留的组合，而不是被某个扁平枚举漏掉。

### 3.2 KV/state ownership

| ID | KV 契约 | 说明 |
|---|---|---|
| `K0` | `NoKv` | worker 不拥有 decoder KV；可执行无状态任务，或由远端状态服务提供所需状态 |
| `K1` | `FullKv` | 单一 token-growing decoder KV/state |
| `K2` | `HybridStateKv` | 每层/每 state-kind 可有不同驻留形状、增长律与峰值律 |
| `K3` | `ModelPartitionedKv<ChildKv>` | 多 resident model 的命名空间、预算与 eviction；child 可为 `K1`、`K2` 或 `K4` |
| `K4` | `CompositeKv<DecoderKv, AuxState>` | decoder KV 与 encoder output、cross-attention state、media state 等复合资源 |

KV 的完整类型不是只有上述 state layout，而是：

```text
Kv = StateLayout × PrefixMode × ReplacementPolicy
```

| 子轴 | ID | 语义 |
|---|---|---|
| Prefix mode | `PC0 NoPrefixCache` | request 只拥有私有 decoder state，不跨请求共享 prefix |
| Prefix mode | `PC1 PrefixCache` | 以 block/radix prefix index 共享 state，维护 refcount、命中长度与可回收 entry |
| Replacement | `RP0 NoEvict` | 已缓存 prefix 不主动驱逐；空间不足时拒绝/延后新的 cache insertion |
| Replacement | `RP1 Lru` | 按最后访问时间驱逐 |
| Replacement | `RP2 Lfu` | 按访问频率驱逐，必须定义 aging，避免永久热 entry |
| Replacement | `RP3 SizeAware` | 按收益/占用字节驱逐，避免一个大 prefix 挤掉大量小热点 |
| Replacement | `RP4 RecomputeCostAware` | 按预计复用概率、重算时间或节省 GPU 时间/byte 驱逐 |
| Replacement | `RP5 SloAware` | 优先保留对 deadline、TTFT/TBT 或高优先级请求最有价值的 prefix |

`PC0` 不选择 replacement；`PC1` 必须选择且只选择一个 `RP0`–`RP5`。实现形态应类似 `ModeledPrefixCacheKv<BaseKv, Index, Replacement>`，而不是为 LRU/LFU/SLO 各复制一份 `FullKv` 或 `HybridStateKv`。`K1/K2/K3/K4` 都能包 `PC1`：hybrid KV 只共享可 prefix 化的 state kind，multi-model KV 在 model namespace 内建 index，composite KV 默认只包装 decoder child；media/encoder cache 若也共享，必须有独立 key 与 replacement domain。

`Held`、`SubsetAdvance`、`VariableAdvance` 是其他 capability，不是新的基础 KV 类型。类似地，`ShardKv` 不应成为独立 KV 品种：AFD 的差异是 FSM 传入 `AdvanceScope::RequestSubset(...)`，KV 只按该 grouping 查询或推进它拥有的 state。

### 3.3 Admission lifecycle

| ID | Admission 契约 | 输入与生命周期 |
|---|---|---|
| `A0` | `NoAdmission` | 上游已经形成 ready task/batch；本 worker 不拥有 request admission |
| `A1` | `LocalPrefillDecodeAdmission` | fresh request → whole prefill → decode → completion |
| `A2` | `PrefillHandoffAdmission` | fresh request → prefill → hold/transfer/ack → release |
| `A3` | `DecodeDirect` | 接收已经 prefill 的 request；reserve/pull/commit 后 decode |
| `A4` | `ChunkedPrefillAdmission` | fresh request 的 prefill 跨多个 iteration，之后 decode 或 handoff |
| `A5` | `FreshRequestSlotAdmission` | 只为 fresh request 预留/提交 state；全局完成由 attention/FFN 流水线协调 |
| `A6` | `EncodeThenPrefill` | media/encoder input → aux state ready → decoder prefill/decode 或 handoff |

SLO-aware 不改变 request 的生命周期阶段，因此不是 `A7`。Admission 的完整类型是：

```text
Admission = Lifecycle × PendingOrderPolicy
```

| Policy ID | Selection policy | 排序/准入依据 |
|---|---|---|
| `AP0` | `FifoThroughput` | FIFO、batch fill、token budget 等非 SLO 基线策略 |
| `AP1` | `StaticPriority` | tenant/request priority class；同级必须定义稳定次序与 starvation 防护 |
| `AP2` | `EarliestDeadline` | 最早 absolute deadline 优先 |
| `AP3` | `PredictedSlack` | `deadline - now - predicted_remaining_service` 最小者优先 |
| `AP4` | `TtftAware` | 优先降低未出首 token 请求的预计 TTFT 违约风险 |
| `AP5` | `TbtAware` | 优先降低 active decode 的预计 inter-token gap/TBT 违约风险 |
| `AP6` | `CompositeSlo` | 明确的 lexicographic/weighted objective，例如 priority → lateness risk → throughput |

`AP1`–`AP6` 是能与 `A1`–`A6` 组合的 typed stateful policy；它们可以各自拥有不同的 request view、prediction query/result、内部 state、shared context 与 completion feedback，不能被假设成只换 comparator 的同形 runtime enum。`A0 NoAdmission` 的 SLO selection 只能由上游 task producer 完成。Policy 通过 model-side typed estimator/oracle 查询自己需要的预测，但 Admission 不应因此拥有 ArchExec 或构造 `ArchInput`。

`PrefixAware`、`ActiveModel`、`TokenBudget` 是其他 admission capability/policy modifier，不应复制七套 lifecycle。Session affinity 是 L6 placement 约束，不属于 L5 admission。

### 3.4 Worker FSM

| ID | FSM 契约 | 主要状态变化 |
|---|---|---|
| `S0` | `Iter` | form batch → predict → wait → complete |
| `S1` | `IterPull` | ingress reserve/pull/commit → `Iter` |
| `S2` | `SlotPipeline` | request 在 layer/slot 间推进，可对 request 子集计费 |
| `S3` | `SlotPipelinePull` | `SlotPipeline` 加 ingress state pull/commit |
| `S4` | `DoubleBufferTask` | task-fed 双缓冲执行，完成后回送 task/result |
| `S5` | `EncoderPipeline` | media/encoder stage → aux-state handoff → decoder 或下游 stage |
| `S6` | `DraftVerify` | draft K token → target verify → accept N token → state 按 N 推进 |

“不抽出 FSM 类型”不等于 worker 没有 FSM：`tick()`、pending pull、slot、double buffer 都是状态机。待决定的是这些状态与 action 是否由独立 `Fsm` 组件拥有，还是直接由一个短 concrete worker 拥有。矩阵先判断行为是否能独立变化，再决定代码形态。

## 4. 完整关系表

基础空间为：

```text
13 Model contracts × 5 KV × 7 Admission × 7 FSM = 3,185 tuples
```

按下列五张关系表机械合成，基础空间中有 `879` 个 tuple 因至少一个明确契约冲突被排除，`2,306` 个被保留；保留项中 `1,788` 个需要表中点名的 capability/adapter，`518` 个无需额外条件。这个计数是矩阵完整性基线，修改任何轴或判定后都应重新计算。

再乘 `SingleModel/MultiArch` 与 `Standard/Speculative` 两个二值修饰项，共 `12,740` 个 model-extended tuple。Admission selection policy 与 KV prefix/replacement 是另外两条子轴：它们只在适用的 `A`/`K` 上展开，因此不使用一个会把“不适用”也算进去的虚假统一乘数。下面的表与 §6 的子轴约束共同覆盖完整空间。

### 4.1 Model × KV

| Model | `K0 NoKv` | `K1 FullKv` | `K2 HybridStateKv` | `K3 ModelPartitionedKv` | `K4 CompositeKv` |
|---|---:|---:|---:|---:|---:|
| `M0 TextDenseFull` | △ remote/recompute | ● | ○ degenerate hybrid | ○ | ○ aux 可为空 |
| `M1 TextMoeFull` | △ remote/recompute | ● | ○ degenerate hybrid | ○ | ○ aux 可为空 |
| `M2 TextDenseHybrid` | △ remote/recompute | △ 非 full state 在外部 | ● | ○ | ○ |
| `M3 TextMoeHybrid` | △ remote/recompute | △ 非 full state 在外部 | ● | ○ | ○ |
| `M4 MultimodalDenseFull` | △ remote/recompute | △ aux state 在外部 | △ aux state 在外部 | ○ | ● |
| `M5 MultimodalMoeFull` | △ remote/recompute | △ aux state 在外部 | △ aux state 在外部 | ○ | ● |
| `M6 MultimodalDenseHybrid` | △ remote/recompute | △ hybrid+aux 在外部 | △ aux state 在外部 | ○ | ● hybrid child |
| `M7 MultimodalMoeHybrid` | △ remote/recompute | △ hybrid+aux 在外部 | △ aux state 在外部 | ○ | ● hybrid child |
| `M8 AttentionShardFull` | △ remote/stateless | ● | ○ degenerate hybrid | ○ | ○ cross-attn |
| `M9 AttentionShardHybrid` | △ remote/stateless | △ 非 full state 在外部 | ● | ○ | ○ |
| `M10 FfnShardDense` | ● | × | × | × | × |
| `M11 FfnShardMoe` | ● | × | × | × | × |
| `M12 EncoderMediaOnly` | ● request-local | × decoder KV | × decoder KV | △ child 必须是 aux/composite | ● aux-only child 可为空 |

硬排除理由只有两个：

- 纯 FFN contract 若拥有 decoder KV，就不再是纯 FFN shard；
- 非 autoregressive encoder-only contract 不应伪装成 `FullKv`/`HybridStateKv` decoder cache。

### 4.2 Model × Admission

| Model | `A0 None` | `A1 P+D` | `A2 P→Handoff` | `A3 DecodeDirect` | `A4 Chunked` | `A5 FreshRequestSlotAdmission` | `A6 Encode→P` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0 TextDenseFull` | ○ ready batch | ● | ○ | ○ | ○ | △ completion 外置 | △ encoder stage 可为空 |
| `M1 TextMoeFull` | ○ ready batch | ● | ○ | ○ | ○ | △ completion 外置 | △ encoder stage 可为空 |
| `M2 TextDenseHybrid` | ○ ready batch | ● | ○ | ○ | ○ | △ completion 外置 | △ encoder stage 可为空 |
| `M3 TextMoeHybrid` | ○ ready batch | ● | ○ | ○ | ○ | △ completion 外置 | △ encoder stage 可为空 |
| `M4 MultimodalDenseFull` | ○ ready batch | ○ aux 已 ready | ○ | ○ | ○ | △ completion 外置 | ● |
| `M5 MultimodalMoeFull` | ○ ready batch | ○ aux 已 ready | ○ | ○ | ○ | △ completion 外置 | ● |
| `M6 MultimodalDenseHybrid` | ○ ready batch | ○ aux 已 ready | ○ | ○ | ○ | △ completion 外置 | ● |
| `M7 MultimodalMoeHybrid` | ○ ready batch | ○ aux 已 ready | ○ | ○ | ○ | △ completion 外置 | ● |
| `M8 AttentionShardFull` | ○ task-fed | △ 全局 completion 回调 | ○ | ● | △ chunk completion 回调 | ● | ○ multimodal attn |
| `M9 AttentionShardHybrid` | ○ task-fed | △ 全局 completion 回调 | ○ | ● | △ chunk completion 回调 | ● | ○ multimodal attn |
| `M10 FfnShardDense` | ● | × | × | × | × | × | × |
| `M11 FfnShardMoe` | ● | × | × | × | × | × | × |
| `M12 EncoderMediaOnly` | ● task-fed | × decoder prefill | × KV handoff | × decoder direct | △ 必须是 encoder chunk variant | × KV reserve | ● |

纯 FFN worker 只能消费上游 task；若它直接拥有 request admission，它就已经变成 attention+FFN orchestration worker。`EncoderMediaOnly` 同理不能使用以 decoder KV 生命周期定义的 admission。

### 4.3 Model × FSM

| Model | `S0 Iter` | `S1 IterPull` | `S2 Slot` | `S3 SlotPull` | `S4 DoubleTask` | `S5 Encoder` | `S6 DraftVerify` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M0 TextDenseFull` | ● | ○ | △ layerwise full-model | △ layerwise full-model | △ task 化 full-model | △ no-op/external encoder | ○ +Speculative |
| `M1 TextMoeFull` | ● | ○ | △ layerwise full-model | △ layerwise full-model | △ task 化 full-model | △ no-op/external encoder | ○ +Speculative |
| `M2 TextDenseHybrid` | ● | ○ | △ layerwise full-model | △ layerwise full-model | △ task 化 full-model | △ no-op/external encoder | ○ +Speculative |
| `M3 TextMoeHybrid` | ● | ○ | △ layerwise full-model | △ layerwise full-model | △ task 化 full-model | △ no-op/external encoder | ○ +Speculative |
| `M4 MultimodalDenseFull` | ○ aux 已 ready | ○ | △ layerwise | △ layerwise | △ task 化 | ● | ○ +Speculative |
| `M5 MultimodalMoeFull` | ○ aux 已 ready | ○ | △ layerwise | △ layerwise | △ task 化 | ● | ○ +Speculative |
| `M6 MultimodalDenseHybrid` | ○ aux 已 ready | ○ | △ layerwise | △ layerwise | △ task 化 | ● | ○ +Speculative |
| `M7 MultimodalMoeHybrid` | ○ aux 已 ready | ○ | △ layerwise | △ layerwise | △ task 化 | ● | ○ +Speculative |
| `M8 AttentionShardFull` | ○ non-pipeline | ○ | ● | ● | △ attention task 化 | ○ cross-attn/media | △ 只参与 verify |
| `M9 AttentionShardHybrid` | ○ non-pipeline | ○ | ● | ● | △ attention task 化 | ○ cross-attn/media | △ 只参与 verify |
| `M10 FfnShardDense` | ○ task/iter | ○ task pull | ○ layer slots | ○ layer slots+pull | ● | × | △ 分布式 verify participant |
| `M11 FfnShardMoe` | ○ task/iter | ○ task pull | ○ layer slots | ○ layer slots+pull | ● | × | △ 分布式 verify participant |
| `M12 EncoderMediaOnly` | ○ one-shot | ○ input pull | ○ layer/tile slots | ○ slots+pull | ○ task 化 | ● | × |

这里特意保留了“非默认 FSM”。例如 full decoder 走 layer slots 虽然现在没有，却是 pipeline-parallel、分层模拟或细粒度 preemption 的合理实现，不能因为当前 worker 没这么写就删掉。

### 4.4 KV × Admission

| KV | `A0 None` | `A1 P+D` | `A2 P→Handoff` | `A3 DecodeDirect` | `A4 Chunked` | `A5 FreshRequestSlotAdmission` | `A6 Encode→P` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `K0 NoKv` | ● | △ stateless/recompute | × 无 KV 可 handoff | × 无 KV 可 pull/commit | △ stateless/recompute | × 无 KV 可 reserve | ○ aux 可由别处持有 |
| `K1 FullKv` | ○ task-fed owner | ● | ● +Held | ● +Held/transfer | ● | ● | △ aux 在外部 |
| `K2 HybridStateKv` | ○ task-fed owner | ● | ● +Held | ● +Held/transfer | ● | ● | △ aux 在外部 |
| `K3 ModelPartitionedKv` | ○ | ○ +ActiveModel | ○ +ActiveModel | ○ +ActiveModel | ○ +ActiveModel | ○ +ActiveModel | ○ |
| `K4 CompositeKv` | ○ | ○ | ○ +Held | ○ +Held/transfer | ○ | ○ | ● |

这张表固定了一个重要边界：Admission 决定生命周期和准入策略，KV 只提供 state footprint、reserve/commit/advance/release 等事实与原语。Admission 不构造 `ArchInput`，KV 也不解释 request 属于 prefill、decode、draft 还是 verify。

### 4.5 Admission × FSM

| Admission | `S0 Iter` | `S1 IterPull` | `S2 Slot` | `S3 SlotPull` | `S4 DoubleTask` | `S5 Encoder` | `S6 DraftVerify` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A0 NoAdmission` | ○ ready batch | ○ upstream pull | ○ upstream slots | ○ upstream slots+pull | ● | ○ encoder tasks | ○ speculative tasks |
| `A1 LocalPrefillDecodeAdmission` | ● | ○ offload/prefetch | △ slot completion hook | △ slot completion+pull | △ request→task adapter | ○ aux 已 ready | ○ draft/target 都可入 batch |
| `A2 PrefillHandoffAdmission` | ● | ○ input/output pull | ○ layerwise prefill | ○ layerwise+pull | △ request→task adapter | ○ multimodal prefill | ○ handoff 到 speculative decode |
| `A3 DecodeDirect` | △ local/zero-copy only | ● | △ local state only | ● | △ task handoff adapter | ○ aux+KV pull | ○ draft/target direct decode |
| `A4 ChunkedPrefillAdmission` | ● | ○ | ○ layerwise chunk | ○ layerwise chunk+pull | △ chunk→task adapter | ○ encoder+prefill chunks | ○ chunked target/draft prefill |
| `A5 FreshRequestSlotAdmission` | ○ whole-iter AFD | ○ | ● | ○ | △ request→task adapter | ○ multimodal attention | ○ AFD speculative participant |
| `A6 EncodeThenPrefill` | △ encoder 在外部完成 | △ media pull | △ encoder/decoder slots | △ slots+pull | △ stage→task adapter | ● | ○ multimodal speculative |

本表没有硬 `×`：admission lifecycle 与执行调度的组合基本都能定义清楚。大量 `△` 正是 FSM 是否值得成为独立轴的压力证据，而不是删除组合的理由。

## 5. Model 修饰项

### 5.1 Multi-Arch

| 组合 | 判定 | 要求 |
|---|---|---|
| `MultiArch + K3 ModelPartitionedKv` | `●` | admission/FSM 传 `active_model`；KV 在 model namespace 内记账 |
| `MultiArch + K1/K2/K4` | `△` | 必须有外层 model namespace 或证明所有 arch 安全共享布局 |
| `MultiArch + K0` | `○` | 无本地 decoder KV，模型选择仍影响 ArchExec 和成本 |
| `SingleModel + K3` | `○` | 可退化为一个 namespace；冗余但不矛盾 |

模型切换只能发生在显式安全边界，默认是 iteration/task boundary。FSM 负责何时允许切换，Admission 负责这一批选择哪个 model，KV 只在给定 model namespace 内操作。

### 5.2 Speculative decoding

| 组合 | 判定 | 要求 |
|---|---|---|
| `Standard + S6 DraftVerify` | `×` | 没有 draft/target/accept 协议时，`DraftVerify` 这个名字没有语义 |
| `Speculative + S6 DraftVerify` | `●` | KV 必须支持 `VariableAdvance(accepted_tokens)` 和 rollback/discard proposal |
| `Speculative + S0/S1` | `○` | ArchExec 可把 draft/verify 暴露为同一 iteration 内的多个 phase |
| `Speculative + S2/S3/S4` | `○` | attention/FFN shard 作为分布式 speculative pipeline participant |
| `Speculative + S5` | `○` | multimodal encoder state 可在 draft/target 间共享或分别命名 |

Speculative 不能通过复制 `FullKv` 来支持。它改变的是一次 iteration 的 state 增长量和 tentative state 生命周期，因此应通过 capability 与 FSM action 表达。

## 6. Capability 组合

### 6.1 KV capabilities

| Capability | `K0` | `K1` | `K2` | `K3` | `K4` |
|---|---:|---:|---:|---:|---:|
| `Held` | × | ○ | ○ | ○ per model | ○ per child |
| `Prefix` | × | ○ | ○ per state kind | ○ per model | ○ decoder child |
| `SubsetAdvance(AdvanceScope::RequestSubset)` | ○ no-op | ● for slot FSM | ● for slot FSM | ● delegated | ● delegated |
| `VariableAdvance(n)` | ○ no-op | ● for speculative | ● per state kind | ● delegated | ● delegated |
| `ModelNamespace` | ○ no state | △ adapter | △ adapter | ● | △ adapter |
| `AuxState` | △ external | △ external | △ external | ○ child-dependent | ● |

`Held` 与 `Prefix` 施加到 `NoKv` 才是完全无意义的组合。其他缺失项都应作为 capability gap 暴露，而不是靠 concrete worker 名称写死。

Prefix replacement 的输入边界必须固定：

| Policy | 允许读取 | 不允许拥有 |
|---|---|---|
| `RP0 NoEvict` | insertion 是否有空间 | request lifecycle、ArchInput |
| `RP1 Lru` | entry last-access timestamp | Admission queue |
| `RP2 Lfu` | aged hit count | concrete worker/FSM state |
| `RP3 SizeAware` | entry bytes、reuse score | model-specific ArchInput |
| `RP4 RecomputeCostAware` | entry bytes、预测重算成本、复用概率 | concrete Admission 类型 |
| `RP5 SloAware` | opaque protection/utility score、entry bytes | deadline/priority 的业务解释 |

各 replacement 可以定义自己的 per-entry metadata、aging/global state、typed context 与 feedback；公共 prefix entry 不应携带所有 policy 的 superset 字段。`RP5` 的 SLO utility 由 admission/controller 侧算好后作为 opaque score 更新到它自己的 entry state；KV replacement 只比较 score 与 bytes。这样 KV 不依赖 `TtftAware`、`TbtAware` 等具体 policy。Session pin/TTL 是 eviction eligibility filter，可叠加在任一 `RP1`–`RP5` 上，不应再复制 replacement。

### 6.2 Admission modifiers

| Modifier | 合法基础 admission | 必需条件 |
|---|---|---|
| `PrefixAware` | `A1/A2/A4/A5/A6` | KV 有 `Prefix`；prefix locality placement 仍属于 L6 |
| `PendingOrderPolicy=AP1..AP6` | `A1`–`A6` | policy-specific typed context/query/state/feedback；必须定义 starvation、tie-break 与 prediction miss 行为 |
| `ActiveModel` | `A1`–`A6` | `MultiArch`；KV 有 model namespace 或 `K0` |
| `TokenBudget` | `A1/A2/A4/A5/A6` | 对 fresh/chunk/encoder token 给出明确计量单位 |
| `BacklogGate` | `A1`–`A6` | 只影响何时 form batch，不改变 KV footprint |

不同 SLO policy 的最低预测输入为：

| Policy | 最低输入 | prediction miss 时的确定性降级 |
|---|---|---|
| `AP1 StaticPriority` | priority class、arrival sequence | 同 priority FIFO |
| `AP2 EarliestDeadline` | absolute deadline、arrival sequence | 无 deadline 请求进入 best-effort class |
| `AP3 PredictedSlack` | deadline、predicted remaining service | 降级为 `AP2` |
| `AP4 TtftAware` | arrival、是否已出首 token、predicted prefill/next-token time | 等待时间最长优先 |
| `AP5 TbtAware` | last-token time、target TBT、predicted next-token time | 距上次 token 最久优先 |
| `AP6 CompositeSlo` | 所选 objective 的完整字段 | 按文档化的下一层 objective，最终 FIFO |

`headroom/tentative admission` 不在基础空间中。若将来确实需要为 transfer 或系统保留容量，它应是 KV 构造时的 pool reserve，或 admission 的显式 `ReservedCapacityPolicy`；不能以一个含糊的 `Tentative` 模式同时改变两边语义。

### 6.3 L6 prefix placement capability

`PrefixAware` 不只影响本 worker admission。为了让 L6 做 cross-worker prefix locality/session-affinity placement，prefix-aware worker 必须额外实现窄的 `PrefixPlacementWorker` capability，按 request/model 查询只读 placement quote。

quote 至少包含 matched tokens、reusable/incremental bytes、session affinity、eviction-required bytes、projected/capacity bytes、fits-now hint、worker status 与 cache epoch；具体模型可使用 associated Quote 扩展，不能暴露 radix node、entry id、replacement metadata 或 commit handle，也不能把这些字段塞进基础 `IterWorker::status()`。

普通 worker/pool 不实现这个 capability。Prefix-aware L6 pool 对 `W` 增加 capability bound，逐 worker 查询后由 typed placement policy 选路。quote 是 hint，不是 reservation；最终 worker admission 必须重新 `propose → try_reserve` 并处理 stale epoch。

## 7. 所有硬排除项

除以下情况外，基础 tuple 全部保留：

1. `M10/M11 FfnShard* × K1/K2/K3/K4`：纯 FFN shard 不拥有 decoder KV；
2. `M10/M11 FfnShard* × A1..A6`：纯 FFN shard 不拥有 request admission；
3. `M10/M11 FfnShard* × S5 EncoderPipeline`：纯 FFN shard 不执行 encoder/media stage；
4. `M12 EncoderMediaOnly × K1/K2`：非 autoregressive encoder-only worker 不拥有 decoder KV；
5. `M12 EncoderMediaOnly × A1/A2/A3/A5`：这些 admission 的契约明确依赖 decoder KV 生命周期；
6. `M12 EncoderMediaOnly × S6 DraftVerify`：encoder-only worker 不拥有 token draft/verify/accept 协议；
7. `K0 NoKv × A2/A3/A5`：没有本地 KV 却声称 handoff、pull/commit 或 reserve KV；
8. `Standard × S6 DraftVerify`：没有 speculative protocol 却选择 draft/verify FSM；
9. `K0 × Held/Prefix`：不存在可 hold 或共享的本地 KV state。
10. `A0 NoAdmission × worker-local AP1..AP6`：worker 声称不拥有 admission，却又在本地做 SLO 排序；SLO 只能在上游 task producer；
11. `PC0 NoPrefixCache × RP0..RP5`：没有 prefix entries 时 replacement policy 没有对象；
12. `K0 NoKv × PC1 PrefixCache`：没有本地 state 时不能维护本地 shared-prefix cache。

这些排除项都是定义矛盾。下列理由**一律不允许**用于排除：

- 当前没有对应 server preset；
- production worker 还没实现；
- 需要新的 ArchInput；
- 需要新的 grouping；
- 只在未来模型中出现；
- 当前测试不覆盖；
- 组合不符合现有六个 worker 的名字。

## 8. ArchInput 与 grouping 的所有权

为了让上述组合真的可替换，数据流必须保持：

```text
Admission ──选择 request / lifecycle──┐
                                      ├─> concrete worker / ArchInputBuilder ─> ArchInput
FSM ──当前 phase + AdvanceScope───────────┤
                                      │
KV ──长度、footprint、state facts─────┘
```

- Admission 回答“哪些 request 现在可以进入、属于什么 lifecycle”；
- FSM 回答“当前执行哪个 phase、哪一组 request/task”；
- KV 回答“这组 request 的 state 事实以及 reserve/advance/release 是否成功”；
- concrete worker（或与 model contract 同属一侧的 `ArchInputBuilder`）组合这些信息形成特定模型的 `ArchInput`。

因此不能有 `Kv::build_arch_input()`。否则 speculative、hybrid-state 或 model-specific input 一出现，就必须复制或修改 KV 实现。FSM 应传 grouping 信息，但不应要求 KV 理解 slot、draft 或 verify 的业务含义。

当前相关代码可从这些入口核对，而不是随机跳文件：

- KV seam：[sketch/worker_compose/kv/mod.rs](../../sketch/worker_compose/kv/mod.rs)
- prefix-cache decorator：[sketch/worker_compose/kv/prefix.rs](../../sketch/worker_compose/kv/prefix.rs)
- Admission seam：[sketch/worker_compose/admission/mod.rs](../../sketch/worker_compose/admission/mod.rs)
- iter-wise FSM 入口：[simulator/src/worker/iter_worker.rs](../../simulator/src/worker/iter_worker.rs)
- AFD attention slot FSM：[simulator/src/worker/disagg_attn.rs](../../simulator/src/worker/disagg_attn.rs)
- AFD FFN double buffer：[simulator/src/worker/disagg_ffn.rs](../../simulator/src/worker/disagg_ffn.rs)

## 9. 用于决定 FSM 是否抽离的最小压力测试

不要再按现有 worker 名字复制实现。至少要让以下 off-diagonal tuple 通过类型设计和独立单测：

| # | Model | KV | Admission | FSM | 验证点 |
|---|---|---|---|---|---|
| `T0` | `M0 TextDenseFull` | `K1 FullKv` | `A1 LocalPrefillDecodeAdmission` | `S0 Iter` | 基线，不证明可组合性 |
| `T1` | `M1 TextMoeFull` | `K1 FullKv` | `A4 ChunkedPrefillAdmission` | `S0 Iter` | MoE 不应复制 admission/KV |
| `T2` | `M3 TextMoeHybrid` | `K2 HybridStateKv` | `A1 LocalPrefillDecodeAdmission` | `S0 Iter` | Hybrid state 与 MoE 正交 |
| `T3` | `M8 AttentionShardFull` | `K2 HybridStateKv` | `A5 FreshRequestSlotAdmission` | `S2 SlotPipeline` | AFD 可以替换为 HybridStateKv，不复制 worker |
| `T4` | `M9 AttentionShardHybrid` | `K2 HybridStateKv` | `A3 DecodeDirect` | `S3 SlotPipelinePull` | AFD + PD + Hybrid 的交叉点 |
| `T5` | `M11 FfnShardMoe` | `K0 NoKv` | `A0 NoAdmission` | `S4 DoubleBufferTask` | task-fed FFN 边界保持纯净 |
| `T6` | `M7 MultimodalMoeHybrid` | `K4 CompositeKv` | `A6 EncodeThenPrefill` | `S5 EncoderPipeline` | multimodal/MoE/hybrid 同时存在 |
| `T7` | `M2 TextDenseHybrid + MultiArch` | `K3 ModelPartitionedKv<K2>` | `A1 + ActiveModel` | `S0 Iter` | active model 不进入 KV 业务逻辑 |
| `T8` | `M3 TextMoeHybrid + Speculative` | `K2 + VariableAdvance` | `A4 ChunkedPrefillAdmission` | `S6 DraftVerify` | accepted length 不要求复制 KV |
| `T9` | `M7 + MultiArch + Speculative` | `K3<K4>` | `A6 + ActiveModel` | `S6 DraftVerify` | 最远未来交叉组合仍能表达 |
| `T10` | `M1 TextMoeFull` | `K1 + PC1<RP1 Lru>` | `A1 + AP3 PredictedSlack` | `S0 Iter` | SLO 排序与 prefix replacement 不互相依赖 concrete 类型 |
| `T11` | `M7 MultimodalMoeHybrid` | `K4<decoder=K2+PC1<RP4>>` | `A6 + AP6 CompositeSlo` | `S5 EncoderPipeline` | hybrid/multimodal prefix benefit 与 SLO prediction 可同时表达 |

判断标准：

- 若 `T0`–`T11` 需要为每个 tuple 新写一个 concrete worker，当前分层失败；
- 若 KV 开始生成 model-specific `ArchInput`，KV seam 失败；
- 若 Admission 必须 match concrete KV 类型而不是 capability，Admission seam 失败；
- 若多个 FSM 的状态/action 无法共享任何稳定契约，让 concrete worker 直接拥有各自 FSM 可能更简单；
- 若同一个 Model/KV/Admission 能无侵入替换 `S0/S1/S2/S3/S6`，独立 FSM 轴才真正“挣得”其复杂度。

## 10. 当前结论（不是最终决定）

这份矩阵不能单独证明必须写成 `Worker<K, A, X, S>`。它只证明：

1. Model、KV、Admission 之间存在大量 off-diagonal 合法组合，不能回到按 worker family 硬编码；
2. FSM 行为也存在真实交叉组合，但是否用泛型类型参数、trait object、短 concrete worker 或私有 helper，必须由 `T0`–`T11` 的实现复杂度决定；
3. 无论 FSM 是否抽离，`ArchInput` 都应由 concrete worker/model-side builder 形成，KV 只接收额外 grouping 并返回 state facts；
4. SLO-aware admission 与 prefix-cache replacement 都是 policy 子轴，不能分别复制 lifecycle 或 state-layout 实现；
5. 可以有多个短 worker variant；“可组合”不等于“一个万能 worker”。

在压力测试前，不再把 FSM 轴写成已定结论，也不再以当前 production worker 的形状否定未来组合。


## 15. Unknown-variant extension test

`T0`–`T11` 只能证明已知组合；真正的抽象质量由一个实现时才第一次看到的 `V_new` 判断。目标变化集是：

```text
Δ(V_new) = new leaf implementation
         + genuinely necessary thin adapter
         + declarative registration
         + focused tests
```

不应包含：修改现有 worker/KV/Admission/IterModelExecution 的 behavior body，或复制它们的 queue、ledger、FSM、input lowering。

### 15.1 变化半径必须与语义新颖度成比例

| 未知变化 | 允许新增 | 必须保持不变 |
|---|---|---|
| 新 KV layout，例如 compressed/tiered state | 一个 layout + 所需 read-view impl | 现有 layout、Admission、shell、replacement |
| 新 prefix replacement，例如 learned/Belady-like | 一个 `PrefixReplacement`，含自己的 EntryState/Context/Feedback | prefix index/ledger、KV layouts、workers |
| 新 admission lifecycle | 一个 lifecycle + 对应 Work | existing policies、KV、已有 shell；除非 cadence 真变了 |
| 新 SLO/fairness policy | 一个 typed policy + context/query/feedback + estimator impl | lifecycle、KV、shell、其他 policies |
| 新 cross-worker prefix placement policy | 一个 L6 typed placement policy + context/feedback | prefix cache、worker shell、Admission、基础 IterWorker |
| 新模型或新 Input | Input + IterModelExecution/estimator adapter | worker FSM、Admission、KV resource logic |
| 新执行 cadence，例如 preempt/resume | 一个 workers/state core | 所有 KV/lifecycle/policy/model adapters，只添加必要 capability bounds |
| 此前不存在的资源操作 | 一个窄 capability trait + 支持它的 leaf impl | 不支持者不加 dummy method，不扩大基础 trait |

若一个“只换 policy”的 variant 需要修改 lifecycle，或“只换 model Input”需要复制 worker，就直接判失败。反过来，真正新增 preemption/rollback 语义时允许增加 capability/shell；强迫它零文件变化只会产生 mega-interface。

### 15.2 什么叫“不改 existing code”

必须满足的 strong baseline：

1. existing behavioral implementation body 零修改；
2. existing behavior tests 零修改且继续通过；
3. 新 variant 删除后，旧系统仍独立编译/运行；
4. 只允许在 declarative provider/registry 增加一行或一个新 provider module；
5. 若新增 variant 需要改中央 match ladder 的多处分支，registration design 失败。

更强的理想形态是 distributed/generated provider registry，使新 module 自注册，从而 existing file 也零修改。是否引入这种 registry 是 build/schema 层的决定；在它落地前，单一 declarative registry row 是唯一允许的现有文件改动，不能夹带行为逻辑。

### 15.3 舒适度与去重检查

每个未知 variant review 都回答：

- 新文件中有多少代码是在描述该 variant 独有语义？
- 是否重新出现现有 `tick` loop、pending queue、reservation map、prefix tree 或 ArchInput lowering？出现即失败；
- 是否可以复用该轴的 generic contract-test suite，只新增 variant-specific case？
- 缺少 capability 时，编译错误是否落在新 recipe/adapter，而不是迫使所有旧 impl 加空方法？
- wiring diff 是否只有 provider/schema registration？

建议在正式迁移前保留五个**不参与接口设计**的盲测题，等 trait 定型后才交给实现者：

1. GPU/CPU 两级、可压缩的 KV layout；
2. 带 tenant virtual time 与 completion feedback 的 fairness/SLO policy；
3. 拥有独立模型状态的 learned prefix replacement；
4. 同时需要 visual encoder state 与 recurrent decoder state 的新 Model Input；
5. 支持 mid-iteration preempt/resume 的新 cadence。

如果实现者能只加 leaf/adapter/provider/test，而且没有阅读后复制 existing worker body，抽象才通过 unknown-variant test。