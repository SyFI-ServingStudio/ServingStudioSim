# L5 Worker 组合式重构 — 最终设计(L5_redesign)

- **状态:validated & locked**。本文是 L5 worker 组合式重构的**单一权威文档**,合并并取代下列已归档文档
  (移至 repo 外 `../../../L5_old_docs/`,保留 git 历史备查):
  `L5_worker_compose_refactor.md`(设计起点/迁移计划)· `L5_worker_compose_interfaces.md`(接口缝定义)·
  `L5_worker_compose_module_apis.md`(模块 API 清单)· `L5_worker_compose_solution.md`(退休的两阶段事务方案)·
  `L5_unified_experiment.md`(barebone 垂直切片实验)· `L5_worker_compose_decision.md`(对账+决定,已并入本文)。
- **仍在 repo 的伴随文档**:[compatibility_matrix.md](L5_worker_compose_compatibility_matrix.md)(§15 未知变体测试 + T0–T11 fit-test 矩阵,fit-test 参考)。
- **实测代码(as-built 事实源)**:`sketch/worker_v2/`(5200 行,`#[cfg(test)]`,`cargo check` 全绿)+ `census.rs`(编译断言的覆盖证明)。
- **一句话**:拍板 commit 四轴 + 能力子 trait 的形状。它对它 scope 的问题是对的;压测找到的每条边界几乎都是**本设计本就画过、sketch 为首个里程碑简化掉的缝** —— 补回是"恢复已设计的缝",不是重设计。

> 阅读顺序:决策者读 Part 0–I + VI;实现者读 Part III + VII;要恢复某条缓建缝时读 Part IV。

---

# Part 0 — 背景与动机

## 0.1 一句话

把 6 个结构相似但重复严重的 worker(barebone / hp_unified / pd_prefill / pd_decode / disagg_attn / disagg_ffn)改成**正交轴的组合**,让未来每个特性只落一条轴,不重写一大片。

## 0.2 要支持的特性(未来落轴目标)

| 特性 | 主落轴 |
|---|---|
| prefix-cache 感知 KV + 准入(F1) | KvStore(`PrefixCacheKv` 能力)+ Admission(`PrefixPrefillDecodeAdmission`) |
| 多轮对话 / prefix tree(F2) | 版本 A:`PrefixCacheKv` 子 trait(轴内);版本 B:L6 + 专用 pool(轴外) |
| chunked prefill(F3) | Admission(`ChunkedPrefillAdmission`)+ `ChunkedPrefillKv` 能力 |
| mixed-attn 多类型 KV(F4,full/SWA/SSM·GDN/MLA) | KvStore 各独立 impl(`HybridStateKv`)+ `HybridKvView` |
| multi-arch worker(F5) | `ModelSwitchKv` 能力 + `SwitchModel` msg + `MultiModelIterExecution` |
| speculative decoding(F6,MTP) | S6 `DraftVerifyWorker` + `DraftVerifyExecution` + `SpeculativeKv` tentative commit/discard |
| PD-for-AFD 三池拆分(F7) | 拓扑(复用 prefill/ffn + 新 decode-attn shell) |
| 并行布局(TP/DP/HP/EP/CP) | `PartitionId` topology(config 轴,不新增接口) |

---

# Part I — 决定与结论(TL;DR)

- **四轴分解成立**:`Worker = ⟨KvStore, Admission, ModelExecution⟩ × Shell(cadence)`,`KvStore` 是唯一跨家族统一轴,Admission/Execution 使用 per-family trait 面,FSM 由 shell 自持(不做泛型轴)。**22(+2) server / 18 distinct type 在 `census.rs` 编译断言全绿**。
- **对它 scope 完备**:结构性/声明式/无状态变化(新 KV layout、新 cadence、新 policy、新 model/Input)都是真 leaf。
- **边界干净且可预期**:唯一过不去的是**反馈驱动/在线学习**类变化(虚拟时间公平、learned eviction)。**关键:这条缝本设计 Part IV.1 本就画了(`type Feedback` + `on_feedback` + Oracle),是 sketch 简化掉的** —— 补回=恢复,不是重构。
- **fixability 收敛**:所有 break 的爆炸形态止步于"机械 reshape"或"加新子轴",没有一个逼到"重定轴"。
- **迁移前第一待办**:把 Part IV.1 的 feedback 缝恢复到接口(SLO/公平必然跨的边界,先画后建)。

---

# Part II — 组件切分与不可动摇的裁决

## 2.1 五组件 + 可组合形状

| 组件 | 一句话职责 | 组合方式 |
|---|---|---|
| `KvStore` | state 事实 + reserve/commit/advance/release 原语 | 静态 type 参数(layout × prefix × replacement × topology) |
| `Admission` | 谁能进、属什么 lifecycle、按什么 policy 排序 | 静态 type 参数(lifecycle × selection) |
| `IterModelExecution` | 消费 grouping+KV 事实 → 拼 `Input` → 出 `Time`;**拥有 ArchInput 拼装** | 静态 type 参数 |
| `WorkerShell` | cadence(tick/slot/pull/double-buffer)+ 整合三轴 | **少数 concrete shell**,不是泛型轴 |
| `PlacementSeam` | L6 跨 worker prefix/亲和路由的只读窄口 | 可选 capability trait(⏸️ 缓建,见 Part IV.2) |

可组合泛型参数 = `⟨Kv, Admission, IterModelExecution⟩`;`WorkerShell` 是 concrete 外层,picks 一个 cadence。

## 2.2 三条不可动摇的裁决(直接来自兼容性矩阵)

1. **无 `KvStore::build_arch_input`**:拼装者是 `IterModelExecution::build_iteration_input`,住模型侧;shell 只给 `AdvanceScope`。✅ 兑现。
2. **`Footprint` 对 Admission 不透明**:Admission 不对它做算术、不重造容量判断;但压力/命中/可行性是 KV 算好的**中性事实**(`KvCapacityPressure` token 当量),policy 可读。✅ 兑现。
3. **FSM 不做泛型轴**:由少数 concrete shell 拥有 cadence(抽成泛型只会把 worker 劈两半 + 一套等大 host/action API)。✅ 兑现(建 5 shell)。

## 2.3 家族边界(family = cadence)

| 家族 | cadence | worker | admission |
|---|---|---|---|
| **iter** | 整迭代 `IterBatchWorker` | barebone·hp·pd_prefill·chunked·spec·multiarch·prefix·hybrid | LocalPrefillDecodeAdmission/Handoff/Chunked/Prefix/MultiModel |
| **PD-decode** | pull + decode 双时间线 `PullDecodeWorker` | pd_decode | 无(thin,inline) |
| **AFD-attn** | layer-wise slot 流水线 `AttentionSlotPipeline` | disagg_attn·pd-for-afd | SlotPipelineAdmission(FreshRequestSlotAdmission)/无(pull) |
| **AFD-ffn** | double-buffer `BufferedFfnWorker` | disagg_ffn | 无 |

- **`KvStore`(+能力子 trait)是唯一跨家族统一轴**:同一 `FullAttnKv` 被 iter 与 AFD 复用 —— AFD 只是一个 shard partition 被 N 个 slot 共享、用 `advance(AdvanceScope::RequestSubset{..})` 部分推进。`PartitionId ⟂ AdvanceScope` 解耦就是为这个。
- **`IterAdmission`/`SlotPipelineAdmission` 与 `IterModelExecution` 是 per-family trait 面**:cadence 驱动不同 hook 点,方法面按家族不同(iter `IterAdmission{form_batch,complete_iteration}`/`IterModelExecution{evaluate_iteration}`;AFD `SlotPipelineAdmission{reserve_fitting_requests}`/`AttentionLayerExecution{evaluate_attention_layer}`)。传输/通信(`submit_gather`/comm group)归 **shell**,不属任何数据轴。
- **修正实测(vs 设计):家族是 4 不是 3** —— PD-decode 的双时间线 `IterBatchWorker` 单 compute FSM 建不了,是独立 cadence。

## 2.4 轴间耦合:capability bound ≠ 替换

> **Model → KV layout(换 KV impl);Admission → KV capabilities(只加 bound,是同一 KV 多实现的方法子集)。**

- KV 的 **layout**(Full/Hybrid/MLA/SSM)由**模型**决定;换模型才换 KV impl。
- KV 的 **能力**(`ChunkedPrefillKv`/`HandoffKv`/`PrefixCacheKv`/view)是同一 impl **顺带多实现**的方法组;换 admission 只换它调的方法子集,**不换 KV**。一个无法部分落地的 layout 配 `ChunkedPrefillAdmission` 会**编译不过**(类型层契约,非运行时 stub)。

---

# Part III — 最终接口(as-built,locked)

> 图例:⏸️ = 缓建缝的完整设计见 Part IV;其余为 as-built 已建、`census.rs` 编译验证。

## 3.1 共享词汇(`shared/`)
```rust
type PartitionId = u16;                 // KV 资源分区键;≠ deployment PoolId
enum AdvanceScope<'a> {                      // 请求选择,与资源分区正交
    WholePartition(PartitionId),
    RequestSubset { partition: PartitionId, request_ids: &'a [RequestId] },
}
struct WorkerContext { id, pool, requests: SharedRequests, log_output_token_times, log_stage_transitions }
// 跨族共享的唯一 lifecycle 事实 = RequestRecord::{record_first_token, record_token, is_complete}
// 各 family 直接内联调用这些方法;不设独立 wrapper(sketch 曾试 emit_iter_token,因 iter 两相 + spec N-token 结构撤回)
```

## 3.2 KV 轴(`kv/`)—— 唯一跨家族统一轴
```rust
trait KvStore {
    type Footprint;                                             // 不透明
    fn num_partitions(&self) -> usize;
    fn footprint(&self, req, prompt: u32, decode: u32) -> Self::Footprint;   // KvStore 不知角色;角色差异=调用方传的(prompt,decode)
    fn fits(&self, partition, &Self::Footprint) -> bool;        // 唯一容量判定(内部做共享预算/多资源判断)
    fn pressure(&self, partition) -> KvCapacityPressure;                // 中性 token 当量,给准入排序
    fn reserve(&mut self, req, partition, Self::Footprint);
    fn commit_resident(&mut self, req, partition, initial_kv: u64, remaining: u32);  // promised→resident,initial_kv=prompt+prefix
    fn release(&mut self, req, partition);
    fn advance(&mut self, AdvanceScope, steps: u32);                // full +N / recurrent 0 / spec 接受 N
    fn sample_submit(&mut self, partition, now);                // 副作用日志(设计原为 sample()->KvSample,实测改副作用)
}
struct KvCapacityPressure { resident_tokens_equiv, reserved_tokens_equiv, capacity_tokens_equiv }

// 能力子 trait(ISP,只由需要的 impl 实现;admission/exec 按需 bound):
trait IterWorkerKv:  KvStore { … }    // iter 家族窄读:prefill_admits / decode_members / drain_ready / release_external …
trait SlotPipelineKv:   KvStore { current_kv(partition,req)->Option<u64>; estimated_peak(partition)->u64 }   // AFD
trait HandoffKv:       KvStore { hold(partition,req,kv_tokens); drop_held(req) }    // PD prefill(实测须带 kv_tokens,非设计的零参 hold(req))
trait ChunkedPrefillKv:    KvStore { append_prefill_chunk(..); finish_chunked_prefill(..) }
trait PrefixCacheKv:     KvStore { probe_prefix(partition, req, prompt, decode) -> PrefixPlacementProbe<Self::Footprint> } // partition-local quote;完整 sharing/index 见 Part IV.2
trait HybridKvView: KvStore { num_kinds()->u16; state_len(partition,req,kind)->Option<u64> }
trait ModelSwitchKv:   KvStore { set_active(ModelId); active()->ModelId }
trait TieredKvView:     KvStore { resident_by_tier(partition) -> (u64,u64) }         // 压测 #1 新增(ISP 加法,未回改核心 trait)

impl: FullAttnKv(base,真 Batch/KvAdmission)· ModeledPrefixCacheKv · HybridStateKv · ModelPartitionedKv · TieredMemoryKv
      —— 后四者皆薄壳 wrap FullAttnKv + 一能力(复用真 Batch/KvAdmission 机器,委托全部核心方法)
```

**负接口(KvStore 绝不做):** 建 ArchInput;解释角色(prefill/decode/draft/verify);暴露标量 `remaining()`/`capacity`/`projected_peak` 作**决策**(会焊死单一同质池 + 线性增长假设;峰值只在 sample 里作不透明日志)。矩阵子轴 `StateLayout × PrefixMode × Replacement` 全落 KvStore 内部作 type 参数,不新增 KV 品种、不扩核心 trait。

**去掉的设计冗余(vs interfaces):** `NoKv` 占位类型(轴可省用"shell 不吃该轴"表达,如 `BufferedFfnWorker<E>`,非空类型);`KvLogicalView` 单一 view(实测拆成 per-family 的 `IterWorkerKv` + `SlotPipelineKv`)。

## 3.3 Admission 轴(`admission/`)—— per-family,lifecycle × policy
```rust
trait IterAdmission<K: KvStore> {                    // iter-family trait；K 为 trait 参数 → impl 可 where K: 能力
    type Msg; type Event;
    fn accept_message(&mut self, &mut K, Msg, &WorkerContext);         // 入队 + stamp Pending;&mut K 因 PD ReleaseKv 要在 iter 外动 KV
    fn form_batch(&mut self, &mut K, &WorkerContext, now) -> bool;     // token 门 + policy + fits + reserve;stamp Prefill
    fn complete_iteration(&mut self, &mut K, &WorkerContext, &mut Vec<Event>, now);   // 记 token + Decode/Done + commit/advance/release
    fn queued_requests(&self) -> u32;
    fn cancel_pending(&mut self, req) -> bool;                     // 取消的 pending 半(容器配 kv.release_external)
}
trait SlotPipelineAdmission<K: KvStore> {
    enqueue_fresh_request(req, &WorkerContext);
    reserve_fitting_requests(&mut K, &WorkerContext, now) -> Vec<RequestId>;
    cancel_or_release_request(&mut K, req);
    queued_kv_tokens();
    queued_requests();
} // AFD-attn

// policy 自持待选队列,每次只交出一个头(见 III.x 性能理由)。⏸️ 完整(Feedback/Oracle)见 Part IV.1
trait PendingOrderPolicy { type Context; push(AdmissionCandidate,&mut Context); peek()->Option<AdmissionCandidate>; pop(&mut Context)->Option<AdmissionCandidate>; remove(RequestId)->bool; len() }
struct AdmissionCandidate { request, arrival_seq, prompt, decode, deadline: Option<Time>, matched_tokens }  // Copy;在 accept 冻结

impl lifecycle: LocalPrefillDecodeAdmission<P> · DraftVerifyAdmission<P> · ChunkedPrefillAdmission<P>· PrefillHandoffAdmission · PrefixPrefillDecodeAdmission<P> · MultiModelAdmission<P> · FreshRequestSlotAdmission(AFD)
impl policy:    FifoOrder · ShortestJobFirst
```

**负接口:** 不持 IterModelExecution;不建 ArchInput;不对 Footprint 做算术;不 match 具体 KV 类型(只按能力 bound);selection policy 不改 lifecycle stage。SLO metadata(deadline/priority)是 request immutable input(需给 trace/workload schema 加 SloSpec),不放 `WorkerConfig`。

## 3.4 IterModelExecution 轴(`execution/`)—— per-family,三 sibling trait
```rust
trait IterModelExecution<K: IterWorkerKv> {                    // K 为 trait 参数(对齐 IterAdmission<K>;sketch 曾回归成方法泛型,Root ② 已修回,见 Part V.3)
    type Input: Default;                             // 不透明;shell 只持有,不读字段
    fn model_kv_layout(&self) -> ModelKvLayout;           // 喂 KvStore 定容
    fn build_iteration_input(&self, &K, &SharedRequests, &mut Input);         // 拼装者;impl 可 where K: TieredKvView/HybridKvView 升级读能力
    fn evaluate_iteration(&mut self, &Input, iter, now) -> Time;
}
trait AttentionLayerExecution { type Input; num_layers; model_kv_layout; attn_to_ffn_bytes_per_token;
                 build_slot_input<K: SlotPipelineKv>(AdvanceScope, &K, &requests, &mut Input)->u64; evaluate_attention_layer(layer,slot,..)->Time }
trait FfnTaskExecution  { type Input; num_dp_groups; ffn_to_attn_bytes_per_token; build_task_input(tokens,&mut Input); evaluate_ffn_task(kind,slot,..)->Time }  // 无 KV 无 store
trait DraftVerifyModel { model_kv_layout; gpus_per_replica; evaluate_draft_verify(&DraftVerifyInput,iter,now)->Time }
trait AcceptanceOracle { accepted_draft_tokens(request,proposal,remaining)->u32 }
trait DraftVerifyModelExecution<K: IterWorkerKv> { type Input; build_draft_verify_input; collect_proposals; evaluate_draft_verify_iteration->DraftVerifyResult }
struct ModelKvLayout { total_kv_bytes_per_token, num_attn_shards }

impl: UnifiedIterExecution<M> · MultiModelIterExecution<M> · TierAwareIterExecution<M>(bound K: TieredKvView)· AttentionLayerExecutionAdapter<M> · FfnSectionExecutionAdapter<M> · DraftVerifyExecution<M,O>
```

**修正(vs interfaces):** `AttentionLayerExecution`/`FfnTaskExecution` 实测是**三个 sibling trait**(形状不同:FfnTaskExecution 无 KV 无 store),非 `IterModelExecution` 子 trait;`make_kv_intent` 未建未用;`ModelSwitchKv` 归 KvStore 能力(非 IterModelExecution)。**负接口:** 不持 pending 队列/KV 账本/FSM cursor/L6 routing;不 match 具体 KV impl(只约束 read-view 能力)。

## 3.5 Shell 轴(`workers/<family>/`)—— cadence,自持 FSM
```rust
IterBatchWorker<K, A, E>        where K: KvStore+IterWorkerKv, A: IterAdmission<K>,     E: IterModelExecution<K>   // iter
PullDecodeWorker<K, E>          where K: KvStore+IterWorkerKv,                      E: IterModelExecution<K>   // PD decode(pull+decode 双时间线)
SlotAttentionWorker<K, A, E>     where K: KvStore+SlotPipelineKv,  A: SlotPipelineAdmission<K>, E: AttentionLayerExecution       // AFD attn(穿 AttentionSlotPipeline)
PullSlotAttentionWorker<K, E>    where K: KvStore+SlotPipelineKv,                       E: AttentionLayerExecution       // PD-for-AFD(穿 AttentionSlotPipeline,无 admission)
BufferedFfnWorker<E>            where E: FfnTaskExecution                                                     // AFD ffn(无 KV 无 admission)
DraftVerifyWorker<K,A,E>        where K: SpeculativeKv, A: DraftVerifyAdmissionLifecycle<K>, E: DraftVerifyModelExecution<K> // S6
AttentionSlotPipeline<K, E>            // SlotAttentionWorker 与 PullSlotAttentionWorker 共享的私有 slot 流水线 + comm seam(不嵌 concrete worker)
```

**文件边界锁定:** cadence family 是 `workers/` 下的一级目录:
`iter/`、`draft_verify/`、`pd_decode/`、`afd_attention/`、`afd_ffn/`。worker/FSM 与共享的
family-private pipeline 各住自己的实现文件;每个具体 `build_*` construction
recipe 住一个同名文件。family 是行为所有权边界,build file 是组合 recipe
边界,避免新增组合持续堆进一个巨型 worker 文件。重复的
allocation/capacity/sampler/cost/context 翻译可进入 family-private build
essentials,但它不选择具体 K wrapper、admission/policy 或 ingress;这些选择
必须留在 build file,且 build file 显式 import 自己的依赖。

对 L6:经 blanket impl 兑现现有 `IterWorker`(`id`/`enqueue`/`tick`/`status`)+ `AfdAttnWorker`,**L6 接口不变**。**新增 shell 门槛**:仅当 ① L6-facing Msg/Event protocol、② compute/transfer/slot overlap cadence、③ completion ownership/barrier、④ 需同时协调两 state core 之一变化才新增。S6 linear draft/verify 已建;tree proposal 可在同一 family 扩展。**仍未建(⏸️)**:`EncoderPipeline`(S5 multimodal)。

## 3.6 负接口汇总(缝的定义)

| 组件 | 绝不做 |
|---|---|
| `KvStore` | 建 ArchInput;解释角色;暴露标量容量/峰值作决策;replacement 读 lifecycle/deadline 业务含义 |
| `Admission` | 持 IterModelExecution;建 ArchInput;对 Footprint 做算术;match 具体 KV 类型;selection 改 lifecycle stage |
| `IterModelExecution` | 持 pending 队列/KV 账本/FSM cursor/L6 routing;match 具体 KV impl |
| `WorkerShell` | 读 `E::Input` 字段;对 Footprint 做算术;存 request→slot 之外的 KV 账本;嵌另一 concrete shell |
| `PlacementSeam` | 露 KV 内部结构;改 access 态;扩胖基础 `IterWorker::status()` |

## 3.7 明确丢弃:`propose → try_reserve + revision`

旧 solution 的乐观并发事务(`propose` `&self` 试算吐 `KvProposal{token,observation+revision}` → `try_reserve` 回校验 revision,过期 `Err(Stale)` 重来)防的是**真并发/async scheduler** 里 propose 与 reserve 之间的漂移窗口。**VibeSim 是单线程离散事件模拟**,一次 `form_batch` tick 内无别的东西改 KV,漂移窗口不存在。其唯一正当用例("policy 只读评估 N 候选、挑最优、只对那个提交")本方案天然支持:`fits`/`pressure`/`probe_prefix`(`&self`)可对多候选连问,`reserve`(`&mut self`)只对选中那个调一次。→ 整套 propose/observation/try_reserve/revision + `KvManager<L,P,T>` 三层内部 trait 全砍,换成一步式 `fits → reserve`。**将来若真要模拟并行/推测准入并带回滚,再按需引入两阶段。**

---

# Part IV — 缓建缝的完整设计(内联保存,归档后不丢)

> 这些是**本设计画过、sketch 为首个里程碑简化掉**的缝。压测的边界(Part V.2 的 #2/#3)正撞在它们上。恢复=按此实现,不是重设计。**迁移待办 #1 就是恢复 IV.1。**

## IV.1 反馈驱动的 PendingOrderPolicy(Feedback + Oracle + Estimator)

sketch 现状是 `PendingOrderPolicy { push / peek / pop / remove / len }`(policy 自持队列,无反馈)。**完整设计:**
```rust
trait PendingOrderPolicy<AdmissionCandidate, Oracle> {
    type Context;    // per-worker 内部态(starvation debt / prediction cache / tenant virtual time)
    type Feedback;   // 完成/违约反馈 ← 盲测 #2/#3 死在缺这个
    fn on_enqueue(&mut self, cand: &AdmissionCandidate, ctx: &mut Self::Context);
    fn select(&mut self, cands: &[AdmissionCandidate], ctx: &mut Self::Context, oracle: &mut Oracle) -> SelectionPlan;
    fn on_feedback(&mut self, fb: Self::Feedback, ctx: &mut Self::Context);   // 完成路径回灌:虚拟时间/命中率在此更新
}
// Oracle 由 Admission engine 临时建在 &kv + &estimator 上,全 &self 只读:
oracle.fits(cand)->bool | oracle.pressure(part)->KvCapacityPressure | oracle.prefix(cand)->PrefixMatch(仅 K:PrefixCacheKv) | oracle.predict(cand,q)->Estimate(仅带 estimator)
// 预测经 typed estimator(住模型侧,policy 不持模型):
trait AdmissionEstimator<K, Work, Query> { type Estimate; fn estimate(&mut self, &Work, &Query, &K, &RequestStore) -> Self::Estimate; }
```
policy 显式选 `StopAtFirstReject`(HOL,保 production 行为)或 `SkipInfeasible`(须带 starvation guard)。

**⚠️ 归属修订(2026-07-25):** 本节原写"policy 的加速结构只存 request reference/generation;**lifecycle** 的 pending membership 是唯一事实源 —— policy index 不成第二份 stage 账本"。sketch 的性能修复把 membership **整体移进 policy**(`push`/`peek`/`pop`/`remove`/`len`),lifecycle 不再持 `pending`。原则的**意图未变且仍成立**:待选 membership 仍然只有**一份**事实源,只是从 lifecycle 换成了 policy;要防的"两份 stage 账本互相漂移"依然被禁止。改归属的理由是原形状 `order(&[AdmissionCandidate])->Vec<usize>` 强制每迭代物化+排序整个 backlog 只为取一个头(2 次堆分配 + O(N log N) + O(N×M) `retain`),而生产 `unified.rs` 是 `front()` O(1);把队列交给 policy 后 `FifoOrder`=VecDeque O(1)、`ShortestJobFirst`=BinaryHeap O(log N) 增量维序。`ChunkedPrefillAdmission` 也遵守这条归属:policy 持有所有尚未开始的 fresh prefill,lifecycle 只额外持有 `active_chunk: Option<AdmissionCandidate>`。一个 partition 的同一 iteration 可以装入多个完整 prefill,但 token budget 的边界至多切出一个跨 iteration 的 partial prefill;下一 iteration 必须先续完这个 `active_chunk`,剩余预算才继续从 policy 取 fresh head。由此 `ShortestJobFirst` 决定"下一个开始谁",但不会重排或抢占已经开始的 chunked prefill。

**恢复动作(一次性缝扩展,B 类):** 给 `PendingOrderPolicy` 补 `type Feedback`+`on_feedback`(带 no-op 默认,FifoOrder/ShortestJobFirst 不改)+ Oracle;`AdmissionCandidate` 补 tenant/来源(SloSpec);完成路径一处调用。

## IV.2 prefix cache 完整设计(index / sharing / replacement)

sketch 现状已经把 placement 窄缝落成 `PrefixCacheKv::probe_prefix(partition,req,prompt,decode)->PrefixPlacementProbe<Footprint>`:probe 把 partition-local 命中量与同一快照算出的 opaque footprint 绑在一起,`PrefixPrefillDecodeAdmission` 从可容纳的 partition 中先按命中量、再按 normalized pressure 选择,随后 `reserve` 固化 request→partition stickiness。当前 `ModeledPrefixCacheKv` 仍只是 modeled hit%:cached prefix 与 request KV 走同一个容量门,但没有 radix index/refcount,所以 matched region 保守地按 request 重复计费。**完整 sharing/replacement 设计(设计过、未建):**
```rust
trait PrefixCacheKv: KvStore {
    fn match_prefix(&self, req) -> PrefixMatch;              // 本地命中:matched_tokens(中性)+ 不透明 handle
    fn insert_prefix(&mut self, req);                        // prefill 完成把 prefix 注册进共享 index
    fn probe_prefix(&self, partition, &PrefixLookup) -> PrefixPlacementProbe;   // L6 只读 quote(见 Part IV.3)
    fn update_prefix_utility(&mut self, PrefixHandle, opaque_score: f32);   // SLO utility 外部算好塞入
}
struct PrefixMatch { matched_tokens: u32 /*policy 可读中性*/, handle: PrefixHandle /*不透明,仅回传*/ }
```
**KV 内部三件套(module_apis §9–11,归 KvManager 内部,不跨模块):**
- **PrefixSharing**:read-only match / retain-release refs / publish / reclaim plan+apply(只从合法未引用 entry 选)/ read-only placement probe。不解释 deadline/TTFT,不读 Admission queue,不向 L6 露 node/entry/cache handle。
- **PrefixReplacement**(PrefixSharing 的 child policy,RP0–RP5):on-insert 初始化 per-entry metadata / on-committed-access 更新 LRU/LFU/cost/SLO 态 / choose-victims(从已过滤 eligible 里选)/ feedback(opaque typed)。**learned eviction(盲测 #3)就落这里**,靠 feedback 通道(同 IV.1 根因)。
- **KvTopology**:validate/enumerate placement / project footprint(logical→partitions/per-HP-rank)/ pressure / reserve-apply-release delta。必须表达的 topology:single pool · attention-DP pools(request sticky 到一 DP group,横跨其 HP ranks)· per-model(`ModelId→InnerTopology`)。

**replacement 负接口:** 只读 opaque per-entry metadata + bytes + eviction eligibility;**不**读 request lifecycle / ArchInput / deadline 业务含义(SLO 由外部算成 opaque utility 经 `update_prefix_utility` 塞入)。矩阵 `Replacement RP0–RP5` 落 `ModeledPrefixCacheKv<Base, R>` typed 参数,共用 index/refcount,不新增 KV 品种。

## IV.3 PlacementSeam(L6 跨 worker,窄能力)
```rust
struct PrefixPlacementQuery { request: RequestId, model: ModelId }
trait PrefixPlacementWorker: IterWorker {
    type Quote;   // matched tokens / reusable+incremental bytes / session affinity / fits-now hint / cache epoch
    fn prefix_placement_quote(&self, &PrefixPlacementQuery) -> Self::Quote;   // 只读,不改 access 态
}
```
只读 quote(hint,非 reservation);内部只调 `PrefixCacheKv::probe_prefix` + `status()`;**不**露 radix node/entry id/replacement metadata/commit handle,**不**塞进基础 `IterWorker::status()`。普通 pool 不实现(无空 stub)。L6 选中 worker 后 Admission **仍须重走** `form_batch`(`fits→reserve`);quote 是 hint,epoch 变了即失效。多轮对话 prefix tree 的**版本 B**(全局专用 prefix pool,类 Mooncake/LMCache)主要住 L6 + 新 pool,碰 worker 仅**放置输入** + 可选**从 store 拉前缀 transfer**(复用 PD pull),不改核心 worker 接口。

---

# Part V — 验证

## V.1 census —— 22(+2) server / 18 distinct type

`census.rs` 用未被调用的泛型 fn 逼编译器 discharge 每个组合的 L6 worker-trait bound(`IterWorker`/`AfdAttnWorker`);**`cargo check` 通过即证明可组合**,无需实例化 concrete model。**server** = (composition, config) 对,多 server 共享一 TYPE(barebone vs HP=N;spec 与 spec-DP=N;policy swap=`P`)。

| # | server | composition(type) |
|---|---|---|
| 1–2 | dense / HP-DP(N) | `IterBatchWorker<FullAttnKv, LocalPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution>` |
| 3 | latency-scheduled(SJF) | `IterBatchWorker<FullAttnKv, LocalPrefillDecodeAdmission<ShortestJobFirst>, UnifiedIterExecution>` |
| 4–5 | speculative(MTP)/ +DP | `DraftVerifyWorker<FullAttnKv,DraftVerifyAdmission<FifoOrder>,DraftVerifyExecution<M,O>>` |
| 6–7 | chunked / +SJF | `IterBatchWorker<FullAttnKv, ChunkedPrefillAdmission<FifoOrder\|ShortestJobFirst>, UnifiedIterExecution>` |
| 8 | PD prefill(handoff) | `IterBatchWorker<FullAttnKv, PrefillHandoffAdmission, UnifiedIterExecution>` |
| 9 | PD decode(pull→decode) | `PullDecodeWorker<FullAttnKv, UnifiedIterExecution>` |
| 10–11 | prefix-cache / +SJF | `IterBatchWorker<ModeledPrefixCacheKv, PrefixPrefillDecodeAdmission<FifoOrder\|ShortestJobFirst>, UnifiedIterExecution>` |
| 12–14 | hybrid / hybrid-DP / +SJF | `IterBatchWorker<HybridStateKv, LocalPrefillDecodeAdmission<FifoOrder\|ShortestJobFirst>, UnifiedIterExecution>` |
| 15–17 | multi-model(2/4)/ +SJF | `IterBatchWorker<ModelPartitionedKv, MultiModelAdmission<FifoOrder\|ShortestJobFirst>, MultiModelIterExecution>` |
| 18 | AFD attn | `SlotAttentionWorker<FullAttnKv, FreshRequestSlotAdmission, AttentionLayerExecutionAdapter>`(→ `AttentionSlotPipeline`) |
| 19 | AFD ffn | `BufferedFfnWorker<FfnSectionExecutionAdapter>` |
| 20 | PD-AFD decode-attn | `PullSlotAttentionWorker<FullAttnKv, AttentionLayerExecutionAdapter>`(穿 `AttentionSlotPipeline`) |
| 21–22 | PD-AFD prefill/decode-ffn | 复用 #8 / #19 |
| **+23** | 两级 KV layout(压测 #1) | `IterBatchWorker<TieredMemoryKv, LocalPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution>` |
| **+24** | tier-aware 成本(Root ② 证明) | `IterBatchWorker<TieredMemoryKv, LocalPrefillDecodeAdmission<FifoOrder>, TierAwareIterExecution>` |

**覆盖(每轴每 impl ≥ 1):** Kv{FullAttnKv·ModeledPrefixCacheKv·HybridStateKv·ModelPartitionedKv·TieredMemoryKv}· Admission{LocalPrefillDecodeAdmission·ChunkedPrefillAdmission·PrefillHandoffAdmission·PrefixPrefillDecodeAdmission·MultiModelAdmission·FreshRequestSlotAdmission}· Policy{FifoOrder·ShortestJobFirst}· IterModelExecution{UnifiedIterExecution·MultiModelIterExecution·TierAwareIterExecution·AttentionLayerExecutionAdapter·FfnSectionExecutionAdapter}· Shell{IterBatchWorker·PullDecodeWorker·SlotAttentionWorker·PullSlotAttentionWorker·BufferedFfnWorker}。能力子 trait 全数被行使(IterWorkerKv/SlotPipelineKv/HandoffKv/ChunkedPrefillKv/PrefixCacheKv/HybridKvView/ModelSwitchKv/TieredKvView)。

## V.2 未知变体压测(matrix §15)—— 5 道盲题

判据:§15.2 strong baseline(现有 body 零改、删 variant 旧系统仍独立编译、wiring 只加一行)+ §15.3 舒适度(新文件里不得重现 tick loop / pending queue / prefix tree / ArchInput lowering)。

| # | 盲测 | 判决 | 根据 |
|---|---|---|---|
| 1 | 两级/可压缩 KV | ✅ PASS | layout=leaf(`TieredMemoryKv`);tier-aware 成本经 Root ② 后也 PASS(`TierAwareIterExecution`) |
| 2 | tenant 虚拟时间 + 完成反馈的公平/SLO | ❌ FAIL(**非 fundamental**) | `PendingOrderPolicy` 无 feedback、`AdmissionCandidate` 无 tenant —— **但 Part IV.1 本就设计了 Feedback+on_feedback+Oracle**;恢复即 leaf |
| 3 | 有模型状态的 learned prefix eviction | ❌ FAIL(**非 fundamental**) | 同 #2 缺 feedback + `probe_prefix` 仍是只读 modeled quote;prefix index/replacement 未建 —— **但 Part IV.2 本就设计了(PrefixReplacement + feedback)** |
| 4 | 视觉 encoder 态 + recurrent decoder 态 Input | ✅ PASS | 新 Input+Exec+`HybridKvView`;Root ② 后 exec 可 `where K: HybridKvView` 读 per-kind 态 |
| 5 | mid-iteration preempt/resume | ⚠️ 半 PASS | 粗粒度(整 worker)= trivial 新 shell;细粒度(批内子集)= iter-exec 整批粒度过不去 |

**三条根因(全部非 fundamental,均为加法):**
- **① 无向上 feedback 通道**(→#2、#3):设计有(Part IV.1),sketch 简化。补 = 恢复 `type Feedback`+`on_feedback`+ 完成路径一处调用。
- **② IterModelExecution 能力不对称**(→#1 成本、#4)【**已修**,Part V.3】:sketch 相对设计回归成方法泛型;恢复 `IterModelExecution<K>`。
- **③ iter-exec 整批 vs AFD-exec per-grouping**(→#5 细粒度):细粒度子集抢占不可计价;修 = 给 iter `build_iteration_input` 加 grouping,或建在 AFD 面上。路由选择,非墙。

## V.3 Root ② 修复(已落地 EXIT=0)

`trait IterModelExecution`(方法泛型 `build_iteration_input<K>`)→ **`trait IterModelExecution<K: IterWorkerKv>`**,K 上提 trait 参数,对齐 `IterAdmission<K>`(也恢复原设计)。**爆炸半径(全加法,现有行为零改):** trait + 2 impl(仍对所有 K blanket-impl → 22-server census 不变)+ 2 shell(bound 与 `<E as IterModelExecution<K>>::Input`)。**证明:** 新 `TierAwareIterExecution` bound `K: IterWorkerKv + TieredKvView`,`build_iteration_input` 读 `resident_by_tier`、`evaluate_iteration` 收 offload 加价 —— 一个 KV 能力真正抵达 cost model。**负向探针:** `TierAwareIterExecution` 配 `FullAttnKv` → `E0277: FullAttnKv: TieredKvView not satisfied`,错误落 census recipe 行,不逼加空方法。

---

# Part VI — 判断:Robustness / Boundary / Comfort / Fixability

## VI.1 Robustness / Boundary / Comfort

- **Robustness —— 稳在"正交 + 可编译验证":** 每个变体恰好落一条轴,不外溢;"拼不拼得起来"由 `cargo check` 回答,不是靠辩。罕见的强 robustness。
- **Boundary —— 一个干净形状:** 抽象把数据**向下**传(config→结构→成本),对**向上反馈**(运行时结果→组件状态)沉默。所有 FAIL 都是需从运行时结果学习的变体。**这条缝设计画过(Part IV.1),sketch 简化掉** → 边界标 scope 边缘,不是设计漏洞。
- **Comfort —— 对"形状"高信心,对"scope 与 fidelity"校准警惕:**
  - 高:正交性经编译验证;唯一真"不一致"(Root ②)一下午修平且对称;缺能力时编译错误落 recipe 而非逼全体加空方法。
  - 校准的不适:(a) feedback 边界比看上去近(SLO/公平是 serving 基本盘,先恢复缝);(b) trait-param reshape 人体工学税(`<E as IterModelExecution<K>>::Input`)会累积;(c) superset stub 藏的是 fidelity 风险不是接口风险(每个能力都是 modeled 数,"能组合"证过、"算得准"未测);(d) census 证 composition 不证 correctness(后者靠两层 golden 门,弱得多)。**舒适度分档:对"能组合"很高、对"行为对"中等、对"预测准"未测。**

## VI.2 Fixability —— 崩了能靠"加接口"保住设计大头吗?能

| 类 | 形态 | 现有代码动了什么 | 实例 |
|---|---|---|---|
| **A 纯加法** | 新文件 + 一行 census | 零 | TieredMemoryKv / 新 cadence shell |
| **B 一次性缝扩展** | 加关联类型/方法(带默认)+ 一处调用点 | 现有 impl 靠默认不变 | feedback 通道(Part IV.1) |
| **C 机械 reshape** | 改 trait 签名;body 逐字保留,只动 impl 头 + bound | 逻辑零重写 | Root ②:`IterModelExecution`→`IterModelExecution<K>` |
| **D 新子轴** | 加一条新轴(trait+impl) | 现有四轴不动 | learned prefix replacement(Part IV.2) |
| **E 重定轴** | 重新决定轴边界 | 大面积重写 | **一次都没碰到** |

**不变量:每个 break 爆炸形态止步于 C/D,没有一个触及 E。** 四条结构原因让 break 恒被逼成 add:① 能力子 trait(ISP)把新需求变新 trait 而非更胖旧 trait;② 轴是数据缝(传值不带回调),扩一条轴局部;③ 泛型 shell + 编译期 census 让 reshape 爆炸半径**机械可定位**;④ "FSM 不泛型"给最坏半径封顶(新 cadence 恒为新文件)。**两个会退化的坑:** feedback 缝若逐消费者 bolt-on 会退化成横切回调网(所以按 Part IV.1 设计成**统一通道**);trait-param reshape 啰嗦会累积。

## VI.3 组合难度规律(一眼预判任意组合)

> **KV 能力*乘法组合*(把 capability 用 `+` AND 到 bound,叠多少条都免费);Admission lifecycle *单选*(一 worker 一条,合并两条行为 = 一个 merged 新文件)。难度 ≈ 合并了几条 lifecycle。**

- **hybrid + prefix cache = LOW(纯 leaf)**:1 条 lifecycle(`PrefixPrefillDecodeAdmission`),hybrid 走能力叠(`HybridKvView`)对 admission 不可见。subtlety(已消除):recurrent 常驻态与 matched 前缀曾共用 `FullAttentionKvFootprint.prefix` 单槽 —— 相加虽自洽,但字段名骗人且让两个 wrapper 事实互斥;现已拆为 `cached_prefix`(归 `PrefixCacheKv`)+ `fixed_state`(归 `HybridKvView`)两槽,`fits` 折 `prompt + resident_fixed()`,叠加即两槽都非零。
- **KV 壳的 delegate 判别式(compose 时必查)**:*wrapper 自己存了资源账本 ⇒ `fits`/`pressure` 的 delegate 必然是假的;只存配置/阈值/路由 ⇒ delegate 成立。* `TieredMemoryKv`(fast+slow 构造时折进 inner capacity)、`ModelPartitionedKv`(一 model 一 partition)、`ModeledPrefixCacheKv`(只存命中率,贡献全走 footprint)三者 delegate 为真;`HybridStateKv` 存了 `recurrent` 这本真实占用账,曾错误地直接 delegate —— 候选自身的 state 经 footprint 进了门,但 `commit_resident` 把它从 `promised` 清掉后从未进 `active_kv`,**每多一个常驻请求超发一份 state**。修法照抄 inner 的 `held`(同形状:占池但不在 `active_kv`):开 `FullAttnKv::fits_with_extra_occupied` 缝,plain 传 0、hybrid 传 `partition_recurrent`。
- **再叠 PD 拆分 = MODERATE**:decode 侧≈0(`PullDecodeWorker<HybridStateKv,UnifiedIterExecution>` 现成);难点集中一处 —— PD-prefill 需合并 lifecycle `PrefixPrefillHandoff`(bound `K: HandoffKv + PrefixCacheKv`),因为 prefix 准入与 PD 握手都是 lifecycle 而 worker 只能选一条。
- **潜在改进:** 把 lifecycle 做成"base + 可组合 mixin(prefix-read/handoff/chunked)"就连这个 merge 都变纯配置 —— 更大重构,当前简单性换来这点 merge 成本。

---

# Part VII — 迁移(每阶段行为无损,golden 守护)

## VII.1 分阶段(Phase 0–5)

每阶段末跑 `just test-all`;吞吐 bit-identical,回归 ±1% 预警(重构阶段**不应**漂移;若漂移必核对后 `just update-golden`)。

- **Phase 0 — 抽 `WorkerContext` + Fsm 轴**:三态转换/`next_wakeup` 提成只产生 action 的 `IterFsm`;具体 worker 保留 action 解释 + `start_iteration` + `build_arch_input`。
- **Phase 1 — `FullAttnKv`**:`Batch`/`KvPool`/`promised`/`finalize`/`release` 原样搬进 impl;worker 经窄查询读 KV 事实,input build 不搬入。
- **Phase 2 — `Admission` 轴**:各 worker 的 `form_batch`+`complete_iteration` 主体抽成 lifecycle 变体;`LoadBalance`/`prefill_fits_budget` 归位;`Admission::Stage` vocab + 它拥有的 record_stage 点(Pending/Prefill/handoff-ready)收进本轴(Transfer/Decode/Done 留 Fsm 轴)。
- **Phase 3 — `IterModelExecution` 轴**:`model + CostBuffers + evaluate_iteration` 收进 `UnifiedArch`/`AttnLayerwiseArch`/`FfnLayerwiseArch`;暴露 `model_kv_layout()`。
- **Phase 4 — family-local 模板构造**:部署只为已知合法的短 worker 模板选组件;不把 AFD/spec 强塞同一通用容器。
- **Phase 5+ — 特性**:稳定四轴上逐个加 F1–F7,每个只碰主轴(prefix→KvStore+Admission、hybrid→KvStore、multi-arch→ModelSwitchKv+MultiModelIterExecution…)。

Phase 0–4 是**纯重构(无行为变化)**;F1–F7 才是新功能。

## VII.2 M1 / M2 里程碑(先验最大风险,再验设计交付)

- **M1 — 具体拆分(先落地,验借用+无损)**:四块拆成具体 struct(`FullAttnKv`/`LocalPrefillDecodeAdmission`/`UnifiedArch<M>`/`IterFsm`)+ 具体容器,方法用具体类型互调,**暂不引 trait**;`BareboneWorker<M>` 留容器别名,`new`/`IterWorker` impl 不变。验**最大风险 = 四路借用拆分 + 字段无损搬迁**。
- **M2 — 抽 trait + 泛型化(验设计交付)**:抽出四 trait + 关联类型,容器改泛型 `Worker<K,A,X,S>`。**两处 M2 硬点:** (a) 需 `K::Input == X::Input` 等式约束;(b) 需 `HandoffKv` 更强 bound 的变体(PrefillHandoffAdmission)—— method-generic `form_batch<K>` 表达不了,`K` 得是 impl 类型参数(这正是 as-built `IterAdmission<K>`/`IterModelExecution<K>` 的由来)。
- **要求:M1 的缝必须已是最终缝**(`render_arch_input(AdvanceScope,ctx)` / `commit_resident` 全参 / reserve 纯 KV / `PartitionId⟂AdvanceScope` 键 / 取消方法),并在 M1 收尾前立一份**只编译的 trait 骨架**,否则 M2 会变搬行为而非机械抬升。

## VII.3 两层等价门 + preservation invariants(无损硬证明)

golden 证明力有限(只 ±1% 吞吐汇总 warn;`promised` 是随机 HashMap,drain 序 run-to-run 不定),故无损须**两层门**:
- **L1 汇总门**:`just test-cpu`/`test-all`(Rust --lib 含 unified.rs 5 单测 + GPU 吞吐 golden ≤±1%)+ `git diff --stat -- deployment orchestrator worker/config.rs` 期望 0 改动。
- **L2 逐行差分门(真正无损证明)**:改前跑固定 preset(开 `log_output_token_times`+`log_stage_transitions`)存 baseline `cost_log`/`request_slo(events+stage_log)`/`kv_snapshot` parquet → 改后同 preset 重跑 → 两侧各按 `(req_id, time)` **canonicalize 排序后逐行 diff**(吸收 promised HashMap 随机序)。

**preservation invariants(bit-identity 硬约束,拆分绝不能改):** `promised` 随机 HashMap 的 drain 序定 arch-input 对序/decode 插入序/同刻完成事件序 → 不得换顺序结构;`Batch::release` 用保序 `Vec::remove`、取消用 `swap_remove` → 不得换 retain/乱序;完成的 decode 先追加、prefill 后追加;`KvSampler` 每 iter 恰一次 submit(位置不变);cost_log 身份(iter_id/wall-start/group 输入序)严于吞吐 —— 保吞吐不证行序,故需 L2 门。

## VII.4 代码锚点(基线 5df422f,现有 file:line)

- iter 参考 `worker/unified_iter.rs`(tick :158 / complete_iter :299 / build_arch_input :383 / promise·drain·group_promised :420–458;record_stage Pending :136 / Prefill :434 / Decode :342 / Done :314·:336)· 多 group `hp_unified.rs`· PD `pd_prefill.rs`/`pd_decode.rs`(advance_pulls :247)· AFD `disagg_attn.rs`(slot)/`disagg_ffn.rs`(double-buffer,Terminal 沿用 attn 属主 :417)。
- 请求生命周期 `shared/request_stage.rs`(`StageEvent`/`UnifiedStage`/`PdStage`/`AfdStage`)+ `request.rs`(record_stage :102 / record_first_token :140 自带 completed :147)。
- 共享叶子 `admission_helpers.rs`(current_kv :119 / advance_decodes :180 / advance_subset :197 / projected_peak :60 / try_admit :275)· `cost_buffers.rs` · `gpu_cluster.rs`。
- L6 构造 `orchestrator/common.rs`(`UnifiedWorkerFactory`/`WorkerBuildFn` :38);vocab 选择 `deployment/config.rs`。

---

# Part VIII — 待办与 open items

## VIII.1 迁移前待办(按优先级)

1. **恢复 Part IV.1 的 feedback 缝到接口**(设计,可后建):`PendingOrderPolicy` 补 `type Feedback`+`on_feedback`(no-op 默认)+ Oracle;`AdmissionCandidate` 补 tenant/来源(SloSpec)。唯一真实需求(SLO/公平)必然跨的边界,现在画好,挡住第一个特性 ad hoc 绕过四轴。
2. **明确 fidelity 是独立未测的断言**:census 只证 composition;迁移用两层 golden 门守行为无损,但那不证成本保真度。
3. **按需再建(都是加法,D 类,不阻塞主体迁移)**:Part IV.2 prefix index/replacement 子轴、Part IV.3 PlacementSeam、`EncoderPipeline`/`DraftVerify` shell。
4. **观察项**:trait-param reshape 人体工学税;若累积,考虑关联类型式能力视图。

## VIII.2 未决 API 边界(H1–H10,实现具体 trait 前须逐项定死)

| ID | 边界 | 未决问题 |
|---|---|---|
| H1 | request ingress → KV | KV candidate 所需 request facts 的类型与生产者(as-built 走 `SharedRequests` 直读) |
| H2 | Admission → KV | reserve API 形态(as-built 确认一步式 `fits→reserve`) |
| H3 | KV → Admission | common observation 与 policy-specific observation 如何组合 |
| H4 | Admission/Worker → KV | reservation handle ownership 与 commit/cancel handoff(as-built:ledger owner=KV) |
| H5 | Worker → Admission | lifecycle 是否共享一 operation set 还是分 role traits(as-built:per-family trait 面) |
| H6 | Worker → KV | growing/chunk/transfer/speculative transition input shapes |
| H7 | Admission/Worker → KvTopology | local attention-DP placement 的 enumeration/choice owner |
| H8 | L6 → Worker | prefix quote 是否含 local placement hint,如何进 Admission(见 Part IV.3) |
| H9 | IterModelExecution → KV views | 最小 read-view capability set;不同 Input 不得推动 mega view(as-built:IterWorkerKv/SlotPipelineKv 分家族) |
| H10 | multi-model → Layout/Topology | 同 worker 内不同 model 用不同 layout/topology 时的 model-set API |

---

*事实源:`sketch/worker_v2/`(as-built)+ `census.rs`(覆盖证明)。fit-test 矩阵与 §15 判据见 [compatibility_matrix.md](L5_worker_compose_compatibility_matrix.md)。归档设计原件见 repo 外 `L5_old_docs/`。*
