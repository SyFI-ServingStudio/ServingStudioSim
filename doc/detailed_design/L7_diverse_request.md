# L7 — 多样请求类型的延展性设计(L7_diverse_request)

- **状态:design analysis(未实现)**。当前里程碑只有**单轮 text**;本文是"未来请求类型越来越多(带 session history / SLO·priority 标签 / 图像 / 触发执行图 / 音频…)时,怎么装进系统而不撑爆 worker"的前瞻设计与判据。它对 request 抽象做的事,正是 [L5_redesign.md](L5_redesign.md) 对 worker 抽象做的事(把 model/modality-specific 的东西关进各自那根轴,共享结构保持 agnostic)。
- **现状事实源**:[L7.md](L7.md)(frontend / run loop / store 的 as-built 事实)。代码锚点:`simulator/src/common/request.rs`(`Request` / `RequestRecord` / `RequestStore`)、`simulator/src/sim/frontend.rs`(唯一 ingress)、`sketch/worker_v2/shared/context.rs`(`WorkerContext` 隔离墙)。
- **一句话**:请求类型会爆炸,但**它们不在同一层**;按"执行形状"而非"modality 名字"分类,多数是加法,只有非自回归输出越界成新 family。request 的充实走一条纪律 —— **agnostic 核 + 类型化、对 worker 不透明、各自只被一根轴读的侧结构**。

> 阅读顺序:决策者读 Part 0 + IV + VIII;想改 `RequestRecord` 的实现者读 Part I + V;想扩 frontend/schema 的读 Part VI;关心 agent/多步编排的读 Part VII。

---

# Part 0 — 背景与问题

未来要支持的请求类型(用户列的问题空间):

1. text
2. text + prefix / session history(多轮)
3. text + 任意 SLO / priority 标签
4. images(图生文 VLM/ITT、文生图 TTI)
5. text 但含**执行路径**(可能触发图生、可能触发文生 —— 本质是系统图遍历)
6. audio 及其它多模态

三个问题:**(a)** 怎么装进系统?**(b)** 会不会 break 已定稿的四轴 worker?**(c)** 怎么把 frontend 隔离出来,不让请求类型的变化顺着接口漏进 worker?

放松条件(用户明确):**有些类型需要新 worker、甚至独立 KV 管理 / 准入,完全 OK**;能共享更好,但不强求。于是真正的问题不是"会不会 break",而是**每类落在栈的哪一层、能不能共享、共享哪根轴**。

---

# Part I — 现状数据流与 worker 的窄读点

```
trace CSV ──► [frontend.rs:143] ──► Request ──► RequestRecord ──► worker 读点
{id,input_len,   TraceEntry→Request   {prompt_len,   {+FSM 字段}     (只读数字)
 output_len,     唯一 ingress          decode_len,
 arrival}                              arrival}
```

`RequestRecord`(`request.rs:42`)= arrival 事实(不可变)+ FSM 工作字段 + 输出记账。worker(`unified.rs` / `pd_*.rs` / `disagg_*.rs`,以及 `sketch/worker_v2`)从 record **只**读这些:

- `(prompt_len, decode_len)` → 准入 gate(`try_admit`)+ KV footprint
- `(prefix_kv, active_chunk_len)` → cost model 的 `prefill_chunk_pairs` → ArchInput
- `tokens_emitted` vs `decode_len` → 完成判定(`is_complete()`,`request.rs:133`)

**关键事实:worker 眼里的 request 已经是纯数字 token 计数,它从不知道"文本"这回事。** 请求面对 worker 的整个表面就是 `(prompt_len, decode_len, prefix_kv, active_chunk_len): u32`。这跟 [L5_redesign.md](L5_redesign.md) 里"KvStore 不知道 role"是同一条原理的延伸——延伸到"不知道 modality"。

**唯一一处 text 假设**:`is_complete() = tokens_emitted >= decode_len`(自回归、逐 token)。焊在 `RequestRecord::is_complete` + serving shell 的 `record_token`。记住它,Part III 会撞上。

---

# Part II — 按执行形状分类(不是按 modality 名字)

modality 的名字会误导;决定能不能塞进现有抽象的是**执行形状**,只有三档:

| 档 | 例子 | 输入 | 输出 | 对现有抽象 |
|---|---|---|---|---|
| **A. 自回归-out,输入即 token 计数** | text、text+session history、audio-in、VLM/ITT | 折成 prompt token 数 | 逐 token | **已经装得下** |
| **B. 自回归-out,输入需先编码** | 图生文(image→N visual tokens) | 非 token,需 vision encoder | 逐 token | **加法扩展** |
| **C. 非自回归-out** | 文生图 TTI(diffusion)、TTS | text token | N 步去噪 / 非 token | **越界(新 family)** |

- **A**:`session history` = 就是 `prefix_kv`(已建模);VLM 的 visual token 一旦在 frontend 折成数,对 worker 就是更长的 `prompt_len`。**零改动**。
- **B**:图片 → vision encoder → N 个 visual token。(a) KV/准入在乎吗?**不在乎**——visual token 就是占 KV 的 prompt token。(b) cost 在乎吗?**在乎**——encoder 是额外算力,成本依赖分辨率/patch 数,而 `RequestRecord` 没有这些。→ 归 **IterModelExecution** + 一段 encoder stage(deferred 的 `EncoderPipeline` shell,blind test #4 验过这个形状)+ request 携带图像事实。
- **C**:完成判定不再是 `tokens_emitted >= decode_len` 而是"N 步去噪完了";KV 可能根本不自回归增长。跟 training 一样是**另一个 family**。

---

# Part III — 会不会 break worker(三读点逐点判定)

| 读点 | A(text/history) | B(VLM/ITT) | C(TTI/diffusion) |
|---|---|---|---|
| **准入/KV**(数字 token 数) | ✅ 不碰 | ✅ 不碰(visual token 就是数) | ✅ 不碰(text 侧仍是数) |
| **cost model**(读 prefix_kv 等) | ✅ 不碰 | ⚠️ 需 encoder 成本 + 图像事实 | ⚠️ 去噪成本 |
| **完成判定** `emitted≥decode_len` | ✅ 成立 | ✅ 成立(仍逐 token out) | ❌ **崩**(步数,非 token) |

- **A/B 不 break worker**——纯加法:新 IterModelExecution(懂 vision 成本)+ 可选 encoder shell + request 挂一个侧 blob。四轴一个 modality match 都不长。
- **C break**,且崩在**跟 training 完全一样的地方**(那唯一一处 text 假设)。→ 归 family 边界,用**兄弟 family/shell(自带 completion)**接,不是做 serving worker 的 variant。这与 L5 的裁决一致:"完成/cadence 是 shell 自持的,不是通用轴" —— TTI 只是又一个 cadence。

---

# Part IV — 6 类问题空间 × 4 层(核心地图)

这 6 类**不在同一层**;按层归位后,"会不会 break worker"大半自己消失,因为多数根本不是 worker 的事。

| # | 类别 | 落在哪层 / 哪根轴 | 共享判决 | 难度 |
|---|---|---|---|---|
| 1 | text | 基准(全共享) | — | — |
| 2 | text + prefix/session history | **KV 轴** + frontend 多轮链接 | **共享 worker**:`PrefixCacheKv` 能力(乘法叠)+ `prefix_kv`(已建模)+ `PrefixPrefillDecodeAdmission` lifecycle | LOW |
| 3 | text + 任意 SLO/priority | **Admission 轴(仅此)** | **共享 worker/KV/exec/shell**;只换 `PendingOrderPolicy`。`AdmissionCandidate.deadline` 已在(policy/mod.rs:24) | **最便宜** |
| 4a | image-in → text-out(VLM/ITT) | **IterModelExecution + encoder shell + payload** | **共享 KV+准入**;IterModelExecution 新 + `EncoderPipeline` shell(deferred) | MODERATE,加法 |
| 4b | image-out(TTI/diffusion) | **新 family/worker** | **新 worker**:完成 ≠ token、KV ≠ 自回归 | 新 worker(OK) |
| 5 | text 含执行路径(图遍历) | **Orchestrator/Flow 层(worker *之上*)** | **既不 break worker 也不碰 request struct**;是 `Flow` 的事 | 另一个抽象(Part VII) |
| 6 | audio 等 | in = 同 4a;out/TTS = 同 4b | in 共享、out 新 worker | 同 4 |

**#3 的关键点**:SLO/priority 不是 modality,是**贯穿性标签**——一个文本请求和一个图像请求都能带 priority。它住在 request 的 agnostic 核,**只被 admission 一根轴读**,KV/exec/shell 全不碰;换 policy 是**最便宜的轴**(纯 type-param swap)。

**正交性收束**:这 6 类归约成 **4 个正交扩展方向**,各打不同层:

1. **贯穿标签**(SLO/priority)→ agnostic 核 + **Admission** 轴
2. **输入编码/状态**(image/audio-in、history)→ **Frontend** + (**KV 能力** | **IterModelExecution + encoder**)
3. **输出形状**(image/audio-out)→ **新 family/worker**
4. **控制流**(执行图)→ **Orchestrator/Flow** 层

因为正交、且落在不同层,**N 种请求类型 ≠ N 种 worker**。最坏例子——"高优先级、带 session history 的 VLM 请求,输出触发一次图生":priority→Admission 换 policy;history→KV 加 `PrefixCacheKv`;VLM-in→IterModelExecution+encoder shell;触发图生→**Flow** 派生子请求给 TTI worker = **四根独立轴各碰一下 + 一个新 worker(TTI)**,全是加法,无叉乘。这跟 L5 四轴分析同一个结论(正交轴组合不组合爆炸),现在在**请求类型**上再次成立。

---

# Part V — 结构决定(request 怎么充实)

**request 的充实走一条纪律:agnostic 核 + 类型化、对 worker 不透明、且各自只被一根轴读的侧结构。**

- **agnostic 核**(留在 `RequestRecord`,共享):token 计数(prompt/decode/prefix)、FSM 工作字段、**完成判据归 shell**。KV / Admission / 完成读这半,保持扁平共享。
- **差异化侧 blob**(类型化,worker 不透明,各自单轴读):
  - `SloSpec` → 只被 **Admission**(`PendingOrderPolicy`)读(`AdmissionCandidate.deadline` 已是接缝)
  - `ModalityInput`(图像分辨率 / patch 数 / 去噪步表 / encoder token 数)→ 只被 **IterModelExecution** 读(和 `IterModelExecution::Input` 同套路)
  - 执行图 → 归 **Orchestrator/Flow** 层(Part VII)

**反模式(必须现在就守的边界)**:`RequestRecord`(`request.rs:42`)是**每个 worker 都读的扁平大结构**。若按 modality/priority 一个个加扁平字段(`image_patches`、`diffusion_steps`、`num_frames`、`priority_tier`…),就退化成"共享结构越加越宽"——每个 worker 的结构体膨胀,modality-blindness 被侵蚀。正确形状 = **agnostic 核 + 类型化 payload,payload 对 worker 不透明**。

**完成判据**:今天焊死在 `RequestRecord::is_complete` + serving shell。非自回归 family 必须让它可变 —— 做成 **per-family(shell 自持)**,和 L5"FSM 不是通用轴、每个 shell 自持 cadence"一致。所以 TTI 不是"加宽 RequestRecord",是"新 family/shell"。

设计已埋好扩展点(边界是"还没建/故意简化",不是"设计错"):`request.rs:5` 头注释("Multi-round fields land alongside future L7 lifecycle work")、`frontend.rs:180` 对 `round_idx` 的显式 bail、以及旧 interfaces 文档早假设过的 `RequestFacts` + `SloSpec` 分离。

---

# Part VI — Frontend 隔离(那道墙)

**frontend 就是 modality 边界,而且它已经站在对的位置**(`frontend.rs:143` 一行 `TraceEntry → Request`):它今天已经在把外部 workload 解析成数字请求事实。扩展 = 保持这个方向。

两条规则(与 worker 的"IterModelExecution 拥有 model-specific Input"完全同构):

1. **modality 在 ingress 就地解析成两半**:agnostic token 事实(KV/Admission/完成读)+ opaque modality payload(只 frontend 填、只 IterModelExecution 读)。KV/Admission 永不碰 payload。
2. **`RequestKind` 枚举住在 frontend,不住在 worker**。"这是 text 还是 image 还是 TTI"的判别发生在 ingress;worker 下游只见数字 + 不透明 payload。tokenizer / chat template / vision preprocessor 全在 frontend 或更上游。

**为什么这道墙成立(和 `WorkerContext` 的关系)**:worker 看请求**只**通过 `WorkerContext.requests`(`ctx.rs:15`,`SharedRequests → RequestStore → RequestRecord`),且每个读点都是窄数字事实。所以只要把 `SloSpec` / `ModalityInput` 挂在 record 上、由**唯一那根轴**去读,worker 四轴一个 modality/priority 的 match 都不用长——`WorkerContext` 这层薄接口就是隔离墙。

**trace schema 扩展**是 frontend/schema 的事,加法:新列(modality、image_tokens、diffusion_steps、round_idx)。今天对多轮(`round_idx`)的显式拒绝(`frontend.rs:180`)正是标记 session-history 扩展点的占位。

---

# Part VII — #5 执行图:不是 worker/request 的问题,是 Flow 的问题

[L6 orchestrator README](../../simulator/src/orchestrator/README.md) 已替我们裁决(Invariant 1):**"Everything above the worker is routing; everything at or below it is cost + lifecycle."** 一个"触发图生 / 再触发文生"的请求 = 一张**动态多节点 Flow 图**:

- **节点** = 一条 leaf 请求,由某个 worker 执行(text-leg→text worker,image-leg→TTI worker)——worker 照旧 modality-blind,一次只跑一条腿;
- **边** = Flow 的路由决策(spawn 子请求 / gather 汇聚 / 选下一跳)。

`PdFlow`("`PrefillDone` → decode 准入")和 AFD gather 就是**静态 2 节点特例**。#5 = 把 `Flow` 从"固定 2 段管线"推广成"动态 DAG",需要:子请求派生、依赖/join、逐节点选 worker。**这些全落在 `Flow` trait,worker 四轴一根都不动。** 换言之,#5 不撑爆 worker,它撑的是**另一个抽象(Flow/orchestrator),值得像我们对 worker、对 request 那样单独做一次表达力压测**(未来工作)。

---

# Part VIII — 裁决 + 现在该锁的一条原则 + 待办

**裁决(robust / boundary / comfort)**:
- **Robust?** 对整个**自回归-serving 家族**(text、session-history、VLM/ITT、audio-in)——**是**。worker 读请求走纯数字 token 接口,天生 modality-blind;多模态-in 是加法(frontend 折成 token + 一个 IterModelExecution payload)。
- **Boundary?** **非自回归输出**(TTI/diffusion、TTS)。唯一一处 text 假设 `tokens_emitted >= decode_len`,是 family 边界(和 training 同类),用兄弟 family 接,不是 variant。
- **Comfort?** 对看得见的扩展(VLM、session-history)——**高**,前提是现在锁一条原则。

**现在就锁的一条原则**(便宜的保险,代码今天不用改——单轮 text 是当前里程碑):

> modality/priority 永远不变成 `RequestRecord` 的扁平字段、也永远不变成 worker 可见的 match;它在 frontend 就地解析成 token 计数 + 一个对 worker 不透明、只被一根轴读的 payload(`SloSpec`→Admission、`ModalityInput`→IterModelExecution、执行图→Flow);完成判据归 workers/family,不 pan-worker 焊死。

**待办(未来里程碑)**:
1. `RequestRecord` 拆分:agnostic 核 + 一个类型化 payload 接缝(与 `IterModelExecution::Input` 同套路)。可先只为 `SloSpec` 开一条,验证接缝。
2. frontend 引入 `RequestKind` + trace schema 扩展(round_idx / modality 列),`frontend.rs:180` 的 bail 改成分派。
3. `EncoderPipeline` shell(B 档 VLM/ITT 的编码 stage),对齐 blind test #4 的形状。
4. TTI/TTS 作为**兄弟 family**(自带 completion cadence),不进 serving worker variant。
5. `Flow` 的动态 DAG 表达力压测(#5),独立于 worker/request 抽象(Part VII)。
