# L7 — 多样请求类型的 typed frontend

- **状态：request/frontend 类型接缝已实现；非 text worker family 尚未实现。**
- **事实源：** `simulator/src/common/request.rs`、
  `simulator/src/common/request_family/`、`simulator/src/sim/frontend/`、
  `simulator/src/orchestrator/mod.rs`。
- **核心裁决：** 请求 family 是 Rust 类型参数，不是贯穿运行路径的 enum。
  runtime kind dispatch 只允许存在于启动期的 `LoadedTrace`；进入 L6/L5 后，
  frontend、`Flow`、`RequestStore`、worker 必须共享同一个 `Definition` 类型。

## 1. Request 的四层结构

```text
Request<Definition>
├── RequestCore                 # id / actual arrival / SLO / scheduling policy
└── Definition                  # immutable family-specific requested work

ActiveRequest<Definition>
├── request                     # core + immutable definition
├── progress: Definition::Progress
├── lifecycle                   # admitted / completed / stage state
└── telemetry                   # first/last/per-output observations
```

`RequestDefinition` 关联自己的 `Progress` 和完成判据。因此 text generation 的
`output_tokens_emitted >= target_output_tokens` 不会被误用到 diffusion/TTS；这些
family 的定义分别使用 generation-step progress。

`Request::new(RequestCore, Definition)` 只是 generic storage constructor：它接收已经
由 replay 解析完成的 core 与 typed definition，不解析 trace，也不提供某个 family 的
positional convenience API。legacy 四列 text trace 的默认 prefix/decoding/scheduling
完全属于 frontend schema；测试需要默认 text request 时使用 test-only helper。

公共 `RequestCore` 只放真正跨 family 的事实：

- `RequestId`
- 实际 release 后的 `arrival_time`
- `SloContract { ttft_slo, tpot_slo, e2e_slo }`
- `SchedulingContract { priority }`

Session-capable definition（当前 autoregressive directional families 与 omni）的
`SessionInput` 在一个 enum variant 内同时保存
`session_id`、该 session 第一条 trace-declared `arrival_time` 和
`declared_prefix_tokens`。因此不存在“有 session start 但没有 session id/prefix
declaration”的半状态。L5 admission 将它归一化为 concrete
`conversation_start_time`；standalone request 使用自己的实际 release time。

prompt/output token 数不在 core。它们属于 `TextGenerationDefinition`。directional
family 使用 `ImageExtent`、`VideoExtent`、`AudioExtent` 具体类型，因此
image-to-video 的消费者不需要 match 无关的 audio variant。只有原生 omni family
内部使用 segment enum，因为“同一个 request 的有序输入/输出确实可异构”就是它的
业务契约，而不是为了绕过 family 类型系统。

## 2. 当前 concrete families

frontend schema 已能产生这些互不兼容的类型：

| trace kind | concrete definition | completion unit |
|---|---|---|
| `text_generation` | `TextGenerationDefinition` | output token |
| `image_to_text` | `ImageToTextDefinition` | output token |
| `video_to_text` | `VideoToTextDefinition` | output token |
| `audio_to_text` | `AudioToTextDefinition` | output token |
| `text_to_image` | `TextToImageDefinition` | generation step |
| `text_to_video` | `TextToVideoDefinition` | generation step |
| `text_to_speech` | `TextToSpeechDefinition` | generation step |
| `image_to_video` | `ImageToVideoDefinition` | generation step |
| `omni_generation` | `OmniGenerationDefinition` | per-output codec/token count |

这些 family 各自在 `common/request_family/` 的同名文件中定义；共享的
autoregressive request vocabulary、durable family progress、generated-media progress
和 concrete extent 才放在小型公共模块中。runtime KV residency 属于 L5 `KvStore`，
future chunked-prefill 的 active chunk 属于 Admission lifecycle；两者都不能复制进
request-family progress。旧的 concrete 名称（例如 `VideoGenerationDefinition`）继续
兼容，同时公开方向明确的 `TextToVideoDefinition` alias；frontend/config vocabulary
使用 `text_to_video`。

`OmniGenerationDefinition` 是特意保留的一个宽 family：

```text
input:  Vec<OmniInputSegment>  # text/image/audio/video，可重复、可混排
output: Vec<OmniOutputSpec>    # text/image/audio/video，可同时要求多个结果
```

其中媒体输出的 completion unit 是模型/codec token。diffusion 式
text-to-image、text-to-video、image-to-video 仍是独立的 step-based family；不能把
`denoise_steps` 假装成 omni token，也不能因为两者都输出 video 就强行共用 worker。

这不等于所有 family 已可执行。当前 unified、HP、PD、AFD worker 都只实现
`TextGenerationDefinition`。`LoadedTrace::into_current_text_frontend` 是显式能力门：
其他 family 会在 PyO3 bridge/L4 build 前失败，不会先降级成 text token 数后假装支持。

## 3. Typed frontend 与 replay 的正交边界

config 用完整的 `input_file_format` 选择 req-frontend concrete loader，并用
`input_file_tags` 添加正交列束；loader 验证并解析后，ServingStudio Sim adapter 直接产出：

```text
ScheduledRequest<TextGenerationDefinition>
ScheduledRequest<ImageTextGenerationDefinition>
ScheduledRequest<VideoGenerationDefinition>
ScheduledRequest<ImageToVideoDefinition>
ScheduledRequest<OmniGenerationDefinition>
...
```

每个文件仍是单一 family，family 已由完整 format 确定，header 必须与声明的 exact
schema 完全一致；不会从列名猜 family，也不允许每行携带 family union。

`ScheduledRequest<Definition>` 拆成四部分：

- `definition`：family-specific immutable work
- `slo`：每个 metric 独立可选的 TTFT / TPOT / E2E duration bound
- `scheduling`：独立的 priority policy
- `release`：`ReleaseMetadata { request_id, trace_arrival_time, session }`

`ReplayScheduler` 的 API 只接受 `ReleaseMetadata` slice，所以 release scheduling
无法读取或 match request definition。它组合三个独立类型：

```rust
ArrivalSchedule::trace_timed(request_rate)
ArrivalSchedule::saturated()

CapacityLimit { max_active_units: Option<usize> }

SessionDependency::Independent
SessionDependency::Chained
```

共享 crate 的 `ArrivalMode` 决定 release time 从哪来；ServingStudio Sim 的
`ArrivalSchedule` 只额外携带 rate-1-normalized trace 所需的 `request_rate` 算术。
`CapacityLimit` 决定同时能有几个 unit 活着；`SessionDependency` 决定一行是否必须等待同 session predecessor completion +
`tool_wait_after_ms`。三轴全组合合法 —— 尤其 `trace_timed + max_concurrency`：
按录制时间线回放、同时限制并发，这是真实 workload，此前被熔在一起的
`ReplayPacing` 表达不了。

**cap 的单位是 unit，不是 request**：`chained` 下一个 session 在 head release 时占住
slot，直到最后一轮完成才释放，**tool wait 期间照占**（此时它一个 in-flight request
都没有）。所以 successor 不再过 capacity 门 —— 它所属 session 早已持有 slot，重复
gate 会直接死锁（唯一能释放 slot 的正是这条 successor 通向的 session completion）。
被 cap 挡住的 head 用「slot 开放的瞬间」打时间戳而非 trace arrival，与实测端
TraceLab「先等 arrival、再拿 permit、然后才发请求」的时钟起点一致。
release 时，frontend 把 metric-specific duration 原样写入 `RequestCore.slo`，并构造
`Request<Definition>`。SLO 不是绝对时间，也不进入 scheduling policy。这也是 production
唯一调用 generic `Request::new` 的位置；
四列 schema 与带 session/SLO/priority/speculative tags 的 schema 最终共享同一个 storage seam。

## 4. trace tags 的真实 owner

tags 仍可独立组合，但 parser 会把字段送到实际 owner，而不是保留一个宽
`ArrivalTags` bag：

| tag | parsed destination | 当前 consumer 状态 |
|---|---|---|
| `session` | release chain + definition 的 `SessionInput` | chain、session-age admission 与 L5 prefix KV 均已消费 |
| `slo` | `RequestCore.slo` 的 TTFT / TPOT / E2E bounds | `request_slo` 已持久化，current admission 不读取 |
| `priority` | `RequestCore.scheduling.priority` | current admission policy 尚未读取 |
| `speculative` | text definition 的 `DecodingStrategy` | current execution 尚未读取 |

frontend 只负责 exact-schema parsing、typed family construction 和 replay，不再声称能
判断某个实际 deployment/worker composition 是否消费这些字段。除 session release chain
外，text worker 的 `PrefixKv` 现在也消费 prefix declaration；SLO selection 与
speculative execution 仍只是被保真存储，在真实 consumer 与 capability contract
落地前不能据此声称模拟已经执行相应策略。

## 5. RequestStore 与到达语义

`RequestStore<Definition>` 把直接寻址和集合迭代拆成两种结构：

- `Vec<Option<ActiveRequest<Definition>>>` 保留 O(1) dense id lookup；
- startup 用 `reserve_slots(count)` 只预留 dense id slot；
- `None` 表示 request 尚未 release；
- Flow 收到 typed request 后以 owned `insert(request)` 填入 slot；
- `arrived_ids` 按 release 顺序只记录 present slot，使 `len` 为 O(1)、
  `iter_arrived` 为 O(arrived)；
- `admitted_ids` 按首次 admission 顺序只记录 admitted request，使
  `num_admitted` 为 O(1)、`iter_admitted` 为 O(admitted)；重复 admission 不会重复入账；
- 不再构造假 arrival record，也不再由后到请求 `upsert` 覆盖占位数据。

这使 session-chain 的乱序 release 保留 O(1) id indexing，同时 arrival/lifecycle
含义保持真实，也不会让大批尚未 release 的预留空槽进入周期性 snapshot 扫描。

## 6. 当前 worker 边界

现有 worker 的字段访问已经显式落到四层：

- admission requested-work 输入：`record.request.definition.*`
- durable prefill/decode 进度：`record.progress.*`
- stage/completion：`record.lifecycle.*`
- TTFT/TPOT 日志：`record.telemetry.*`

当前 text worker 已通过 L5 `PrefixKv` 实现 prefix-aware KV，chunked prefill 仍未实现。
`TextGenerationDefinition.prompt_tokens` 只表示本次 fresh suffix；
`SessionInput::Session.declared_prefix_tokens` 是 immutable requirement，不是 observed hit。
`FullAttnKv` 在目标 attention partition 上解析：

```text
resident = min(local retained session KV, declared prefix)
prefill_tokens_to_compute = fresh prompt + declared prefix - resident
post_prefill_context_tokens = fresh prompt + declared prefix
```

`ResolvedPrefillContext` 只存于 `FullAttnKv`。Execution 从 `PrefixKv` 读取
resident 与 `prefill_tokens_to_compute`；`TextGenerationProgress` 仍只保存真实 processed prefill
work 与 emitted output tokens。cache disabled/evicted/placement miss 都会重算缺失 prefix，
不会把 declaration 当成 guaranteed hit。admission 成功时会把实际 resident 数一次性复制到
`RequestTelemetry.prefix_cache_hit_tokens`，L7 再把 declaration 与这个 observation 一起写入
`request_slo`。同一行还独立保留 immutable `fresh_prompt_tokens` 与 runtime
`prefill_processed`：`NULL` hit 表示从未完成 prefix resolution，`0` 表示真实 miss；miss
token 数与 hit rate 由 declaration/hit 推导。已经产出首 token 的 request 必须满足
`hit + prefill_processed = fresh_prompt_tokens + declared_prefix_tokens`，使 Analyzer 能在
动态 hit rate 下检查 token、causal-attention 与 decode-KV 三层守恒。

`request_slo` 同时原样持久化 nullable 的 `declared_ttft_slo_ms`、
`declared_tpot_slo_ms`、`declared_e2e_slo_ms`。每列只描述对应 metric；某列为 `NULL`
表示该 request 没有声明该项 obligation，不可解释成零，也不可由另一项 SLO 推导。

retained prefix 与 normal active/promised/PD-held KV 共享同一个 total attention capacity。
默认 `prefix_cache_mode: opportunistic`，completed-session KV 可以使用当下全部空闲 attention
KV；`prefix_cache_max_gpu_memory_gb` 是可选 ceiling，不填写就不额外设固定上限。新
reservation 始终可驱逐 retained prefix；需要 no-reuse baseline 时显式选择
`prefix_cache_mode: disabled`。一次 hit 会把整个 session entry 的所有权移出 cache，request
完成后再把 physically resident KV 放回，因此当前不模拟 inter-request sharing、radix
dedup 或 ref-counted pages。PD prefill 在 handoff 后继续把 source KV 记为 held，直到 decode
pull ack 才转成 retained prefix；handoff 携带 full initial-context token count，decode 不从
request definition 重新推导。

barebone、HP、PD prefill 与 AFD attention 都接通该 capability，并通过 worker selector
暴露 `prefix_cache_mode`、`prefix_cache_policy` 与可选
`prefix_cache_max_gpu_memory_gb`；launcher 从 Rust schema 自动生成对应 CLI/sweep 参数。
PD decode 消费 prefill 传来的 exact KV 数，但不另外维护 session prefix cache。默认类型参数
仍指向 `TextGenerationDefinition`，且类型参数继续贯穿 `Flow`、`SharedRequests` 与 store；
prefix support 没有把 current worker 扩成 request-family variant。

## 7. 新 family 的实现检查表

1. 在 `common/request_family/<direction>.rs` 定义 concrete `Definition`、`Progress`
   和完成判据；只把真正跨 family 的 vocabulary 放入 shared primitive module，
   不要把 family 字段重新堆回 `common/request.rs`。
2. 在 `frontend/schema.rs` 增加 exact columns 和 `TraceDefinition` parser；每个 row
   直接生成 concrete `ScheduledRequest<Definition>`。
3. 在 `LoadedTrace` 增加启动期 dispatch variant。
4. 实现接受同一 `Definition` 的 L5 worker family、L6 `Flow<Definition>`、typed
   `SharedRequests<Definition>`。
5. 第 4 步完成后，把 `LoadedTrace` variant 窄化到 matching typed execution path；当前
   `into_current_text_frontend` 仍会拒绝所有 non-text family。
6. 若只是增加 workload pacing，只改 `ReplayPacing`/`ReplayScheduler`；若只是增加
   session causality，只改 `SessionDependency`/`ReplayScheduler`；若只是增加 scheduling
   policy，只改 admission consumer。不要让这些轴与 request family 叉乘。
7. 只有当一个 model/worker 原生接受异构有序 segment 时才建 omni family；不要用
   `OmniInputSegment` 取代能由方向类型表达的 specialized request。
