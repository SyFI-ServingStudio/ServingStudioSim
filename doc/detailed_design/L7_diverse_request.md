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
├── RequestCore                 # id / actual arrival / scheduling contract
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
- `SchedulingContract { priority, completion_deadline }`

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

schema 根据 config 的 `trace_kind` 选择 concrete parser，然后直接产出：

```text
ScheduledRequest<TextGenerationDefinition>
ScheduledRequest<ImageTextGenerationDefinition>
ScheduledRequest<VideoGenerationDefinition>
ScheduledRequest<ImageToVideoDefinition>
ScheduledRequest<OmniGenerationDefinition>
...
```

每个文件仍是单一 kind，header 必须与声明的 exact schema 完全一致；不会从列名
猜 kind，也不允许每行携带 kind union。

`ScheduledRequest<Definition>` 拆成三部分：

- `definition`：family-specific immutable work
- `scheduling`：priority 和相对 completion deadline
- `release`：`ReleaseMetadata { request_id, trace_arrival_time, session }`

`ReplayScheduler` 的 API 只接受 `ReleaseMetadata` slice，所以 pacing 无法读取或
match request definition。open-loop、closed-loop、session-chain 与 modality 正交。
release 时，frontend 才把相对 deadline 解成 absolute simulated deadline，并构造
`Request<Definition>`。这也是 production 唯一调用 generic `Request::new` 的位置；
四列 schema 与带 session/SLO/speculative tags 的 schema 最终共享同一个 storage seam。

## 4. trace tags 的真实 owner

tags 仍可独立组合，但 parser 会把字段送到实际 owner，而不是保留一个宽
`ArrivalTags` bag：

| tag | parsed destination | 当前 consumer 状态 |
|---|---|---|
| `session` | release chain + definition 的 `PrefixInput` | chain 已消费；prefix KV 只携带、尚未消费 |
| `slo` | `RequestCore.scheduling` | current admission 尚未读取 |
| `speculative` | text definition 的 `DecodingStrategy` | current execution 尚未读取 |

frontend 只负责 exact-schema parsing、typed family construction 和 replay，不再声称能
判断某个实际 deployment/worker composition 是否消费这些字段。除 session release chain
外，当前 worker 尚未实现上述可选语义；在真实 consumer 与 capability contract 落地前，
这些字段只是被保真存储，不能据此声称模拟已经执行相应策略。

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

当前 text worker 没有 prefix-aware KV 或 chunked prefill：完整 fresh prompt 直接从
definition lowering，KV membership/residency 只从 `KvStore` 读取。未来实现这些能力时，
必须通过对应 L5 owner 的窄 capability 交给 Execution，不能把 runtime 状态塞回
`TextGenerationProgress`。

barebone、HP、PD prefill/decode、AFD attention/FFN 的 cadence、KV、admission、
execution 语义没有改变；这次只重塑 shared request seam。默认类型参数仍指向
`TextGenerationDefinition`，但类型参数贯穿 `Flow`、`SharedRequests`、store，未来
family 不需要把当前 worker 扩成 variant。

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
6. 若只是增加 pacing discipline，只改 `ReplayMode`/`ReplayScheduler`；若只是增加
   scheduling policy，只改 admission consumer。不要让它们与 request family 叉乘。
7. 只有当一个 model/worker 原生接受异构有序 segment 时才建 omni family；不要用
   `OmniInputSegment` 取代能由方向类型表达的 specialized request。
