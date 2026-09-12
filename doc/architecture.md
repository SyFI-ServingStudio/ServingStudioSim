# Architecture

ServingStudio Sim is organised as a strict seven-layer stack. Each layer answers exactly one
question, consumes only the layer directly below it, and hides everything under it
from the layer above. Two properties fall out of that discipline:

- **Timing flows up, state lives at the top.** L1 is the only layer that produces a
  number; L2–L4 compose those numbers into a per-iteration cost with no simulation
  state of their own; mutable simulation state first appears at L5 and above.
- **Parallelism is invisible below L4.** L1–L3 see only per-rank shapes. Which
  ranks exist, and how a request is partitioned across them, is decided at L3's
  `resolve_config` and consumed at L4; the ops and kernels underneath never read a
  parallel config.

## The layer stack

### L1 — Kernel / Profiling

The measured floor. L1 turns a *kernel shape* into *metrics* (time, flops, bytes,
energy) and it is the only layer that ever produces a number. It splits into two
halves that meet at a PyO3 bridge:

- **Profiling (Python, `profiling/`)** launches the real kernel on a real GPU and
  records a row into `profile.db`, keyed by kernel kind and shape. The `perf_api`
  facade is the single entry point the Rust side calls.
- **Timing (Rust, `simulator/src/timing/`)** fits an interpolating cache over the
  profiled samples once at build time, then evaluates that cache millions of times.
  It also owns the **CostTree** — the compile-once / evaluate-per-iteration
  structure that every layer above uses to compose leaf metrics into one total.

*Sees:* a kernel config + a per-call shape, and `profile.db`.
*Doesn't see:* ops, models, parallelism, or the simulation.

### L2 — Op

The first layer that gives a cost leaf a **semantic name** (`model.attn.o_proj`).
An op is a named composition of one or more L1 kernels with `compile` / `eval`
entry points into the CostTree. Atomic ops (`qkv`, `o_proj`, `gate_up`, `down`,
`rms_norm`) are a generic single-kernel wrapper instantiated at the L4 wiring site
with no file of their own; a **compound op** (e.g. `FlashInferAttentionOp`) gets a
file because one logical operation fans out to several kernels with op-specific
cost math.

*Sees:* per-rank shapes; forwards to L1 kernels.
*Doesn't see:* parallel config, simulation state, or which worklet called it.

### L3 — Worklet

One model-module-level unit of a decoder layer (pre-attention, the attention
block, the MLP block) assembled from ops. L3 is where a parallelism scheme's
**per-rank partition is derived** (`resolve_config` does the Megatron sharding) and
where cross-op composition — sum, overlap, pipeline — is expressed. The group
suffix names the sync grain: `Local` (one GPU, no collective) vs `TP` (one
tensor-parallel sync section whose boundary is an all-reduce). A worklet still
returns only metrics; it owns no simulation state.

*Sees:* the global config + parallelism degree, and its ops.
*Doesn't see:* an op's sub-kernels; simulation state; the model as a whole.

### L4 — Model arch

Wires worklets into a full model or one disaggregated model section and compiles
its stable CostTree section(s) **once**. A homogeneous iter-wise decoder layer is
folded with `Scale{num_layers}` rather than materialised N times. L4 exposes
separate typed contracts for whole-iteration, layer-wise attention, and
layer-wise FFN cost queries, plus the topology/KV/transfer facts L5 needs.

*Sees:* worklets, the model config, the parallel config.
*Doesn't see:* the KV pool, admission, or the clock — those are the worker's.

### L5 — Worker

The first stateful layer. A worker composition is
`<KV, Admission<Selection>, Execution> × Shell(cadence)`. KV owns resource
lifecycle; selection policy is the sole owner of fresh pending membership;
execution lowers state into a typed L4 input; the concrete shell owns cadence,
overlap, and completion. Production shells cover whole-iteration, pull+decode,
layer-slot attention, and buffered FFN timelines.

*Sees:* one arch contract, its selected components, request state, and local clock.
*Doesn't see:* other workers, or how requests reach it.

### L6 — Orchestrator

Routes a deployment's requests across **pools** of workers. The deployment forms a
hierarchy — deployment → pool → group → worker — and the **pool boundary is L6's
only abstraction**: routing above the pool (L6b, inter-pool) is separate from
routing within a pool (L6a, pool-local). L6 exposes a single object to the layer
above: the `Flow` trait (`on_arrival` / `tick` / `cluster`).

*Sees:* pools of workers, the request stream.
*Doesn't see:* the tick loop, the clock, or trace I/O — it is driven by L7.

### L7 — Sim infrastructure

The deployment-agnostic run machinery: the single-threaded tick loop with one
global clock (`run_sim`), the trace frontend that replays a request arrival trace,
the streaming parquet logging, and the launcher that presents all of this as a run
interface. The **deployment layer** sits at the L6/L7 seam: a `Deployment` trait
plus a `build_flow` dispatch is the single `dyn` erasure point that turns a preset
into the concrete orchestrator + workers L7 then drives.

*Sees:* a `Flow`, a clock, an arrival trace, the log sink.
*Doesn't see:* what kind of deployment it is driving — that is erased behind `Flow`.

## Module → layer map

Which directory implements which layer. The Rust core is `simulator/src/`; the
Python profiling and the launcher live at the repo top level, and the analyzer is a
standalone crate.

| Directory | Layer | What it holds |
|---|---|---|
| `profiling/` | L1 (Python) | Kernel runners + `profile.db` + the `perf_api` facade; the only source of measured numbers. |
| `simulator/src/timing/` | L1 (Rust) | Kernel caches, the PyO3 bridge, and the CostTree (`compile`/`eval`, flatten/aggregate). |
| `simulator/src/op/` | L2 | Atomic `Op<K>` (no file) and compound ops: `attention/` (FlashInfer, DSA indexer, sparse MLA), `moe/` (dispatch, combine, grouped FP8 GEMM), `gemm/`; `comm/` and `ssm/` are still module-header stubs. |
| `simulator/src/worklet/` | L3 | Per-module sync sections (`pre_attn_local`, `attn_block_tp`, `mlp_block_tp`, …). |
| `simulator/src/arch/` | L4 | Iter-wise and layer-wise model assembly (Llama3, Qwen3-MoE, GLM-5.2 DSA/MoE); compiled CostTree sections and model contracts. |
| `simulator/src/worker/` | L5 | KV/admission/execution components, concrete cadence shells, build recipes, and per-worker logging. |
| `simulator/src/orchestrator/` | L6 | Deployment → pool → group → worker hierarchy; the `Flow` trait; pool-local vs inter-pool routing. |
| `simulator/src/deployment/` | L6/L7 seam | The `Deployment` trait + `build_flow` dispatch — the single `dyn` erasure point. |
| `simulator/src/sim/` | L7 | `run_sim` tick loop, the trace frontend, the run summary. |
| `simulator/src/log/` | L7 | Streaming parquet writer, the cost-log + manifest sidecar. |
| `simulator/src/schema/` | cross-cutting | The preset/config schema shared across layers. |
| `simulator/src/common/`, `introspect/` | cross-cutting | Shared helpers; the kernel-query / introspection subcommands. |
| `launcher/` (top level) | L7 run interface | Builds the release binary + PyO3 env, prewarms the cache, expands sweeps, validates presets. |
| `gpu/spec.json` (top level) | cross-cutting | Per-GPU peaks, bandwidth, and the alias table every component normalizes GPU names against. |
| `model/` (top level) | outside the stack | HF `config/*.json` (verbatim downloads) plus `work/`, an **independent** necessary-work accountant: theoretical-minimum FLOPs/bytes derived from a model config alone. |
| `analyzer/` (top level) | post-run | Standalone crate (no PyO3): Rust computes analysis subjects, Python renders them. |
| `alignment/` (top level) | validation | Normalizes shared nsys evidence for both simulator validation against real serving and simulator-guided attribution of real-framework changes. |

The last three sit **outside** the seven layers on purpose. `analyzer/` and
`alignment/` read a finished run's artifacts rather than participating in it, and
`model/work/` never reads the simulator's kernel tree — a lower bound derived
from the sim's own decomposition could not detect the sim doing redundant work.

## The spine: build once, evaluate per iteration

Every cost layer (L1 timing through L4 arch) shares one shape. At **build** time
the arch walks its worklets and ops, each op mints its fixed leaf slot(s) into a
`CostTreeBuilder`, and the result is flattened once into a `Vec<FlatCostNode>` plus
a `CostManifest` sidecar that carries the human-readable names. At **eval** time —
the hot path, run every simulated iteration — each leaf interpolates its cache for
the current batch shape, the evaluator streams those metrics into the slot buffer
in visit order, and a single reverse pass aggregates them to one total. Names never
touch the hot path; the analyzer replays the taxonomy from the manifest afterwards.

This is what lets a run evaluate millions of iterations cheaply: the expensive
structural work happens once, and each iteration is pure interpolation plus a
bottom-up fold.

## The cross-language boundary

The only language seam is the PyO3 bridge inside L1, between `simulator/src/timing/`
(Rust) and `profiling/` (Python). Identity is carried by a string that must stay
equal on both sides: a kernel's `KernelSpec::KIND` equals its Python `profile.db`
table name equals its Python facade stem, and the bridge derives the call name as
`get_{kind}_times`. Everything above L1 is pure Rust; everything the profiler
measures is Python. The launcher wires the env so this boundary is invisible at run
time.
