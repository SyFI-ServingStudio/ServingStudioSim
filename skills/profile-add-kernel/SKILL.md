---
name: profile-add-kernel
description: Use when adding a full MLSim L1 kernel (a new kernel kind or a new backend of an existing kind) end to end — both the Python profiling side (per-kernel module + runner + registry) and the Rust timing side (KernelSpec + cache wiring). Covers docs-first design, naming/wire-string rules, the per-kernel files on both sides, lazy RunnerRef wiring, generated perf_api facades, tests, validation, and a mandatory progress + output report.
---

# Profile Add Kernel (L1, full stack)

Adds one L1 kernel across **both** sides of the boundary:

- **Python profiling** (`profiling/`) — measures the kernel on real GPUs and
  stores rows in `profile.db`.
- **Rust timing** (`simulator/src/timing/`) — turns those rows into a fast
  cache lookup for the simulator.

Per-kernel knowledge is **one file per side**, mirrored:
`profiling/kernels/<kind>.py` ⇄ `simulator/src/timing/kernels/<kind>.rs`. The
two MUST agree on the `KIND` wire string and the per-row field set. Runners stay
in their own files because the main process must never eager-import torch/cuda.

> **Single-agent skill.** This skill assumes you are the only agent editing the
> tree. If several kernels are being added in parallel, the orchestrator isolates
> each agent in its own git worktree (see the `orchestrate-parallel-subagents`
> skill) — you still run every step here normally inside your worktree.

---

## Naming & wire string (the #1 trap — read before you name anything)

The §8 design tables label kinds in **CamelCase for display** (`AttnPrefill`,
`EltWise`, `Norm`). That is NOT the wire string. The wire `KIND` is:

- **snake_case**, and it is the SINGLE source of truth used three ways at once:
  the DB `table_name`, the Rust `KernelSpec::KIND`, and the generated facade stem
  `get_<kind>_times` / `count_missing_<kind>`.
- The registry validator **enforces `table_name == kernel_kind`** (see
  `profiling/db/registry.py`), so they cannot diverge.

So `AttnPrefill` → `KIND = "attn_prefill"`; `EltWise` → `"elementwise"` (follow
the orchestrator/repo name, not the doc label). The §8 table's runner *path* and
*family* names are likewise **illustrative, not literal** — the real runner path
follows repo convention (`profiling/runners/<family>/<backend>.py`).

---

## 0. START HERE — Report Protocol (mandatory, do this first)

You MUST keep a live report. Do not write any code before step 0 is done.

1. The orchestrator gives you a report path (e.g.
   `/m-coriander/coriander/kanzhu/MLSim_workspace/agent-trace/<kind>.md`). If
   none was given, default to that path.
2. **Before editing anything**, copy the entire **Checklist** (section 2) into
   that file verbatim under a `## Subagent Notes` heading, boxes unchecked.
3. As you finish each item, flip `[ ]` → `[x]` **immediately** (not in a batch)
   and append a one-line note of what you did / what command you ran. If an item
   is N/A, mark `[~]` and say why.
4. **Idempotency / resume:** if the report or any target file already exists
   (e.g. a prior pass, or your own context was compacted mid-run), do NOT blindly
   recreate or overwrite. VERIFY the file against the skeleton, mark the item
   `[~] verified pre-existing`, and continue from the current checkbox state.
5. When all boxes are resolved, append the **Output Report** (section 5).
6. The task is NOT done until every box is resolved and the Output Report is
   present. Your final message is a ≤10-line summary pointing at the report file.

---

## 1. Required reading (read before editing; cite the anchors you used)

- `docs/file_structure.md` — L1 `profiling/` and `simulator/src/timing/` layout.
- `docs/detailed_design/L1/design.md`:
  - §2.3 runner examples + ownership boundary; §2.5 / §2.5b `Timer` / `Energy`.
  - §3.2.1 registry / `KernelProfilerSpec` / args-schema ownership. Also read
    `profiling/db/registry.py`'s `BackendSupport` (the per-backend dtype/GPU
    capability that drives backend selection — step B2c below).
  - §8.1 / §8.2 compute / comm kernel tables — locate your kernel's row. **But
    these tables only give family / runner / cache / metric family / the Timer
    to use. They do NOT list the config fields, nor the Config-vs-Input split.**
    For those you must also read the **per-kind sweep-shape table** (the
    `*KernelInput` field list) and the **cache-variant table**, and confirm
    against the reference pair below. If the row is missing OR partial OR its
    naming diverges from the request, that is a "stop and ask" (section 6).
  - §8.3 the `(kernel_kind, backend)` coverage invariant.
- **Reference implementation** for the *measurement* (what the runner actually
  does): `ref/profile/<family>/` — e.g. attention →
  `ref/profile/attention/flashinfer_profiler.py`, norm →
  `ref/profile/norm/rmsnorm_flashinfer.py`, elementwise →
  `ref/profile/elementwise/elementwise_triton.py`. Mirror the real kernel call
  from here; do not invent the library API.
- **Shape-of-truth reference pair** for the *file structure* (open both, mirror):
  - Python: `profiling/kernels/single_gemm.py` + `profiling/runners/gemm/torch.py`.
  - Rust: `simulator/src/timing/kernels/single_gemm.rs` (+ how it is wired in
    `simulator/src/timing/kernels/mod.rs`).
- **Multi-GPU comm reference pair** (ONLY if your runner spawns a rank-group
  launcher — see the runner-contract callout in §3): `profiling/kernels/all_reduce.py`
  + `profiling/runners/comm/{nccl,nvshmem}.py` + the shared spawn-once orchestration
  in `profiling/runners/comm/_batch.py`. A single-GPU compute kernel does NOT need
  these — its single_gemm reference above is complete.

---

## 2. The Checklist (copy verbatim into your report, then check off as you go)

```
## Subagent Notes
Kind: <kind>   Backend: <backend>

Phase A — Design
[ ] A1 Read required docs; list the exact §anchors that define this kernel.
[ ] A2 Classify: NEW kind  |  NEW backend for existing kind. (Different paths below.)
[ ] A3 Decide names. KIND = snake_case wire string (== table_name == facade stem;
       NOT the §8 CamelCase label). Plus backend string, <Kind>Args fields,
       runner module+function, Rust <Name>{Spec,Config,Input,Kernel}.
[ ] A4 Decide shape split (derive from the sweep-shape + cache tables AND the
       reference pair — §8.1 does NOT encode it): static Config fields vs runtime
       Input (sweep) fields; metric family (COMPUTE/COMM); the Timer the §8.1 row
       pins (e.g. cuda_event vs do_bench); the Rust CacheKind (Cache1DLinear /
       Cache1DDirect / Cache2DLinear — Log/Cliff are UNBUILT, build_cache returns
       FitFailed).
[ ] A5 Confirm Python <Kind>Args fields == Rust per-row enumerate fields (minus
       `backend`) == runner kwargs. This contract is load-bearing; write it down.
       (Runner kwargs = the single-spec fn's `**kwargs` for a compute kernel;
       for a list-native comm kernel, the fields each spec dict carries and the
       `_per_rank_batch` fn reads out per size — same field set either way.)

Phase B — Python profiling side
[ ] B1 Write runner profiling/runners/<family>/<backend>.py (add an empty
       __init__.py if the <family>/ package dir is new). Lazy-import heavy libs
       INSIDE the fn (mirror ref/profile/<family>/ for the real call), use the
       Timer the §8.1 row specifies, call Energy.perf(per_iter_time_ms=time_ms)
       for compute, return ComputeMetrics/CommMetrics.
       Contract split (see §3 runner-contract callout): COMPUTE kernel = write the
       single-spec `profile_<kind>(**kwargs) -> Metrics` shown here; the registry
       auto-wraps it via `batched` (do nothing extra). LAUNCHER/COMM kernel (spawns
       TorchMpLauncher / NvshmemLauncher) = ALSO expose `profile_<kind>_batch(
       kwargs_list) -> list[RunnerResult]` that spawns the rank group ONCE and loops
       sizes inside, else you re-init per shape.
[ ] B2 NEW kind: create profiling/kernels/<kind>.py — KIND const, <Kind>Args
       frozen dataclass(KernelArgs), register(KernelProfilerSpec(...)) with a
       lazy RunnerRef and table_name=KIND. Compute: RunnerRef function_name =
       "profile_<kind>", leave list_native defaulted False. List-native comm:
       function_name = "profile_<kind>_batch" AND list_native=True.
       NEW backend: add a second register(...) call in the existing
       profiling/kernels/<kind>.py; reuse <Kind>Args; no new file/barrel line.
[ ] B2c Declare each backend's capability on its register(...):
       supports=BackendSupport(compute={dtypes}|None, kv={dtypes}|None,
       gpus={gpu_names}|None). This is the SINGLE SOURCE OF TRUTH for the
       `--emit-backends` options column AND the launcher backend-map validator
       (NOT profile.db row presence). compute=None = dtype-agnostic (comm is
       size-keyed, elementwise byte-keyed); `kv` only for attention (independent
       of compute — bf16-query/fp8-KV is valid); `gpus` only to gate a backend to
       specific parts (e.g. trt = frozenset({"NVIDIA B200"})). Mirror an existing
       kernel: single_gemm.py (compute-only), flashinfer_attn_prefill.py
       (compute+kv+gpus), all_reduce.py (compute=None). MUST match the Rust dtype
       tags in C2b.
[ ] B3 NEW kind only: add `from profiling.kernels import <kind>  # noqa: F401`
       to profiling/kernels/__init__.py (match the existing single_gemm style;
       ruff will re-sort the import block — that is expected).
[ ] B4 Add tests/test_<kind>.py importing the kernel module DIRECTLY (so it
       passes before/independent of the barrel): args field-set+coercion, KIND
       wire string, register-spec shape (table_name==kernel_kind), lazy-import
       (runner module absent from sys.modules after importing the kernel module).
       List-native comm: also assert spec.list_native is True and
       spec.runner_ref.function_name == "profile_<kind>_batch" (mirror
       tests/test_all_reduce.py).
[ ] B5 ONLY if an Args field is a tuple/list (distribution-sensitive op, §4): add
       the tuple/list branch to db/batch.py::_coerce_value so the inbound JSON
       list coerces to the declared origin (tuple → frozen args stay hashable).
       DB storage already handles it (db/table.py); scalar-only kernels skip this.

Phase C — Rust timing side
[ ] C1 Create simulator/src/timing/kernels/<kind>.rs: <Name>KernelConfig
       (#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug)], backends +
       static fields), <Name>KernelInput (#[derive(SweepCoords)], sweep fields),
       <Name>Spec impl KernelSpec (KIND, sweep_grid, cache_kind, enumerate),
       `pub type <Name>Kernel = Kernel<<Name>Spec>`.
[ ] C2 enumerate emits the SAME field set as Python <Kind>Args + `backend`
       (verify against A5).
[ ] C2b If the config carries a dtype, tag its field so the enumerate record emits
       a TYPED dtype for the launcher capability gate: #[compute_dtype] on the
       compute field (a GEMM/norm's `dtype`, attention's `q_dtype`), #[kv_dtype]
       on attention's KV field. #[derive(KernelConfig)] generates the
       compute_dtype()/kv_dtype() accessors from the tags (untagged = None =
       dtype-agnostic, e.g. comm/elementwise). These MUST agree with B2c's
       BackendSupport compute/kv axes — the launcher reads the typed field, never
       re-parses describe_config. Mirror single_gemm.rs (compute) /
       flashinfer_attn_prefill.rs (compute+kv).
[ ] C3 Wire simulator/src/timing/kernels/mod.rs: `pub mod <kind>;` + `pub use`.
[ ] C4 Add #[cfg(test)] tests in <kind>.rs mirroring single_gemm.rs (config
       identity, describe_config, input coords flatten, enumerate emits all wire
       fields; for 2D also assert sweep_grid len==2 and cache_kind).
[ ] C5 Register the kernel's Input in the closed cost-log SlotInput enum in
       simulator/src/timing/slot_input.rs: add the `use` import for
       <Name>KernelInput and one `log_inputs!` line (`<Variant> =>
       <Name>KernelInput`). MANDATORY for every kernel — Op::eval's
       `K::Input: Into<SlotInput>` bound makes a missing entry a compile error the
       moment the kernel is composed into an Op<K>; add it up front (harmless: the
       Input already derives Clone + Serialize).

Phase D — Validate + report
[ ] D1 ruff: uv run ruff check profiling/ tests/   (clean)
[ ] D2 pytest: uv run pytest tests/test_<kind>.py
[ ] D3 registry/facade: uv run python -m profiling list --json shows the new
       (table, backend); uv run python -c "from profiling.perf_api import
       get_<kind>_times, count_missing_<kind>" resolves.
[ ] D4 cargo test --lib timing   (Rust kernel compiles + its tests pass)
[ ] D5 NO GPU profiling run (would write the shared profile.db) unless the
       orchestrator explicitly approved it. State this explicitly.
[ ] D6 Write the Output Report (section 5).
```

> Run all Python tools via `uv run <tool>` — NOT bare `.venv/bin/<tool>` (often
> permission-blocked in sandboxes). `uv` reuses the synced env / cache.

---

## 3. Step details + skeletons

### B1 runner (`profiling/runners/<family>/<backend>.py`)
Mirror `profile_single_gemm` for structure and `ref/profile/<family>/` for the
real kernel call: lazy `import torch` / `import flashinfer` inside the function,
raise `ProfilerNotImplemented` if the lib/CUDA is unavailable, time the kernel
closure with **the Timer the §8.1 row pins** (`Timer.do_bench` for GEMM,
`Timer.cuda_event` for attention, etc. — read the table's Timer column, don't
copy the reference's blindly), compute energy with
`Energy.perf(..., per_iter_time_ms=time_ms)`, return `ComputeMetrics(time_ms,
tflops, memory_bandwidth_gbps, energy_j)`. Raise `KernelLaunchFailed` on
`RuntimeError`. Do NOT touch DB / env / CUDA_VISIBLE_DEVICES. If `<family>/` is a
brand-new dir, add an empty `__init__.py` package marker.

### The runner contract: single-spec (default) vs list-native (launcher kernels)
The worker calls every runner as a **list runner**:
`run(kwargs_list: list[dict]) -> list[RunnerResult]` (1:1, in order). You almost
never write that signature yourself:

- **Compute kernels (the common case) — write single-spec, unchanged.** Keep
  `profile_<kind>(**kwargs) -> Metrics` exactly as `single_gemm` shows. The
  registry wraps it via `batched()` automatically (`list_native` defaults False),
  which loops the specs and captures per-item errors. The worker subprocess
  already amortizes the expensive setup (import / CUDA context) once per chunk, so
  the per-spec in-process call is cheap. **Do nothing extra.**

- **Multi-GPU comm kernels that spawn a launcher — make it list-native.** A runner
  that does `TorchMpLauncher(n).run(...)` / `NvshmemLauncher(n).run(...)` pays a
  full `mp.spawn` + `init_process_group` / `nvshmem.init` (seconds to ~12 s) on
  EVERY call. Left single-spec, the adapter loops it → one spawn **per message
  size** (~19× wasted init). Instead expose `profile_<kind>_batch(kwargs_list) ->
  list[RunnerResult]` that spawns the rank group **once** and loops the sizes
  inside the live group (a `_per_rank_batch` fn reads each spec's fields and
  allocates/frees per size). Reuse `profiling/runners/comm/_batch.py::run_comm_batch`
  (spawn-once + whole-chunk failure semantics: the ranks run sizes in lockstep, so
  a per-size abort would deadlock the collective — fail the whole chunk instead).
  `RunnerResult(metrics | error)` is the per-item element. Set `list_native=True`
  and point `function_name` at the batch entry in B2.

### B2 per-kernel module (`profiling/kernels/<kind>.py`)
Copy `single_gemm.py`: `KIND: str = "<kind>"` (snake_case), a
`@dataclass(frozen=True) class <Kind>Args(KernelArgs)` with the per-row fields,
and one `register(KernelProfilerSpec(kernel_kind=KIND, backend="<backend>",
runner_ref=RunnerRef("profiling.runners.<family>.<backend>", "profile_<kind>"),
table_name=KIND, args_schema=<Kind>Args, metric_family=MetricFamily.COMPUTE,
batch_outlier_policy=BatchOutlierPolicy(), supports=BackendSupport(...)))`.

The `supports=` (step B2c) is the load-bearing new field: it declares which
dtypes / GPU this `(kind, backend)` can run, and is what the launcher's
`--emit-backends` skeleton (the `options` column) and its backend-map validator
consult — *not* whether `profile.db` has a row (a cold cache still counts as
supported). Get it from the reference kernel comment / the library's real
constraints, not from what happens to be measured. A compute-agnostic kernel
(comm, elementwise) passes `compute=None`.

For a **list-native comm** kernel (see the runner-contract callout above), mirror
`all_reduce.py` instead: `metric_family=MetricFamily.COMM`, point the `RunnerRef`
`function_name` at `"profile_<kind>_batch"`, and add `list_native=True` to the
`KernelProfilerSpec(...)`. Everything else (table_name==KIND, args_schema, lazy
RunnerRef) is identical.

### C1 Rust kernel (`simulator/src/timing/kernels/<kind>.rs`)
Copy `single_gemm.rs`. Static dims → `<Name>KernelConfig` fields (besides
`backends`); runtime sweep dims → `<Name>KernelInput` fields (one field per sweep
axis). If the config has a dtype field, tag it so the enumerate record carries a
typed dtype (step C2b): `#[compute_dtype]` on the compute field, `#[kv_dtype]` on
attention's KV field. `#[derive(KernelConfig)]` reads the tags to generate the
`compute_dtype()`/`kv_dtype()` accessors (untagged → `None`). These typed fields
must match the Python `BackendSupport` axes (B2c) — the launcher reads them
directly instead of parsing the `describe_config` string.
- **1D** (e.g. rms_norm, elementwise): `sweep_grid` = `SweepGrid::new(vec![
  Axis::token_axis()])`; `cache_kind` = `Cache1DLinear`; `enumerate` uses
  `grid.expand_1d`.
- **2D** (e.g. attention seq×kv): `sweep_grid` =
  `SweepGrid::new(vec![Axis::token_axis(), Axis::token_axis()])` (one preset per
  axis); `cache_kind` = `Cache2DLinear` (Log/Cliff are unbuilt → `FitFailed` in
  `cache/mod.rs::build_cache`); `enumerate` uses `grid.expand_2d`, **row-major**
  (axis0 outer, axis1 inner). Mirror the 1D structure for the rest.
`enumerate` always does `.with("backend", backend)` + one `.with(field, value)`
per Args field (snake_case keys matching the Python dataclass).

---

## 4. Ownership rules (don't cross these)

- Runners only allocate/launch/measure and return metrics — a single `Metrics`
  dataclass (compute; the registry wraps it in the list contract via `batched`) or
  a `list[RunnerResult]` (list-native comm; see the §3 runner-contract callout).
  They never touch the DB, env, or `CUDA_VISIBLE_DEVICES`.
- Do NOT hand-write `get_<kind>_times` in `perf_api.py` — `facade.py` generates
  it from the registry.
- Do NOT add a central enum on either side — `KIND` is the single source of
  truth; the bridge derives Python facade names from it.
- Reuse existing `Timer` methods, `Energy`, metric dataclasses, default env, and
  built `CacheKind` variants. Adding a new metric family / Timer method / cache
  variant / launcher / subprocess env is a **stop and ask** (section 6).

### The only wiring beyond the per-kernel files

The Rust side has NO central kind registry to edit: `register_kernel!` makes each
kernel self-register via `inventory`, and `introspect::lookup` iterates that set
at link time. So the **only** Rust wiring for the L1 kernel itself is the
`kernels/mod.rs` `pub mod` + `pub use` (step C3). There is intentionally no
exhaustive "all kinds" list, match, or guard test to update — do not add or
recreate one (a redundant `introspect::registry_covers_every_kernel_kind` test
that hardcoded the kind set was removed 2026-05 precisely because it taxed every
new kernel for zero coverage the per-kernel tests + the build don't already give).

The one Rust file beyond the per-kernel pair that you DO always touch is the
cost-log `SlotInput` enum in `timing/slot_input.rs` (mandatory step C5). It is a
deliberately *closed* enum (inline capture, no per-slot box), and `Op::eval`'s
`K::Input: Into<SlotInput>` bound turns a missing entry into a **compile error**
the moment the kernel is composed into an `Op<K>`. Add one `log_inputs!` line
(`<Variant> => <Name>KernelInput`) + the matching `use` import up front for every
new kernel — it is harmless (the Input already derives `Clone + Serialize`) and
saves a surprise compile error during later L2/L3/L4 wiring.

### Vector / tuple-valued args field (distribution-sensitive ops)

Most kernels' Args are scalars. A distribution-sensitive op (e.g. `grouped_gemm`'s
`per_group_batches: tuple[int, ...]`, where the Rust `enumerate` emits a JSON
array via `RoutingDistribution::to_per_expert_counts`) needs one extra touch the
scalar path doesn't:
- DB storage already works: a `tuple[...]` / `list[...]` args field maps to a JSON
  `TEXT` column (`db/table.py::_sqlite_type` → `_to_db_value` json-dumps it), and
  it participates in the UNIQUE key like any scalar — no schema change.
- But `db/batch.py::_coerce_value` MUST handle the `tuple`/`list` origin so the
  inbound JSON list from the Rust facade is rebuilt into the declared origin
  (coerce to a **tuple** so the frozen `KernelArgs` stays hashable, and coerce
  each element by the inner type). Add that branch if it is missing.
- Rust: `ArgsPayload::with(key, vec)` already accepts a `Vec<u32>` (→ `Value::Array`),
  and `#[derive(KernelConfig)]` accepts a `Vec<_>` identity field (e.g. `local_ppm`).

---

## 5. Output Report template (append to your report file when done)

```
## Output Report
- Kind / backend: <kind> / <backend>   (NEW kind | NEW backend)
- Doc anchors used: L1 design §..., §...   (table + sweep-shape + cache anchors)
- Field contract (A5): Args fields = [...]; Rust enumerate = [...] (+backend);
  runner kwargs = [...] — MATCH: yes/no
- Files created/edited: <list, marking any [~] verified pre-existing>
- Barrel lines applied (or deferred to orchestrator if isolated): __init__.py +
  mod.rs lines.
- Public facades: get_<kind>_times, count_missing_<kind>
- Example input spec: {"...": ...}
- Validation results: D1 ruff <pass/fail>; D2 pytest <n passed>; D3 list/facade
  <shown?>; D4 cargo <n passed>; D5 GPU run <skipped/why>.
- Skill friction (REQUIRED, be candid): which steps were unclear, missing, wrong,
  or out of order? Where did you read source instead of the skill? One change
  that would have helped most?
- Remaining work: <e.g. real GPU profiling, other library variants, none>.
```

---

## 6. Stop and ask

Pause and ask the user (or note loudly in the report and adopt the nearest sane
choice) when:

- The kernel's §8 row is missing, partial (no fields/axes/metric), OR its naming
  diverges from the requested kind/backend.
- One §8 row maps to **many library variants** (e.g. FlashInfer prefill =
  {ragged, paged, block, chunked, rect}) and it is unclear whether each variant
  is a separate `backend` string, a separate runner function, or a config field.
- It needs a new metric family, cache interpolation type, `Timer` method,
  launcher protocol, or subprocess env.
- A real GPU run would write the shared `profile.db` without explicit approval.
