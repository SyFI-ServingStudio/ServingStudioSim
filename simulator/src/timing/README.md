# L1 Timing — the Rust cost layer

This is where the simulator answers **"how long did this work take?"** It owns
no event loop and no model semantics; it turns *kernel shapes* into *metrics*
(time / flops / bytes / energy) and composes those leaf metrics into a per-iter
total. It never measures a GPU itself — it asks the Python profiling layer for
profiled samples, fits an interpolating cache over them once, then evaluates that
cache millions of times.

This is the practical, code-matching reference. For the layer overview see
`doc/detailed_design/L1.md`, and `COST_TREE.md` for the CostTree spec. If this file
disagrees with the code, the code wins — open an issue.

## What it exposes / what it requires

**Exposed upward** (re-exported from `crate::timing`, see `mod.rs`) — higher
layers (L2 ops, L3 worklets, the cost-tree compile) consume these:

- **Kernel surface:** the `KernelConfig` trait and the `Probe` / `CacheProbe`
  traits (`eval(&Input) -> LeafMetrics`, the per-leaf cost query). Note
  `Kernel<S>` and the per-kind `*KernelConfig` / `*KernelInput` are **not**
  re-exported here — import them from `crate::timing::kernels::*`.
- **The cost tree:** `CostTree`, `CostNode`, `CostTreeBuilder`, `Evaluator`,
  `FlatCostNode`, `CostManifest`, `LeafDesc` — build a cost structure once, then
  stream per-iter metrics through it.
- **Metric / coverage types:** `LeafMetrics`, `Metrics4`, `CoverageFlags`.
- **cost_log capture:** `SlotInput`, `AttnPrefillLog`.
- **Sweep grid:** `SweepGrid`, `Axis`, `Coords`, the `SweepCoords` trait + derive.
- **The bridge:** `PerfApiBridge`, `BuildError`, `KernelMissing`.
- **Derives:** `KernelConfig`, `SweepCoords` (re-exported under the trait names).

**Required from below:** the Python **`profiling.perf_api`** facade (the L1
profiling package), reached over PyO3 by `bridge`. The bridge calls
`get_{kind}_times` / `count_missing_{kind}` / `enable|disable_jit_profiling` /
`get_current_gpu_name`. The host must therefore have the PyO3 env wired (the
launcher does this) and a populated `profile.db`.

## The two phases (the central idea)

The structure of a cost query is stable across iterations; only the leaf numbers
change with batch shape. So the work splits cleanly:

```
BUILD (once per run)
  KernelSpec::sweep_grid  ─→  enumerate(config, grid, backend) ─→ Vec<ArgsPayload>
        │                                                              │
        │                                  PerfApiBridge.get_times ────┘  (PyO3 → profile.db)
        ▼                                          │
  Kernel<S>::build  ◄── fits a Cache over the profiled samples per backend
  CostTreeBuilder   ─→  CostTree (recursive CostNode tree + ordered leaf slots)
                              │  flatten()
                              ▼
                         Vec<FlatCostNode>  + CostManifest sidecar

EVAL (every iteration, hot path)
  for each leaf:  Kernel::eval(&Input) ─→ LeafMetrics   (cache interpolation, best-of-N backend)
        │  Evaluator streams them into buf[slot] in visit order
        ▼
  CostTree::aggregate(flat, buf, scratch) ─→ one LeafMetrics   (single reverse pass, reused scratch)
```

**Build** profiles the sweep grid and fits a cache; **eval** is pure
interpolation + a bottom-up fold. Names exist only at build time (in the slot
list / manifest), never on the hot path or in log rows (INV-5).

## Directory map

```
mod.rs            Public re-exports for crate::timing (the upward surface above).

bridge/           The PyO3 boundary to Python profiling.perf_api.
  core.rs           PerfApiBridge: get_times / count_missing / JIT toggle /
                    get_current_gpu_name + dry-run report accumulation.
  payload.rs        Wire types: KernelKind (=&'static str), ArgsPayload, DType,
                    KernelMetrics (raw profiled row → flops()/bytes() derivations).
  error.rs          BuildError / PerfApiError.

kernels/          One file per kernel kind + the generic engine.
  engine.rs         KernelSpec + KernelConfig traits, Kernel<S> (build/eval/Probe),
                    and the `register_kernel!` inventory hook (no central enum).
  single_gemm.rs    Example kind: declares Config/Input, KIND, sweep_grid,
  rms_norm.rs       cache_kind, enumerate. All the generic machinery is in engine.
  elementwise.rs  · flashinfer_attn_{prefill,rect,decode}.rs · all_reduce.rs

cache/            Fit profiled samples → an interpolating cache; eval off-grid.
  mod.rs            Cache trait, CacheKind enum, build_cache dispatch, OutlierWarning.
  interp.rs         Metrics4 (f32, SIMD-packed), LeafMetrics, CoverageFlags, the
                    branchless `locate` (the measured hot-path floor).
  linear_1d.rs · linear_2d.rs · direct_1d.rs · cliff_2d.rs · log_2d.rs · backend.rs

sweep.rs          SweepGrid + Axis presets (e.g. token_axis, a profiler↔sim
                  contract curve) + Coords / SweepCoords (Input → coord projection).
cost_tree.rs      CostNode/FlatCostNode/CostTree/CostManifest + Evaluator +
                  aggregate(). The compile-once-eval-many structure.
slot_input.rs     SlotInput: closed enum of leaf inputs, captured for cost_log.
result.rs         Probe (typed per-leaf eval) + CacheProbe (dyn, JSON, for
                  the kernel-query introspection subcommand).
routing.rs        RoutingDistribution: MoE expert ppm + Hamilton apportionment.
```

## Adding a kernel

One file: `kernels/<kind>.rs` declaring `Config` (`#[derive(KernelConfig)]`, must
carry `backends` + `gpu_name`), `Input` (`#[derive(SweepCoords)]`), a unit
`Spec: KernelSpec` with `const KIND`, and `sweep_grid` / `cache_kind` /
`enumerate`; end with `register_kernel!(FooKernel, FooSpec)`. The generic
`Kernel<S>` supplies build/eval/`Probe`, and `inventory` wires it into the
`kernel-query` dispatch with no central match. Add the kind to `kernels/mod.rs`
and, if its input reaches a leaf, one line in `slot_input.rs`. Full end-to-end
procedure: skill `top-add-kernel`; the Rust timing/cache wiring alone is skill
`impl-wire-kernel-to-rust`, and its cache-fidelity check is `impl-validate-kernel-cache`.

Two optional `KernelSpec` hooks (default no-ops, but load-bearing when needed):

- **`profile_kind()`** — the profiler facade / `profile.db` table the specs are
  profiled against (`get_{profile_kind}_times`); defaults to `KIND`. Override when
  a cache *variant* (same physical kernel, different cache axes) reuses an existing
  kind's profiled rows: the variant keeps its own `KIND` for registry/identity but
  profiles through the base kind's facade.
- **`infeasible_mask()`** — row-major mask (aligned with the grid / `enumerate`)
  of physically unreachable cells; their sample is forced non-finite at build so
  the cache drops + renormalizes instead of fabricating a value. `flashinfer_attn_prefill`
  is the shipped example: its `(A,B)` re-axis grid has an unreachable `A < B/2`
  corner that the mask strips.

## Cross-language identity (must stay equal)

```
KernelSpec::KIND  ==  Python table_name  ==  Python facade stem
   e.g.  "single_gemm"
→ bridge calls   format!("get_{kind}_times")  → "get_single_gemm_times"
→ Python exposes get_single_gemm_times (generated by profiling/facade.py)
```

A kernel's `ArgsPayload` field set (from `enumerate`) is the wire schema that the
Python `*Args` dataclass validates — the two must match field-for-field. The
sweep `Axis` presets are equally a contract: every grid point must be a profiled
row, or build fails with `MissingEntry` (a `build-cache-only` run can
`enable_jit_profiling` to fill them).

## Key types at a glance

- **`KernelMetrics`** (bridge) — a raw profiled row (f64 + optional rate fields).
  `flops()` / `bytes()` derive absolute work from the rate × time (comm rows use
  `busbw × time`; compute rows use `mem_bw × time`).
- **`Metrics4`** (cache) — the f32, 16-byte-SIMD-packed `{time,flops,bytes,energy}`
  stored/blended inside caches and rolled up by the aggregate.
- **`LeafMetrics`** = `Metrics4` + `CoverageFlags` — the per-leaf eval result.
  Coverage (`EXTRAPOLATED` / `JIT` / `NO_COVERAGE`) ORs up the tree, so an
  off-grid leaf anywhere surfaces at the root.

The compose/aggregate machinery (`CostNode`, `FlatCostNode`, `CostTree`,
`Evaluator`, `CostManifest`) has its own write-up — see **[COST_TREE.md](COST_TREE.md)**.

## Build-time vs runtime safety

`PerfApiBridge::new()` calls `disable_jit_profiling` so the default state is
"sim-runtime-safe": a cache miss raises `MissingEntry` rather than launching a GPU
profile mid-simulation. Only the `build-cache-only` entry point re-enables JIT.
`enable_dry_run()` switches build to count-missing-only (no cache fit), feeding
the launcher's `--cache-report`.
