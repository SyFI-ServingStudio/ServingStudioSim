# `optimality` — how far a run is from optimal GPU usage

Emits a **sub-optimality waterfall** in unit **GPU·seconds** (= wall time × the
worker's physical GPU count). Optimal = the minimal GPU·s to do the same work
with no idle, perfect load balancing, and kernels at their best batching / rate.
Rather than one optimal, a **ladder of increasingly-idealized lower bounds**, so
each successive gap is an attributable source of sub-optimality:

| Rung | leaf value / structure | gap vs previous = cause |
|---|---|---|
| R0 Real            | `span × G`            | — (GPU·s actually held) |
| R1 Busy            | `Σ total_time × G`   | **idle** (scheduler gaps) |
| R2 Balanced        | real times, `Max`→mean | **imbalance** (DP/EP straggler) |
| R3 per-config best | unlocked: selected grid-peak throughput; locked: R2 unchanged | **batching** |
| R4 ignore network  | R3, comm leaves → 0  | **communication** |
| R5 hardware limit  | active regime's work unit / matching GPU-spec peak | **profiled↔hardware** |

Buckets telescope and sum exactly back to Real, drawn as a stacked bar at five
levels — cluster / pool / worker / iteration (idle 0 by construction) / per-kernel.
For unlocked analysis, the independent `model.work` labeler adds segmented and
global necessary-work bounds below R5. The R5 green band then splits into
`excess_over_necessary`, `fusion`, and `hardware_necessary`; their sum remains
exactly R5. Locked analysis does not compute these bounds because its observed
operating points are fixed and cannot be globally rebatchable.

The payload also carries one per-worker kernel rung ladder. Its R0/R1 bars reuse
the additive R2 kernel baseline and append two explicit aggregate chunks
(`imbalance = R1-R2`, `idle = R0-R1`); R2..R5 carry each location's attributable
value. The renderer keeps a kernel's color across all six bars and connects it
with a ribbon, without pretending the two aggregate gaps have a per-kernel
critical-path attribution.

The UI service exposes two separate on-demand contracts for one selected
`(pool_tag, worker_id, iter_id)`: a complete one-row waterfall and an R0-R5
per-kernel ladder. Both fold every matching row rather than sampling and define
R0=R1 because an individual iteration has no scheduler holding-span boundary;
`R1-R2` remains one aggregate imbalance chunk. In batch-locked mode only, the
waterfall reconstructs that iteration's exact workload from `groups` and splits
R5 at the segmented and fully fused necessary-work floors. These independent
`model.work` bounds never enter the kernel-ladder payload because they cannot be
assigned to simulator kernels without an attribution rule.

## Files

The subject is split by pipeline stage and aggregation tier — `mod.rs` is a thin
hub (module doc + shared rung constants/helpers + `pub use run::run_optimality`):

- `run.rs` — orchestration. `run_optimality` reads `cost_log`, manifests,
  `run_meta` GPU counts and `gpu/spec.json`, wires the stages below, and assembles
  the report/payload JSON (+ the `unavailable` degrade paths, `definitions`).
- `prepare.rs` — preparation. Interns every manifest leaf into a global location
  and precomputes each `(pool, worker, section)`'s fold weights `α` + rate ceilings
  (`build_section_fold_plans`); no folding here.
- `fold.rs` — the algorithm. Exact R0/R1 SQL sums (`read_exact_worker_totals`) + the
  stride-sampled R2..R5 mean-fold hot loop (`accumulate_fold`,
  `leaf_selected_throughput_ms`).
- `levels.rs` — the worker / pool / cluster tiers. `assemble_tiers` turns the fold's
  per-worker accumulators into R0..R5 rung arrays and rolls them up; `levels_json`
  + `level_entry_json` + `rung_report_json` emit the level/report JSON.
- `kernel.rs` — the kernel tier. Per-location bars (`kernel_levels_json`, single-leaf so
  only batching/communication/hw), the report's `worst_batching`, and the
  per-worker kernel rung ladders (`worker_kernel_ladders_json`).
- `grid_peaks.rs` — the R3 ceiling. Enumerates unique `(kind, config)` from the
  manifests, asks the simulator for each config's fitted-grid peak rate in one
  batched `kernel-query peak` call (including maximum grid arithmetic intensity,
  used with the GPU ridge point to select one throughput basis), caches it as
  `raw/kernel_grid_peaks.json`. Absent + un-generatable → R3 = R2 with a caveat.
- `spec.rs` — the R5 hardware ceilings. Resolves the run's `gpu_name` to a `gpu/spec.json`
  entry by its explicit `aliases`, then dense peak TFLOP/s by dtype + HBM GB/s.
- `floors.rs` — bridge to the independent `model.work` labeler. It computes
  unlocked per-level bounds and the locked-only exact-iteration pair; failure is
  additive-only and degrades to the plain R5 ladder.

## Cost model notes

- `G_worker` is **read** from `run_meta` (`workers[].gpu_ids.len()`), never inferred
  from tp×dp×ep — the degree→physical-GPU mapping is deployment-defined.
- R0/R1 are exact SQL sums over every row; R2..R5 fold a 1-in-`stride`-iteration
  sample (rates are near-constant across iterations) **anchored to the exact R1**
  by the sampled ratio, so the ladder stays monotone and the buckets stay exact.
- The mean-mode fold (`trace::manifest::fold_mean`) is linear ⇒ a precomputed
  per-leaf weight `α` (`∏ 1/(child·overlap)` over `Max` × `∏ n` over `Scale`) gives
  both the per-worker totals and the additive per-kernel attribution in one walk.
- R3 is not a roofline of the current point. It approximates large-batch
  operation: if any fitted point's algorithmic intensity reaches the GPU-spec
  ridge, the config uses only its peak TFLOP/s; otherwise it uses only peak GB/s.
- `analyze run --lock-batch-size` disables that counterfactual (`R3 = R2`) and
  skips grid-peak generation. R5 then classifies every observed leaf separately
  from its current `FLOPs / bytes` versus the GPU ridge point. It also disables
  the global necessary-work floors, which assume the workload can be rebatchable.
- Unlocked output uses `optimality_report.json` and
  `optimality_waterfall.json`; locked output uses the separate
  `optimality_batch_locked_report.json` and
  `optimality_batch_locked_waterfall.json`. A normal analyzer invocation generates
  both whenever optimality is selected; `--lock-batch-size` recomputes only the
  locked variant. Exact iteration requests carry the mode explicitly.
- R5 uses the active regime: unlocked analysis reuses the R3 grid classification;
  locked analysis uses the current-point classification.
