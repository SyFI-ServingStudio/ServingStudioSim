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
| R6 segmented necessary | sum of per-location semantic rooflines | **redundant work** |
| R7 scope-fused necessary | fused semantic roofline within the allowed batch boundary | **fusion** |

Buckets telescope and sum exactly back to Real, drawn as a stacked bar at five
levels — cluster / pool / worker / iteration (idle 0 by construction) / per-kernel.
For both run-level modes, the independent `model.work` labeler adds segmented and
scope-fused necessary-work bounds below R5. The R5 green band then splits into
`excess_over_necessary`, `fusion`, and `hardware_necessary`; their sum remains
exactly R5. Unlocked mode may compose work into a saturated batch. Locked mode
preserves every observed iteration boundary: equal shapes are deduplicated only as
an evaluation optimization, then restored by occurrence weighting before rollup.

The payload also carries per-worker kernel rung ladders. Its R0/R1 bars reuse
the additive R2 kernel baseline and append two explicit aggregate chunks
(`imbalance = R1-R2`, `idle = R0-R1`); R2..R5 carry each location's attributable
value. The renderer keeps a kernel's color across all six bars and connects it
with a ribbon, without pretending the two aggregate gaps have a per-kernel
critical-path attribution. In unlocked mode each worker workload is saturated by
scaling its compressed additive totals 10,000× and normalizing the label back. This
is the efficient equivalent of summing saturated iterations once each semantic op's
bound has stabilized. The analyzer maps the label to locations as R6 segmented
necessary work and retains R7 scope-fused work as an aggregate-only rung. In locked
mode it evaluates both rooflines for each distinct fixed-batch shape before weighting
and addition.

The analyzer—not the UI—then reduces complete worker ladders into explicit pool and
cluster `aggregate_kernel_ladders`. R0..R5 GPU seconds are additive. For unlocked
composition, the reducer adds each location's `(FLOPs, bytes)` first, then reevaluates
the location rooflines for R6 and one scope-wide roofline for R7; a pool or cluster
R7 is therefore not `Σ worker R7` when active bounds differ. For locked composition,
the allowed boundary is each observed iteration, so the reducer adds the already
evaluated worker R6/R7 values and never lets a compute-bound iteration offset a
memory-bound one. It checks every scope before emission: `Σ_location R2..R6` must
match the corresponding scope rung, and `R6 = R7 + fusion`. The UI only selects the
requested scope and renders it.

The UI service exposes two separate on-demand contracts for one selected
`(pool_tag, worker_id, iter_id)`: a complete one-row waterfall and a per-kernel
ladder. Both fold every matching row rather than sampling and define
R0=R1 because an individual iteration has no scheduler holding-span boundary;
`R1-R2` remains one aggregate imbalance chunk. In both modes, the waterfall
reconstructs that iteration's workload from `groups` and splits R5 at the segmented
and fully fused necessary-work floors. Locked mode labels the exact batch. Unlocked
mode replicates each independent batch entry 10,000 times, labels that large-batch
counterfactual, then divides every result by 10,000; it never multiplies sequence length.

For an exact iteration, a versioned semantic-location map provides the attribution
rule. The independent labeler emits minimum FLOPs/bytes per semantic
operation; the map assigns each row exactly once to an exact manifest location.
Maps are selected by both model `arch_type` and the exact non-communication
manifest location set, because unified and PD layouts may share one model arch
type while using different location namespaces.
Only a complete, reconciling map extends the kernel ladder with a
location-attributed segmented-necessary rung. Otherwise the endpoint remains the
unchanged R0..R5 ladder and records a caveat. Per-location redundancy is
`max(R5 - necessary, 0)`; the opposite sign is retained as `under_accounted`
rather than clamped away. Because R5 is stride-sampled while R6 uses exact groups,
the payload retains the positive raw difference but classifies it as material
`under_accounted_gpu_s` only when it exceeds 0.5% of R6.

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
  per-worker accumulators into named R0..R5 values and rolls them up; `levels_json`
  + `level_entry_json` + `rung_report_json` emit the level/report JSON.
- `kernel.rs` — the kernel tier. Per-location bars (`kernel_levels_json`, single-leaf so
  only batching/communication/hw), the report's `worst_batching`, and the
  per-worker kernel rung ladders.
- `ladder.rs` — typed kernel-ladder/work domain and the sole worker → pool →
  cluster reducer. It serializes at the publication boundary only. Unlocked parents
  recompute R6/R7 from additive work; locked parents add child rooflines that were
  already evaluated at fixed-iteration boundaries.
- `grid_peaks.rs` — the R3 ceiling. Enumerates unique `(kind, config)` from the
  manifests, asks the simulator for each config's fitted-grid peak rate in one
  batched `kernel-query peak` call (including maximum grid arithmetic intensity,
  used with the GPU ridge point to select one throughput basis), caches it as
  `raw/kernel_grid_peaks.json`. Absent + un-generatable → R3 = R2 with a caveat.
- `spec.rs` — the R5 hardware ceilings. Resolves the run's `gpu_name` to a `gpu/spec.json`
  entry by its explicit `aliases`, then dense peak TFLOP/s by dtype + HBM GB/s.
- `floors.rs` — bridge to the independent `model.work` labeler. One batched call computes
  unlocked per-level/saturated-worker labels or deduplicated fixed-iteration labels;
  exact-iteration detail uses the same contract. Transport failure degrades the
  request, while one unsupported/heterogeneous scope degrades only that scope to the
  plain R5 ladder. Locked responses expose shape, iteration, affine-basis, and direct
  fallback counters in payload meta.
- `location.rs` — strict exact-iteration semantic-location mapping. It validates
  complete coverage, computes per-location `max(FLOPs/TFLOPS, bytes/BW)`, and
  attaches R6 plus redundant/under-accounted diagnostics atomically.

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
  from its current `FLOPs / bytes` versus the GPU ridge point. R6/R7 remain available,
  but they preserve fixed iteration boundaries rather than assuming global rebatching.
- Unlocked output uses `optimality_report.json` and
  `optimality_waterfall.json`; locked output uses the separate
  `optimality_batch_locked_report.json` and
  `optimality_batch_locked_waterfall.json`. A normal analyzer invocation generates
  both whenever optimality is selected; `--lock-batch-size` recomputes only the
  locked variant. Exact iteration requests carry the mode explicitly.
- R5 uses the active regime: unlocked analysis reuses the R3 grid classification;
  locked analysis uses the current-point classification.
