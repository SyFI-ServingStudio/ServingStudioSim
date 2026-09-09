# Optimality analysis: rung and plot semantics

This document defines the current `optimality` analyzer from the implementation.
It covers the hierarchy from a CostTree kernel location through worker, pool, and
cluster, and explains every stacked-bar section shown by the offline matplotlib
renderer and `viz-ui`.

The implementation is authoritative:

- computation: `analyzer/rust/src/optimality/`;
- offline plots: `analyzer/python/optimality/optimality_plot.py`;
- interactive plots: workspace `VibeSimUI/app/src/features/metrics/optimality*.ts`.

## 1. What “optimal” means

The analyzer does not claim to find one executable optimal schedule. It constructs
a ladder of increasingly idealized lower bounds for completing the same modeled
work. Removing one constraint at a time makes adjacent rung differences
attributable:

```text
R0 Real
  - idle
R1 Busy
  - imbalance
R2 Balanced
  - batching
R3 Per-config best
  - communication
R4 Ignore network
  - hardware gap
R5 Hardware limit
  - excess over necessary work
R6 Segmented necessary
  - fusion opportunity
R7 Scope-fused necessary
```

All rung and bucket values use **GPU·seconds**, not wall-clock seconds:

```text
GPU·seconds = wall time × physical GPU count held by the worker
```

The physical GPU count `G_w` comes from
`run_meta.workers[].gpu_ids.len()`. It is not inferred from `tp`, `dp`, `ep`, or
another parallel degree. If the roster is unavailable, the analyzer degrades to
`G_w = 1` and records a caveat.

The base ladder is monotone after defensive clamping:

```text
R0 ≥ R1 ≥ R2 ≥ R3 ≥ R4 ≥ R5 ≥ 0
```

R6 and R7 are optional independent bounds. In the intended accounting they obey
`R5 ≥ R6 ≥ R7`, but R6 can exceed sampled R5 when the simulator under-accounts
model work or because the two paths use different sampling. Scope waterfalls
clamp their displayed floors to `0 ≤ R7 ≤ R6 ≤ R5`; kernel ladders retain the raw
difference and report `under_accounted`. A missing GPU spec, grid-peak sidecar, or
necessary-work label does not make the analysis fail; the affected adjacent rungs
collapse and the corresponding bar section becomes zero or the ladder ends at R5.

## 2. CostTree fold used by R2–R5

Each `cost_log` row selects a worker CostTree manifest section. The ordinary tree
fold used for runtime is:

- `Leaf`: the leaf value;
- `Sum`: sum of children;
- `Max`: `max(children) / overlap`;
- `Scale(n)`: `n × child`.

The optimality fold replaces `Max` with a balanced mean:

```text
Max(children) / overlap  →  mean(children) / overlap
```

This fold is linear. Each leaf slot `l` therefore has a precomputed coefficient
`α_l`:

```text
α_l =
  product of 1 / (number_of_children × overlap) over Max ancestors
  × product of n over Scale ancestors
```

For any chosen leaf value `v_l`, the balanced row value is:

```text
Σ_l α_l × v_l
```

The same `α_l` is used both for a worker total and for the leaf/location
attribution. Consequently, per-location R2–R5 values add back to the scope rung.

R0 and R1 are exact SQL aggregates over all rows. R2–R5 use iterations selected
by:

```text
stride = clamp((max(iter_id) + 1) / 80, 1, 50)
```

For worker `w`, the sampled fold is anchored to exact R1 with:

```text
A_w = exact_busy_ms_w × G_w / sampled_busy_ms_w
Rk_w = A_w × sampled_fold_Rk_w / 1000,  k ∈ {2, 3, 4, 5}
```

This anchor includes both the sample upscaling and the worker GPU count. If a
worker has no sampled rows, R2–R5 are set to R1 rather than zero.

## 3. Exact definition of every rung


### R0 — Real

For worker `w`:

```text
span_w =
  max(wall_start_ms + total_time_ms)
  - min(wall_start_ms)

R0_w = span_w × G_w / 1000
```

This is the GPU capacity held between the worker's first logged start and last
logged completion. The implementation holds the numerator as an intermediate
GPU·ms value and divides by 1000 when publishing GPU·seconds. It includes
scheduler gaps between logged rows.

For an exact selected iteration, there is no separate holding-span boundary.
The analyzer explicitly sets `span = Σ total_time`, so `R0 = R1` and idle is
zero.

### R1 — Busy

For worker `w`:

```text
raw_R1_w = Σ_rows total_time_ms × G_w / 1000
R1_w = min(raw_R1_w, R0_w)
```

It keeps the CostTree row totals exactly as logged but removes time between rows.
The final `min` is the common defensive monotonicity clamp; ordinary
non-overlapping worker rows already satisfy `raw_R1 ≤ R0`. The `R0 - R1`
difference is scheduler idle.

### R2 — Balanced

For each sampled row, use the observed `slot_time_ms` at every leaf and fold with
the balanced `Max → mean` coefficients:

```text
sampled_R2_w = Σ_rows Σ_leaves α_l × observed_time_ms_l
R2_w = A_w × sampled_R2_w / 1000
```

It preserves observed kernel timings, `Sum`, `Scale`, and overlap, while replacing
straggler selection inside every `Max` by equal load. The `R1 - R2` difference is
critical-path load imbalance. It is not assigned to individual kernels because
there is no unique additive leaf attribution for a `max - mean` critical-path
gap.

### R3 — Per-config best

R3 asks what the same fixed structural kernel configuration could cost at its
best profiled rate over the batch/free axis.

In **batch-unlocked** mode, the analyzer queries one grid ceiling for each unique
`(kind, kernel_config)`. It chooses exactly one rate basis:

- compute basis if any fitted grid point reaches the GPU ridge point;
- bandwidth basis otherwise;
- communication leaves use the bandwidth basis.

For a compute-classified leaf:

```text
t3_l = min(observed_time_l, FLOPs_l / grid_peak_TFLOP/s)
```

For a bandwidth-classified leaf:

```text
t3_l = min(observed_time_l, bytes_l / grid_peak_GB/s)
```

The units are converted to milliseconds in code before folding. The `min` ensures
that an incomplete or noisy ceiling never makes a lower-bound rung slower than
the observed leaf.

In **batch-locked** mode, the large-batch counterfactual is disabled:

```text
t3_l = observed_time_l
R3 = R2
```

Therefore the batching section is exactly zero in batch-locked output.

If the grid-peak sidecar cannot be loaded or generated, the affected leaf falls
back to its observed time, also making its R3 contribution equal to R2.

### R4 — Ignore network

R4 retains each non-communication leaf's R3 value and sets communication leaves
to zero:

```text
t4_l = 0      if l is communication
t4_l = t3_l   otherwise
```

Communication kinds include the named collectives and point-to-point operations
in `prepare.rs`, plus kinds prefixed by `p2p`, `nccl`, or `comm`. `moe_alltoall`
is on that list: it is the expert-parallel exchange itself, and omitting it made
R4 count hundreds of milliseconds of fabric traffic as optimizable local compute.
Its sibling `moe_alltoall_prepare` is deliberately **not** — its time is the
local atomics/index kernels that build the send layout, wrapped around a tiny
metadata exchange, so classifying it as communication would erase real local
work.

A leaf whose kind is not on that list but which is genuinely a collective will be
silently misclassified, so a new comm kind must be added there in the same change
that introduces it.

The `R3 - R4` difference is the modeled communication cost under the R3 batching
assumption.

### R5 — Hardware limit

R5 replaces the profiled grid ceiling with the matching dense compute peak or HBM
bandwidth from `gpu/spec.json`. Communication leaves remain zero.

In batch-unlocked mode, R5 reuses the compute-versus-bandwidth classification
chosen for R3. In batch-locked mode, it classifies every observed leaf from its
current arithmetic intensity relative to the hardware ridge point.

For a compute-classified non-communication leaf:

```text
t5_l = min(t3_l, FLOPs_l / dtype_hardware_peak_TFLOP/s)
```

For a bandwidth-classified non-communication leaf:

```text
t5_l = min(t3_l, bytes_l / hardware_HBM_GB/s)
```

The dtype peak is selected from the leaf config (`dtype`, `q_dtype`,
`input_dtype`, then `kv_dtype`; default `bf16`). The `R4 - R5` difference is the
profiled-kernel versus spec-sheet-hardware gap under that selected regime. It is
a kernel-maturity headroom estimate, not a prediction that the spec peak is
attainable by the current implementation.

If the GPU spec or required dtype peak is absent, the affected leaf uses its R4
upper bound, so R5 collapses onto R4 and hardware gap is zero.

### R6 — Segmented necessary work

R6 is independent of CostTree `slot_flops` and `slot_bytes`. Rust reconstructs
the workload from logged workload groups, and `model.work` computes the minimum
semantic FLOPs and bytes for the model.

For semantic segment `s` inside one allowed composition boundary:

```text
C_s = min_FLOPs_s / hardware_peak_FLOP/s(dtype_s)
M_s = min_bytes_s / hardware_HBM_byte/s
N_s = max(C_s, M_s)

R6_boundary = Σ_s N_s
```

`dtype_s` is **per segment**, reported by the labeler on the wire, not one dtype
for the whole pool. A checkpoint is mixed: an FP8 MoE still keeps its router,
its norms, and (for GLM-5.2 DSA) its BF16 FlashMLA kernel off the FP8 tensor
cores. The labeler resolves each segment's precision from the model config's own
`quantization_config` — matched per weight matrix against `modules_to_not_convert`
— and from mechanism facts that are independent of the checkpoint. Note this is a
different question from R5's `dtype`, which comes from the *measured leaf's*
config in the manifest; R6 asks what the minimum work would run at, not what the
run happened to launch.

Thus R6 keeps semantic operations segmented and pays a separate roofline for each
one. It removes work present in the simulator/kernel accounting but absent from
the model's minimum semantic work, such as repeated weight traffic or avoidable
activation traffic. Batch-unlocked analysis applies this to the aggregated scope
workload. Batch-locked analysis evaluates each fixed iteration shape first, applies
its occurrence count, and then sums those boundary results.

For a kernel ladder, a versioned semantic-location map must cover every
non-communication manifest location and consume every semantic segment exactly
once. Only then is `N_s` attached to locations. Communication locations receive
zero necessary local work. Attribution is all-or-nothing; a partial mapping does
not silently turn missing work into zero.

### R7 — Scope-fused necessary work

R7 adds the semantic work before applying one roofline inside the allowed
composition boundary:

```text
R7_boundary = max(
  Σ_s min_FLOPs_s / hardware_peak_FLOP/s(dtype_s),
  Σ_s min_bytes_s / hardware_HBM_byte/s
)
```

It is the lowest and most permissive bound: compute-bound semantic work may
offset memory-bound semantic work as if the boundary were fully fused or
overlapped. The compute term still sums *per segment* rather than dividing one
global FLOP total by one peak — fusing every leaf into a single kernel does not
fuse precisions. In batch-unlocked mode the boundary can be the worker, pool, or
cluster workload. In batch-locked mode the boundary remains each fixed iteration,
and the scope R7 is the occurrence-weighted sum of those per-iteration results.
The `R6 - R7` difference is fusion/overlap opportunity within that policy.

R7 is aggregate-only. It has no honest per-kernel decomposition, so the kernel
ladder draws it as one global segment rather than inventing leaf shares.

## 4. Plain-language rung definitions

All values below are GPU·seconds.

- **R0 — Real:** `(last iteration finish time − first iteration start time) × physical GPU count`.
- **R1 — Busy:** `sum of all logged iteration durations × physical GPU count`.
- **R2 — Balanced:** R1 after replacing each parallel group's slowest branch with its average branch time.
- **R3 — Per-config best:** R2 with every fixed kernel configuration run at its best measured grid rate.
- **R4 — Ignore network:** R3 with all communication time removed.
- **R5 — Hardware limit:** R4 with compute and memory work run at the GPU's theoretical hardware limits.
- **R6 — Segmented necessary:** the minimum semantic compute and memory work, with each semantic segment paying its own roofline cost.
- **R7 — Scope-fused necessary:** the same minimum semantic work, with all segments in the allowed scope fully fused or overlapped before one roofline cost is paid.

## 5. Meaning at each hierarchy level

### Kernel location

A “kernel” row is a manifest **location name**, not necessarily one unique
runtime launch or one kernel kind. Equal location names are interned across
sections/workers; `Max` siblings and DP replicas of that location pool together.

The additive kernel baseline starts at R2:

```text
kernel R2 = anchored Σ α_l × observed_time_l
kernel R3 = anchored Σ α_l × per_config_best_l
kernel R4 = anchored Σ α_l × ignore_network_l
kernel R5 = anchored Σ α_l × hardware_limit_l
```

There is no per-kernel R0 or R1. Idle and imbalance are scope-level critical-path
effects. If necessary-work mapping succeeds, a location also has R6; R7 remains
aggregate-only.

The legacy payload field `kernels[].real` means **balanced R2**, not R0 Real.

### Worker

A worker owns the base computation:

- R0 and R1 are exact worker SQL aggregates multiplied by `G_w`;
- R2–R5 are sampled balanced folds anchored to exact worker R1;
- R6/R7 come from the worker workload label when available.

The complete worker kernel ladder reconciles as:

```text
Σ_kernel R2 = worker R2
Σ_kernel R3 = worker R3
Σ_kernel R4 = worker R4
Σ_kernel R5 = worker R5

worker R1 = Σ_kernel R2 + aggregate imbalance
worker R0 = worker R1 + aggregate idle
```

### Pool

For every base rung:

```text
Rk_pool = Σ_workers_in_pool Rk_worker,  k ∈ {0, 1, 2, 3, 4, 5}
```

Pool idle and imbalance are therefore sums of worker gaps, not a newly measured
pool wall-span or a second cross-worker critical-path fold.

For R6/R7, aggregation follows the selected batch policy described below. The
analyzer emits a complete pool ladder; the UI selects it and does not reconstruct
it from worker JSON.

### Cluster

For every base rung:

```text
Rk_cluster = Σ_all_workers Rk_worker,  k ∈ {0, 1, 2, 3, 4, 5}
```

As at pool level, cluster R0 is the sum of worker-held GPU capacity, not
`cluster wall span × total cluster GPUs`. This is important when pools have
different active windows.

R6/R7 again follow the batch policy. The analyzer emits the complete cluster
ladder and validates that every additive location rung reconciles with its scope
total.

### Iteration

Iteration is a detail scope rather than another roll-up tier. An exact selected
`(pool, worker, iter_id)` folds every matching row, without stride sampling, and
sets R0=R1. The run payload also carries a synthetic `iteration` level made from
cluster rungs with idle forced to zero; the offline overview intentionally omits
that synthetic row because it is Busy-anchored while the other bars are
Real-anchored.

## 6. Batch-unlocked versus batch-locked necessary work

The mode changes two independent counterfactuals.

| Concern | Batch unlocked | Batch locked |
|---|---|---|
| R3 batching | Best fitted grid rate for the fixed config | Disabled: R3=R2 |
| R5 regime | Reuses R3's large-grid compute/bandwidth class | Classifies each current leaf |
| Public worker/pool/cluster floors | Add workload totals at that scope, then evaluate scope rooflines | Evaluate each distinct observed fixed-batch shape, multiply by occurrences, then add |
| Kernel-ladder worker attribution | Label a 10,000× replicated workload, then normalize; this amortizes weights without changing sequence geometry | Preserve each fixed-iteration boundary |
| Parent R6/R7 | Add FLOPs/bytes and reevaluate rooflines at the wider scope | Add child rooflines already evaluated at fixed-batch boundaries |

The locked policy prevents a compute-bound iteration from canceling a
memory-bound iteration. The unlocked policy deliberately permits that
counterfactual at wider scopes.

## 7. What every plot and bar section means

All stacked charts are drawn left-to-right from the lowest available floor.
Recoverable headroom accumulates to the right until the full bar reaches its
baseline total.

### Scope optimality waterfall

This is the cluster/pool/worker view. `viz-ui` shows the selected scope followed
by its immediate children:

- cluster view: cluster, then each pool;
- pool view: pool, then its workers;
- worker view: the selected worker;
- exact iteration view: one on-demand iteration row.

Without R6/R7, the left-to-right sections are:

| Bar section | Exact value | Meaning |
|---|---:|---|
| `hardware_optimal` | R5 | Hardware-roofline work remaining after all modeled headroom above it |
| `hardware_gap` | R4−R5 | Profiled kernel versus spec-sheet compute/HBM peak |
| `communication` | R3−R4 | Communication leaves removed at R4 |
| `batching` | R2−R3 | Loss relative to the best fitted rate for the fixed config |
| `imbalance` | R1−R2 | `Max` critical path versus balanced mean |
| `idle` | R0−R1 | Gaps between worker rows inside its held span |

When R6/R7 are available, the displayed, R5-clamped floors replace the R5
section:

| Bar section | Exact value | Meaning |
|---|---:|---|
| `hardware_necessary` on the Rust wire / `scopeFusedNecessary` in `viz-ui` | R7 | Irreducible scope-fused semantic work |
| `fusion` | R6−R7 | Benefit available from fusing/overlapping semantic segments |
| `excess_over_necessary` | R5−R6 | Simulator/kernel work beyond minimum segmented semantic work |

The replacement is telescoping after that display clamp:

```text
R5 = (R5 - displayed_R6) + (displayed_R6 - displayed_R7) + displayed_R7
```

Therefore every scope bar still sums to R0. Exact iteration bars sum to R1
because their idle section is zero by construction.

`optimality_ratio` is `R5 / bar_total`. `necessary_ratio`, when available, is
`R7 / bar_total`. Neither ratio means “fraction of wall time spent executing
useful kernels”; both are lower-bound GPU-work ratios.

The offline files are:

- `optimality_waterfall.png`: absolute GPU·seconds;
- `optimality_waterfall_normalized.png`: every bar independently scaled so its
  own Real is 100%.

The offline overview includes cluster, pool, and worker rows but omits the
synthetic iteration row.

### Per-kernel optimality bars

Each bar's total is the location's **R2 Balanced**, not R0 Real. This is why
there are no idle or imbalance sections.

Without mapped necessary work:

```text
R2 = (R2-R3) + (R3-R4) + (R4-R5) + R5
```

The sections are batching, communication, hardware gap, and hardware optimal,
with R5 drawn at the left.

With mapped necessary work, `viz-ui` splits R5 per location:

| Bar section | Value |
|---|---:|
| `necessaryCovered` | `min(R5_location, R6_location)` |
| `redundant` | `max(R5_location - R6_location, 0)` |
| `hardwareGap` | `max(R4_location - R5_location, 0)` |
| `communication` | `max(R3_location - R4_location, 0)` |
| `batching` | `max(R2_location - R3_location, 0)` |

If the independent R6 location floor exceeds R5 by more than 0.5% of R6, the
stack cannot include the excess without becoming longer than R2. `viz-ui` keeps
the stack telescoping through R5 and draws a red diamond at R6 to mark
`under_accounted` work. Smaller positive differences are retained in payload
diagnostics but treated as sampling-scale noise.

The interactive view shows at most 16 named locations and sums lower-ranked
locations into `other`. It can toggle:

- **Real scale**: absolute GPU·seconds;
- **Normalized**: each kernel's own R2 is 100%.

The offline `optimality_kernels*.png` files use the run-wide legacy top-16
location array and currently show only the R2–R5 four-section form, even when
mapped R6 data exists elsewhere in the payload.

### Kernel rung ladder

The ladder uses one horizontal stacked bar per rung. A kernel location keeps the
same categorical color across rungs, so its shrinking contribution can be
followed vertically.

For the complete-scope view:

- R0 uses all R2 kernel pieces, then appends aggregate imbalance and idle;
- R1 uses all R2 kernel pieces, then appends aggregate imbalance;
- R2 uses the attributable balanced kernel pieces;
- R3 uses each kernel's per-config-best value;
- R4 uses each kernel's ignore-network value; communication locations disappear;
- R5 uses each kernel's hardware-limit value;
- R6, when present, uses each location's mapped necessary work;
- R7, when present, is one aggregate globally fused segment.

R0 and R1 deliberately reuse the R2 kernel baseline. Their colored kernel
pieces must not be read as per-kernel R0/R1 attribution; the only exact statement
is that the appended aggregate chunks reconcile the totals.

When `viz-ui` filters the ladder to one selected kernel, it omits aggregate idle
and imbalance. The displayed R0 and R1 rows then equal that kernel's R2 merely to
keep the same rung layout; they are not standalone Real/Busy measurements for the
kernel.

`viz-ui` renders analyzer-owned worker, pool, cluster, and exact-iteration ladders
through R7 when available. The offline matplotlib
`optimality_workers/<worker>_kernel_ladder.png` currently renders worker ladders
only through R5 and adds translucent ribbons between non-zero contributions.

## 8. Reconciliation identities

These identities are the quickest way to interpret or audit a payload:

```text
R0 = R1 + idle
R1 = R2 + imbalance
R2 = R3 + batching
R3 = R4 + communication
R4 = R5 + hardware_gap
R5 = displayed_R6 + excess_over_necessary
displayed_R6 = displayed_R7 + fusion   (in the scope waterfall)
```

At worker, pool, and cluster scope:

```text
Σ_kernel R2..R5 = scope R2..R5
Σ_kernel raw_R6 = scope raw_R6          (when strict attribution exists)
```

R7 is intentionally excluded from the second identity because it is
scope-fused and aggregate-only.

## 9. Degraded cases that change interpretation

- Missing `run_meta` GPU roster: values use `G=1` for affected workers and are
  not physical GPU·seconds.
- Missing grid peaks in unlocked mode: R3 collapses toward R2; batching headroom
  is understated.
- Batch-locked mode: R3=R2 by definition; zero batching is policy, not evidence
  that the observed batches are large.
- Missing GPU spec or dtype peak: R5 collapses toward R4; hardware gap is
  understated or zero.
- Missing `model.work` label: the bar ends at R5 and uses
  `hardware_optimal`; this does not prove all R5 work is semantically necessary.
- Missing or non-reconciling semantic-location map: scope waterfall floors may
  still exist, but the kernel ladder cannot show per-location R6.
- Partial worker necessary-work coverage: the parent kernel ladder remains
  R0–R5 rather than treating missing children as zero.

Always read `meta.caveats`, `meta.peaks_source`, `meta.gpu_spec_matched`, and the
necessary-work availability fields before comparing plots across runs.
