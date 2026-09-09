---
name: operate-run-alignment
description: >-
  Run, resume, or inspect the shared VibeSim↔framework evidence pipeline for
  vLLM or SGLang. Its artifacts support both simulator validation through
  top-align-with-framework and framework optimization through
  top-compose-real-framework-from-sim.
---

# Run Alignment

Operate the existing alignment pipeline to produce one reproducible evidence
exchange between VibeSim and a measured vLLM or SGLang run. Keep every matching
decision in the labeled kernel-sequence JSON; never create a separate mapping
YAML.

Read `alignment/README.md` and `alignment/profiler/README.md` first — they own
the phase configs, artifact roots, and commands. This skill covers only the
operator judgment those docs leave open.

## Choose the consuming direction

Run the same capture, normalization, labeling, and comparison path in both
directions. Choose the consumer from the user's question before interpreting the
result:

- **Align VibeSim to framework:** determine whether the simulator faithfully explains
  the measured framework. Hand the completed artifacts to
  `top-align-with-framework` for judgment and simulator-side repair routing.
- **Align framework to VibeSim:** use the simulated target and its measured-only /
  simulated-only gaps to locate framework implementation opportunities. Hand
  the same artifacts to `top-compose-real-framework-from-sim`, which owns trial
  selection, implementation, controlled baseline/trial measurement, and the
  retain/reject decision.

This operate skill produces and checks evidence. It does not decide which side
is wrong, select a framework optimization, or edit either implementation.

## Respect the supported boundary

The server pipeline supports one `deployment: unified` target, one main group,
and one replica. The real engine may be `vllm` or `sglang`; its explicit world
size is `server.tp_size * server.dp_size` and must equal both the visible-device
population and the simulation topology. Use the shared `engine_text` builder
(`vllm_text` and `sglang_text` remain accepted adapter spellings):
`group_assignment: single` for one attention-DP group, or `per_dp_rank` when
each attention-DP rank schedules its own batch. Stop and report an unsupported
replica, deployment, topology, or input-builder request; never silently reduce
it to TP1 or collapse DP batches.

## Run the shared phases

From `VibeSim/`, run each phase through the launcher (`uv run python -m launcher
alignment {profile,timing-predict,analyze,sim}`; see README for the configs).
Each phase is an explicit checkpoint with a disjoint artifact root; no phase
launches the next. The duty-cycle `gpu_time_multiplier` is no longer hand-derived
before the simulation — the analyzer's **kernel-align** pass emits
`recommended_gpu_time_multiplier`
(`Σ measured_gpu_cycle_ms / Σ measured_ms`),
and the simulation phase injects it automatically. So `analyze` splits into two
semantic passes and runs on both sides of the simulation. Dry-run profile and sim
before the real launch.

Require launcher metadata to declare the producer/artifact kind explicitly
(`framework_capture`, `timing_predict`, or `simulation`). Never infer it from an
incidental file such as `raw/run_meta.json`. A consumer must reject absent or
unknown producer kinds rather than choosing semantics from directory contents.

When several cases have independent artifact roots, run the same phase for all
ready cases concurrently. Complete shared prerequisites once (build, profile DB,
labels or common configuration), then fan out the cases and collect their exit
statuses separately. Do not serialize independent experiments merely to make
failure attribution easier: their disjoint output roots already provide that
attribution. Respect real resource conflicts, such as a bounded GPU pool or an
exclusive launcher lease, but let the repository's resource manager schedule
those constraints instead of serializing the whole campaign in advance.

**Do not hand-drive a multi-case matrix, and do not write a generator script for
one.** `launcher alignment-campaign` already implements the paragraph above:
declare the matrix in a pack under `presets/alignment/<pack>/`, then

```bash
uv run python -m launcher alignment-campaign check  --pack <pack>
uv run python -m launcher alignment-campaign render --pack <pack> --host <host> --out-root <root>
uv run python -m launcher alignment-campaign run    --pack <pack> --out-root <root> --phase <P>
```

One `run` invocation is **one phase across every ready case**, scheduled through
the repository's resource manager, with each case's exit status journaled
separately. Completed case×phase pairs are skipped, so a labeling or analysis
failure never costs a repeated capture; `--refresh` re-runs one deliberately.
Preview the plan and each excluded case's reason with `--dry-run`.

There is no `--all` and you must not build one. The batch axis is one phase over
many cases; chaining phases would delete the explicit checkpoint between them.
Advance the pipeline by issuing the next `--phase` yourself after inspecting the
previous one, exactly as for a single case.

`launcher/alignment_campaign/README.md` owns the pack schema. Reach for the
single-config path below when there is one workload and no matrix.

1. **Profile** — instrumented vLLM or SGLang. Preflight the GPU, port, NSYS,
   model cache, and fork venv (`fork_python` must import `torch` and the
   selected engine).

   `profile_kind` selects one of three passes, and they are not
   interchangeable — each buys one kind of evidence at the cost of another, so
   pick by which check will consume it (see `alignment/README.md` for the
   config fields):

   | `profile_kind` | NSYS | Buys | Costs |
   |---|---|---|---|
   | `nsys` (default) | yes, bounded window | per-kernel/segment truth for Check 1 | a short window, plus capture/flush pauses; must disable popularity logging |
   | `workload_metrics` | no | the complete scheduler timeline and per-request timing, uncontaminated, for the whole run | no kernel detail |
   | `expert_popularity` | no | logical expert counts/probabilities by layer, for MoE routing demand | its EPLB sync and D2H logging must never enter timing evidence |

   Use `cuda_profiler_api` + CUDA graph node tracing for the `nsys` pass.

   Judging E2E (Check 3) from the `nsys` pass alone is a common mistake: that
   window covers a slice of the run and carries profiler pauses, so a TTFT/TPOT
   distribution read off it is neither the whole population nor an undisturbed
   one. Run `workload_metrics` for that, with identical model, topology,
   backend, and workload settings so the two passes are comparable.

   Run `expert_popularity` for **any** MoE model, not only an EP deployment.
   Expert parallelism is what makes routing skew produce dispatch/combine
   traffic, but a grouped GEMM is charged per expert group whatever the
   placement: skew leaves the total token-expert selections unchanged while
   redistributing them into fuller and emptier groups. A local/TP1 MoE arch
   that assumes balanced routing will therefore mis-cost its expert GEMM, and
   the deviation surfaces in Check 1 as an unexplained MoE slot gap that is
   easy to misattribute to the kernel cost model.
2. **Timing prediction** — set the typed input builder. It reads the simulation
   *preset* (`simulation.yaml` via `simulation_preset`) for gpu/arch/backends, so
   it runs before any completed simulation. `measured_phase: forward` only
   reconstructs input shapes, it does not limit later analysis. Inspect the
   emitted CostTree leaf slots before mapping. A cold profile DB may JIT-fill and
   need the matching GPU.

   **Do not manually pre-fill the cache for an alignment campaign.** Run
   timing-predict or simulation on the real case inputs and let their ordinary
   `perf_api` lookups JIT-fill the exact demanded shapes. Never invent a broad
   Cartesian profiling sweep as a prerequisite. If demand-driven execution
   leaves misses, use `operate-profile-existing-kernel` only to diagnose or
   retry those exact missing shapes, then rerun the owning timing-predict or
   simulation stage. This keeps cache population tied to production demand and
   prevents speculative axes from dominating the experiment.
3. **Label + kernel-align** — initialize a labeled inventory with explicit
   unmapped decisions, label every stored occurrence per the rules below, and
   reference only that file from `iteration.labeled_kernel_sequences_file`:

   ```bash
   uv run python -m alignment label initialize \
     logs/<experiment>/profile/kernel_sequences.json \
     logs/<experiment>/kernel_sequences_labeled.json
   uv run python -c 'from pathlib import Path; from launcher.alignment_config import load_labeled_kernel_sequences; load_labeled_kernel_sequences(Path("logs/<experiment>/kernel_sequences_labeled.json"))'
   ```

   Use `initialize --unfold` only when an implementation-identical repeated
   layer boundary and one-off model boundary need different labels. It preserves
   the same kernel-align entry point while making each occurrence addressable as
   `literal-v1`; use `before_name` to state the distinguishing successor.

   With a pack, `alignment-campaign label --pack <pack> --run-dir <case>` does
   the initialize-apply-check loop against the pack's stored rules. It applies
   them until the label **state** stops changing rather than a fixed number of
   passes — `apply_rules` counts an overwrite as applied even when the label is
   unchanged, so a fired-count loop would never terminate. Read the reported
   `unfired` rules: one that matches nothing in any case is dead weight from an
   older inventory.

   The rules are a set, not a sequence: every file the manifest names is loaded
   and the union applied, so no label may depend on which rule is tried first.
   When you add a rule, narrow its matchers until nothing else claims the same
   positions — do not rely on placing it before or after another rule.
   `subsumptions` (in `check`) proves an order-dependent pair from the rule text
   alone; `disagreements` (here, at the fixpoint) catches the overlaps it cannot
   prove but a real capture exhibits.

   Run `analyze` with only `iteration.enabled` (the kernel-align config, no
   `simulation_log_dir`). The strict analyzer expands the folded inventory
   losslessly, validates names/categories against `parsed.json` and mapped slots
   against the CostTree, and writes `recommended_gpu_time_multiplier` into
   `reports/alignment_iteration_report.json`. This pass needs no DES simulation;
   fix the labeled source on failure, never the analyzer output.
4. **Simulation** — ordinary VibeSim preset with explicit worker settings:

   ```bash
   uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml
   ```

   Kernel alignment is not a prerequisite. If adopting a reported recommendation,
   set `worker.gpu_time_multiplier` in the preset explicitly and record its source
   and any cross-workload reuse in experiment notes. Confirm the value baked into
   the run's `params.json`; the launcher never reads a report to override it.
   The longer wall time can
   change batching, TTFT, TPOT, E2E, and throughput. Require a complete artifact
   set before continuing.
5. **e2e-align, render** — run `analyze` again with `workload`/`e2e` enabled (the
   e2e-align config, pointing `simulation_log_dir` at the completed sim) and then
   render. These subjects consume the sim that already baked in the multiplier.

   The complete `workload_metrics` run is the authority for real serving
   throughput and request latency. Report its
   `measured_client_completion_tps`. Both `alignment-workload` and
   `alignment-e2e` consume this full run directly and must not name or inspect
   the bounded NSYS capture. The profile records req-frontend's host-monotonic
   replay window so workload analysis can exclude startup/preflight iterations
   independently. NSYS belongs only to the earlier kernel-align phase; never
   divide full-run tokens by an NSYS span or disable workload analysis merely
   because the kernel capture used a smaller representative trace.

Do not use the phase NVTX envelope as GPU E2E (host submission ranges; graph
kernels run after the marker closes), and do not subtract an iteration's own busy
union from its host span and call the remainder idle or CPU overhead — it can be
queue time occupied by a prior iteration.

## Inspect emitted artifacts

Use the artifact that owns the requested granularity; do not infer a
per-physical-kernel error from rows that are joined many-to-one by semantic
operation.

| Question | Artifact and field |
|---|---|
| Per-iteration total error | `payloads/alignment_iteration_series.json` → `iterations[].delta_ms` / `relative_diff_pct` |
| Per-iteration, per-operation error | `payloads/alignment_iteration_breakdowns.jsonl` → one iteration record's `operation_summary[]` |
| Measured physical-kernel detail | The same breakdown record's `measured_kernels[]` |
| Simulated CostTree-leaf detail | The same breakdown record's `simulated_kernels[]` |
| Directional mapping gaps | The same breakdown record's `unmapped_measured_ms` / `unmapped_simulated_ms` |
| Per-stream intervals and reduced occurrences | `payloads/alignment_timeline_iterations.jsonl` |
| Mapping decisions and unresolved measured rows | `kernel_sequences_labeled.json` |
| Whole-analysis aggregates and recommended multiplier | `reports/alignment_iteration_report.json` |

`operation_summary[]` is the direct measured-versus-simulated comparison: it
contains `measured_ms`, `simulated_ms`, `delta_ms`, and `relative_diff_pct` for
each semantic operation in that iteration. Use `measured_kernels[]` and
`simulated_kernels[]` to explain that row, but keep their different
decompositions visible rather than manufacturing a one-to-one kernel join.

The render step may sample iterations for PNG output; the JSON/JSONL payloads
remain the machine-readable authority for every emitted iteration. When an
Analyzer resource for the result is ready, user-visible numerical reporting
must instead follow `operate-use-analyzer` and its typed-resource citation
contract.

## Preserve one raw-evidence layer

Reuse one capture-evidence parser for full serving runs and bounded repetitive
units reached through the same framework entry point. Keep profiler schema,
process/device ownership, correlation joins, interval union, ordered events, and
coarse kernel taxonomy in a shared raw layer. Keep framework scheduling
interpretation and CostTree semantics in separate consumers. Do not introduce a
second command-only capture path or an alignment-only fixed-shape input builder
merely to make a small reproduction convenient. Explicit shapes already belong
to the existing `operate-run-timing-predict` entry point.

The same evidence must expose both directions:

- measured operations with no simulated owner;
- simulated work with no measured counterpart.

Do not add labels merely to improve coverage. Every mapped or profiled owner
must match the measured production operation's backend specialization, layout,
page contract, numeric contract, and shape. A common operation name does not
make ragged, paged, calibration, or cache-free paths equivalent.

Treat trace-derived workload as evidence, not scheduler policy. Request lengths,
arrival times, and physical kernel inputs may be replayed. Observed DP placement,
equal-shaped adjacency, and profiler-created synthetic chunks must not become
predictive worker constraints. Implement chunked prefill as a generic lifecycle
over the original requests. If an analysis conditions on observed placement or
another realized decision, report that counterfactual separately from the
predictive alignment.

## Match measured kernels to simulated slots

Matching is semantic alignment between two decompositions: measured CUDA kernel
occurrences (grouped by engine phase, folded by exact repetition) and named L1 leaf
slots in the timing-predict CostTree. The join key is a stable model operation,
not a demangled kernel name. Weigh evidence in this order:

1. **Phase** — `preprocess`, `forward`, `postprocess`, or `sample`.
2. **Track** — one concurrent CUDA stream. A sequence holds `tracks[]`; track 0
   is the one that opened the range, the rest are side streams the framework
   overlapped onto the same device (vLLM runs the Qwen3-Next shared expert this
   way during decode, and inline on the main stream during prefill).
3. **Folded position** — prefix/repeat-body/suffix *within a track*; a repeat
   body matching the layer count is one transformer layer, prefix/suffix is
   model-level work.
4. **Ordered neighbors** — the surrounding norm/proj/attn/act/proj pattern.
5. **Kernel semantics** — what it computes; `suggested_category` is a hint only.
6. **Slot contract** — the candidate `simulated_slots` jointly own the same work.
7. **Cross-sequence consistency** — the same position across prefill/mixed/decode
   gets compatible labels.

One name may sit at different positions and one operation may launch several
kernels, so never map a name globally. Before editing, reason one row per folded
occurrence (phase, track, folded path, name + category, inferred operation,
evidence, slots, decision) and resolve every unresolved row.

Never reason across a track boundary. `after` / `after_name` / `before_name`
stop there by construction, because two tracks ran at the same time and have no
"before" between them — the neighbour on the other side of the edge is not
evidence about anything. Expect the same operation to sit on the main stream in
one sequence and a side stream in another; that is a scheduling fact about the
framework, not a reason to label it differently.

## Make mapping decisions

Give every stored occurrence — including inside a repeat body, where the label
applies to every expansion — exactly one nested `label`: `{"status":
"unmapped"}`, or a mapped label with `operation`, `type`, `role`, and a non-empty
`simulated_slots` (see README for the shape). Keep the measured `name` and
`suggested_category` everywhere; introduce no kernel IDs or catalog.

- **Map exact ownership.** Map when the occurrence computes all or a defined
  component of one operation and the chosen slots jointly own that work.
- **Many-to-one is fine.** Cache update, attention mainloop, and combine may share
  one operation/slot when the CostTree models them as one leaf.
- **One-to-many follows CostTree attribution.** The analyzer counts the measured
  duration once and attributes selected leaves through `Sum` / `Scale` / `Max`;
  parallel siblings contribute only through the critical child, while raw
  folded workload remains a mapping-coverage audit. If the slots are separate
  operations or need an unjustified split, leave the row unresolved.
- **Keep helpers explicit.** Mark bookkeeping, alloc/fill/copy, launch prep, and
  sampling `unmapped`; never attach them to a nearby op for coverage.
- **Keep simulator-only work visible.** A simulated leaf with no measured
  counterpart stays an unmapped slot; never invent a zero-duration kernel.
- **Stay consistent.** One operation keeps the same `type`, `role`, and ordered
  `simulated_slots`. A slot may be shared by several operations (e.g. a fused vs.
  unfused all-reduce boundary owning the same `tp_allreduce` slot); the analyzer
  resolves the per-iteration owner from the operations actually present.

When reusing labels from an earlier capture, strip only `label` and provenance
from both inventories and require exact equality of the remaining
phase/sequence/fold/name/category structure before transferring; otherwise review
every changed position and never fuzzy-match.

## Reduce logical occurrences, not raw rows

Identify one logical occurrence by phase and operation ordinal across folded
sequences and devices, never by a raw row ID. Preserve the ordered raw inventory
for audit, then reduce durations according to synchronization semantics:

- independent replicated compute contributes the critical-rank duration;
- a synchronizing collective uses arrival-to-completion semantics while keeping
  arrival wait separate from kernel cost;
- never sum the same replicated logical occurrence across ranks in a stacked
  critical path.

`unmapped` and cross-rank synchronization are independent labels. An unmapped
collective may still require synchronizing reduction.

Produce an auditable unmapped inventory grouped by semantic family and logical
critical-path contribution. Classify with phase, folded position, ordered
neighbors, semantics, and tensor/collective ownership — never a remembered
ordinal or demangled name alone. Alignment reports facts and candidate owners;
it does not select the next framework optimization.

For multi-device, multi-stream timing, construct one complete reduced path per
device first: mapped work, unmapped work, and stream overlap must share the same
device timeline. Select the critical device only after that composition. Never
take separate maxima for mapped, unmapped, and overlap components and splice
them into a path that no device executed. Collective residency or arrival wait
is timeline evidence, not kernel duration.

Stream plots use the same reduced logical work as the numerical report, not raw
residency intervals. Show material streams separately and aggregate small
streams into a lossless remainder whose total is explicit; aggregation must not
drop work or inflate the critical path.

## Preserve routing and workload equivalence

For MoE baseline reproduction, capture logical routing/popularity outside the
timed range — the `profile_kind: expert_popularity` pass above — and inject that
same demand into the simulator with `routing: custom` and the arch's
`expert_popularity_file`.
Preserve original layer and step/request identity, token count, top-k, logical
expert counts, route-weight mass, and temporal variation. A single model-wide
histogram erases layer skew and bursts and cannot support per-iteration
critical-path alignment.

If the arch has no `expert_popularity_file` field, it is costing a hardcoded
balanced assumption. That is a gap to report, not a reason to skip the pass: an
arch whose grouped GEMM reads a routing distribution can consume a measured one,
and adding the selector field is a small, patterned change.

Keep logical demand separate from framework realization. The reusable input is
the token-to-expert assignment and route weight before EP placement, padding,
duplicated sends, local permutation, and collective wait. Use realized rows and
bytes as validation evidence, not simulator demand; otherwise the framework's
inefficiency becomes a simulated requirement.

Communication-sensitive analysis also needs joint structure: source/destination
owner unique-token counts, owner-hit multiplicity, and expert co-occurrence.
Marginal expert counts size grouped compute but cannot determine dispatch
duplication or bottleneck message bytes.

Version the routing artifact with model/router/checkpoint revision,
token/request digest, sampling policy, expert namespace, observation window,
and schema. Store logical expert IDs independently from placement so target
exploration can remap experts. Validate end-to-end consumption: assignment and
weight totals, post-placement per-layer rank distributions, padding/drop policy,
generated kernel shapes, and simulated message sizes. Reading the artifact or
setting a config field is not proof that the CostTree consumed it.

## Verify completion

The experiment is complete only when: `analysis/alignment_manifest.json` uses the
current schema and points at the snapshotted labeled inventory; no separate
mapping file exists; iteration metadata lists every captured phase; payloads
carry a `phase_summary` and per-kernel phase; mapped operations include
model-level suffix work, not just the repeated body; coverage keeps both unmapped
measured kernels and unmapped simulated slots; at least one prefill/mixed and one
decode plot are visually checked for phase boundaries and operation arrows; and
E2E pairing has no unexplained missing requests.

`alignment-campaign extract` enforces the mechanizable half of that list and
refuses to emit metrics for a case that fails it, so run it before reporting.
It also reads every number below out of the reports by a fixed formula table —
**never transcribe a metric by hand**, in a report or a results table. `--pack`
is optional here: `--runs <one-off run dir>` works with no pack at all.

```bash
uv run python -m launcher alignment-campaign extract --runs <dir> --out /tmp/metrics.json
uv run python -m launcher alignment-campaign compare --measured /tmp/metrics.json
```

With `--pack`, `compare` additionally judges each metric against that pack's
declared tolerances (non-zero exit on FAIL) and warns on drift from the recorded
golden; `--markdown` writes the human-readable matrix.

Report the experiment dir, exact commands, request/iteration counts, captured
phases, mapping coverage, total iteration error, E2E latency/throughput error,
unmapped gaps, intended consuming direction, and links to the labeled inventory,
reports, and plots.

## Resume and failure policy

Resume from the last verified artifact root; never rerun the expensive GPU
profile because a later label or analyzer step failed. If an artifact disagrees
with the current schema, prefer regenerating the owning phase; migrate metadata
only when the old and new fields are provably identical, and record it. Never
alter measured timings, kernel identities, request results, or CostTree values to
make validation pass.
