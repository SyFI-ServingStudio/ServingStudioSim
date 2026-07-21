---
name: operate-run-alignment
description: Use when the user wants to run, rerun, resume, or inspect an end-to-end VibeSim-to-vLLM alignment experiment. Covers the explicit phases (instrumented vLLM/NSYS profile, measured-shape timing prediction, analysis split into kernel-align and e2e-align, and simulation), the analyzer-derived GPU duty-cycle multiplier that the simulation auto-injects, and semantic labeling of folded measured kernel positions onto simulated CostTree slots. This skill runs and labels; interpreting the results to judge alignment quality (kernel deviation, duty cycle, TTFT/TPOT) belongs to top-align-with-framework. NOT for implementing alignment infrastructure, running an unrelated deployment simulation, or using timing-predict without measured vLLM alignment.
---

# Run Alignment

Operate the existing alignment pipeline to produce a reproducible comparison
between one VibeSim run and one measured vLLM run. Keep every matching decision
in the folded labeled kernel-sequence JSON; never create a separate mapping YAML.

Read `alignment/README.md` and `alignment/profiler/README.md` first — they own
the phase configs, artifact roots, and commands. This skill covers only the
operator judgment those docs leave open. To then judge whether the model is well
aligned from the produced artifacts, see `top-align-with-framework`.

## Respect the supported boundary

Alignment v1 supports one `deployment: unified` target, one main group + one
replica, one vLLM rank (`server.tp_size: 1`), and the `vllm_text` builder with
`group_assignment: single`. Stop and report an unsupported TP, replica,
deployment, or input-builder request; never silently reduce it to this boundary.

## Run the phases

From `main/`, run each phase through the launcher (`uv run python -m launcher
alignment {profile,timing-predict,analyze,sim}`; see README for the configs).
Each phase is an explicit checkpoint with a disjoint artifact root; no phase
launches the next. The duty-cycle `gpu_time_multiplier` is no longer hand-derived
before the simulation — the analyzer's **kernel-align** pass emits
`recommended_gpu_time_multiplier` (`Σ measured_gpu_cycle_ms / Σ measured_ms`),
and the simulation phase injects it automatically. So `analyze` splits into two
semantic passes and runs on both sides of the simulation. Dry-run profile and sim
before the real launch.

1. **Profile** — instrumented vLLM/NSYS. Preflight the GPU, port, NSYS, model
   cache, and fork venv (`fork_python` must import `torch` and `vllm`); use
   `cuda_profiler_api` + CUDA graph node tracing.
2. **Timing prediction** — set the typed input builder. It reads the simulation
   *preset* (`simulation.yaml` via `simulation_preset`) for gpu/arch/backends, so
   it runs before any completed simulation. `measured_phase: forward` only
   reconstructs input shapes, it does not limit later analysis. Inspect the
   emitted CostTree leaf slots before mapping. A cold profile DB may JIT-fill and
   need the matching GPU.
3. **Label + kernel-align** — copy `profile/kernel_sequences.json` to
   `kernel_sequences_labeled.json`, label every stored occurrence per the rules
   below, and reference only that file from
   `iteration.labeled_kernel_sequences_file`. Validate it before the analyzer run:

   ```bash
   uv run python -c 'from pathlib import Path; from launcher.alignment_config import load_labeled_kernel_sequences; load_labeled_kernel_sequences(Path("logs/<experiment>/kernel_sequences_labeled.json"))'
   ```

   Run `analyze` with only `iteration.enabled` (the kernel-align config, no
   `simulation_log_dir`). The strict analyzer expands the folded inventory
   losslessly, validates names/categories against `parsed.json` and mapped slots
   against the CostTree, and writes `recommended_gpu_time_multiplier` into
   `reports/alignment_iteration_report.json`. This pass needs no DES simulation;
   fix the labeled source on failure, never the analyzer output.
4. **Simulation** — ordinary VibeSim preset with the kernel-align multiplier
   auto-injected:

   ```bash
   uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml \
     --gpu-time-multiplier-from logs/<experiment>/<kernel-align-analysis-dir>
   ```

   The launcher reads `recommended_gpu_time_multiplier` and overrides
   `pools.main.groups.0.worker.gpu_time_multiplier`; confirm the printed override
   and the value baked into the run's `params.json`. The longer wall time can
   change batching, TTFT, TPOT, E2E, and throughput. Require a complete artifact
   set before continuing.
5. **e2e-align, render** — run `analyze` again with `workload`/`e2e` enabled (the
   e2e-align config, pointing `simulation_log_dir` at the completed sim) and then
   render. These subjects consume the sim that already baked in the multiplier.

Do not use the phase NVTX envelope as GPU E2E (host submission ranges; graph
kernels run after the marker closes), and do not subtract an iteration's own busy
union from its host span and call the remainder idle or CPU overhead — it can be
queue time occupied by a prior iteration.

## Match measured kernels to simulated slots

Matching is semantic alignment between two decompositions: measured CUDA kernel
occurrences (grouped by vLLM phase, folded by exact repetition) and named L1 leaf
slots in the timing-predict CostTree. The join key is a stable model operation,
not a demangled kernel name. Weigh evidence in this order:

1. **Phase** — `preprocess`, `forward`, `postprocess`, or `sample`.
2. **Folded position** — prefix/repeat-body/suffix; a repeat body matching the
   layer count is one transformer layer, prefix/suffix is model-level work.
3. **Ordered neighbors** — the surrounding norm/proj/attn/act/proj pattern.
4. **Kernel semantics** — what it computes; `suggested_category` is a hint only.
5. **Slot contract** — the candidate `simulated_slots` jointly own the same work.
6. **Cross-sequence consistency** — the same position across prefill/mixed/decode
   gets compatible labels.

One name may sit at different positions and one operation may launch several
kernels, so never map a name globally. Before editing, reason one row per folded
occurrence (phase, folded path, name + category, inferred operation, evidence,
slots, decision) and resolve every unresolved row.

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
- **One-to-many only for additive workload.** The analyzer counts the measured
  duration once and sums the selected leaf workloads — not overlap-aware wall time
  under `Max`. If the slots are separate operations or need an unjustified split,
  leave the row unresolved.
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

## Verify completion

The experiment is complete only when: `analysis/alignment_manifest.json` uses the
current schema and points at the snapshotted labeled inventory; no separate
mapping file exists; iteration metadata lists every captured phase; payloads
carry a `phase_summary` and per-kernel phase; mapped operations include
model-level suffix work, not just the repeated body; coverage keeps both unmapped
measured kernels and unmapped simulated slots; at least one prefill/mixed and one
decode plot are visually checked for phase boundaries and operation arrows; and
E2E pairing has no unexplained missing requests.

Report the experiment dir, exact commands, request/iteration counts, captured
phases, mapping coverage, total iteration error, E2E latency/throughput error,
unmapped gaps, and links to the labeled inventory, reports, and plots.

## Resume and failure policy

Resume from the last verified artifact root; never rerun the expensive GPU
profile because a later label or analyzer step failed. If an artifact disagrees
with the current schema, prefer regenerating the owning phase; migrate metadata
only when the old and new fields are provably identical, and record it. Never
alter measured timings, kernel identities, request results, or CostTree values to
make validation pass.
