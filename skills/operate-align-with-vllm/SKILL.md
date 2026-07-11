---
name: operate-align-with-vllm
description: Use when the user wants to run, rerun, resume, or inspect an end-to-end VibeSim-to-vLLM alignment experiment. Covers the four explicit phases (simulation, instrumented vLLM/NSYS profile, measured-shape timing prediction, and analysis/rendering), semantic matching from folded measured kernel positions to simulated CostTree slots, and explicit embedded-label decisions. NOT for implementing alignment infrastructure, running an unrelated deployment simulation, or using timing-predict without measured vLLM alignment.
---

# Align VibeSim With vLLM

Operate the existing alignment pipeline and produce a reproducible comparison
between one VibeSim run and one measured vLLM run. Keep matching decisions in a
folded labeled kernel-sequence JSON; never create a separate kernel-mapping YAML.

Read `alignment/README.md` and `alignment/profiler/README.md` before preparing a
run. Defer config and artifact details to those files and the current launcher
schemas when they change.

## Respect the supported boundary

Check the current boundary before allocating a GPU. Alignment v1 supports:

- one direct `deployment: unified` simulation target;
- one main group and one replica;
- one vLLM model rank (`server.tp_size: 1`);
- the `vllm_text` input builder with `group_assignment: single`.

Stop and report the unsupported dimension instead of silently reducing a TP,
replica, deployment, or input-builder request to this boundary.

## Match measured kernels to simulated slots

Treat matching as semantic alignment between two different decompositions:

- measured side: ordered CUDA kernel occurrences grouped by indexed vLLM phase
  and folded exact repetition;
- simulated side: named L1 leaf slots inside a timing-predict CostTree;
- join key: a stable model operation chosen from structural and semantic
  evidence, not a demangled kernel name alone.

Use this evidence in descending order:

1. **Phase** — establish whether the occurrence belongs to `preprocess`,
   `forward`, `postprocess`, or `sample`.
2. **Folded position** — use prefix, repeat body, and suffix placement. A repeated
   body matching the model layer count is strong evidence for one transformer
   layer; prefix and suffix kernels usually belong to model-level work.
3. **Ordered neighbors** — identify an occurrence by the surrounding norm,
   projection, attention, activation, and projection pattern.
4. **Kernel semantics** — inspect what the implementation computes. Treat
   `suggested_category` as a hint, never as sufficient proof.
5. **CostTree slot contract** — confirm that the candidate `simulated_slots`
   jointly represent the same model work and multiplicity.
6. **Cross-sequence consistency** — require the same semantic position across
   prefill, mixed, and decode sequence variants to receive compatible labels.

Do not globally map a kernel name to one operation. The same implementation name
may occur at different model positions, while one model operation may launch
several different kernels.

### Build a decision worksheet

Before editing the JSON, reason through one row per stored folded occurrence:

| Field | Purpose |
|---|---|
| phase | Own the occurrence by runtime phase |
| sequence and folded path | Identify prefix/repeat/suffix position |
| name and suggested category | Preserve measured identity and parser hint |
| inferred model operation | State the semantic role |
| evidence | Record position, neighbors, implementation, and shape reasoning |
| simulated slots | Name the exact CostTree leaves, if they exist |
| decision | `mapped`, `unmapped`, or unresolved |

Resolve every unresolved row before analysis. The worksheet is a reasoning aid;
the labeled folded JSON remains the only analyzer input.

## Make mapping decisions

Apply these rules in order:

1. **Map exact semantic ownership.** Map a measured occurrence when it computes
   all or a well-defined component of one modeled operation and the chosen slot
   set jointly owns that same work.
2. **Allow many measured kernels to one operation.** For example, cache update,
   attention mainloop, and combine launches may jointly implement one modeled
   attention operation. Give them the same operation and slot when the CostTree
   intentionally models them as one leaf.
3. **Keep helpers explicit.** Mark framework bookkeeping, allocation/fill/copy,
   launch preparation, sampling, or other runtime work `unmapped` when no
   equivalent simulated slot exists. Do not attach it to a nearby model op just
   to improve coverage.
4. **Use one-to-many only for additive workload ownership.** Map a fused
   measured operation to multiple simulated slots when their combined leaf
   workloads represent exactly that operation. The analyzer counts measured
   duration once and sums the selected folded slot workloads. This sum is not
   overlap-aware wall time under `Max` branches. If the slots represent separate
   operations or require an unjustified duration split, leave the row unresolved.
5. **Keep simulator-only work visible.** A simulated leaf with no measured
   counterpart remains an unmapped simulated slot in the report; never invent a
   zero-duration measured kernel.
6. **Require metadata consistency.** One operation must keep the same `type`,
   `role`, and ordered `simulated_slots` list; one simulated slot must not
   represent multiple operation names.

Every stored occurrence, including occurrences inside a repeat body, must have
exactly one nested `label`:

```json
{"label": {"status": "unmapped"}}
```

or:

```json
{
  "label": {
    "status": "mapped",
    "operation": "model.operation_name",
    "type": "gemm",
    "role": "concise semantic ownership",
    "simulated_slots": [
      "unified.slot_main",
      "unified.slot_epilogue"
    ]
  }
}
```

Keep the full measured `name` and `suggested_category` at every sequence
location. Do not introduce kernel IDs, a catalog, or materialized iteration
assignments. A label in a repeat body intentionally applies to every exact
expansion of that body.

When reusing labels from an earlier capture, first remove only `label` and
source-provenance fields from both inventories and require exact equality of the
remaining phase, sequence, fold, kernel-name, and category structure. Transfer
labels only after that equality check passes. Otherwise review every changed
position; never fuzzy-match names or ordinals.

## Run the four phases

Run all commands from the repository `main/` directory with `uv run`.

### 1. Simulation

Create a fresh `logs/YYYYMMDD_N_<short_name>/` directory and keep these sibling
configs in it:

```text
simulation.yaml
profile.yaml
timing_predict.yaml
analyze.yaml
```

Give the four phases disjoint artifact roots: `simulation/`, `profile/`,
`timing_predict/`, and `analysis/`. Make the simulation trace and profile
frontend path resolve to the same source trace.

Dry-run the ordinary simulation first and inspect the expanded run count and
output directory:

```bash
uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml --dry-run
uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml
```

Require a completed simulation artifact set before continuing.

### 2. Instrumented vLLM/NSYS profile

Preflight the selected GPU, port, NSYS installation, model cache, and the
instrumented fork environment. Keep `fork_python` pointed at the literal venv
path; verify that interpreter imports both `torch` and `vllm`.

Use `cuda_profiler_api` and CUDA graph node tracing unless diagnosing a known
capture issue. Validate the profile config before launching:

```bash
uv run python -m launcher alignment profile logs/<experiment>/profile.yaml --dry-run
uv run python -m launcher alignment profile logs/<experiment>/profile.yaml
```

For a long launch, monitor the server log and process state without interrupting
NSYS finalization. After completion require:

- `profile_result.json` with successful capture validation;
- `parsed.json` with the intended iteration range and indexed phases;
- `kernel_sequences.json` using schema v2 and `folded-v1`;
- a nonempty replay JSONL for E2E comparison.

### 3. Measured-shape timing prediction

Configure the typed input builder explicitly. `measured_phase: forward` selects
the phase used to reconstruct model input shapes; it does not limit later
analysis to forward kernels.

Run:

```bash
uv run python -m launcher alignment timing-predict \
  logs/<experiment>/timing_predict.yaml
```

Expect a matching case map plus `raw/cost_manifest/` and `raw/cost_log/`.
Inspect the emitted CostTree leaf slots before finalizing mapping decisions. A
cold profiling database may JIT-fill missing rows and therefore require the
matching GPU.

### 4. Label, analyze, and render

Copy `profile/kernel_sequences.json` to the experiment-level
`kernel_sequences_labeled.json`. Apply the matching and decision rules above to
every stored occurrence. Reference only that JSON from
`iteration.labeled_kernel_sequences_file` in `analyze.yaml`.

Validate the labeled inventory before the full analyzer run:

```bash
uv run python -c 'from pathlib import Path; from launcher.alignment_config import load_labeled_kernel_sequences; load_labeled_kernel_sequences(Path("logs/<experiment>/kernel_sequences_labeled.json"))'
```

Then run:

```bash
uv run python -m launcher alignment analyze logs/<experiment>/analyze.yaml
```

The strict analyzer must expand the folded inventory losslessly, match every
stored name/category against `parsed.json`, validate every mapped slot against
the CostTree, and retain all captured phases. Fix the labeled source when this
validation fails; do not patch the analyzer output.

## Verify completion

Do not call the experiment complete until all of the following hold:

- `analysis/alignment_manifest.json` uses the current schema and points to the
  snapshotted labeled inventory;
- no separate kernel-mapping YAML or JSON is generated or consumed;
- iteration metadata lists every captured phase;
- payload breakdowns contain a `phase_summary` and measured kernels carry their
  phase;
- mapped operations include model-level suffix work when the simulator models
  it, rather than silently stopping at the repeated layer body;
- coverage reports retain both unmapped measured kernels and unmapped simulated
  slots;
- at least one prefill/mixed and one decode breakdown plot are visually checked
  for phase boundaries and operation arrows;
- E2E request pairing has no unexplained missing requests.

Report the experiment directory, exact commands, request and iteration counts,
captured phases, measured and simulated mapping coverage, total iteration error,
E2E latency/throughput error, explicit unmapped gaps, and links to the labeled
inventory, reports, and representative plots.

## Resume and failure policy

Treat each phase as an explicit checkpoint. Resume from the last verified
artifact root; do not rerun an expensive GPU profile merely because a later
label or analyzer step failed.

When a cross-stage artifact disagrees with the current schema, determine whether
the workspace changed during the run. Prefer regenerating the owning phase.
Perform a metadata-only migration only when the old and new fields have provably
identical semantics and record that migration with the experiment; never alter
measured timings, kernel identities, request results, or CostTree values to make
validation pass.
