---
name: impl-add-model-work-label
description: >-
  Use when implementing independent model.work necessary-work support for a
  Transformer architecture or checkpoint so Optimality can produce R6/R7.
  Covers attention and FFN specs, heterogeneous LayerStack composition, exact
  architectures[0] registration, hand-derived parameter/FLOP goldens, semantic
  location maps, and exact-iteration verification. Not for simulator L1-L4
  timing composition.
---

# Implement Add Model Work Label

Own the independent `model.work` accountant for one architecture family. The
accountant answers what work the model must perform from the model config and
workload alone. Never derive minimum FLOPs or bytes from simulator kernels,
`CostTree` shapes, profiler rows, or achieved-work logs; doing so would make
redundancy self-referential.

Use the simulator tree only after the semantic work inventory is fixed, to map
semantic rows onto stable, exact location names.

## Preconditions

- `top-split-model-into-kernels` has established the true architecture from HF
  modeling code, a paper, or another authoritative implementation.
- The model config exists and its exact `architectures[0]` value is known.
- Before writing a location map, the L4 arch is wired and its manifest/location
  names are stable. The independent builder and golden tests may be written
  earlier.

If the architecture math is unresolved, return to `top-split-model-into-kernels`.
If location names are still changing, finish the L4 wiring before mapping them.

## Read First

- `model/work/README.md` for the accountant, segment, parameter, and validation
  contracts.
- The target `model/config/*.json` and the architecture evidence produced by
  exploration/splitting.
- The closest files under `model/work/attention/`, `model/work/ffn/`, and
  `model/work/models/`.
- `model/work/registry.py` and `tests/test_model_work.py`.
- For location maps, `analyzer/rust/src/optimality/location.rs`, the final
  simulator manifest/location names, and the closest file under
  `model/work/location_maps/`.

## Workflow

### 1. Establish the independent inventory

Write down, from the true model implementation:

- exact architecture string and layer schedule;
- attention mechanism, head geometry, cache/state shape, and prefill/decode
  behavior;
- FFN dimensions, routed/shared experts, top-k, and router behavior;
- embeddings, output head, norms, parameter tying, and dtype assumptions;
- formulas for compulsory FLOPs, parameter bytes, and persistent state traffic.

Do not use ServingStudio Sim's partitioned or fused kernel shapes as evidence for these
formulas.

### 2. Compose or extend the accountant

- Reuse an existing `AttentionSpec` or `FFNSpec` when its math matches.
- Add a mechanism under `model/work/attention/` or `model/work/ffn/` only when
  the math is genuinely new.
- Keep the per-model builder under `model/work/models/` thin: parse config fields
  and compose specs.
- Represent heterogeneous schedules with `LayerStack`; use stable tags that
  produce meaningful semantic segment prefixes.

### 3. Register the architecture

Register the builder in `model/work/registry.py` under the exact
`architectures[0]` string. Preserve the hard failure for unknown architectures;
silent fallback would label unsupported models incorrectly.

### 4. Add hand-derived goldens

Extend `tests/test_model_work.py` with independently calculated expectations for
representative prefill and decode workloads:

- total and activated parameter counts;
- FLOP buckets and compulsory byte buckets;
- scaling invariants for sequence length, sampled positions, layer schedules,
  and expert activation where applicable.

Do not calculate expected values by calling the implementation under test.

Run:

```bash
uv run pytest tests/test_model_work.py
uv run python -m model.work <config> --prefill 8192@0 --json
uv run python -m model.work <config> --decode 256x4096 --json
uv run python -m model.work.parameter_counts <config>
```

Choose smaller representative shapes when the architecture requires different
inputs, but cover both prefill and decode semantics.

### 5. Add semantic location maps

Add one versioned map per supported architecture/deployment location set under
`model/work/location_maps/`.

- Match exact `arch_types` and exact non-communication CostTree locations.
- Consume every semantic row exactly once.
- List every non-communication location exactly once.
- Use an empty semantic list when a location has zero compulsory work under the
  cross-leaf-fusion convention.
- Never infer minimum-work formulas from the simulator location being mapped.

If a deployment has no correct map, report it as unsupported instead of claiming
full R6/R7 readiness.

### 6. Verify the analyzer handoff

Run a small simulation for each mapped deployment and inspect one exact,
`batch_locked` iteration through the analyzer. Confirm:

- `necessary_work_available=true` and the expected mapping id is reported;
- every non-communication kernel receives the intended necessary-work mapping;
- aggregate minimum work reconciles with `model.work` for the same workload;
- R6 necessary-work coverage and R7 redundancy are present and numerically sane.

Use `operate-run-simulation` to launch the run and `operate-use-analyzer` to read
the exact iteration; do not substitute sweep-level or UI-derived numbers.

## Completion Report

Return the architecture string, reused/new specs, builder and registry paths,
golden-test results, supported location-map/deployment ids, and exact-iteration
R6/R7 evidence. Explicitly list any deployment that remains unmapped.
