---
name: dev-compose-kernel
description: >-
  Develop or integrate a real serving kernel, fused boundary, or communication
  path. Not for ServingStudio Sim L1 cost kernels.
---

# Compose A Production Serving Kernel

Follow six steps: choose a measured hotspot, try an existing kernel, build one
if needed, integrate the complete boundary, measure it in the server, and decide
whether to keep it.

## 1. Choose the real target

Start from a measured hotspot. Record the exact boundary, production shapes,
dtypes, layouts, graph mode, per-rank workload, and frequency. Use its measured
time share to estimate the largest possible end-to-end gain.

Test one change at a time. Unmapped work is an investigation lead, not proof
that a kernel should be removed or rewritten.

## 2. Use an existing kernel when possible

Search the current framework and maintained libraries first. Route read-only
source discovery through `dev-explore-kernel`.

Check that a candidate supports the required operation, shapes, precision,
layout, graph mode, and hardware. Compare it with the current kernel in a small
correctness and performance test. Compile and autotune before timing.

If it passes, continue at Step 4. If no maintained kernel fits, continue at
Step 3. Do not weaken production requirements to make a nearby implementation
fit.

## 3. Build a kernel when reuse is not enough

Use kernel-design resources to implement and profile a new candidate. One option
is MIT HAN Lab's
[`kernel-design-agents`](https://github.com/mit-han-lab/kernel-design-agents),
including `KernelWiki` and `ncu-report-skill`. Follow its official setup, record
the revision, and keep it task-local rather than silently copying it into
ServingStudio Sim. Re-profile on the target GPU.

Develop outside the server first. Test against a trusted reference and the
current kernel using production dtypes, layouts, maximum shapes, edge values,
and graph replay when applicable. Use real per-rank distributions for routed
work. When performance is poor, profile the cause instead of guessing from
elapsed time.

Passing this step proves a kernel candidate, not a serving improvement.

## 4. Integrate the whole boundary

Define which inputs use the new path and keep an explicit fallback for the rest.
Finish compilation and autotuning before the server reports readiness.

Compare the complete old and new paths, including conversions, copies, packing,
workspace, and consumer work. Removing a kernel or launch is not a win if work
merely moves elsewhere. Preserve outputs, tensor layout, buffer lifetime, and
distributed behavior.

For a communication change, compare the complete old and new data paths. Count
what each rank actually sends and receives, how often it communicates, and the
local packing or reduction work added around the collective. If several messages
are combined, verify that every receiver reconstructs the same tensors and
meaning as before, including under graph replay.

Move setup, validation, synchronization, and logging out of the timed path.
Instrumentation must not add or reorder work.

## 5. Measure in the production execution path

Use `dev-build-serving-repetitive-unit` for fast kernel and layer timing, then
run a bounded full-model test. Both must use the production loader, backend,
graph mode, and request path.

Capture comparable before/after evidence with `operate-profile-serving-run`.
Use the unprofiled run as the performance result and a bounded profile to
explain it. Verify the full replaced boundary and any fallback use.

Do not extrapolate a reduced-model result across layers, requests, or
concurrency. Do not call a kernel-count reduction a performance result.

## 6. Promote or stop

Keep the change only after correctness, production graph/backend checks, the
bounded full-model test, and the canonical unprofiled workload pass. When
precision or numerical behavior changes, also run the appropriate sequence or
quality check.

Close the trial as `retain`, `reject`, or `inconclusive`. Record the exact diff,
commands, workload, hardware, revisions, results, fallback coverage, and
remaining uncertainty. Keep failed results as evidence; do not mix another
optimization into the same trial.

## Boundaries

- Real serving kernel work: this skill.
- Read-only implementation search: `dev-explore-kernel`.
- Serving-system techniques outside the kernel boundary: `dev-llm-serving`.
- ServingStudio Sim L1 profiling/cost-model kernels: `top-add-kernel`, not this skill.
