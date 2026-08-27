---
name: operate-align-moe-kernel
description: >-
  Diagnose alignment of one existing fused MoE kernel through exact-input,
  compressed-popularity, and full-run comparisons. Not for whole-model MoE
  attribution, routing-model design, or adding a new kernel backend.
---

# Align One MoE Kernel

Use this workflow after generic kernel alignment identifies a discrepancy in one
fused MoE kernel. Keep the kernel implementation, backend, numeric format,
weights, topology, and profiler settings fixed while changing only the source of
its routed input.

The subject is one registered fused MoE kernel kind. Its repeated invocations
across layers and iterations remain the same subject. Do not include router,
dispatch, combine, or communication work unless those operations are part of
the registered kernel's measured boundary.

## Build the comparison ladder

Run the comparisons in order. Preserve every earlier result; a later aggregate
comparison must not replace a tighter one.

1. **One layer: vLLM versus exact-input isolated measurement.** Select one
   captured layer invocation. Measure its fused MoE interval in vLLM and replay
   the actual per-rank expert inputs through the same kernel in isolation.
   Compare corresponding ranks when physical identity is retained, and always
   compare critical rank to critical rank.
2. **One iteration: vLLM versus added exact-input measurements.** Repeat the
   exact isolated replay for every MoE layer in the selected iteration. Reduce
   ranks exactly as on the vLLM side, then add the sequential layer times. This
   comparison shows whether exact standalone measurements reproduce the whole
   iteration's real MoE work.
3. **One iteration: vLLM versus iteration-popularity measurements.** Reconstruct
   every layer from that iteration's compressed expert-popularity record,
   measure the resulting kernel inputs, and aggregate them identically. Compare
   this result directly with the same vLLM iteration.
4. **One iteration: iteration popularity versus full-run popularity.** Rebuild
   that iteration using the full-run popularity summary while retaining its
   token count and other kernel inputs. Compare these measurements with step 3,
   not only with vLLM. The delta isolates temporal information lost by replacing
   iteration-specific popularity with the full-run summary.
5. **Full capture: vLLM versus full reconstructed measurement.** Apply the
   validated full-run-popularity reconstruction to every captured iteration.
   Compare every reconstructed MoE total with the corresponding complete vLLM
   MoE total, then report the aggregate result. Only after the measured ladder
   is intact may cached interpolation or timing prediction replace an isolated
   measurement; label that result as predicted.

Iteration totals add layer invocations only after reducing each invocation to
its physical critical-rank time. For a fused operation implemented by several
PDL-overlapped CUDA kernels, use interval busy union on each device rather than
summing subkernel durations. Apply the same reduction and aggregation rules to
both sides of every comparison.

## Preserve input meaning

Record which representation feeds every rung:

- **Exact input:** the real per-iteration, per-layer, per-rank expert workload
  captured from the serving invocation.
- **Iteration popularity:** a compressed per-iteration, per-layer logical expert
  distribution reconstructed into kernel inputs.
- **Full-run popularity:** a distribution aggregated over the capture and then
  used to reconstruct one iteration at a time.

Never call a reconstruction from aggregate popularity an exact vLLM replay.
Aggregate expert marginals cannot recover iteration-level skew, expert
co-occurrence, or the critical EP-rank tail. Keep the sampler seed and algorithm
in the artifact, but do not treat seed averaging as recovery of discarded
information.

## Report each comparison

For every pair, report the left and right time, signed difference, absolute
difference, relative error, covered layer/iteration count, and any unmatched
work. Keep per-rank and per-layer rows available even when the main report shows
only iteration totals.

The ladder localizes the first material divergence:

- step 1: serving environment versus isolated kernel execution;
- step 2: whether that execution difference persists over a real iteration;
- step 3: iteration-level input compression;
- step 4: full-run temporal aggregation;
- step 5: coverage and generalization across the capture.

Do not advance a correctness claim past a failed rung. Later agreement can be
cancellation between signed errors. If exact routing inputs are unavailable,
mark steps 1 and 2 unavailable and state that execution-environment error cannot
be separated from input-reconstruction error.

## Reuse existing operators

Use `operate-run-alignment` for the vLLM capture and normalized kernel intervals.
Use `operate-profile-existing-kernel` for isolated measurements and a separate
experiment profile database. Use Analyzer comparison output when it already
contains the required pair; add a deterministic helper script only when the
same extraction or aggregation would otherwise be rewritten across experiments.

Completion requires all available rungs, the first divergent rung identified,
and links to the raw capture, exact or popularity input artifacts, isolated
measurements, and pairwise reports.
