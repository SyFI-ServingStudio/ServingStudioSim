---
name: top-align-with-framework
description: >-
  Use as the top entry point when the user wants to evaluate how well VibeSim
  matches a real serving framework (vLLM today) — judging alignment quality from
  one measured alignment run, not just producing it. Covers the three evaluation
  levels from tightest to loosest: per-iteration kernel-only timing (per-kernel
  deviation and missing large-chunk coverage, with cause diagnosis — different
  kernel vs thermal/clock artifact), GPU duty-cycle / kernel-GPU-ratio
  correctness, and end-to-end TTFT/TPOT (where a small sim-faster TTFT gap is
  expected from vLLM's async scheduler). This is a routing/judgment skill; it
  delegates the actual running to operate-run-alignment and cost-model fixes to
  the kernel skills, and does not run phases or edit code itself.
---

# Top Align With Framework

Top-level skill for **judging** whether VibeSim is well aligned to a real serving
framework (vLLM today), from one measured alignment run. `operate-run-alignment`
*produces* the comparison; this skill *interprets* it — what to compare, what a
healthy gap looks like, and how to diagnose a bad one. You route and judge here;
you do not run phases or edit cost code from this skill.

## The through-line: evaluate tightest → loosest

Alignment is judged at three levels of aggregation, and you must work from the
tightest to the loosest — a failure at a tight level explains the loose ones, so
diagnosing loose-level gaps first wastes effort:

1. **Per-iteration kernel-only timing** — is each individual modeled kernel close
   to the measured kernel, and is any large kernel missing entirely? (tightest,
   most diagnostic)
2. **GPU duty cycle** — did the kernel/GPU-ratio correction land reasonably, so
   summed kernel time becomes the right wall time?
3. **End-to-end TTFT / TPOT** — do request-level latencies track, allowing for
   one *expected* structural gap? (loosest)

A clean tight level under a dirty loose level points at structure (scheduling,
async), not the cost model.

## Step 0 — Run the alignment

Route to `operate-run-alignment` for the phases. It produces the artifacts every
check below reads: the per-iteration labeled kernel breakdown (measured duration
vs simulated CostTree slot per mapped operation) plus the kernel-align pass's
`recommended_gpu_time_multiplier` (in `alignment_iteration_report.json` meta), and
the E2E TTFT/TPOT/throughput overlays. Interpret those artifacts; do not re-derive
them here.

## Check 1 — Per-iteration kernel-only timing

The tightest and most diagnostic check. Compare measured kernel duration against
the simulated CostTree slot time per mapped operation, per iteration, on
**kernel-only** time (pure compute, *before* the `gpu_time_multiplier` wall
inflation).

**First, coverage — is a large chunk missing?** Before scoring deviations, scan
for a big **measured** kernel with no simulated counterpart (an unmapped measured
kernel), or a large **simulated** slot with no measured kernel. A genuinely
missing large-chunk kernel is a structural hole in the model, not a mistimed
leaf, and it must be resolved before any finer per-kernel comparison is
trustworthy.

But treat every apparent gap as **guilty until proven** — the kernel
classification and the alignment labeling both readily manufacture *false
positives*. A big "unmapped measured" kernel is often just a labeling miss or a
parser mis-category (`suggested_category` is a hint, not proof), and a large
"simulated slot with no measured kernel" is often a real kernel that was
mis-classified or mapped to the wrong operation and so never attributed here.
Before declaring a structural model hole, re-verify the occurrence's phase,
folded position, and label — route back to `operate-run-alignment`'s
*Match measured kernels* / *Make mapping decisions* — and only conclude "missing"
once the classification and mapping are confirmed correct. Small unmapped helpers
(bookkeeping, alloc/fill/copy, launch prep, sampling) are expected and fine; a
large unmapped duration is a signal to *audit the labels first*, then the model.

**Then, per-kernel deviation — and judge the cause.** Flag any operation whose
modeled time deviates a lot from measured, and attribute it:

- **Different kernel / implementation.** The sim profiled a different backend or
  algorithm than vLLM actually launched (a different GEMM tile, a different
  attention impl, a different quant path). This is a real model gap — align the
  backend or re-profile. Cross-check the measured kernel name/semantics against
  what the CostTree slot assumes.
- **Thermal / clock (throttle or boost).** A broad, roughly *uniform* slowdown
  (or speedup) across essentially all kernels, with no single structural cause, is
  a measurement artifact of the captured run's clocks — not a cost-model bug. Do
  **not** retune the cost model to chase it; note it, and prefer a cleaner
  capture. Distinguish it from a real gap by its uniformity: a real gap is
  concentrated in specific ops.
- **Shape / quantization / cache-interpolation edge.** The modeled kernel is the
  right kernel but the cost cache mispredicts at this shape or bucket boundary →
  route to `impl-validate-kernel-cache` (and `top-add-kernel` if a row/backend is
  actually wrong or missing).

## Check 2 — GPU time / duty cycle

Kernel-only sums exclude the inter-kernel gaps (launch overhead, sync, scheduling)
that real wall time contains. The kernel-align pass derives the duty-cycle
correction `recommended_gpu_time_multiplier = Σ measured_gpu_cycle_ms / Σ
measured_ms` — the same per-occurrence `measured_ms` reduction the breakdown
reports, so numerator and denominator are one consistent metric. Judge whether
that correction is reasonable:

- its inverse (the pooled kernel/GPU busy fraction) is plausible for the workload
  (a well-batched run spends most of the GPU cycle in kernels, so the multiplier
  is near 1; a large multiplier means big host bubbles that deserve an
  explanation, not silent absorption);
- the per-iteration `measured_gpu_cycle_ms` vs `measured_ms` gaps differ in the
  expected direction — decode is more launch-bound (larger gap), prefill more
  compute-bound (smaller);
- after the simulation bakes in the multiplier, the simulated iteration cycle time
  tracks the measured GPU cycle. If Check 1 is clean (kernels themselves match)
  but the iteration cycle is still off, the **duty-cycle correction**, not the
  kernels, is the suspect — revisit the multiplier and its population.

## Check 3 — TTFT and TPOT

The loosest check; read the overlays as distributions, not per-request ratios,
and expect one structural gap.

- **TPOT** should track closely once Checks 1–2 pass — it is dominated by
  steady-state decode iteration time, which those checks already validated.
- **TTFT** is expected to be **a bit faster in the simulator** than in measured
  vLLM. vLLM's async scheduler adds first-token latency (request queueing and
  per-step scheduling overhead) that the simulator does not model, so a small
  sim-faster TTFT gap is *expected and not a defect* — do not tune the model to
  close it. A **large** TTFT gap, or the **wrong direction** (sim slower), is a
  real signal: trace it back to prefill kernels (Check 1) or to scheduling.

## Delegation boundary

You judge; you do not run or fix from here. Diagnose in the tight → loose order:
coverage (no missing large kernel) → per-kernel deviation + cause → duty-cycle
ratio → TTFT/TPOT. Then route the fix:

- rerun, resume, or re-label a capture → `operate-run-alignment`;
- a wrong-shape or missing kernel cost → `impl-validate-kernel-cache` /
  `top-add-kernel`;
- a mis-mapped or missing simulated slot → `operate-run-alignment`'s labeling.

Stop and return to the user when a deviation's cause is genuinely ambiguous
(model gap vs thermal artifact) rather than guessing — the attribution decides
whether the cost model changes at all.
