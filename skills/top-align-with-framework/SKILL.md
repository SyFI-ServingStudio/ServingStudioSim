---
name: top-align-with-framework
description: >-
  Align VibeSim to a real framework across kernel timing, GPU duty cycle, and TTFT/TPOT.
  Routes execution and fixes; does not run phases or edit code.
---

# Top Align With Framework

Top-level skill for **judging** whether VibeSim is well aligned to a real serving
framework (vLLM or SGLang), from one measured alignment run. `operate-run-alignment`
*produces* the comparison; this skill *interprets* it — what to compare, what a
healthy gap looks like, and how to diagnose a bad one. You route and judge here;
you do not run phases or edit cost code from this skill.

## The through-line: evaluate tightest → loosest

Alignment is judged at three levels of aggregation, and you must work from the
tightest to the loosest — a failure at a tight level explains the loose ones, so
diagnosing loose-level gaps first wastes effort:

0. **Request/correctness eligibility** — did both sides execute the same prompt
   population successfully, with explicit output-evidence provenance?
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

Route to the **Align VibeSim to framework** side of `operate-run-alignment`. It produces the artifacts every
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

When the discrepant operation is one fused MoE kernel and the question is
whether the error comes from isolated execution, compressed routing input, or
full-run popularity aggregation, route the focused diagnosis through
`operate-align-moe-kernel`. Keep that evidence separate from whole-model MoE
attribution.

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

Before scoring timing, require the same request population, prompt-token
identity, successful execution, and explicit output-evidence provenance. This
is an eligibility check, not a claim that simulator output establishes
downstream task accuracy.

Trace equivalence does not authorize replaying framework scheduler decisions as
simulator policy. Reject alignments that force observed DP ranks, equal-shaped
adjacent requests, or synthetic chunks to manufacture iteration agreement;
report observed-conditioning experiments separately.

For every material unmapped semantic family, make one explicit disposition:

- label an existing simulated owner;
- add a missing simulator decomposition;
- remove or fuse framework-only plumbing;
- retain a justified noise tail.

Rank these families by logical critical-path contribution, not raw cross-rank
duration sum. Route label/decomposition repairs to their owners; this top skill
chooses the disposition but does not edit the artifacts or implementation.

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

- multi-device evidence was reduced as complete per-device paths before the
  critical device was selected; mapped, unmapped, and overlap maxima were not
  chosen independently;
- collective residency/wait remains timeline evidence rather than being counted
  as CUDA kernel duration, and stream plots preserve reduced work totals even
  when small streams are aggregated;
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

Use phase transitions as falsification evidence. If a final pure-decode drain
retains the gap after new prefills stop, the fact that most total time was in
mixed iterations cannot support a scheduling-only explanation.

Keep overall and kernel conclusions separate. Per-kernel deviation is reliable
only for iterations with equivalent phase, shapes, specialization, numeric
contract, and graph mode. If framework and simulator scheduling produce
different iteration populations, do not force their kernel occurrences to sum
to the E2E gap or call the remainder host overhead. Compare matched strata,
report unmatched work and scheduling structure separately, and leave causal
percentages unresolved without a controlled A/B. Mixed-region occupancy is not
attribution. Backend, scale strategy, and other numeric differences qualify the
conclusion but do not prove a throughput cause.

## Delegation boundary

You judge; you do not run or fix from here. Diagnose in the tight → loose order:
request/correctness eligibility → coverage (no missing large kernel) →
per-kernel/collective deviation + cause → duty-cycle ratio → iteration wall →
TTFT/TPOT/throughput. Then route the fix:

- rerun, resume, or re-label a capture → `operate-run-alignment`;
- one fused MoE kernel with uncertain routing-input fidelity →
  `operate-align-moe-kernel`;
- a wrong-shape or missing kernel cost → `impl-validate-kernel-cache` /
  `top-add-kernel`;
- a mis-mapped or missing simulated slot → `operate-run-alignment`'s labeling.

Stop and return to the user when a deviation's cause is genuinely ambiguous
(model gap vs thermal artifact) rather than guessing — the attribution decides
whether the cost model changes at all.
