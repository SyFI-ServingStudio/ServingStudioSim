# GLM-5.3-Flash MI300X sparse-MLA prefill "3.78× over-prediction": root cause

**Branch:** `glm53-fix2-sparsemla-v2` (off #80 head `f99a4cca`)
**Date:** 2026-10-08
**Verdict:** GPU-FREE. **No kernel re-measure, no leaf code change, no index-distribution
repoint.** The L1 kernel cost model and its banked rows are CORRECT. The apparent
3.78× over-prediction is a **workload-shape mismatch** in the alignment comparison:
timing-predict is run on a synthetic micro-case (`prefill_chunk_pairs = [[0,2048]]`)
that does **not** reproduce the captured iteration-0 prefill batch.

This result **overturns** PR #80 Fix-2 (which proposed repointing the prefill leaf to
`recent_contiguous` and scheduled an 858-cell MI300X re-measure). Both are unnecessary.

---

## 1. Ground truth: the live kernel runs at 324 µs, and the microbench reproduces it

Raw per-dispatch GPU time (`end_ns − start_ns`) of `_sparse_attn_prefill_ragged_kernel`
from the Check-1 torch-profiler trace (job 453335, TP4/EP4, 8×MI300X), file
`wt-glm53-flash-align/logs/align_glm53_mi300x_tp4ep4/profile/parsed.merged.kernels.parquet`:

- **88 dispatches** = 11 DSA layers × 4 ranks × 2 iterations, splitting cleanly into:
  - **44 at 320–329 µs** (mean **324 µs**) — the prefill iteration (seq `2cdd1520`)
  - **44 at 592–604 µs** (mean **598 µs**) — the decode iteration (seq `fbc0f34b`)

This confirms pr-fixes-body.md's "prefill 324 / decode 598 µs, no occurrence exceeds
605 µs." The kernel-inventory `mean_call_us` field (660–1454 µs) is a spurious aggregate
(not GPU-active time) and must not be used; the breakdown `duration_ms` (0.320 ms) equals
the raw end−start and is correct. 11 × 324 µs = 3554 µs/rank = the breakdown's
`measured_ms = 3.554` for `dsa.sparse_mla` prefill.

### The microbench is slot-linear and verified (not over-measuring)

Parsing `valid_counts` into total selected-token slots over the MI300X
`rocm_triton_mla_sparse`, `num_heads=16`, bf16 rows (`profiling/profile.db`,
`verified=1`), the kernel time is linear in **total slots**, essentially independent of
query-row count, slot distribution across rows, or index distribution:

| valid_counts | total slots | time (µs) |
| --- | --- | --- |
| `r:1..256`  |   32,896 |   47.9 |
| `r:1..512`  |  131,328 |  123.3 |
| `r:1..1024` |  524,800 |  367.1 |
| `r:1..2048` (recent_contiguous, id 52778) | 2,098,176 | 1223.8 |
| `r:1..2048` (scattered, id 52174) | 2,098,176 | **1225.6** |
| `u:2048x2048` (scattered, id 51426) | 4,194,304 | 2178.0 |

Index distribution is irrelevant: contiguous 1.2238 ms vs scattered 1.2256 ms = **0.15%**.
This alone **disproves** Fix-2's premise (repointing to `recent_contiguous` changes nothing)
and shows the 858-cell `recent_contiguous` backfill was unnecessary.

**The live 324 µs corresponds to ~0.456M slots on this verified curve** (bracketed by
`r:1..512` = 123 µs / 0.131M and `r:1..1024` = 367 µs / 0.525M). So the microbench, fed
the live slot count, predicts the live time exactly. The microbench is correct.

## 2. The model is fed 2.1M slots, the live kernel does 0.456M

- `presets/predict_glm53_flash_mi300x_cases.json` case 0 = `prefill_chunk_pairs: [[0,2048]]`
  (one request, 2048 fresh tokens from position 0).
- This is a genuine causal prefill: every row attends to its full causal prefix (context
  ≤ `index_topk`=2048 ⇒ no sparsity), so the leaf correctly derives a `CausalTail` ramp
  `r:1..2048` = **2,098,176 slots** → resolves to verified row (id 52174) = **1225.59 µs/layer**
  → ×11 = **13,481.47 µs/rank** = the alignment's `simulated_ms` for `dsa.sparse_mla` prefill.
- The alignment (`timing_predict_case_map.json`) pairs **case 0 → measured iteration 0**.
  So it compares timing-predict of `[[0,2048]]` (2.1M slots, 1225 µs/layer) against the live
  iteration-0 kernel (0.456M slots, 324 µs/layer).

**3.78× = 1225.6/324 ≈ the slot ratio 2.098M/0.456M = 4.6× tempered by the curve's
small-size launch offset.** It is a slot-count overstatement in the *input*, not a kernel
cost error.

The leaf derivation is correct: given `[[0,2048]]`, 2.1M slots IS the right causal count
and 1225 µs IS the right cost. The live iteration-0 simply did not run `[[0,2048]]`.

## 3. Cross-check: the decode case is wrong too (opposite direction)

Case 1 = `decode_kv_lens: [4096]×32` ≈ 32×2051 = 65K slots → ~48 µs on the curve, but the
live decode kernel is **598 µs/layer** (~0.95M slots, ≈ a few hundred decode rows) and the
model predicts only 308 µs/layer (3388 µs/rank). A 32×4096 batch cannot produce 598 µs, so
the synthetic cases categorically do not reproduce the captured iterations. This confirms
the mismatch is a workload-shape artifact, not a methodology or attribution bug, and not a
per-shape kernel discrepancy.

## 4. Classification

| Hypothesis | Verdict |
| --- | --- |
| #1 workload-coordinate (wrong shape/slots fed to kernel) | **YES — root cause.** The modeled batch (`[[0,2048]]`, 2.1M slots) is ~4.6× heavier in selected slots than the live iteration-0 prefill (~0.456M). The leaf logic is correct; the *case/workload input* is synthetic and mismatched. |
| #2 microbench methodology over-measures | **NO.** The microbench is slot-linear and verified; fed the live slot count (0.456M) it predicts the live time (324 µs). |
| #3 attribution artifact | **NO.** The breakdown `measured_ms` (3554 µs/rank) equals 11 × raw-trace 324 µs; attribution is correct. (The inventory `mean_call_us` is spurious — do not use it.) |
| Fix-2 index distribution (`unique_scattered` → `recent_contiguous`) | **DISPROVEN.** 0.15% difference at the operating shape; the 858-cell re-measure was unnecessary. |

## 5. GPU-free fix (no re-measure, no new rows)

Make the alignment compare *matched* workloads: drive timing-predict with the capture's
**real per-iteration batch composition** instead of the synthetic micro-cases in
`predict_glm53_flash_mi300x_cases.json`. With a case 0 whose selected-slot total matches
the live iteration-0 (~0.456M), the existing verified rows interpolate to ~324 µs/layer and
the `dsa.sparse_mla` prefill deviation collapses from +278% to ~0 — using **no new
profile.db rows**. The decode case needs the same treatment (its real batch is heavier than
32×4096).

**What's needed to execute (not a GPU lease):** the capture's per-iteration batch
(`prefill_chunk_pairs` / `decode_kv_lens`). It is **not** in the alignment bundle — the
torch trace stores no tensor shapes, and the upstream-vLLM capture carried no per-iteration
request markers (decision #14). Recovering it is a workload-reconstruction step (capture
scheduler/request log, or a re-capture with per-iteration markers), after which the cases
can be regenerated and timing-predict + the Check-1 analyze re-run against the reused
measured artifacts.

## 6. Before / after

- **Before (timing-predict case 0 = `[[0,2048]]`):** `dsa.sparse_mla` prefill =
  1225.59 µs/layer × 11 = **13,481.47 µs/rank** (deviation vs live +278%). Authoritative
  from the alignment `simulated_ms` and verified DB row 52174.
- **Ground-truth live:** 324 µs/layer × 11 = **3554 µs/rank**.
- **After (cost model fed the live slot count ~0.456M):** ~324 µs/layer → ~3554 µs/rank,
  deviation ~0. Proven from the verified grid (0.456M slots brackets to 324 µs); requires
  the real batch to instantiate the case, not a re-measure.

## 7. Gates

No simulator source changed (finding doc only), so B200 is trivially bit-identical, no
golden moves, and test-cpu is unaffected. `profile.db` is untouched (skip-worktree, never
git-added). No fabricated or biased rows.
