# Tune loop: blind validation of the aligned Sim on the engine (2026-10-25)

The aligned Sim (`ALIGNMENT.md`, at `691917e9`) predicted configs that the
flashnext-neuron engine had never run. The predictions were committed before any of
those configs ran, and then the engine ran them. Tool: `tools/trn2-tune-loop/`
(propose / predict / run / compare / headroom); skill: `operate-tune-trn2-with-engine`.
Loop directory: `logs/trn2/tune/blind_w64/` (not tracked). Committed predictions:
`tune/blind_w64_predictions.json` (uniform acceptance, commit `335d00ec`, plus
tp8_chunk4096 in `0e818b51`) and `tune/blind_w64_predictions_accdist.json` (drawn
acceptance, `01a309e4` / `0e818b51`).

## Setup

- **Workload.** W64: 64 requests, 2048 in / 512 out, 50 ms apart. All 16 chips are
  used: two 32-core replicas (cores 0-31 and 32-63, comm ports 61000 / 63000), and
  requests are split by rid parity.
- **Engine knobs.** Decode buckets 1-16, context buckets 2048 / 4096, and prefill
  buckets up to the chunk budget (128, 512 [, 2048]). Every shape is warmed before the
  clock starts.
- **MTP.** k = 1. The input is acceptance 0.67, the engine's k=1 rate on random
  prompts.
- **Candidates.** All share block regions, RS/AG (`reduce_scatter_all_gather`) and EP32
  from the search's recommended set (`search_v2/s3_mtp`, `fixed_2k_512-a0.67`).

| Label | Role | Layout | MoE dispatch | Chunk budget / prefill buckets |
|---|---|---|---|---|
| top1 | search's best (a) | dp8-tp4-g4 | all_gather | 512 / 128, 512 |
| chunk4096 | (b) budget above the largest bucket | dp8-tp4-g4 | all_gather | 4096 / 128, 512, 2048 |
| tp8_chunk4096 | (b) at the layout that loads it (added after chunk4096 failed, predicted before it ran) | dp4-tp8-g8 | all_gather | 4096 / 128, 512, 2048 |
| tp8 | layout contrast | dp4-tp8-g8 | all_gather | 512 / 128, 512 |
| a2a | (c) predicted loser | dp8-tp4-g4 | all_to_all, exact capacity | 512 / 128, 512 |

**Attention TP4 needed no engine change.** `serve.py --tp 4` builds 8 DP groups of one
chip each. The ops already shard 24 Q heads by 4 and replicate the KV head 2x. Its MTP
acceptance was 0.66, the same as at TP8, so the numerics are sane. (The ops'
correctness tables in `ops/qsa.md` and `ops/gdn.md` cover tp=4.)

## Predicted vs measured

**Prediction sets.**

- **Uniform:** acceptance 0.67 for every request. This set is fully blind.
- **Drawn:** each request takes the per-request acceptance of an earlier engine W64 k=1
  run. The table gives the median of three draws: `block_k1`, `layer_rsag_k1`,
  `layer_k1_2rep`.
  - It was made after top1 ran and before the other configs ran, so it is blind for
    them.
  - Why it was made: top1 showed that duration follows the slowest requests, and a
    uniform mean hides them.

**Duration and stalls.**

- **Raw** is the engine's duration.
- **Stalls** are its first-use stalls: steps 0-2 above the expected wall of their
  shapes, read off the timeline only (`first_use_stall_s`). The Sim does not model
  them.
- The error columns compare against the duration without stalls.

| Config | Duration pred: uniform | Duration pred: drawn | Measured raw (s) | Stalls (s) | Measured w/o stalls (s) | Error: uniform / drawn |
|---|---|---|---|---|---|---|
| top1 | 53.2 | 65.8 | 70.4 | 9.9 | 60.5 | -12% / +9% |
| tp8_chunk4096 | 50.9 | 63.3 | **62.8** | 3.1 | **59.7** | -15% / +6% |
| tp8 | 57.3 | 70.0 | 66.7 | 3.1 | 63.6 | -10% / +10% |
| a2a | 62.0 | 74.6 | 96.4 | 3.0 | 93.4 | -34% / -20% |
| chunk4096 | 53.8 | 66.0 | did not load | | | |

| Config | Out tok/s: uniform / drawn / measured raw | TTFT p50 (s): pred / measured (minus stalls) | TPOT p50 (ms): uniform / drawn / measured | Acceptance measured |
|---|---|---|---|---|
| top1 | 616 / 498 / 465 | 18.0 / 27.1 (17.1) | 72.5 / 69.1 / 70.9 | 0.66 |
| tp8_chunk4096 | 644 / 517 / 522 | 12.6 / 16.3 (13.2) | 70.4 / 68.5 / 70.9 | 0.66 |
| tp8 | 572 / 468 / 491 | 17.5 / 20.9 (17.8) | 76.8 / 73.3 / 73.3 | 0.70 |
| a2a | 528 / 439 / 340 | 23.1 / 37.0 (34.1) | 82.5 / 78.9 / 100.4 | 0.64 |

**Re-simulated with each run's own measured per-request acceptance**
(`compare --resim`; errors against the run without stalls):

| Config | Duration | Out tok/s | TPOT p50 |
|---|---|---|---|
| top1 | 61.4 s (+1.6%) | 534 (-1.5%) | 70.4 ms (-0.8%) |
| tp8_chunk4096 | 60.3 s (+1.0%) | 544 (-1.0%) | 69.9 ms (-1.4%) |
| tp8 | 62.4 s (-1.9%) | 525 (+1.9%) | 72.5 ms (-1.2%) |
| a2a | 75.0 s (-19.8%) | 437 (+24.6%) | 79.6 ms (-20.8%) |

The engine's step-level timing replays within the alignment's residuals:

- **top1 and a2a:** every step through `timing-predict`. The decode step is 8-10%
  long. For top1 the prefill forward is 5% short (T = 512 at dp8: 1347 ms measured
  vs 1319 predicted). The summed step time is within 2% once stalls are removed.
- **a2a** is the exception (below).

### Did the ranking hold?

- **Yes, once first-use stalls are removed.**
  - Measured order: tp8_chunk4096 (59.7 s) < top1 (60.5) < tp8 (63.6) < a2a (93.4).
  - Every prediction set gives the same order (Kendall tau 1.0 on duration, out
    tok/s and TPOT p50), and the predicted best is the measured best.
  - tp8_chunk4096 vs top1 is a tie (1.3% apart, under the 3% noise band).
- **On raw durations top1 and tp8 swap** (tp8_chunk4096 62.8 < tp8 66.7 < top1 70.4 <
  a2a 96.4).
  - The TP4 engine stalls 9.9 s after warm-up, against 3.1 s for TP8. Step 0 is a 9 s
    host-side stall on the first chunked prefill.
  - The Sim does not model it, and the dp8-tp4 advantage the search predicted (+6..10%
    over dp4-tp8) is smaller than that stall.
- **The contrast held by a wide margin:** a2a is the worst measured and predicted, but
  the Sim under-predicts how much worse it is.
- **Not confirmed: chunk budget above the bucket at dp8.** The engine cannot load the
  dp8 x 2048 block prefill. At dp4-tp8 the budget-4096 config is the measured best
  (-6% duration and -22% TTFT p50 vs tp8 at chunk 512), as predicted (-11% and -28%
  uniform).

### Verdict: re-align before trusting a2a or dp8 predictions further

`compare` returns `realign`. Read by error source:

| Source | Evidence | Size | What to do |
|---|---|---|---|
| MTP acceptance as a uniform input | uniform 0.67 predicts all-gather configs 10-15% short; the measured per-request rates (0.37-0.84) make the Sim match within ±2% | the largest error for every all-gather config | predict with `--accept-from` draws from earlier runs (median of 3: +6..+10%, the spread of a single draw is ±5%); keep acceptance an input |
| First-use stalls | 3 s per run at TP8, 10 s at TP4 (not modeled) | 5-14% of W64 duration; flips top1 / tp8 on raw durations | compare without stalls (`compare` does), and treat the TP4 stall as an engine warm-up bug |
| all_to_all packing cost | prefill forward at dp8 x T = 512: 2465 ms measured vs 1682 predicted (all_gather: 1347). The fitted 0.8 µs per received row reproduces dp4 x T = 2048 (+837 ms) but dp8 needs ~2.7 µs/row (+1118 ms on half the rows); decode +11% | -20% duration and TPOT with measured inputs | re-fit the a2a cost with a per-call / per-destination term from both layouts before trusting any a2a prediction (incl. the capacity-factor what-if) |
| Engine HBM | dp8 x 2048 block prefill (16384 gathered MoE rows) fails NEFF load: `Allocation Failure` | config infeasible; the Sim accepted it | `propose` refuses block prefill > 8192 gathered rows; the Sim's HBM check needs activation / scratch memory |
| Decode / prefill composition | step replay within the alignment's residuals at dp8-tp4 (decode +8-10%, prefill -5%) | ±2% end-to-end with measured inputs | none |

## Best measured config

**tp8_chunk4096** has the knobs dp4-tp8-g8, block regions,
`reduce_scatter_all_gather`, all-gather MoE, EP32, MTP k = 1, `max_batch_tokens` 4096,
prefill buckets 128 / 512 / 2048, and two replicas.

- **Measured:** 62.8 s, 522 out tok/s (2611 tok/s in + out on 16 chips), TTFT p50
  16.3 s, TPOT p50 70.9 ms.
- **Against the engine's best before this loop** (two replicas, layer, RS/AG, k = 2,
  chunk 512: 85.8 s, 382 out tok/s, TPOT p50 92.4 ms, TTFT p50 23.1 s): 1.37x the
  throughput, -23% TPOT, -30% TTFT.
- **Against the step-5 baseline** (8 chips, layer k = 0, chunk 2048: 169 s, 194 tok/s):
  2.7x.

### Headroom (aligned Sim, this config, its own measured acceptance)

R0-R7 cluster ladder (`tune.py headroom blind_w64 tp8_chunk4096 --accept-from
tune_blind_w64_tp8_chunk4096_r0 tune_blind_w64_tp8_chunk4096_r1 --tag meas`). Units
are GPU-seconds over all 64 ranks.

| Rung | GPU-s | of R0 |
|---|---|---|
| R0 real | 3848 | 100% |
| R1 busy | 3847 | 100% |
| R2 balanced | 3847 | 100% |
| R3 per-config best | 2487 | 64.6% |
| R4 ignore network | 2456 | 63.8% |
| R4a/b sustained / achievable | 1182 | 30.7% |
| R5 hardware limit | 796 | 20.7% |
| R6/R7 necessary work | 22 | 0.6% |

The gaps between rungs:

- **Launch + batching (R2-R3), 35%.** Region floors, padded decode buckets, and idle
  groups running the padded shape.
- **Kernel gap (R4-R4a), 33%.** `moe_cte` at 8192 gathered rows, and the MoE decode
  kernels.
- **Excess over necessary work, 20%.**
- **Hardware gap, 10%.**
- **Communication, 0.8%.**

The Sim's what-ifs on these exact knobs:

| What-if | Duration (s) | Out tok/s | TTFT p50 (s) | TPOT p50 (ms) |
|---|---|---|---|---|
| base (Sim, measured acceptance) | 60.3 | 544 | 12.6 | 69.9 |
| MoE prefill kernel 3x faster (`moe_block` / routed experts at >= 256 rows) | 45.2 (-25%) | 725 (+33%) | 4.2 (-66%) | 57.8 (-17%) |
| + idle-DP padding skip | 45.3 (-25%) | 723 (+33%) | 4.0 (-68%) | 58.0 (-17%) |
| all-to-all at capacity factor 1.25 | 49.7 (-17%) | 659 (+21%) | 6.1 (-51%) | 62.6 (-10%) |
| host 2x faster | 56.2 (-7%) | 583 (+7%) | 12.5 (-1%) | 65.8 (-6%) |
| fused decode + prefill forward | 58.3 (-3%) | 562 (+3%) | 13.2 (+5%) | 65.9 (-6%) |
| idle-DP padding skip | 58.7 (-3%) | 558 (+3%) | 11.1 (-12%) | 69.7 (-0%) |
| 16 chunks per DP group | 61.7 (+2%) | 531 (-2%) | 18.2 (+44%) | 64.4 (-8%) |

### Top optimization targets for this config

1. **`moe_cte` prefill 3x faster** (its static per-block cost at 2048-8192 gathered
   rows). Predicted: +33% out tok/s, TTFT p50 -66%, TPOT -17%. This is the largest
   kernel gap and the one with the most support: the MoE rows are measured L1 rows,
   and the composition replays within ±5% at prefill.
2. **A faster host path** (input assembly, H2D, id readback, the per-forward ~19 ms).
   Predicted: +7% out tok/s, TPOT -6%. Decode at block granularity is close to its
   host floor with 4 groups.
3. **Remove the first-use stalls after warm-up.** Measured, not predicted: 3.1 s of
   the 62.8 s here (+5%), and 9.9 s (+16%) at TP4. Fixing it would also make dp8-tp4
   (top1) worth its predicted edge.

The **idle-DP padding skip** gives only +3% here (TTFT -12%): at dp4 with a 4096 budget,
every group has a chunk in nearly every prefill step. It mattered more at dp8 / chunk
512 (+6%, TTFT -15% for top1). The **all-to-all with a capacity factor** (+21%) is a
model prediction resting on the all-to-all cost that this loop just found wrong at
dp8. Do not build it on this number before the a2a re-fit.

## How to rerun

```bash
cd wt-trn2-flashnext/ServingStudioSim
source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH
T="uv run --no-sync python tools/trn2-tune-loop/tune.py"
$T propose blind_w64 --exp logs/trn2/search_v2/s3_mtp --workload fixed_2k_512-a0.67 --top 1 \
  --variant 'chunk4096:max_batch_tokens=4096' --variant 'tp8:attn_dp=4,attn_tp=8,gdn_tp=8' \
  --contrast 'a2a:moe_dispatch=all_to_all'        # chunk4096 is now refused (HBM guard)
$T predict blind_w64
$T predict blind_w64 --tag accdist --accept-from w64_2048_512_block_k1
$T run blind_w64; tmux new -s tune-blind "$T run blind_w64 --execute"   # ~25-35 min per new layout
$T compare blind_w64 --resim
$T headroom blind_w64 tp8_chunk4096 --accept-from tune_blind_w64_tp8_chunk4096_r0 \
  tune_blind_w64_tp8_chunk4096_r1 --tag meas
```

Engine compile and warm-up per candidate (two replicas in parallel, shared cache):

| Candidate | Warm-up |
|---|---|
| dp8-tp4 block, all-gather | 26 min |
| dp8-tp4 block, a2a | 30 min |
| dp4-tp8 block, chunk 512 | 15 min |
| T = 2048 prefill shapes at dp4-tp8 block | +32 min |

Each serving run then takes ~65-100 s.

## Open issues

- **all_to_all cost model** fails at dp8 (-20% with measured inputs). It needs a
  re-fit over both layouts. Until then a2a predictions, including the capacity-factor
  what-if, are untrusted.
- **The Sim's HBM model** lacks per-region activation and scratch memory. The guard in
  `propose` is empirical: block prefill of 8192 rows loads and 16384 does not. Layer
  granularity at 16384 rows is unmeasured.
- **First-use stalls** after warm-up (3 s at TP8, 10 s at TP4) are an engine artifact
  and not modeled.
- **MTP acceptance** stays an input. Per-request values do not carry over between runs
  on the seeded random prompts (correlation <= 0.26); only their spread does.
- **TTFT p50 on W64** reads low in the raw comparison because it includes the stalls.
  Minus stalls, top1 is 17.1 measured vs 18.0 predicted, and tp8_chunk4096 is 13.2 vs
  12.6.
- **Not run:** dp8-tp4 at a chunk budget above the bucket with buckets <= 512 (the Sim
  does not cap chunks at the bucket; `propose` refuses it), dp2 layouts, and k >= 2.
