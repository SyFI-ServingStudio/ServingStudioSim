# Config search: GLM-5.3-Flash FP8 on one trn2.48xlarge, InferenceX 8k1k

**These are simulator predictions from the aligned model.** They predate the 2026-11-17 fixes
(InferenceX output lengths U[819, 1024] and the main-run window; the measured 45 ms host floor;
the R1 service's wave scheduler): read them as the seed engine's continuous batching with exact
1024-token outputs. The glm5-trn2 campaign Ticks search the service model instead
(`campaigns/glm8k1k/round*/tick*/` in glm5-trn2).

- **Host constants and prefill factor** are fitted on the seed engine's measured C = 16 run
  (`ALIGNMENT.md`): `region_host_ms` 1.1, `region_gap_ms` 0.1, `step_overhead_ms` 0.12,
  `forward_overhead_ms` 19 (unidentified), and a prefill-forward device factor of 1.78.
- **The factor was fitted at one layout** (dp4 x tp8, chunk 2048) and is applied to every layout.
- **L1 rows:** 477 measured, 393 derived (`l1_coverage.json`). The glm5-trn2 benchmarks are complete
  for KDA (tp 4/8/16), DSA (tp 4/8, batch <= 64, context <= 16384) and MoE (EP32/EP16, M <= 16384).
- **Predictions that lean on derived rows:**
  - KDA at tp 1/2 and DSA at tp 2/16 (kind-median ratio);
  - DSA decode above 64 rows;
  - context 32768;
  - MoE above 16384 gathered rows (prefill at dp >= 8 with chunk >= 4096).

Tool: `tools/trn2-config-search/search.py`, with stages `main`, `whatif` and `kernels`. Raw results
are in `search/results_*.json` (one InferenceX point per layout and C) and
`search/breakdown_main.json`.

Workload: InferenceX 8k1k, closed loop.

- input U[6553, 8192], output 1024;
- C in {4, ..., 256} over the instance (2 replicas x 32 LNC2 ranks = 16 chips);
- 2C warm-up + 10C measured requests.

Metrics: tok/s per chip = (input + output) tokens over the measured window / 16; interactivity =
1000 / median TPOT.

## Search space (stage `main`, 336 runs)

| axis | values |
|---|---|
| `attn_tp` (dp = 32 / tp per replica) | 1, 2, 4, 8, 16 |
| prefill chunk (= the one prefill bucket) | 512, 1024, 2048, 4096 |
| `dmax` = largest decode bucket = requests per DP group (slots dmax + 1) | 8, 16, 32, 64 |
| C | 4 ... 256 |

Refused by the HBM check (24 GiB per LNC2 core, 2 GB reserve; FP8 FFN weights, BF16 attention /
mHC / head, the engine's preallocated `slots x cmax` pools):

| layout | weights / rank | pools / rank |
|---|---|---|
| tp1, any dmax | 24.47 GB | 2.7 - 19.7 GB |
| tp2 d32 / d64 | 17.36 GB | 7.6 / 14.9 GB |
| tp4 d64 | 13.80 GB | 12.5 GB |

At tp8 the weights are 12.0 GB per rank, at tp16 11.1 GB. Attention weights are BF16 and scale
with 1 / tp. The FFN's FP8 experts are 9.6 GB per rank at EP32, whatever the tp.

## Answer: best tok/s/chip subject to median interactivity >= X

| X (tok/s/user) | engine as it is | + one region per forward, host x 0.25 | + MoE / mHC x 0.25, DSA / KDA x 0.5 | + MoE / mHC x 0.1, DSA / KDA x 0.25 |
|---|---|---|---|---|
| 20 | **none** (max 9.7) | none (max 10.5) | 340 (tp16, chunk 2048, d16, C 32) | 1702 (tp8, chunk 2048, d16, C 256) |
| 30 | none | none | none (max 30.0, 57 tok/s/chip) | 1092 (tp16, chunk 2048, d16, C 64) |
| 40 | none | none | none | 654 (tp16, chunk 2048, d16, C 32) |
| 50 | none | none | none | 206 (tp8, chunk 2048, d16, C 8) |

**No layout of the seed engine reaches 20 tok/s/user.**

- **Best interactivity:** 9.7, at tp8, chunk 2048, d8, C 4 (18.8 tok/s/chip).
- **Throughput winner, no floor:** tp4, chunk 2048, d16, C 256: 445 tok/s/chip at 3.7 tok/s/user.
- **At >= 5 tok/s/user:** tp4, chunk 4096, d8, C 256: 355 tok/s/chip.
- **At >= 8 tok/s/user:** tp16, chunk 2048, d16, C 16: 67 tok/s/chip.

Pareto front at layer regions with the aligned constants:

| layout | C | tok/s/chip | intvty | median TTFT (s) |
|---|---|---|---|---|
| tp8 chunk2048 d8 | 4 | 18.8 | 9.70 | 5.7 |
| tp8 chunk2048 d8 | 8 | 35.5 | 9.47 | 6.4 |
| tp16 chunk2048 d16 | 16 | 66.7 | 8.51 | 4.9 |
| tp16 chunk4096 d16 | 32 | 120.2 | 7.86 | 4.5 |
| tp8 chunk4096 d8 | 128 | 235.8 | 7.78 | 140 (queued: 64 slots per replica) |
| tp4 chunk4096 d8 | 256 | 354.5 | 5.91 | 188 (queued) |
| tp4 chunk2048 d16 | 256 | 444.5 | 3.65 | 10.3 |

**Regime caveat.** Every point here is 2 replicas behind one closed loop. In that regime the
waves of same-length requests desynchronize, and most prefill steps run one group's chunk with
three padded idle groups (`README.md`, preset results). The measured single-replica engine run
stays synchronized: 137 tok/s/chip at 8.8 tok/s/user, against 108 / 7.0 for the same per-replica
load in the 2-replica sweep. A router that keeps replicas in lockstep, or an engine that skips
idle groups' prefill, recovers that gap.

Rules the search shows:

- **TP4 for throughput, TP8 / TP16 for interactivity.**
  - TP1 does not fit.
  - TP2 fits only at d <= 16 and loses on both ends (370 tok/s/chip; 7.0 tok/s/user).
  - Best per TP, throughput / interactivity: tp4 445 / 8.3, tp8 425 / 9.7, tp16 348 / 9.3.
- **dmax caps throughput.** Each instance holds 2 x dp x dmax requests; past that, C only queues
  and TTFT runs into minutes. d8 keeps decode at the 8-row bucket, which gives the best
  interactivity. d16 at tp4 holds all 256 requests, which gives the best throughput.
- **The chunk trades TTFT against TPOT.** At tp4 d16 C256, chunk 512 / 1024 / 2048 / 4096 gives
  343 / 391 / 445 / 414 tok/s/chip. Above 2048, each prefill step stalls decode longer, and with
  dp 8 the 32768 gathered MoE rows run on derived rows.

## Bottlenecks

Mean time per engine step by L1 kind, in ms, folded over layers (`search/breakdown_main.json`):

| step | tp8 chunk2048 d8, C 4 | tp4 chunk2048 d16, C 256 |
|---|---|---|
| decode-only step | **100.2** | **127.6** |
| - moe_partial (42 layers) | 50.2 | 61.1 |
| - dsa decode (11) | 12.0 | 19.2 |
| - mhc_pre + mhc_post (90) | 16.5 | 19.9 |
| - kda decode (34) | 9.7 | 15.1 |
| - launch + host beyond device | 8.0 + 0.1 | 7.0 + 0.1 |
| - collectives (all) | 3.1 | 4.2 |
| step with a prefill chunk (x 1.78 prefill factor) | **1260** | **2032** |
| - prefill moe_partial | 557 | 793 |
| - prefill kda | 333 | 603 |
| - prefill dsa | 162 | 280 |
| - prefill world AG + RS | 48 | 89 |

- **Decode is device-bound on the MoE partial.** Every decode layer pays about 1.2 ms of
  `moe_partial`, even at 8 gathered rows. The region upcasts all 9 local experts' FP8 weights
  (about 226 MB) to BF16 on every call. That is about 50 ms of a 100 ms decode step, and it caps
  interactivity near 10 tok/s/user. Next come mHC (a 0.1 ms floor x 90 regions), DSA decode and
  KDA decode.
- **Host time is not the decode limit.** 48 regions x 1.1 ms + 19 ms is 72 ms of host per decode
  forward, and it overlaps about 92 ms of device. Collapsing to one region and quartering host
  time moves interactivity only from 9.7 to 10.5.
- **Prefill steps stall decode.** A step that carries a 2048 chunk per group costs 1.2-2.0 s
  (measured on the engine: 1.17-1.29 s for the prefill forward at dp4 x tp8). Every running
  request's TPOT absorbs the other requests' prefills, which is why interactivity falls with C.
- **What reaching 20 / 30 / 40 / 50 tok/s/user takes** (stage `kernels`, all with one region per
  forward and host time x 0.25):
  - 20: MoE and mHC 4x faster, plus DSA and KDA 2x faster (340 tok/s/chip at tp16 C 32).
  - 30-50: MoE and mHC about 10x faster, plus DSA and KDA about 4x faster. For example, a native
    FP8 expert GEMM without the per-call upcast, fused mHC regions, and NKI KDA / DSA. Then tp8
    C 256 gives 1702 tok/s/chip at 28 tok/s/user, and tp16 C 64 gives 1092 at 34.8.

## MTP placeholder (k = 3, golden acceptance)

`presets/trn2/glm53_flash_fp8_trn2_8k1k_mtp3.yaml` uses the seed layout with per-draft
acceptance 0.8105 (acceptance length 3.00):

| C | tok/s/chip | interactivity (tok/s/user) |
|---|---|---|
| 4 | 31.4 | 17.2 |
| 16 | 93.7 | 12.5 |
| 128 | 279.4 | 4.8 |

The engine has no MTP yet, so treat this as a sketch. The MTP pass is costed as one DSA + MoE
layer plus embed and head.
