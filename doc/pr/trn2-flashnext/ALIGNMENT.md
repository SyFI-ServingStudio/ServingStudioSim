# Alignment with the flashnext-neuron engine (2026-10-25)

`qwen4_exp_neuron_bf16[_speculative]` was aligned with the full 48-layer flashnext-neuron
engine (`framework/flashnext-neuron`, step 5) serving on 32 LNC2 cores with the NRT
execution barrier off. This file records what was compared, what changed in the
composition, the results, and what is still off. Tool: `tools/trn2-engine-align/`.

## Data

| Source | What it gives |
|---|---|
| `logs/engine/serve_*.jsonl` (16 runs, 18 logs) | per step: kind, each DP group's decode requests and prefill tokens, padded shapes, per-forward wall (decode, MTP decode, prefill, MTP prefill), rank-0 device time in a profiled window of ~40 steps; per request: arrival (absolute s), TTFT, TPOT, ITLs; the header in the Sim's knob names |
| `logs/engine/full_decode_bench.jsonl`, `full_prefill_bench.jsonl` | decode / prefill step wall, host launch, and per-region device time per granularity (op, mixer, layer, block) on a quiet host |
| `kernel-profile run qsa_mixer2` (chips 8-15, this work) | 118 new `qsa_mixer2` rows (below) |

The serving runs are W64 (64 requests, 2048 in / 512 out, 50 ms apart) at 11 settings
and Poisson 0.25 / 1 req/s (lognormal 512 in, 64-256 out) at 5. Host load averages
were 11-65 during the runs, which moves decode steps by 5-10% between otherwise
equal runs.

## What changed

| # | Finding (engine evidence) | Change in the Sim |
|---|---|---|
| 1 | With the barrier off a region's NRT dispatch is 0.10-0.15 ms, collective or not; per region the host spends ~0.55 ms (op) to ~1.3 ms (layer) assembling inputs and launching, while the device runs; decode at layer granularity is host-bound (63 regions x 1.3 + ~19 ms of per-forward host work ≈ 101 ms; measured 89-107 ms), block decode is device-bound (host 52 + 19 < device 62-97) | `launch_model: pipelined` (default): forward = `max(Σ_regions max(device_r + region_gap_ms, region_host_ms), regions x region_host_ms + forward_overhead_ms)`; step + `step_overhead_ms`. Fitted: `region_host_ms` per_op 0.3 / mixer 1.0 / layer 1.3 / block 3.7, `forward_overhead_ms` 19, `mtp_forward_overhead_ms` 2, `region_gap_ms` 0.17, `step_overhead_ms` 2.5. The old 1.25 ms collective launch floor (the barrier) is gone; `calibrated` / `l1_additive` remain, now per region and per forward |
| 2 | The device of a region holding a world collective waits for the slowest rank's launch: a GDN layer region is 1.27-1.31 ms at B = 1 and 16 (quiet) while its rows sum to 0.8-1.2 ms | every region's device is floored at its dispatch time (item 1's `max`); the region-level floored device matches the bench's per-region sums (layer decode 69-75 vs 69-74 ms) |
| 3 | Decode and prefill are separate forwards per step, each with its own MTP passes | `step_forwards: separate` (default; `fused` = one forward, a what-if) |
| 4 | SPMD: every DP group runs the forward's padded shape; an idle group pays the full shape; the prefill context bucket is that of the furthest start + T (a 16-token tail chunk at 2032 padded to 2048 makes every group's context 4096) | `spmd_padding: true` (default); token-wise regions run at the padded rows exactly (no re-bucketing); context from the furthest prefix + padded T |
| 5 | MoE and PLE all-gather every group's rows over the world and reduce-scatter back; a group batch not divisible by its 8 TP ranks is gathered from each of them ("dup": k=2 B=4, 12 rows, runs MoE on 384 rows; decode step 161 vs 109 ms at B=8) | world gathers count dup rows; RS/AG applies only to a forward whose rows split over the TP group (else the all-reduce form) |
| 6 | One prefill chunk per DP group per step | worker knob `max_prefill_chunks_per_partition` (presets: 1) |
| 7 | The QSA prefill mixer steps from 2.4 to 36 ms (2048 rows) once its context passes the 2048-token selection budget; the prior grid {0, 32768} interpolated 5.4 ms at context 4096, so every C=4096 prefill forward was 430 ms short | `qsa_mixer2` caches prefill on `max(0, C - budget)` over {0, 32, 2048, 6144, 30720}; decode adds C = 2048; the 118 rows profiled (tp 8/4/2, torch_sparse and nki); default context buckets start at 2048 as the engine's do |
| 8 | Block granularity (cut only at QSA front \| back + the head gather): 14 regions, decode 62-71 ms device | `region_granularity: block`; regions counted per forward from the engine's cut points (layer 63, mixer 112, block 14; MTP 4 / 2) |
| 9 | The engine's layer region runs `hc_mix` and `hc_combine` as separate stages; the fused `hc_combine_mix` L1 row is 5x below `hc_mix` at 2048 tokens, which made prefill 15-20% short | HC is `combine` + `mix` leaves at every granularity; f_g = 1.0 |
| 10 | MTP: verify L = k+1 rows per request (GDN q = k+1, QSA and MoE B·L rows), catch-up over the verify rows, k-1 single-row draft passes, MTP over every prefill chunk, k-1 first drafts after a final chunk; acceptance 0.65-0.69 at k = 1, (0.63, 0.77) per position at k = 2 | each pass its own forward (MTP overhead per pass); per-request acceptance read off the ITLs for the E2E traces |
| 11 | all_to_all with exact capacity computes the same 4N rows per rank as all_gather and adds slot packing / scatter-add: prefill forward 4332 vs 3495 ms, decode +12 ms | experts see `ranks x cap` rows (cap = S exact, or `min(S, ceil(cf S k / ranks))` with `moe_a2a_capacity_factor`); the <2-rows fallback; `moe_a2a_pack_us_per_row` 0.8 µs per received row on each a2a leaf (fitted; no L1 row for the packing) |
| 12 | Arrival times are absolute | presets set `request_rate: 1.0` (it was the default 10, replaying the 50 ms trace at 5 ms) |

## Step-level alignment (replay)

Every engine step's shape went through `timing-predict` (`align.py replay`). Cells are
the median step wall, measured / predicted (ms). "Total" sums every step; "stalls" are
the engine's first executions of shapes after warm-up (steps 0-2 of every run, 1.2-2.4
s each), which the Sim does not model.

| run | steps | decode step | mixed step | prefill forward | total | stalls (s) | total w/o stalls |
|---|---|---|---|---|---|---|---|
| w64 layer k0 | 543 | 100 / 103 (+3%) | 3609 / 3595 (-0%) | 3495 / 3491 (-0%) | -2.0% | 3.0 | -0.1% |
| w64 layer k1 | 509 | 128 / 121 (-5%) | 3743 / 3708 (-1%) | 3496 / 3491 (-0%) | -5.8% | 4.4 | -3.4% |
| w64 layer k2 | 483 | 171 / 161 (-6%) | 3793 / 3766 (-1%) | 3495 / 3491 (-0%) | -5.0% | 3.7 | -3.1% |
| w64 block k1 | 481 | 101 / 112 (+11%) | 3705 / 3671 (-1%) | 3496 / 3481 (-0%) | -1.0% | 2.6 | +0.6% |
| w64 RS/AG k1 | 430 | 120 / 116 (-3%) | 3438 / 3272 (-5%) | 3209 / 3068 (-4%) | -5.8% | 3.3 | -3.7% |
| w64 a2a k1 | 396 | 140 / 137 (-2%) | 4606 / 4455 (-3%) | 4332 / 4197 (-3%) | -4.8% | 2.8 | -3.4% |
| w64 layer k0 chunk 512 | 591 | 100 / 103 (+4%) | 942 / 898 (-5%) | 837 / 795 (-5%) | -4.7% | 4.1 | -1.7% |
| w64 layer k1 chunk 512 | 504 | 129 / 121 (-6%) | 990 / 936 (-5%) | 838 / 795 (-5%) | -9.4% | 2.7 | -7.6% |
| w64 RS/AG k1 chunk 512 | 496 | 118 / 116 (-2%) | 861 / 826 (-4%) | 716 / 693 (-3%) | -6.9% | 4.7 | -3.2% |
| w64 2 replicas layer k1 (r0, r1) | 459, 467 | 113 / 111 (-3%), 112 / 111 (-2%) | 3719 / 3708, 3723 / 3708 (-0%) | 3492 / 3491 (-0%) | -4.4%, -3.9% | 2.8 | -1.9%, -1.3% |
| w64 2 replicas RS/AG k2 chunk 512 (r0, r1) | 376, 393 | 122 / 118 (-3%) | 913 / 893, 915 / 893 (-2%) | 716 / 693 (-3%) | -6.4%, -6.9% | 3.0 | -2.9%, -3.4% |
| poisson 1 layer k0 | 287 | 107 / 103 (-4%) | 3091 / 3188 (+3%) | 2971 / 3085 (+4%) | -5.0% | 4.5 | -0.7% |
| poisson 1 layer k1 | 228 | 111 / 111 (-0%) | 3206 / 3277 (+2%) | 2986 / 3085 (+3%) | -1.5% | 2.4 | +0.9% |
| poisson 1 layer k0 chunk 512 | 444 | 97 / 103 (+7%) | 937 / 899 (-4%) | 833 / 795 (-4%) | -4.1% | 3.7 | -0.2% |
| poisson 0.25 layer k0 | 836 | 107 / 103 (-3%) | 957 / 898 (-6%) | 834 / 795 (-5%) | -7.8% | 7.0 | -3.1% |
| poisson 0.25 layer k1 | 684 | 101 / 111 (+10%) | 983 / 926 (-6%) | 835 / 795 (-5%) | +0.5% | 4.2 | +4.0% |

Without the stalls, the summed step time is within ±4% for 17 of 18 logs (chunk-512
k = 1: -7.6%). Per phase, per granularity:

| Granularity | Decode (serving) | Prefill forward (serving) | Decode bench (quiet host) | Prefill bench |
|---|---|---|---|---|
| layer | -6% .. +10% | T=2048: -0% (C=4096), +3..4% (C=2048); T=512: -5%; RS/AG -3..-4% | -2% .. +14% (2 runs, the second 10% faster than the first) | T=128/512/2048: -1% / -3% / +6% |
| block | +11% (device-bound; the Sim's leaves sum 7% over the fused block) | -0% | -17% .. -5% (against p10: -3% .. +4%) | +11% / -0% / +5% |
| mixer | not served | | +5% .. +11% | |
| per_op | not served | | -8% .. +7% | |

Device time (rank 0, profiled window) is the engine's busy time including the waits
inside cross-group collectives; the Sim's leaves exclude those waits (they are in each
forward's launch leaf), so the replay's raw leaf sum is 10-25 ms below it at layer
decode while the floored region sum matches the bench's per-region sums.

## End-to-end alignment

Each run simulated with its knobs, its requests at their engine arrival times, and each
request's acceptance from its ITLs (`align.py e2e`; nearest-rank percentiles as
`serve.py`).

| run | duration (s) meas / sim | out tok/s | TTFT p50 (s) | TPOT p50 (ms) |
|---|---|---|---|---|
| W64 layer k0, chunk 2048 | 169.3 / 163.7 (-3.3%) | 193.6 / 200.2 (+3.4%) | 62.0 / 58.6 (-5.5%) | 211.1 / 206.3 (-2.3%) |
| W64 layer k1 | 183.4 / 170.8 (-6.9%) | 178.7 / 191.8 (+7.4%) | 65.2 / 60.4 (-7.3%) | 191.3 / 178.4 (-6.7%) |
| W64 layer k2 | 192.5 / 181.4 (-5.8%) | 170.2 / 180.6 (+6.1%) | 65.3 / 60.9 (-6.6%) | 185.9 / 178.1 (-4.2%) |
| W64 block k1 | 161.6 / 160.1 (-0.9%) | 202.7 / 204.7 (+0.9%) | 62.7 / 59.8 (-4.7%) | 167.4 / 168.9 (+0.9%) |
| W64 RS/AG k1 | 157.6 / 146.5 (-7.1%) | 207.9 / 223.7 (+7.6%) | 58.9 / 53.0 (-9.9%) | 163.3 / 156.0 (-4.5%) |
| W64 all_to_all k1 | 197.9 / 186.7 (-5.6%) | 165.6 / 175.5 (+6.0%) | 77.9 / 72.9 (-6.4%) | 216.9 / 203.6 (-6.1%) |
| W64 layer k0, chunk 512 | 135.3 / 126.7 (-6.4%) | 242.1 / 258.6 (+6.8%) | 45.0 / 39.1 (-13.2%) | 173.7 / 167.5 (-3.5%) |
| W64 layer k1, chunk 512 | 141.9 / 126.8 (-10.7%) | 230.9 / 258.5 (+12.0%) | 45.4 / 40.4 (-11.1%) | 148.8 / 132.5 (-10.9%) |
| W64 RS/AG k1, chunk 512 | 126.4 / 115.8 (-8.4%) | 259.3 / 283.0 (+9.1%) | 42.0 / 35.7 (-14.9%) | 130.3 / 121.9 (-6.5%) |
| W64 2 replicas, layer k1 | 111.5 / 106.2 (-4.7%) | 294.0 / 308.4 (+4.9%) | 33.6 / 30.8 (-8.4%) | 115.9 / 110.4 (-4.8%) |
| W64 2 replicas, RS/AG k2, chunk 512 | 85.8 / 77.8 (-9.3%) | 381.8 / 421.1 (+10.3%) | 23.1 / 19.3 (-16.4%) | 92.4 / 84.3 (-8.8%) |
| Poisson 1, layer k0 | 106.6 / 102.9 (-3.5%) | 98.6 / 102.2 (+3.6%) | 5.4 / 4.7 (-14.4%) | 349.2 / 337.9 (-3.2%) |
| Poisson 1, layer k1 | 100.3 / 99.3 (-0.9%) | 104.9 / 105.9 (+0.9%) | 4.9 / 4.7 (-4.3%) | 312.2 / 312.8 (+0.2%) |
| Poisson 1, layer k0, chunk 512 | 95.5 / 96.7 (+1.2%) | 110.1 / 108.8 (-1.2%) | 2.1 / 1.8 (-16.1%) | 260.0 / 250.8 (-3.5%) |
| Poisson 0.25, layer k0 | 146.0 / 145.6 (-0.3%) | 38.8 / 38.9 (+0.3%) | 3.1 / 2.6 (-15.8%) | 163.6 / 148.9 (-9.0%) |
| Poisson 0.25, layer k1 | 135.3 / 137.1 (+1.3%) | 41.9 / 41.4 (-1.3%) | 3.1 / 2.7 (-11.1%) | 102.8 / 101.7 (-1.1%) |

Median |error|: duration 5.2%, out tok/s 5.4%, TTFT p50 10.5%, TPOT p50 4.3%; 6.0%
over all 64 cells. Worst cell: 16.4% (TTFT, 2 replicas RS/AG k2 chunk 512).

**Ranking.** The 11 W64 rows sort by duration in the same order measured and
simulated: 2 replicas RS/AG k2 chunk 512 < 2 replicas layer k1 < RS/AG k1 chunk 512 <
layer k0 chunk 512 < layer k1 chunk 512 < RS/AG k1 < block k1 < layer k0 < layer k1 <
layer k2 < all_to_all k1. One adjacent pair is fragile: layer k0 vs k1 at chunk 512
are 6.6 s apart measured and 0.1 s apart simulated (k1 at chunk 512 is the most
under-predicted row, see residuals).

**Before the alignment.** At the same shapes the old composition priced a layer decode
step at 97-106 ms through 50 collective-launch floors of 1.25 ms (the barrier; right
total, wrong mechanism, and wrong at block), a T=2048 prefill forward at 2372 ms
(engine 2985 at C=2048, 3495 at C=4096: -21% / -32%), ran decode and prefill rows in
one forward without padding idle groups, and replayed the trace 10x compressed.

## Residuals

- **The Sim is 5-10% fast on most W64 rows.** About half of it is the engine's
  first-use stalls (1.2-2.4 s on each of steps 0-2: 2.4-7 s per run, 2-4% of a W64
  run). They also make TTFT p50 10-16% low where the first requests wait behind them
  (Poisson 0.25 / 1, chunk 512). Not modeled (an engine warm-up artifact).
- **Prefill at T = 512 is 4-6% short** (all runs), T = 2048 within ±4%: the MoE block
  L1 row at 2048 gathered rows is ~6% below the in-engine MoE, at 8192 rows ~2% above.
  One f_g cannot fit both; chunk-512 rows inherit -5%.
- **Device-bound decode** (k = 1/2 at B = 16, rows 32-48 per group) is 4-6% short;
  block decode is 11% long: the Sim sums isolated leaf rows, the fused 4-layer block
  runs ~7% under them at decode (fixed per-NEFF costs), and its floors are spread
  evenly over the block's regions.
- **k = 1, chunk 512** is the worst row (-10.7%): the two above compound (many T = 512
  prefill forwards, every decode at B = 16 verify).
- **all_to_all** relies on the fitted packing cost (0.8 µs per received row per a2a
  leaf); there is no L1 row for the packing / scatter-add. The capacity-factor path is
  a model, not a measurement: it assumes the expert kernel's time follows the received
  rows; the engine's `moe_a2a_cf` drops tokens under real routing skew.
- **Acceptance is an input.** Each request's per-position acceptance is read off its
  ITLs, not predicted.
- **mixer and per_op** were checked only against the quiet-host decode bench (+5..11%,
  -8..+7%); the Sim's per_op leaves split the engine's ops further (~820 vs 457
  regions), absorbed into `region_host_ms` 0.3.
- **Not modeled:** the engine's 16-request cap per DP group (never binding in these
  runs), the speculative worker's DP placement (round-robin; the engine places on the
  least-loaded group), the PD prefill worker's one prompt per iteration (its spread
  groups pay their own shapes, not SPMD-padded; P/D is not in the engine), host load.

## Rerun

```bash
source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH
uv run --no-sync python tools/trn2-engine-align/align.py replay logs/align/final_replay
uv run --no-sync python tools/trn2-engine-align/align.py e2e logs/align/final_e2e
```
