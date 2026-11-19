# Alignment with the glm5-trn2 seed engine

**Source.** `glm5-trn2/results/seed/serve_c16_replica0.jsonl`: one replica (32 LNC2 ranks, dp4 x tp8,
EP32), chunk 2048, prefill bucket 2048, decode buckets 8 / 16, context buckets 2048 / 4096 / 8192 /
9216, 17 slots per group, cmax 9216, closed-loop 8k1k at C = 16 for the replica (16 warm-up + 48
measured requests). It records 4120 steps, each with the decode forward (rows per group, ms), the
prefill forward (rows and starts per group, ms) and wall ms. It also records every request and the
summary: 1099 tok/s, median TPOT 114.1 ms, interactivity 8.77, median TTFT 6.63 s.

**Tool.** `tools/trn2-config-search/engine_align.py` (output: `alignment.json` here):

1. It reads each forward from the timeline. A forward slower than 3x its shape's median is a
   first-use compile or NEFF load and is dropped. That drops 8 forwards, among them the
   first prefills at 145 s, 44 s and 49 s and the first decode at 74 s.
2. It rebuilds the context of each decode step from the request rows.
3. It replays every distinct shape through `launcher timing-predict` with all host constants
   at 0, so each case's cost log is pure device time.
4. It groups the leaves into the forward's regions: embed, 45 layers, head and combine.
5. It fits `region_host_ms`, `region_gap_ms` and `forward_overhead_ms`, plus a device factor
   per forward side, by weighted least squares on relative error.
6. `step_overhead_ms` is the measured median host time between forwards: step wall minus
   its forwards, plus the gap to the next step.

## Forwards: measured vs predicted

| shape (per DP group) | forwards | measured ms | device (sum of L1 rows) | predicted, trn2-flashnext constants | predicted, fitted |
|---|---|---|---|---|---|
| decode, 4 rows -> bucket 8, context 8192 | 249 | 99.2 | 91.6 | 103.7 | 99.6 |
| decode, 4 rows -> bucket 8, context 9216 | 3865 | 100.8 | 92.7 | 104.9 | 100.7 |
| prefill 2048, start 0 (context 2048) | 15 | 1172.2 | 631.1 | 641.7 | 1129.5 |
| prefill 2048, start 2048 (context 4096) | 15 | 1210.2 | 671.5 | 682.1 | 1201.4 |
| prefill 2048, start 4096 (context 8192) | 15 | 1269.6 | 725.3 | 735.9 | 1297.1 |
| prefill <= 2048, start 6144 (context 8192) | 16 | 1285.6 | 725.3 | 735.9 | 1297.1 |

Fitted constants, now the arch defaults:

| constant | trn2-flashnext | fitted (default now) | identifiability |
|---|---|---|---|
| `region_host_ms` (layer) | 1.3 | **1.1** | weak (decode is device-bound) |
| `region_gap_ms` | 0.17 | **0.1** | from the decode residual (48 regions) |
| `forward_overhead_ms` | 19 | **19** (kept) | not identifiable: 48 x 1.1 + F stays under the ~100 ms device for F < ~45 |
| `step_overhead_ms` | 2.5 | **0.12** | measured directly |
| decode `region_device_factor` | 1.0 | **1.0** | the decode L1 rows sum to the measured forward |
| prefill `prefill_device_factor` (new knob) | 1.0 | **1.78** | 4 prefill shapes, +-4% |

**Decode.** The decode forward is device-bound at this load. Its L1 rows sum to 92 ms, and 48
region gaps plus the dispatch floor of the small regions (embed, head, combine) add 8 ms. The
prediction lands within 0.5%.

**Prefill.** The prefill forward runs 1.78x the sum of its L1 rows. About 540 ms per prefill
forward, roughly 12 ms per layer, is not explained by the isolated op rows. The candidates:

- in-region interference: the layer region compiles attention and FFN into one NEFF;
- the world collectives at 8192 gathered rows, a 64 MiB per-rank reduce-scatter input, extrapolated
  past the 16 MiB collective grid from the Qwen campaign's rows;
- the KDA prefill barriers inside a larger graph.

The bench rows cannot separate these, so the factor is fitted per side and applied to the
prefill-forward compute leaves. It is the main fidelity limit: a change that shifts work between
prefill kinds is predicted at the same 1.78.

## End to end (in sample)

Preset `presets/trn2/glm53_flash_fp8_trn2_align_seed_c16.yaml` replays the engine's own 64
requests at C = 16 on one replica. Metrics come from `inferencex.py --warmup 1`.

| | engine (measured) | Sim, fitted defaults | Sim, trn2-flashnext constants |
|---|---|---|---|
| total tok/s (one replica, 8 chips) | 1099 | 1107 (+0.7%) | 1116 (+1.5%) |
| median TPOT (ms) | 114.1 | 113.3 (-0.6%) | 114.7 |
| interactivity (tok/s/user) | 8.77 | 8.82 | 8.72 |
| median TTFT (s) | 6.63 | 6.47 (-2.5%) | 4.06 (-39%) |
| median E2E (s) | 123.1 | 122.3 | 121.3 |

This is **in sample**: the constants are fitted on this run's forwards. The unfitted constants
already matched TPOT, because they over-charged the decode host and under-charged prefill device
time by similar amounts. The fit fixes TTFT, which prefill sets.

Open: validation at another load, for example C = 64 per replica (decode bucket 16, more prefill
interleaving), and at another layout (tp4). `forward_overhead_ms` stays unidentified until a run is
host-bound: decode buckets at 1-2 rows, or block regions.

## Re-alignment on the round-1 service (official measurement, 2026-11-17)

**Source.** The protected evaluator's official round-1 measure of the glm5-trn2 R1 service
(`d41eb54`, `campaigns/glm8k1k/round1/evidence/measure-20261116-223250-*`): two replicas of the
seed engine behind one frontend, decode-protected waves, C = 8 (one request per DP group),
decode bucket 8, context buckets 2048 / 4096 / 6144 / 6656 ... 9216 (512 apart), InferenceX 8k1k
(80 measured requests). Measured: 40.20 tok/s/chip, median TPOT 98.66 ms (10.14 tok/s/user),
median TTFT 9.70 s, in-service decode step 98.47 / 98.05 ms per replica, prefill step 1229 /
1234 ms.

**Two workload facts the simulator had wrong** (exogenous, from the InferenceX source
`infx/bench_serving/benchmark_serving.py::sample_random_requests` and the run's own request
list): output lengths are uniform in [0.8 OSL, OSL] like the inputs (mean 926 here, not 1024),
and the 2C warm-up requests finish before the measured run starts on an idle server. The trace
generator now draws both lengths and lists the 10C measured requests only, and
`inferencex.py` counts every completion by default.

**Host floor (measured).** The serving timelines are device-bound, so they cannot identify the
host constants (the fit above left `forward_overhead_ms` at the trn2-flashnext 19 ms). A probe
(`glm5-trn2/campaigns/glm8k1k/round2/tick2/host_probe.py`, one replica of the unchanged R1
engine) timed decode forwards with the engine's 12-launch throttle and with it lifted (a whole
forward's 48 launches fit the 63-deep NRT queue, so the launch returns after the host has
dispatched every region):

| decode (bucket 8) | step, LAG 12 | host launch of the forward, LAG 60: per-rank median (min / median / max over 32 ranks) |
|---|---|---|
| 1 row per group, context 8192 | 99.8 ms | 15 / 24-30 / 45-46 ms |
| 8 rows per group, context 8192 | 99.6 ms | 14-16 / 36-38 / 45 ms |
| 1 row per group, context 9216 | 101.4 ms | 15-16 / 17-29 / 42-44 ms |
| 1 row per group, context 2048 | 91.7 ms | 15 / 17-21 / 30-37 ms |

The SPMD forward waits for the slowest rank at every collective, so the host floor is about
45 ms for 48 regions: `region_host_ms` 0.9 (layer) and `forward_overhead_ms` 2 are the new
defaults (48 x 0.9 + 2 = 45.2 ms; before: 48 x 1.1 + 19 = 71.8 ms). The device-bound
predictions do not change; a what-if that makes the device faster than ~45 ms per forward now
hits the measured floor instead of a 72 ms prior. Real rows per group do not change the decode
step (8 vs 1: same bucket, same static shapes).

**Forwards** (`engine_align.py <measure>/metrics.json --host-ms 0.9,2.0`: the warm-up's steady
forwards of every compiled shape, both replicas): decode device factor 0.98, region gap
0.15 ms, **prefill device factor 1.76** (1.78 on the seed's offline run: the prefill composition
excess reproduces across runs). Decode shapes within +-1%, prefill shapes -4% ... +3%.

**End to end, out of sample** (`presets/trn2/glm53_flash_fp8_trn2_service_r1.yaml`: the new
`wave_admission` worker mode with the service's 100 / 300 ms quiet window, at most 8 requests
per group, the frontend's `least-loaded` replica routing; point `r1_official` replays the 80
measured requests in order):

| | measured (official) | Sim (host 0.9 / 2.0) |
|---|---|---|
| tok/s per chip | 40.20 | 40.6 (C 8 trace) / 39.5 (replay, host 1.1 / 19) |
| median TPOT (ms) | 98.66 | 99.1 |
| interactivity | 10.14 | 10.09 |
| median TTFT (s) | 9.70 | 8.2 (C 8 trace) / 11.1 (replay) |

Residuals: decode +0.5% (run-to-run spread of the decode step is about +-1.5%: 97.1-101.4 ms
for the same shapes across the service warm-up, the in-service mean and the probe); TTFT
depends on which requests wait for a draining replica (random output lengths), +-15%.

## Re-alignment on the round-3 service (official measurement, 2026-11-17)

**Source.** The official round-3 measure of the glm5-trn2 R3 service (`ddac04a`,
`campaigns/glm8k1k/round3/evidence/measure-20261117-135039-*`). It runs the R1 waves plus three
decode changes: an NKI decode-MoE kernel over FP8 experts (M <= 64), a DSA pooled-key cache, and
lockstep first token. A start-up cap probe measured the 16-row step at 92 ms, so the run served 8
rows per group, C = 64.

Measured: 312.0 tok/s/chip, median TPOT 66.29 ms (15.09 tok/s/user), median TTFT 44.5 s,
in-service decode step 63.9 ms, prefill step 1236 ms. A second official measure of the same code
gave 312.22 / 66.21.

**Device evidence** (glm5-trn2 `campaigns/glm8k1k/round3/probe/`, one replica, rank-0 NRT profile
per region):

- The decode forward takes 62.0 ms at 8 rows and 91.0 ms at 16 (ctx 8192), against 96.2 ms for
  the round-1 engine.
- Op benches, new against old:

  | op | 8 rows | 16 rows | 32 rows |
  |---|---|---|---|
  | MoE, M = 4 x rows | 0.491 / 1.163 (M 8) | 0.505 / 1.195 (M 32) | 1.115 / 1.314 (M 64) |
  | DSA decode, B = rows, ctx 8192 | 0.878 / 1.041 | 1.247 / 1.597 | 1.840 / 2.568 |

- A KDA layer's non-MoE time is flat from 8 to 16 rows (0.69 -> 0.66 ms), so mHC and KDA decode do
  not grow with rows on device.

**Model** (`presets/trn2/glm53_flash_fp8_trn2_service_r3.yaml`):

- The measured kernel ratios are `kind_time_scale` entries (`moe_partial`, `dsa`).
- mHC and KDA are held flat between 8 and 16 rows.
- `region_gap_ms` is 0.05: the measured TPOT is the device step + ~4.4 ms, and 0.10 overshot by
  ~2.4 ms per forward.
- `prefill_device_factor` is 1.88, fitted to the in-service prefill step. The round-1 value of
  1.78 was fitted to single warm-up shapes, and the sim's prefill step was then 5.5% short of
  the in-service mean.

| | measured (official) | Sim |
|---|---|---|
| tok/s per chip | 312.0 | 310.9 |
| median TPOT (ms) | 66.29 | 66.38 |
| interactivity | 15.09 | 15.07 |
| median TTFT (s) | 44.5 | 44.95 |
| prefill step (ms) | 1236 | 1227 |
| 16-row step (ms) | 92.0-92.7 (cap probe), 91.0 device | 94.9 |

Disagreements recorded:

- Tick 3's prediction at 16 rows (75-81 ms) assumed the MoE kernel at 0.35x for every decode M.
  On device it is 0.85x at M 64: per-expert cost grows with M, and the kernel multiplies all M
  rows for every active expert.
- Before the mHC/KDA calibration, the sim was 5-8% pessimistic at 16 rows.

## Re-alignment on the round-4 service (official measurement, 2026-11-17)

**Source.** The official round-4 measure of the R4 service (`6ce3143`,
`campaigns/glm8k1k/round4/evidence/measure-20261117-180112-*`): 339.0 tok/s/chip, median TPOT 57.63 ms
(17.35 tok/s/user), C 64. The 16-row cap probe measured 82 ms. Round 4 added a routed-row decode-MoE kernel and
fused decode mHC.

**Device evidence** (round-4 probe):

- Decode forward: 52.7 ms at 8 rows and 78.9 ms at 16.
- MoE new / `dense_local`: 0.504 / 1.194 ms at M 32, 0.890 / 1.317 ms at M 64.
- mHC isolated, fused / torch at 1 row per rank: pre 0.061 / 0.126 ms, post 0.025 / 0.055 ms.

**Model** (`presets/trn2/glm53_flash_fp8_trn2_service_r4.yaml`): the measured MoE ratios. Fused mHC is 0.45x in the
composed region, fitted to the official measure; that matches the isolated benches.

| | measured (official) | Sim |
|---|---|---|
| tok/s per chip | 339.0 | 338.1 |
| median TPOT (ms) | 57.63 | 57.88 |
| interactivity | 17.35 | 17.28 |
| 16-row step (ms) | 78.9 device, 82.0-82.7 cap probe | 77.5 |

Disagreement: at 16 rows the model is now ~4 ms optimistic. Before round 4's kernels it was 4-5 ms pessimistic,
so the 16-row composition is not yet captured beyond about +-5%.

**Prefill device profile** (glm5-trn2 `campaigns/glm8k1k/round5/tick5/prefill_devprof_summary.json`): one prefill
forward of rank 0, 1.27 s.

- 46.7 GB of SBUF spill traffic (24.8 GB saved, 21.9 GB reloaded) out of 84 GB of HBM traffic.
- Collectives active 0.21 s.
- TensorE active 0.18 s, VectorE 0.33 s, DMA 0.49 s.

This is the first direct evidence for the prefill composition excess, which `prefill_device_factor` 1.88 absorbs.
The likely causes are spills and collective time in the composed region, not the isolated op rows.

## Re-alignment on the round-7 service (official measurement, 2026-11-18)

**Source.** The official round-7 measure of the R7 service (`466700d`,
glm5-trn2 `campaigns/glm8k1k/round7/evidence/`): 346.0 tok/s/chip, median TPOT 55.92 ms (17.88 tok/s/user), C 64,
mean decode step 54.2 ms. The 16-row cap probe measured 73.6-74.0 ms. Round 7 changed only the decode-MoE kernel
(block-interleaved fused evacuation).

**Model** (`presets/trn2/glm53_flash_fp8_trn2_service_r7.yaml`): the R4 model with round 7's measured decode-MoE
ratios, 0.394x of `dense_local` at M 32 and 0.512x at M 64.

| | measured (official) | Sim |
|---|---|---|
| tok/s per chip | 346.0 | 343.0 |
| median TPOT (ms) | 55.92 | 56.48 |
| interactivity | 17.88 | 17.71 |
| 8-row decode step (ms) | 54.2 | 56.4 |
| prefill step, 2048 x 4 groups (ms) | 1,236 | 1,227 |

**Round 8, sparse DSA decode** (measured, not retained; `r8_dsa_sparse`): an NKI kernel gathers each row's top-512
pools and attends over them only.

- Isolated DSA decode at ctx 8192: 0.762 / 0.879 / 1.226 ms at B 8 / 16 / 32, against round 7's 0.879 / 1.252 /
  1.839.
- Official measure 352.86 @ 18.51 (TPOT 54.04 ms, decode step 52.9 ms). With the isolated ratios the model gives
  348.4 @ 18.16.
- At 8 rows the model's decode step stays ~2 ms slower than the device (55.1 vs 52.9 ms), so throughput reads -1.3%
  and interactivity -1.9%.

**Round 9, KDA prefill as one NKI kernel** (rejected on correctness, so no official measure). In the evaluator's
warm-up the composed prefill step was 942 / 960 / 1,036 / 1,052 / 1,085 ms at ctx 2048 / 4096 / 6144 / 8192 / 9216,
against 1,171 / 1,187 / 1,262 / 1,275 / 1,308 (-17 to -20%). In the model this is `prefill_device_factor` 1.88 ->
~1.53: 375 tok/s/chip at C 64 with round 8's DSA kernel. The kernel failed the evaluator's concurrent passkey check:
deterministically, in DP groups 1-3, on a prompt whose fp32 first-token margin is 0.81 logits. Cause (Tick 10): the
kernel rounds the KDA gate GEMM products to bf16, as the torch source reads. The compiled torch path keeps them fp32
(neuronx-cc drops the cast). That doubled passkey misses (15 against 7 of 128 prompt-group pairs). With fp32 gate
products the kernel matches the compiled torch block (state 3.6e-6) and the miss rate returns to 7 / 128. Round 10
carries the repaired kernel; the prefill step model above stands.

## Re-alignment on the round-11 and round-13 services (official measurements, 2026-11-18 / 2026-11-19)

**Sources** (glm5-trn2 `campaigns/glm8k1k/round{10,11,13}/`), all at C 64:

| round | change | official |
|---|---|---|
| 10 (`fe7d0e4`) | round 9's KDA prefill kernel with fp32 gate products, round 8's sparse DSA decode | 382.47 @ 18.48 |
| 11 (`e4f3578`) | KDA decode as an NKI kernel | 398.54 @ 19.74 (TPOT 50.66 ms, 16-row cap probe 62.5-62.8 ms) |
| 13 (`990cfa2`) | deterministic expert placement (9 per EP rank), fp32 cross-rank partials, prefill chunk 4096 | 419.45 @ 19.82 (TPOT 50.44 ms, TTFT 31.9 s); same-code samples 419.45-420.86 |

**Model** (`presets/trn2/glm53_flash_fp8_trn2_service_r13.yaml`): the calibrated round-11 model of the campaign's
layout study, with round 13's changes.
- The round-11 per-kind ratios: `kda` 0.33 is round 11's decode kernel; `prefill_device_factor` 1.53 is round 10's.
- The measured host floor (`host_time_scale` 0.28), the wave scheduler and the lockstep first token.
- A decode `region_device_factor`:
  - 1.12 reproduces round 11.
  - The excess is mostly the wait at the MoE combine. Expert load is imbalanced, and the rank with the most active
    local experts sets the layer: ~170 us per layer at 8 rows (Ticks 12-13, measured on the evaluator's prompts).
  - Round 13's placement removes 30 us of it; its fp32 partials add ~0.3 ms per step. That gives 1.1052.

| | round 11 measured | Sim | round 13 measured | Sim |
|---|---|---|---|---|
| tok/s per chip | 398.54 | 394.9 | 419.45 | 419.0 |
| interactivity | 19.74 | 19.69 | 19.82 | 19.93 |
| median TPOT (ms) | 50.66 | 50.8 | 50.44 | 50.2 |
| median TTFT (s) | 36.1 | - | 31.9 | 32.4 |
| 16-row decode step (ms) | 62.7 | 62.5 | 62.9 | - |

The round-13 prediction was made before the run (Tick 13: 419.0 @ 19.93). It landed within 0.1% on throughput and
0.6% on interactivity.

**Round 14 what-ifs** (`r14_dsa_x05`, `r14_dsa_x07`): the DSA decode front in NKI. The DSA phase of a decode layer
measured ~0.82 ms at 8 rows and ~0.89 ms at 16 on the round-13 code.

| DSA decode time | C 64 |
|---|---|
| 0.5x | 473.1 @ 22.12 |
| 0.7x | 453.1 @ 21.19 |

At C 128 the 16-row step stays above round 14's 49 ms cap (TPOT <= 50 ms at the target of 20 tok/s/user).
