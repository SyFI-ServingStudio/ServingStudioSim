# Config search: Qwen3.8-Flash-Next BF16 on one trn2.48xlarge

This file reports simulator-only results (2026-10-25). It covers the best serving
configurations for three workload families on 16 Trainium2 chips (64 LNC2 cores), which
knobs matter, where the remaining time goes, and how far the conclusions move if the
uncertain constants change. The tool is `tools/trn2-config-search/` (its README has the
stage grammar), and the agent workflow is skill `operate-search-trn2-configs`. Every
number comes from `qwen4_exp_neuron_bf16[_speculative]` over `profiling/profile.db` at
commit `7d8ac4a2`, with the loaded-host launch constants unless a what-if row says
otherwise. Nothing was profiled for this search.

## Method

| Stage | Configs | What it varies | Runs |
|---|---|---|---|
| `s1_layout` | 1200 planned, 114 buildable | every 32- and 64-core layout x {all_reduce, RS/AG} x {all_gather, all_to_all} x {mixer, layer} | 1026 |
| `s2_knobs` | 280 planned, 238 buildable | one knob at a time around the 8 stage-1 winners: chunk size, granularity, collectives, PLE residency, three bucket inventories, every per-op impl list | 2142 |
| `s3_unified` | 20 | the 5 best layouts x chunk {2048, 4096} x context buckets {default, fine} | 180 |
| `s3_mtp` | 12 planned, 9 buildable | MTP k = 1..4 on 3 layouts, acceptance 0.5 / 0.71 / 0.85 | 243 |
| `s3_pd` | 9 | P/D, 8 + 8 chips, prefill layout x decode layout | 81 |
| `s4_whatif_*` | 456 + 36 + 36 | the s1 grid, the MTP and the P/D configs under 4 scenarios | 4752 |

Every run uses 2 replicas of 32 cores. Each P/D run uses one 32-core prefill instance and
one 32-core decode instance. Every unified config therefore uses all 16 chips, and
`tok/s` compares directly across configs (divide by 16 for tok/s/chip).

**Workloads** (`tools/trn2-config-search/stages/workloads.yaml`, seeded):

| Name | Requests | Input / output | Arrivals |
|---|---|---|---|
| `fixed_2k_512` | 64 | 2048 / 512 | 50 ms apart (the step-4 trace, `presets/trn2/trn2_fixed_2k_512_c64.csv`) |
| `agentic_lo`, `agentic_hi` | 48 | 16-32k / 200-500 | bursts of Poisson(6) requests within 2 s, one burst per ~90 s / ~30 s |
| `chat_r{0.25,0.5,1,2,4,8}` | 96-256 | 1-2k / 200-400 | Poisson at 0.25 to 8 req/s |

The engine has no prefix cache, so the agentic prompts are fully prefilled.

**SLOs** (parameters; `collect --slo name=TTFT_ms,TPOT_ms` recomputes without
re-simulating):

| SLO | TTFT | Mean TPOT |
|---|---|---|
| `chat` | 2 s | 100 ms |
| `chat_strict` | 1 s | 60 ms |
| `agentic` | 30 s | 150 ms |
| `agentic_relaxed` | 60 s | 250 ms |

Goodput is the number of requests that meet both limits per second of simulated span.

> **Arrival-time note.** Under `trace_timed` the simulator divides trace arrival times by
> `workload.request_rate`, whose default is 10. The step-4 presets do not set it, so the
> README's "First results" replayed the 50 ms trace at 5 ms spacing. This search sets
> `request_rate: 1.0`.

## Results

### Best configuration per workload

All of the best configs share four settings:

- `region_granularity: layer`;
- `tp_collective: reduce_scatter_all_gather`;
- `moe_dispatch: all_to_all`;
- MoE EP32 (`moe_tp: 1`).

Other knobs are at their defaults unless noted. "ctx fine" means context buckets
`[4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072]`. The MTP rows use
acceptance 0.71.

| Workload | Objective | Config | tok/s | TTFT p50 / p90 (s) | TPOT p50 / p90 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| fixed_2k_512 | max tok/s, chat goodput | MTP k=3, dp4-tp8-g8, ctx fine | 5660 | 0.98 / 1.18 | 48 / 52 | 2.21 @chat, 1.21 @strict |
| fixed_2k_512 | no MTP | dp4-tp8-g4 (any dp4/dp8 layout within 0.5%) | 3326 | 0.84 / 1.02 | 91 / 93 | 1.30 @chat |
| agentic_hi | max tok/s | MTP k=3, dp2-tp8-g16, ctx fine | 3118 | 8.1 / 16.4 | 50 / 87 | 0.131 @agentic |
| agentic_hi | TTFT | MTP k=3 or k=1, dp4-tp8-g8 / dp8-tp4-g4 | 3055-3114 | 6.5 / 10.2-10.8 | 44-61 | 0.131 |
| agentic_hi | no MTP | dp4-tp8-g8, ctx fine | 2987 | 6.2 / 10.2 | 100 / 129 | 0.126 |
| agentic_lo | max tok/s, goodput | MTP k=3, dp8-tp4-g4, ctx fine | 2665 | 8.5 / 11.0 | 49 / 92 | 0.108 |
| chat_r1 | chat_strict goodput | MTP k=3, dp4-tp8-g8 | 1863 | 0.36 / 0.41 | 47 / 52 | 1.04 (all requests) |
| chat_r8 (saturated) | max tok/s | MTP k=2, dp8-tp4-g4 | 8394 | 0.63 / 0.81 | 98 / 130 | 2.46 @chat |
| chat_r8 | no MTP | dp2-tp8-g16, fine decode buckets | 6377 | 0.35 / 0.49 | 131 / 155 | 0.04 @chat |

**Against the step-4 layout.** The step-4 layout is dp4-tp8-g8-ep32 with mixer regions,
all_reduce and all_gather, run as 2 replicas.

| Workload | step-4 layout tok/s, TTFT p50, TPOT p50 | best unified | best MTP (0.71) |
|---|---|---|---|
| fixed_2k_512 | 1617, 13.1 s, 169 ms | 3329 (2.06x), 1.3 s, 90 ms | 5660 (3.5x), 1.0 s, 48 ms |
| agentic_hi | 2468, 29.7 s, 250 ms | 2987, 6.2 s, 100 ms | 3118, 8.1 s, 50 ms |
| chat_r1 | 1341, 1.0 s, 228 ms | 1633, 0.36 s, 100 ms | 1863, 0.36 s, 47 ms |
| chat_r8 | 3541, 21.8 s, 254 ms | 6377 (1.8x), 0.35 s, 131 ms | 8394 (2.4x), 0.63 s, 98 ms |

### Throughput-latency frontier

The fixed and agentic traces fix the offered load, so their frontier collapses to one or
two points, and MTP dominates both. The useful frontier is the chat rate sweep: one line
per config, with throughput rising with offered load until it saturates.

Each row lists tok/s, TTFT p50 / p90 (s), TPOT p50 / p90 (ms), and goodput @chat /
@chat_strict (req/s).

| Rate (req/s) | unified dp4-tp8-g8 (ctx fine) | MTP k=2 (0.71) | MTP k=3 (0.71) | P/D: P dp2-tp8-g16, D dp4-tp8-g8 | step-4 layout |
|---|---|---|---|---|---|
| 0.25 | 472, 0.30/0.37, 91/92, 0.26/0 | 491, 0.29/0.33, 45/48, 0.27/0.27 | 492, 0.29/0.34, 41/43, 0.27/0.27 | 472, 0.19/0.19, 89/90, 0.26/0 | 444, 0.96/1.04, 171/177, 0/0 |
| 1 | 1629, 0.34/0.37, 98/102, 0.56/0 | 1832, 0.35/0.40, 53/57, 1.02/1.02 | 1863, 0.36/0.41, 47/51, 1.04/1.04 | 1622, 0.19/0.33, 94/97, 0.91/0 | 1341, 1.03/1.77, 228/253, 0/0 |
| 2 | 2525, 0.34/0.39, 112/120, 0.23/0 | 2843, 0.36/0.45, 59/63, 1.58/0.96 | 2871, 0.38/0.46, 53/58, 1.60/1.57 | 2505, 0.19/0.33, 99/100, 1.37/0 | 2064, 1.33/2.09, 271/377, 0/0 |
| 4 | 4688, 0.37/0.54, 140/155, 0.04/0 | 5681, 0.41/0.59, 78/92, 3.12/0.25 | 5130, 0.44/0.66, 85/116, 1.93/0 | 4604, 0.51/1.57, 109/113, 0.09/0 | 3434, 6.8/12.9, 261/359, 0/0 |
| 8 | 6322, 0.42/0.56, 135/163, 0/0 | 8204, 0.48/0.64, 96/118, 2.69/0.09 | 5367, 0.65/1.35, 186/212, 0/0 | 5221, 9.6/14.6, 113/117, 0.03/0 | 3541, 21.8/33.5, 254/354, 0/0 |

How to read it:

- Without MTP, decode TPOT has a floor of about 90 ms at layer granularity. That floor
  is launch, not kernels: 48 collective-bearing regions per step at `launch_coll_ms`
  1.25, plus `step_overhead_ms`. No non-MTP config meets the 60 ms strict SLO.
- MTP roughly halves TPOT at every load below saturation, and adds 30% to 70% throughput
  where decode dominates.
- At high load, k=3 collapses (chat_r8: 5367 tok/s, TPOT 186 ms) while k=2 keeps scaling.
  Use k=2 above about 4 req/s and k=3 below. This is not root-caused; see open issues.
- P/D gives the lowest TTFT at light and medium load (0.19 s). It loses capacity once
  prefill saturates the 8 prefill chips: chat_r8 at 5221 tok/s with 9.6 s TTFT,
  fixed_2k_512 at 2647 tok/s, agentic TTFT 20-29 s. With two 32-core units, 8 + 8 is the
  only split: one replica needs at least 32 cores for its weights, and 64-core replicas
  lack world-64 rows.

`report_main.md` (below) has the full per-workload frontiers.

### Which knobs matter

These are one-at-a-time changes from the stage-1 winners (`s2_knobs`), averaged over the
seeds. Each cell is tok/s / TTFT p90 / TPOT p50 relative to the seed.

| Knob = value (seed: layer, RS/AG, a2a, chunk 2048, defaults) | fixed_2k_512 | agentic_hi | chat_r1 | chat_r8 |
|---|---|---|---|---|
| region_granularity = mixer | -57% / +41% / +139% | -12% / +15% / +142% | -27% / +56% / +139% | -39% / +38% / +76% |
| region_granularity = per_op (tp8 layouts only) | -81% / +533% / +443% | -36% / +140% / +486% | -54% / +313% / +445% | -70% / +2544% / +298% |
| moe_dispatch = all_gather | -30% / +1153% / +20% | -11% / +362% / +52% | -4% / +225% / +43% | -32% / +3949% / +44% |
| moe_routed_impls = [nki] (instead of auto) | -10% / +96% / +11% | -4% / +68% / +22% | -4% / +185% / +35% | -9% / +596% / +20% |
| ple_table_shards = dp_group (where it fits) | -32% / +1343% / 0% | -30% / +127% / -4% | 0 / 0 / 0 | -33% / +7484% / -8% |
| tp_collective = all_reduce | -1% / +34% / 0% | 0% / +15% / +3% | -1% / +25% / +5% | -1% / +31% / +5% |
| max_batch_tokens = 512 / 1024 | -5% / +182% / +1% ; -2% / +42% / +1% | -1% / +45% ; 0 / +18% | 0 / +90% ; 0 / +27% | -1% / +275% ; 0 / +19% |
| max_batch_tokens = 4096 / 8192 | 0 / -7% ; +1% / +11% | 0 / +15% / +3% ; -1% / +27% / +8% | 0 | 0 |
| context_buckets = [32768, 131072] (coarse) | -5% / +239% / +2% | -1% / +31% / +4% | -2% / +196% / +29% | -6% / +364% / +15% |
| context_buckets = fine (10 sizes) | 0 | 0 / -8% / -2% | 0 | 0 |
| hc_mix_impls = [torch_cat] | -6% / +3% / +7% | -1% / +3% / +11% | -2% / +6% / +8% | -4% / +3% / +5% |
| gdn_decode_impls = [nki] | -1% / 0 / +1% | 0 | 0 | -25% / +15% / +51% |
| gdn_prefill_impls = [torch] | -3% / +11% / +2% | 0 / +5% / +2% | 0 / +11% / +2% | 0 / +10% / +1% |
| qsa_prefill_impls = [nki] | 0 | 0 / +14% / +4% | 0 | 0 / +2% / 0 |
| decode_buckets = [8, 32, 128, 256] | -3% / 0 / +4% | 0 / 0 / +1% | -1% / 0 / +4% | -1% / +1% / +1% |
| decode_buckets fine, prefill_buckets coarse or fine, router / shared / QSA-decode impls, best-of-N impl lists | within 1% | within 1% | within 1% | within 1% |

**Layout** matters much less once the switches above are set. With layer regions,
RS/AG and a2a fixed, every buildable EP32 layout is within 1% to 4% on tok/s:

| Workload | tok/s across layouts | Best layout | Worst layout |
|---|---|---|---|
| fixed_2k_512 | 2945 to 3326 | dp4-tp8-g4 | dp2-tp8-g16 (-11%) |
| agentic_hi | 2953 to 2981 | | |
| chat_r8 | 6259 to 6354 | dp2-tp8-g16 | |

The layout mostly moves TTFT. GDN TP 1 is the worst, and TTFT p90 is up to 2x worse
with gdn_tp 1 than with gdn_tp 4 to 8. EP16 x TP2 is 5% to 15% slower than EP32 in every
case. The layout knob ranking is:

1. region granularity;
2. MoE dispatch;
3. MoE routed impl (`auto`);
4. PLE residency (keep `replica`);
5. context-bucket coverage and chunk size (TTFT only);
6. TP collective (TTFT);
7. GDN TP >= 4;
8. everything else (within a few percent).

### Headroom ladder

These are the analyzer's cluster buckets for the R0 GPU-time of the best configs, as
fractions of R0.

| Workload | Config | R0 GPU-s | idle + imbalance | launch + batching (R2-R3) | comm | kernel gap | hardware gap | excess over necessary |
|---|---|---|---|---|---|---|---|---|
| fixed_2k_512 | step-4 layout | 6483 | 0% | 69% | 0% | 16% | 8% | 7% |
| fixed_2k_512 | best unified | 3147 | 1% | 65% | 1% | 2% | 17% | 14% |
| fixed_2k_512 | best MTP | 1845 | 2% | 60% | 3% | 3% | 22% | 8% |
| agentic_hi | step-4 layout | 29326 | 11% | 49% | 1% | 27% | 5% | 7% |
| agentic_hi | best unified | 24277 | 46% | 39% | 2% | 2% | 4% | 5% |
| chat_r8 | step-4 layout | 8263 | 1% | 35% | 1% | 43% | 9% | 10% |
| chat_r8 | best unified | 4527 | 6% | 60% | 5% | 5% | 14% | 10% |
| chat_r8 | best MTP | 3464 | 23% | 27% | 17% | 6% | 18% | 8% |

What this says:

- **The kernel gap mostly closes by choosing the right path.** In the step-4 layout,
  16% to 43% of time is kernel gap: the all-gather MoE block runs `moe_cte` at 256 to
  8192 tokens, well below its bandwidth bound. All-to-all dispatch sends each expert
  rank about T_all/ep rows, so the routed experts stay at 256 rows or fewer, and the
  kernel gap drops to 2% to 6%.
- **What remains is mostly the launch + batching rung (R2 to R3), 40% to 65%.** Under
  the calibrated launch model, each region's launch floor is part of its leaf, and R3
  removes it. So this rung is chiefly the 1.25 ms floor of collective-bearing regions
  at decode. It is the biggest lever, and it rests on the least certain constant.
- **The hardware gap (best measured kernel against spec) is 14% to 22%**, and excess over
  necessary work is 8% to 14%. Necessary work itself (R6) is under 1% of R0.
- **For agentic, idle is 46%.** Bursts arrive every ~30 s and the host drains each burst
  well before the next one. More hardware would not help this workload; lower TTFT
  would.

## Sensitivity to the uncertain constants

Stage 4 re-runs the 114 stage-1 configs (32-core layouts), the MTP configs and the P/D
grid under four scenarios:

- `base`;
- `launch_half`: `launch_coll_ms` 0.625, `launch_plain_ms` 0.0375, `step_overhead_ms` 2.0;
- `moe_prefill_3x`: `kind_time_scale: [moe_block@256=0.3333, moe_routed_experts@256=0.3333]`.
  MoE calls of 256 rows or more, the cte path, run 3x faster;
- `both`.

`kind_time_scale` is a new what-if arch knob (`qwen4_exp_neuron_bf16.rs`). It never
changes the measured rows.

The table gives the best tok/s per family and scenario, written as tok/s / TPOT p50 (ms)
/ TTFT p50 (s).

| Workload | Family | base | launch_half | moe_prefill_3x | both |
|---|---|---|---|---|---|
| fixed_2k_512 | unified | 3326 / 91 / 0.8 | 5107 / 58 / 0.7 | 3338 / 90 / 1.3 | 5107 / 58 / 0.7 |
| fixed_2k_512 | MTP | 5660 / 48 / 1.0 | 7775 / 34 / 0.9 | 5666 / 48 / 0.9 | 7827 / 34 / 0.8 |
| fixed_2k_512 | P/D | 2647 / 97 / 4.7 | 3804 / 64 / 3.8 | 2647 / 97 / 4.7 | 3804 / 64 / 3.8 |
| agentic_hi | unified | 2981 / 102 / 6.9 | 3071 / 61 / 6.9 | 2981 / 102 / 6.7 | 3075 / 64 / 6.2 |
| agentic_hi | MTP | 3118 / 50 / 8.0 | 3163 / 35 / 7.4 | 3118 / 50 / 8.0 | 3163 / 35 / 7.4 |
| chat_r1 | unified | 1631 / 99 / 0.3 | 1793 / 61 / 0.2 | 1631 / 99 / 0.3 | 1793 / 61 / 0.2 |
| chat_r8 | unified | 6354 / 131 / 0.4 | 7763 / 99 / 0.3 | 6354 / 131 / 0.4 | 7763 / 99 / 0.3 |
| chat_r8 | MTP | 8394 / 98 / 0.6 | 8989 / 93 / 0.6 | 8394 / 98 / 0.6 | 8989 / 93 / 0.6 |
| chat_r8 | P/D | 5288 / 108 / 9.6 | 6722 / 78 / 6.3 | 5288 / 108 / 9.6 | 6722 / 78 / 6.3 |

The next table gives the best tok/s per switch combination (granularity, dispatch,
collective), across the four scenarios (base / launch_half / moe_prefill_3x / both).

| Workload | layer + a2a + RS/AG | layer + all_gather + RS/AG | mixer + all_gather + RS/AG | mixer + a2a + RS/AG |
|---|---|---|---|---|
| fixed_2k_512 | 3326 / 5107 / 3338 / 5107 | 2332 / 3087 / 2783 / 3942 | 1649 / 2646 / 1874 / 3283 | 1424 / 2640 / 1427 / 2640 |
| agentic_hi | 2981 / 3071 / 2981 / 3075 | 2724 / 2837 / 2950 / 3051 | 2522 / 2708 / 2784 / 2969 | 2624 / 2919 / 2626 / 2919 |
| chat_r8 | 6354 / 7763 / 6354 / 7763 | 4439 / 5076 / 5909 / 7066 | 3686 / 4662 / 4811 / 6609 | 3849 / 5839 / 3849 / 5839 |

Kendall tau of the 114-config ranking against `base`:

| Scenario | tau |
|---|---|
| launch_half | 0.68-0.90 |
| moe_prefill_3x | 0.53-0.94 |
| both | 0.51-0.93 |

**Robust conclusions.** These hold under every scenario:

- Layer granularity beats mixer and per_op.
- MTP beats non-MTP on throughput and TPOT.
- EP32 beats EP16 x TP2.
- PLE `replica` beats `dp_group`.
- P/D trails unified on throughput, but wins TTFT at light load.
- The best layout changes only within a ~1% band. For example, dp4-tp8-g8 gives way to
  dp4-tp8-g4 or dp2-tp4-g16. The base winner stays in the top 3 to 16 of 114.

**Conclusions that hinge on the constants.**

1. **Absolute throughput and TPOT hinge on the launch constants.** Halving them gives
   +54% on fixed_2k_512 (3326 to 5107), +22% at chat_r8 and +10% at chat_r1, and cuts
   TPOT from 91 ms to 58 ms. The non-MTP strict-TPOT SLO (60 ms) becomes nearly
   reachable. The MTP gain shrinks from +70% to +52% on fixed_2k_512, and from +32% to
   +16% at chat_r8. MTP's advantage is partly launch amortization. Every TPOT number in
   this file inherits the loaded-host fit; re-fit `launch_coll_ms` and `step_overhead_ms`
   on a quiet host before quoting absolute numbers.
2. **The all-to-all advantage on prefill-heavy work hinges on the MoE prefill kernel.**
   With a 3x faster `moe_cte`:
   - all_gather at layer granularity ties with a2a on agentic_hi (3051 against 3075
     under `both`);
   - it narrows the gap to 7% at chat_r8 and 16% on fixed_2k_512.

   Mixer + all_gather gains 10% to 30%. If the engine ships only all-gather dispatch,
   fixing `moe_cte`'s static block cost is the second-largest lever after launch.
3. **The a2a result also depends on how the arch models a2a.** The model assumes routed
   experts at T_all/ep equivalent tokens on the critical-path rank, with real layer-0
   routing, and it uses the c4 all_to_all rows. The engine has no all-to-all dispatch
   yet (`flashnext/neuron/engine/program.py` traces all_reduce, all_gather and
   reduce_scatter only), and RS/AG sequence parallelism is not in the engine either.
   Treat both as design targets that need an engine measurement.

## Configs not evaluated (need profiling)

`probe` lists these, with the missing kernels, in each stage's `probe.jsonl`. They were
not profiled.

| Configs | Missing rows |
|---|---|
| 64-core replicas (1 replica over 16 chips), every layout | world-64 `all_gather` / `reduce_scatter`, `moe_block` at 8 local experts, PLE gather at 64 shards |
| `attn_tp` 1 | `qsa_mixer2` at 24 heads, `lm_head_argmax` vocab rows |
| `attn_tp` 2 | `qsa_mixer2` torch_sparse prefill at (2048, 34816) / (8192, 40960); some also `all_gather` c1 |
| MoE EP <= 8 (moe_tp >= 4) | `moe_block` shapes |
| per_op granularity at tp 2/4 | GDN / QSA per-op rows |
| MTP k=4 | `gdn_decode_mixer` at query_len 5 |
| `head_impls` with nki | `lm_head_argmax` nki rows (5 of 14) |
| `moe_routed_impls` best-of-N lists | `moe_routed_experts` rows for nki_tkg / cte variants at the a2a shapes |
| `ple_table_shards: dp_group` at G <= 8 | invalid: the per-rank table does not fit HBM. It runs only at G = 16 (dp2) and costs ~30% |

## How to rerun

```bash
cd wt-trn2-flashnext/ServingStudioSim
source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH SERVINGSTUDIO_NO_GPU=1
S="uv run --no-sync python tools/trn2-config-search/search.py"
for s in s1_layout s2_knobs s3_unified s3_mtp s3_pd s4_whatif_unified s4_whatif_mtp s4_whatif_pd; do
  $S plan tools/trn2-config-search/stages/$s.yaml logs/trn2/search/$s && $S probe logs/trn2/search/$s
done
tmux new -s search-all "bash logs/trn2/search/run_all.sh s1_layout s2_knobs s3_unified s3_mtp s3_pd s4_whatif_unified s4_whatif_mtp s4_whatif_pd"
$S report logs/trn2/search/s1_layout logs/trn2/search/s2_knobs logs/trn2/search/s3_unified \
  logs/trn2/search/s3_mtp logs/trn2/search/s3_pd --out logs/trn2/search/report_main.md
$S report logs/trn2/search/s4_whatif_unified --out logs/trn2/search/report_whatif_unified.md
```

Notes on the run:

- The loop runs `run` then `collect` per stage, at `--parallelism 48`.
- `s2_knobs` seeds from `stages/s1_top.yaml`, written by
  `$S top logs/trn2/search/s1_layout tools/trn2-config-search/stages/s1_top.yaml --n 2 --metric total_tok_s --metric goodput_req_s@chat`.
- The whole search took about 40 minutes of wall time on this host (simulations are
  ~0.03 to 1 s each; analysis dominates).
- Generated reports are in `logs/trn2/search/report_*.md`. Those files are ignored by
  git, like all of `logs/`.

## Open issues

- **Launch constants.** They are fitted on a loaded host. The ranking is robust to them,
  but absolute TPOT and throughput are not. Re-fit them on a quiet host. That is
  engine-side work and needs devices.
- **MTP k=3 at high load.** It degrades sharply at chat_r8 (TPOT 186 ms against 96 ms for
  k=2), with no retractions. A suspect is verify rows (B x 4) overflowing the 256-row
  decode bucket or the token buckets. Not root-caused.
- **a2a and RS/AG are not in the engine.** The two biggest knob wins after layer
  granularity are modeled but not implemented in flashnext-neuron.
- **No Analyzer MCP.** The numbers come straight from the run directories
  (`summary.json`, `raw/request_slo.parquet`, `reports/optimality_report.json`), not
  through the Analyzer service the `operate-use-analyzer` skill prefers. That service is
  not available in this session.
- **Agentic workload.** It has no prefix reuse, because the engine has no prefix cache.
  If one lands, add `prefix_len` to the agentic traces (`trace/README.md`).
