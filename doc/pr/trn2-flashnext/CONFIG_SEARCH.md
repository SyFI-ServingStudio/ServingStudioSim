# Config search: Qwen3.8-Flash-Next BF16 on one trn2.48xlarge

Simulator-only results (2026-10-25) on the arch **after its alignment with the
flashnext-neuron engine** (`ALIGNMENT.md`: end-to-end median error 6.0% over 16 serving
runs, the measured W64 ranking reproduced). The best serving configurations for three
workload families on 16 Trainium2 chips (64 LNC2 cores), which knobs matter, what
the engine would gain from changes it does not have yet, and how this differs from the
first search (before the alignment). Tool: `tools/trn2-config-search/`; skill:
`operate-search-trn2-configs`. Runs: `logs/trn2/search_v2/` (reports `report_*.md`).
The only rows profiled for this work are the `qsa_mixer2` grid rows of the alignment.

## Method

| Stage | Configs run | Varies | Runs |
|---|---|---|---|
| `s1_layout` | 240 of 2400 (the rest are 64-core layouts without world-64 rows, or refused) | every 32- and 64-core layout x {all_reduce, RS/AG} x {all_gather, all_to_all} x {layer, block} x chunk {512, 2048} | 2160 |
| `s2_knobs` | 482 | one knob at a time around the 14 stage-1 winners: chunk 256..8192, granularity incl. mixer, collectives, dispatch, a2a capacity factor, PLE residency, bucket inventories (incl. the engine's), every per-op impl list | 4338 |
| `s3_unified` | 60 | 5 best layouts x {layer, block} x chunk {512, 1024, 2048} x context buckets {default, fine} | 540 |
| `s3_mtp` | 18 | k = 1..3 x 3 layouts x chunk {512, 2048}, acceptance 0.67 (the engine's on random prompts) and 0.87 (its chat prompts) | 324 |
| `s3_pd` | 9 | P/D 8 + 8 chips, prefill x decode layout | 81 |
| `s4_whatif_*` | 840 + 108 + 36 | stage-1 grid (32-core, chunk 512), MTP and P/D configs under 7 scenarios (below) | 8856 |

Every config uses all 16 chips (two 32-core replicas, or 8 + 8 for P/D), with the
engine's scheduler: one prefill chunk per DP group per step, separate decode and
prefill forwards, SPMD padding, new requests to the least-loaded DP group. Workloads
(`stages/workloads.yaml`) and SLOs are unchanged from the first search: `fixed_2k_512`
(the W64 trace), `agentic_lo/hi` (16-32k prompts in bursts), `chat_r{0.25..8}`
(1-2k / 200-400, Poisson); SLOs `chat` 2 s / 100 ms, `chat_strict` 1 s / 60 ms,
`agentic` 30 s / 150 ms, `agentic_relaxed` 60 s / 250 ms. `tok/s` is prefill +
decode tokens over the span on 16 chips.

## Results

### Best configuration per workload

Every winner shares the switches **block regions, RS/AG (sequence-parallel HC),
all-gather MoE, EP32**. "ctx fine" is context buckets
`[2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072]`.

| Workload | Objective | Config | tok/s | TTFT p50 / p90 (s) | TPOT p50 / p90 (ms) | Goodput (req/s) |
|---|---|---|---|---|---|---|
| fixed_2k_512 | max tok/s, no MTP | dp2-tp8-g16, chunk 4096 | 2668 | 11.0 / 18.2 | 94 / 110 | |
| fixed_2k_512 | no MTP, chunk <= 2048 | dp8-tp4-g4, chunk 512 | 2555 | 15.1 / 23.8 | 92 / 110 | |
| fixed_2k_512 | MTP @0.67 | k=1, dp8-tp4-g4, chunk 512 | 3082 | 15.5 / 24.6 | 68 / 88 | |
| fixed_2k_512 | MTP @0.87 | k=3, dp8-tp4-g4, chunk 512 | 3409 | 15.9 / 25.4 | 57 / 76 | |
| agentic_hi | max tok/s | MTP k=3 @0.87, dp2-tp4-g16, chunk 2048 | 2558 | 43.9 / 87.5 | 149 / 364 | 0.016 @agentic |
| agentic_hi | no MTP | dp4-tp8-g8, chunk 1024, ctx fine | 2459 | 44.0 / 78.9 | 196 / 353 | 0.002 |
| agentic_lo | max tok/s | MTP k=1 @0.87, dp4-tp8-g8, chunk 2048 | 2328 | 51.6 / 97.8 | 199 / 522 | 0.002 |
| chat_r0.25 | chat goodput | MTP k=1 @0.87, dp2-tp4-g16, chunk 512 | 489 | 1.44 / 1.90 | 51 / 66 | 0.26 @chat (all) |
| chat_r0.25 | no MTP | dp2-tp8-g16, chunk 512 | 477 | 1.38 / 1.81 | 84 / 94 | 0.25 @chat |
| chat_r1 | chat goodput | MTP k=3 @0.87, dp2-tp4-g16, chunk 512 | 1847 | 1.95 / 2.83 | 68 / 88 | 0.55 @chat |
| chat_r1 | MTP @0.67 | k=1, dp2-tp4-g16, chunk 512 | 1767 | 1.8 / 2.5 | 97 / 122 | 0.36 @chat |
| chat_r1 | no MTP | dp2-tp8-g16, chunk 512 | 1647 | 1.75 / 2.50 | 137 / 175 | 0.07 @chat |
| chat_r8 (saturated) | max tok/s | dp8-tp2-g4, chunk 512 | 4019 | 27.1 / 43.0 | 206 / 322 | 0 |

No configuration meets `chat_strict` (1 s TTFT): a 1-2k prompt costs at least one T =
2048 prefill forward (~3 s) or four T = 512 forwards (~0.8 s each) on a SPMD engine
whose MoE runs on every group's padded rows.

**Against the engine's measured W64 baseline layout** (dp4-tp8-g8, layer, all_reduce,
all_gather, chunk 2048, two replicas, k = 0):

| Workload | Baseline layout (Sim) | Best, today's engine knobs | Best MTP @0.67 |
|---|---|---|---|
| fixed_2k_512 | 1542 tok/s, TTFT 28.2 s, TPOT 148 ms | 2553-2668 (1.7x), 11-15 s, 92-94 ms | 3082 (2.0x), 68 ms |
| agentic_hi | 2276, TTFT p90 108 s, TPOT 353 ms | 2459 (1.08x), 79 s, 196 ms | 2520, 158 ms |
| chat_r1 | 1450, 5.0 s, 264 ms | 1647 (1.14x), 1.75 s, 137 ms | 1767, 97 ms |
| chat_r8 | 3099, 39 s, 270 ms | 4019 (1.30x), 27 s, 206 ms | 3563 |

The engine's best measured W64 run (two replicas, layer, RS/AG, k = 2, chunk 512) is
1910 tok/s in + out (86 s); the Sim's best configuration with knobs the engine has is
3082 tok/s (block regions instead of layer, dp8-tp4-g4 instead of dp4-tp8-g8, k = 1).

### Which knobs matter

Stage 1, every pair of configs that differ in one switch only (median over layouts),
as tok/s / TTFT p90 / TPOT p50 change:

| Switch (all else equal) | fixed_2k_512 | agentic_hi | chat_r1 | chat_r8 |
|---|---|---|---|---|
| region_granularity layer -> block | +19% / -2% / -22% | +3% / -2% / -15% | +7% / -1% / -9% | +5% / -2% / -5% |
| tp_collective all_reduce -> RS/AG | +6% / -11% / -5% | +4% / -10% / -19% | +1% / -10% / -3% | +10% / -14% / -9% |
| moe_dispatch all_gather -> all_to_all (exact) | -11% / +25% / +8% | -8% / +28% / +37% | -2% / +25% / +5% | -17% / +36% / +20% |
| max_batch_tokens 2048 -> 512 | +21% / -35% / -11% | -1% / +3% / -3% | +2% / -32% / -9% | +2% / -6% / -3% |
| moe_ep 32 -> 16 (moe_tp 2) | -8% | -7% | -1% | -14% |

Stage 2 (one knob around the winners):

- **Chunk size.** fixed_2k_512: 256 / 512 / 1024 / 2048 / 4096 / 8192 = 2519 / 2555 /
  2385 / 1953 / 2668 / 2668 tok/s. A 2048 budget is the worst: the decode rows take
  16-32 tokens of it, so each prompt runs a 2016-token chunk and then a 32-token tail
  that SPMD pads to a full T = 2048 forward at context 4096 (the engine's W64 runs show
  exactly this: 29 of 31 prefill forwards at C = 4096). A budget above the largest
  bucket plus the decode rows (4096) runs each prompt in one chunk. chat: 512 best
  (TTFT p90 2.2 s vs 2.5 at 2048); agentic: flat in tok/s, 512 has the best TTFT p90
  on the best layout (33.7 s).
- **Granularity:** block > layer (-7..-16%) > mixer (-10..-30%).
- **Layouts** at the winning switches spread 5-25% (they spread 1-4% before):
  fixed_2k_512 and chat_r4/r8 want dp8 (tp4/tp2, g4): 2553 vs 2400 for dp4-tp8-g8 and
  2150 for dp2; light chat wants dp2-tp4-g16 (fewer groups pay the padded shape: TTFT
  p90 2.4 vs 6.5 s at dp8); agentic is flat (2359-2410). gdn_tp 1 costs 2-10%.
- **moe_routed_impls [nki]** -26..-72% (as before, keep `auto`); `ple_table_shards
  dp_group` -8..-30%.
- Within ±5%: context-bucket inventories (keep a 2048 bucket: without it every short
  chunk runs QSA's sparse path), decode/prefill bucket inventories, every other impl.

**Knob ranking:** block regions; chunk budget (512, or above the largest bucket); layout
(DP width); RS/AG; all-gather MoE (not all-to-all); EP32; `auto` MoE; everything else
within a few percent.

### MTP

At the engine's acceptance on random prompts (0.67) **k = 1 is best** (fixed_2k_512:
k = 1 / 2 / 3 = 3082 / 3043 / 2939 tok/s, TPOT 68 / 68 / 71 ms): each extra draft adds
an MTP forward of ~7 ms host + device, which 0.67 per draft does not repay. At 0.87
(the chat prompts) k = 3 is best (3409, TPOT 57 ms). MTP adds 15-21% on fixed_2k_512,
4-12% on chat and agentic, and loses at chat_r8 (3563 vs 4019: verify rows widen
every decode forward at saturation).

### P/D

8 + 8 chips lose everywhere on throughput (1206-1437 tok/s on the long-prompt
workloads) and on TTFT except at the lightest load (chat_r0.25: TTFT p50 1.19 s vs
1.38 unified). The prefill instance is prefill-bound on 8 chips, and the Sim's P/D
prefill worker runs one prompt per iteration (its DP groups pay their own shapes, not
padded ones; the engine has no P/D).

### What the engine would gain (stage 4)

Each cell: best tok/s under the scenario (median change over the configs in tok/s /
TTFT p90 / TPOT p50).

| Scenario (engine change) | fixed_2k_512 | agentic_hi | chat_r1 | chat_r8 |
|---|---|---|---|---|
| base | 2553 | 2410 | 1638 | 3981 |
| `moe_prefill_3x`: MoE kernel 3x faster on >= 256 rows (moe_cte static per-block cost) | 3284 (+26% / -48% / -12%) | 2870 (+18% / -51% / -49%) | 1682 (+3% / -51% / -19%) | 6608 (+51% / -65% / -31%) |
| `a2a_cf125`: all-to-all at capacity factor 1.25 (all-to-all configs) | 3020 (+25% / -52% / -11%) | 2706 (+20% / -53% / -55%) | 1659 (+2% / -51% / -15%) | 5419 (+57% / -68% / -35%) |
| `no_pad`: idle DP groups skip the padded shape | 2675 (+5% / -13% / -2%) | 2604 (+4% / -7% / -21%) | 1661 (+2% / -26% / -22%) | 4366 (+6% / -9% / -5%) |
| `host_half`: host 2x faster (dispatch + per-forward work) | 2948 (+13% / -1% / -15%) | 2439 (+2% / -2% / -9%) | 1701 (+3% / -1% / -5%) | 4028 (+2% / -1% / -1%) |
| `fused`: one decode + prefill forward | 2551 (+0% / -0% / -1%) | 2385 (-1% / +3% / +5%) | 1661 (+0% / -1% / -2%) | 4067 (+1% / -3% / -1%) |
| `multi_chunk`: up to 16 chunks per DP group per step | 2621 (+5% / -11% / -3%) | 2405 (-0% / +1% / -0%) | 1633 (+0% / +1% / +0%) | 3343 (-10% / +15% / +9%) |

With MTP (@0.67) the same picture: MoE prefill 3x +6..+67%, no_pad +4..+22% (fixed
+22%), fused +0..+6%, host_half +1..+5%, multi_chunk -16..+14%.

The headroom ladder of the best configs agrees: the **kernel gap is now the largest
bucket** (29% of R0 GPU-time on fixed_2k_512, 43-60% on chat and agentic: `moe_cte` at
2048-8192 gathered rows), then launch + batching (19-38%: region floors and padded
idle groups), excess over necessary work 13-24%, communication 1%.

## Old vs new conclusions

The first search (`git show 811d2d48:doc/pr/trn2-flashnext/CONFIG_SEARCH.md`) used the
barrier-on, loaded-host launch constants (1.25 ms per collective region), one fused
forward per step without SPMD padding, the prior-axis QSA grid, the fused HC row, and
an all-to-all priced at T_all/ep rows.

| Question | First search | Now |
|---|---|---|
| Granularity | layer (mixer -57%) | **block** (+3..+19% over layer; the engine measured +13% on W64) |
| MoE dispatch | **all-to-all** (+30% fixed, closes the kernel gap) | **all-gather**: exact-capacity all-to-all computes the same rows and adds packing (-2..-17%, TTFT +25..36%; engine: W64 duration +8%). A capacity factor would help (+20..57%) only if the expert kernel scales with received rows and the dropped tokens are acceptable |
| TP collective | RS/AG, TTFT only (-1% tok/s) | RS/AG, +1..+10% tok/s and -10..-14% TTFT p90 |
| Chunk size | 2048; 512 costs TTFT (+182%) | **512** (+21% tok/s, -35% TTFT p90 on fixed) or a budget above the largest bucket (4096: +4% more); 2048 is the worst |
| What bounds decode | launch: 48 collective regions x 1.25 ms, TPOT floor ~90 ms | layer decode is host-bound at ~101 ms (63 regions x 1.3 ms + 19 ms per forward); block is device-bound (70-110 ms); best unified TPOT p50 84-94 ms |
| Biggest lever | launch constants (halving: +54%) | the MoE prefill kernel (3x: +3..+67% tok/s, TTFT -48..-80%); a 2x faster host gives +1..+13% |
| MTP | k = 3 everywhere below saturation (+70% on fixed @0.71) | k = 1 at the engine's 0.67 (+15..21% on fixed); k = 3 only at 0.87; loses at saturation |
| Layout | within 1-4%; dp4-tp8-g8 | 5-25%: dp8-tp4-g4 for prompt-heavy and saturated loads, dp2-tp4-g16 for light chat |
| P/D | wins TTFT at light load | loses throughput everywhere, wins TTFT only at 0.25 req/s |
| fixed_2k_512 best | 3326 unified, 5660 MTP (k=3 @0.71) | 2553-2668 unified, 3082 MTP (k=1 @0.67) |
| chat_r8 best | 6377 unified, 8394 MTP | 4019 unified, 3726 MTP |
| Robust either way | EP32 > EP16, `auto` MoE, PLE `replica`, MTP lowers TPOT | same |

## Predicted wins the engine still needs to implement

Ordered by predicted gain on these workloads; each is a Sim prediction the engine has
not measured.

1. **A faster MoE prefill kernel** (`moe_cte`'s static per-block cost at 2048-8192
   gathered rows): +3..+67% tok/s (most on prompt-heavy and saturated loads), TTFT
   -48..-80% at 3x. The largest remaining gap.
2. **Block regions with RS/AG and dp8-tp4-g4** (each knob exists; the combination and
   the dp8 x tp4 layout have not been run): 2553 tok/s on fixed_2k_512 vs the Sim's 1542
   for the engine's baseline layout.
3. **A chunk budget above the largest prefill bucket** (`max_batch_tokens` 4096 with
   buckets up to 2048), or 512: avoids the padded tail chunk at context 4096 (+21..+37%
   over 2048 on fixed_2k_512). A configuration change, no code.
4. **MTP k = 1 by default**; k >= 2 only for workloads with acceptance >= ~0.85.
5. **Skip the padded shape on idle DP groups** (a group without a chunk or decode rows
   does not run the forward's work): +2..+22% tok/s, TTFT -9..-36%.
6. **All-to-all with a capacity factor and drop-safe routing** (or variable-size
   all-to-all that sends only routed rows): +20..+57% in the model, which assumes the
   expert kernel's time follows the received rows; at exact capacity keep all-gather.
7. **A faster host path** (input assembly, H2D, id readback): +2..+13%.
8. Not worth it here: a fused decode + prefill forward (0..+6%) and several chunks per
   DP group per step (-16..+14%).

## How to rerun

```bash
cd wt-trn2-flashnext/ServingStudioSim
source ../neuron_env.sh; export PATH=$HOME/.cargo/bin:$PATH SERVINGSTUDIO_NO_GPU=1
S="uv run --no-sync python tools/trn2-config-search/search.py"
bash logs/trn2/search_v2/run_all.sh s1_layout                     # tmux: align-search
$S top logs/trn2/search_v2/s1_layout tools/trn2-config-search/stages/s1_top.yaml \
  --n 2 --metric total_tok_s --metric goodput_req_s@agentic
bash logs/trn2/search_v2/run_all.sh s2_knobs s3_unified s3_mtp s3_pd s4_whatif_unified s4_whatif_mtp s4_whatif_pd
R=logs/trn2/search_v2
$S report $R/s1_layout $R/s2_knobs $R/s3_unified $R/s3_mtp $R/s3_pd --out $R/report_main.md
$S report $R/s4_whatif_unified --out $R/report_s4_whatif_unified.md
```

`run_all.sh` plans, probes, runs (`--parallelism 48`) and collects each stage; the
whole search is ~16.3k runs and took about 50 minutes here. Generated reports are
under `logs/` (not tracked).

## Open issues

- **Engine-unmeasured combinations.** Block + RS/AG, dp8-tp4-g4 and dp2-tp4-g16
  layouts, chunk budgets above the largest bucket: predicted, not run on the engine.
- **All-to-all with a capacity factor** is a model (row-driven expert cost, packing
  fitted on one run); dropped tokens are not accounted for.
- **Workload acceptance.** MTP conclusions hinge on acceptance: 0.65-0.69 measured on
  random-token prompts, 0.87 on chat prompts; real traffic is in between.
- **64-core replicas, attn_tp 1/2, EP <= 8, MTP k = 4** still need rows (`probe.jsonl`).
- **P/D** runs one prompt per prefill iteration; it is not an engine path.
- **No Analyzer MCP**: numbers come from the run directories.
