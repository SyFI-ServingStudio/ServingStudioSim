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
