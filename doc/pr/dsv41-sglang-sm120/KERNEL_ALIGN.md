# Kernel alignment, first pass (2026-11-06, no new GPU time)

**Inputs:**
- the B2 nsys captures in `logs/b2_reextract/{07_balanced_anchor_c16, 01_decode_graph_sizes_c16}` (git-ignored);
- the pack rules in `presets/alignment/dsv41_flash_rtxpro6000_sglang_tp4_ep4/label_rules/` (`gen_rules.py` generates
  `rules.json`).

The simulator was not changed. Commands, from the worktree root with `PATH=$HOME/.cargo/bin:$PATH`:

```bash
P=presets/alignment/dsv41_flash_rtxpro6000_sglang_tp4_ep4
uv run python $P/label_rules/gen_rules.py
uv run python -m launcher alignment-campaign label --pack $P --out-root logs/b2_reextract --case 07_balanced_anchor_c16 --refresh
uv run python -m launcher alignment-campaign run --pack $P --out-root logs/b2_reextract --phase analysis_kernel --case 07_balanced_anchor_c16
uv run python -m alignment label coverage logs/b2_reextract/07_balanced_anchor_c16/analysis_kernel/reports/alignment_iteration_report.json
```

## Coverage

The rule set has 93 rules for 44 operations, with no disagreements or check findings.

| case | iterations | measured coverage (duration) | critical-path coverage | simulated coverage |
|---|---:|---:|---:|---:|
| 07_balanced_anchor_c16 | 94 | 97.10% | 97.97% | 99.80% |
| 01_decode_graph_sizes_c16 | 115 | 97.44% | 98.09% | 99.69% |

## Errors (barrier critical path; the analyzer's measured_ms vs timing-predict)

| case / stage | n | measured ms/iter | simulated ms/iter | signed | abs |
|---|---:|---:|---:|---:|---:|
| 07 decode (14-16 req x 7 verify rows) | 67 | 55.8 | 56.4 | +1.2% | 1.6% |
| 07 prefill 2048-token chunks (iterations 1-2, warm-up and JIT, excluded) | 25 | 205.1 | 343.7 | **+67.5%** | 67.5% |
| 01 decode (13-16 req) | 101 | 57.4 | 58.5 | +1.9% | 2.1% |
| 01 prefill chunks (iterations 1-2 excluded) | 12 | 212.9 | 327.1 | **+53.7%** | 53.7% |

`recommended_gpu_time_multiplier` is 1.256 (07) and 1.250 (01). The pooled number is inflated:
- Iteration 2 is a 2.9 s warm-up step and passes the outlier screen.
- Per stage, the multiplier is 1.04 for decode and 1.35 for prefill (1.35 includes iteration 2).
- Do not adopt it before prefill is fixed.

## The prefill overestimate (the ~2x cache-miss TTFT): decoder SWA bounded replay

The cause is not a cost row. SGLang's DSV4 model runs **layers 20-39 over only the last `SWA_WINDOW` (128) extend
tokens of each request** (`late_layer_start` = the last kv_source layer). The code is
`enable_decoder_swa_bounded_replay`, in `models/deepseek_v4.py:3867` and `layers/attention/deepseek_v4_backend.py`
(`LateLayerTail`, `late_layer_num_tokens`). DSpark only reads tail rows, so the whole extend is not needed.

The capture shows it directly. In every 2048-token chunk:
- **Layers 0-19 run at full M:**
  - all-reduce 1.7-1.9 ms;
  - wq_b 0.44 ms;
  - grouped GEMM 0.83 ms.
- **From layer 20 on, the same kernels run at tail size:**
  - all-reduce 0.13-0.27 ms;
  - wq_b 0.03 ms;
  - grouped GEMM 0.29 ms.

The arch charges all 40 layers at 2048 tokens.

Per 2048-token chunk (07, ms per iteration), measured / simulated:

| operation | measured | simulated | sim / meas |
|---|---:|---:|---:|
| tp_all_reduce | 74.6 | 136.6 | 1.83 |
| ffn.routed_moe | 40.7 | 57.6 | 1.41 |
| indexer.logits_prefill | 0.5 (+ ~2 ms of unmapped torch elementwise/reduce) | 11.5 | ~5 |
| attn.sparse_mla | 8.9 | 17.3 | 1.95 |
| mhc_mix_stats | 8.6 | 16.0 | 1.87 |
| attn.wq_b / wo_b / fused_wqa_wkv | 9.2 / 9.2 / 9.7 | 16.3 / 16.1 / 15.7 | 1.6-1.8 |
| ffn.shared_gate_up / shared_down | 7.0 / 2.5 | 10.8 / 4.3 | 1.5-1.7 |
| mhc_post / mhc_combine_norm | 2.8 / 1.0 | 6.5 / 4.6 | 2.4 / 4.8 |
| engram.wkv (layers 1, 14: before the tail) | 12.3 | 11.9 | 0.97 |

Per launch, the rows are right where the token count matches:
- the 20 MB all-reduce row is 1.74 ms, against 1.7-1.8 ms measured in layers 0-19;
- wq_b is 0.416 ms simulated against 0.438 ms measured;
- engram.wkv, which runs only before the tail, is within 3%.

The ~1.5-1.9x per-operation ratios are the early half at the right cost plus the late half charged at 2048 rows.
Modeling the tail (late layers at `min(extend_len, 128)` rows per request) is the fix. It is an arch change, so it is
not made here.

## Decode: right in aggregate, with compensating per-operation errors (07, ms per iteration)

| operation | measured | simulated | note |
|---|---:|---:|---|
| ffn.routed_moe + ffn.shared_gate_up | 16.7 + 10.6 | 26.0 + 3.1 | The shared expert runs on a side stream under the grouped GEMM and is charged its contended time. Combined: 27.3 vs 29.1 (1.07). |
| tp_all_reduce | 9.2 | 10.1 | 1.10 |
| dspark.markov.vocab_all_gather | 2.0 | 0.11 | 18x under. The arch gathers 448 B per round; SGLang gathers vocab-sharded rows (~340 us per round, x6). |
| head.logits_all_gather | 1.18 | 2.11 | 1.79. It is modeled with the all_reduce kind at 25 MB; an all-gather moves about half. |
| attn.fused_wqa_wkv / wq_b / wo_b | 3.6 / 1.5 / 1.4 | 3.1 / 1.3 / 1.4 | 0.87-0.95 |
| attn.sparse_mla | 1.67 | 2.06 | 1.23. Every decode/verify launch took `sparse_mla_prefill_mg_dual` (112 rows > 64). |

## Unmapped residue (on purpose, no simulated slot)

Decode leaves 1.6% unmapped and prefill 2.1%:
- **Decode:**
  - two folded draft side-stream `[2x: quant, GEMM]` bodies, 0.2 ms/iter;
  - torch elementwise, index and copy bookkeeping;
  - verify broadcasts;
  - the routed+shared residual add.
- **Prefill:**
  - the torch elementwise/reduce of the prefill indexer einsum, ~2 ms/iter;
  - a clamp kernel, 0.6 ms;
  - the residual add;
  - the Engram cub id sort.

# Second pass (2026-11-07): model fixes, re-align, end-to-end check

## Changes (simulator only)

1. **Decoder SWA bounded replay** (`deepseek_v41_sglang.rs` `BoundedReplay`; arch flag
   `decoder_swa_bounded_replay`, default on like SGLang v0.5.23). On prefill/extend forwards, layers 20-39 run each
   request's last 128 extend rows. Layer 20's compressor and index-K writes keep every row
   (`write_tail_source_kv`), and the DSpark KV inject reads tail rows only. Decode and verify are unchanged. The
   full-row `mhc.combine` before the source write is not priced.
2. **Logits all-gather** priced at bf16 (`enable_fp32_lm_head` off; the gather precedes the fp32 cast).
3. **DSpark Markov gather** payload set to SGLang's: 32 fp32 (value, index) pairs = 256 B per request per rank
   (`_launch_partial`, `gather_stacked`). This does not close its timing gap: measured ~345 us per round is fixed NCCL
   SHM latency, not size. ~1.9 ms per decode step stays unmodeled.
4. **Dense-blocks decode indexer at the graph width** (`SglangDsv41LayerWorklet::bookkeeping_tokens`). On SM12x,
   `decode_scores` builds `[requests, lmax]` tensors, where `lmax` is the decode graph's page-table width
   (`--context-length` / ratio) and not the real context. The alignment cases ran at `max_model_len` 16384, so they
   could not see it; the live server ran at 1M. **CALIBRATED** (`DENSE_INDEXER_PASSES = 10`, about 0.45 ns per
   request x lmax element), from:
   - the torch-profiler `decode_bs1_ctx64k` / `decode_bs8_ctx16k` traces: ~2 ms and ~30 ms per step in these torch
     kernels;
   - the live log's decode steps.

   Replace with profiled rows from a long-context nsys capture.

## Kernel alignment after the changes (multiplier excludes the warm-up iterations 1-2)

| case / stage | n | measured ms | before | after |
|---|---:|---:|---:|---:|
| 07 prefill 2048-token chunks | 25 | 205.1 | +67.5% | **+2.6%** (abs 4.2%) |
| 01 prefill chunks | 12 | 212.8 | +53.7% | **-2.1%** (abs 6.9%) |
| 07 decode | 67 | 55.8 | +1.2% | **+0.3%** |
| 01 decode | 101 | 57.4 | +1.9% | **+0.5%** |

These are kernel critical paths (`measured_ms` vs timing-predict). The **multiplier is a different quantity**: the
duty cycle `Σ measured_gpu_cycle_ms / Σ measured_ms`, i.e. the GPU idle between kernels that the simulator does not
model. The analyzer first recommended 1.256 / 1.250. That came from iteration 2, a JIT warm-up prefill: 2.9 s of
critical path in a 5.0 s cycle. Its factor is 1.71, under the screen's 2.0 floor, yet its 2.07 s of idle is about 80x
the stage median.

**Analyzer fix** (`analyzer/rust/src/alignment_iteration/mod.rs`, `screen_duty_cycle_outliers`): a second, idle-based
screen. An iteration is also dropped when all three hold:
- its idle is an Iglewicz-Hoaglin outlier in its stage;
- it is at least 10x the stage's median idle;
- it is longer than the stage's median iteration.

The report now adds `gpu_time_multiplier_by_stage`. Two new tests cover the screen; all 70 alignment_iteration tests
pass.

| case | pooled (was) | pooled (now) | decode | mixed (prefill) |
|---|---:|---:|---:|---:|
| 07 | 1.256 | 1.105 | 1.040 | 1.151 |
| 01 | 1.250 | 1.084 | 1.034 | 1.204 |

**Adopted in the Decycle presets:** `gpu_time_multiplier: 1.04` and `prefill_gpu_time_multiplier: 1.17` (prefill
time-weighted over the two captures). The earlier "keep 1.0" was wrong: it compared critical paths, not cycles.

Caveat: `DENSE_INDEXER_PASSES` was fitted against live decode wall time at multiplier 1.0, so decode now carries about
4% of idle twice. That is inside that calibration's ±10% scatter; refit it with the long-context capture.

## Decode steps vs the live server at 1M context

Live step times are from consecutive `Decode batch` lines in `serve_full.log` (40 steps each, 1-s timestamps).

| requests | context/req | live ms | sim ms (no change 4) | sim ms (final) |
|---:|---:|---:|---:|---:|
| 1 | 40-150K | 27.5-27.9 | 28.1-28.3 | 30.9-31.0 |
| 2 | 40K | 32.1 | 29.1 | 34.6 |
| 3 | 150K | 46.4 | 33.2 | 41.5 |
| 4 | 150K | 50.0 | 36.7 | 47.7 |
| 6 | 40-90K | 56.2-58.3 | 41.3-42.1 | 57.9-58.7 |
| 8 | 90K | 92.5 (n=10) | 47.0 | 69.1 |

## End-to-end: the measured parts of the 2026-11-05 windows

Requests are filtered by issue time to the replayed span. "Matched acceptance" is the live random-token run's mean
accept length: 5.0 typical (p 0.885) and 5.45 busy (p 0.915), uniform per position.

| window | run | requests | TTFT p50 / p90 / p99 s | miss TTFT p90 s | TPOT p50 / p99 ms |
|---|---|---:|---|---:|---|
| typical 18 min | live | 384 | 0.44 / 2.5 / 13.1 | 14.0 | 12.9 / 135 |
| | sim, matched acceptance | 388 | 0.29 / 1.7 / 11.4 | 13.7 | 12.9 / 66 |
| busy dense 15 min | live | 350 | 1.79 / 21.7 / 56.6 | 34.2 | 25.1 / 121 |
| | sim, matched acceptance | 493 | 3.4 / 13.9 / 37.4 | 32.7 | 21.3 / 113 |

- **Typical:** TPOT p50 and cache-miss TTFT match. Sim TTFT on cache hits is about 30% low, and TPOT p99 is half of
  live: decode stalls behind prefill chunks are under-modeled.
- **Busy:** still optimistic. The sim completes 41% more requests in the span, TTFT p90 is 36% low and TPOT p50 15%
  low. Likely causes:
  - the 8-request decode step (25% low);
  - the unmodeled DSpark collective latency (~2 ms per step);
  - the prefill/decode interleave.

## Decycle full days at real-text acceptance (p 0.613), 4x RTX PRO 6000

| day | done | TTFT p90 hit / miss | TTFT p99 | TPOT p50 / p99 |
|---|---|---|---:|---|
| typical | 1658/1658 | 1.3 / 8.8 s | 6.0 s | 14.7 / 62 ms |
| peak | 9020/9020 | 3.7 / 16.2 s | 36.6 s | 26.9 / 94 ms |

The busy dense window alone at p 0.613: TTFT p90 34.7 s on cache hits, against the 10 s target, and TPOT p99 178 ms.
Together with the busy-window optimism above, one node does not hold the busy day's dense hours.

## With the per-stage multipliers (v4 runs)

| window | run | requests | TTFT p50 / p90 / p99 s | miss TTFT p90 s | TPOT p50 / p99 ms |
|---|---|---:|---|---:|---|
| typical 18 min | live | 384 | 0.44 / 2.5 / 13.1 | 14.0 | 12.9 / 135 |
| | sim, matched acceptance | 386 | 0.35 / 2.1 / 13.3 | 16.8 | 13.6 / 112 |
| busy dense 15 min | live | 350 | 1.79 / 21.7 / 56.6 | 34.2 | 25.1 / 121 |
| | sim, matched acceptance | 446 | 3.9 / 15.9 / 43.3 | 44.4 | 21.4 / 84 |

| full day, p 0.613 | done | TTFT p90 hit / miss | TTFT p99 | TPOT p50 / p99 |
|---|---|---|---:|---|
| typical | 1658/1658 | 1.5 / 10.3 s | 7.0 s | 15.3 / 63 ms |
| peak | 9020/9020 | 6.5 / 26.5 s | 42.7 s | 29.6 / **106 ms** (target <= 100) |

# RunPod (GPU peer-to-peer) profile and alignment (R4, 2026-11-09)

**Machine:** RunPod pod `syy2ojjzsqhnsp` (deleted). The R3 replays ran on a different RunPod host; both are
4x RTX PRO 6000 Server Edition, P2P OK on all pairs. The simulator keeps the two machines apart with the arch flag
`gpu_p2p` (fabric `pcie_p2p` vs `pcie`).

## Collective rows (19 x `all_reduce` / `nccl`, 4 ranks, merged into profile.db, 0 conflicts)

| message size | no-P2P (Massed) | P2P (RunPod) |
|---|---|---|
| 4-32 KB | 0.016-0.018 ms | 0.028-0.047 ms (slower: higher latency) |
| 4 MB - 1 GB | 18 GB/s bus bandwidth | 31 GB/s (1.7x faster) |

## Kernel alignment (P2P variant `tp4_ep4_sglang_dspark_p2p`; warm-up iterations screened)

| case | decode meas / sim | prefill meas / sim | duty cycle decode / prefill |
|---|---|---|---|
| 20 (case 07 workload, 16K context length) | 53.5 / 53.4 ms (-0.2%) | 177.0 / 181.5 ms (+2.6%) | 1.029 / 1.145 |
| 21 (1M context length, 8 decoding, 32-66K context) | 78.0 / 71.0 ms (**-9.0%**) | 195.4 / 205.7 ms (+5.3%) | 1.102 / 1.108 |

**Case 21 is the first measured 1M-context-length decode.** It showed two more ops sized by the graph's `lmax`, now
modeled:
- `indexer.logits_decode`: the width is now `lmax`. It was 1.84 ms simulated against 4.65 measured; now 2.42. Its
  rows still under-price wide grids.
- `indexer.torch_topk`: `DENSE_TOPK_PASSES = 1` over `[requests, lmax]`. It was 0.42 ms simulated against 2.07; now
  2.54.

The remaining case 21 decode gap is the logits kernel (about 2.2 ms) and small decode all-reduces (8.11 ms measured
vs 5.93 ms simulated, likely rank skew). The parse fix that made these captures readable:
`alignment/nsys/parse.py` chunks the correlation-id query, because a 1M-context iteration exceeds SQLite's 32,766
bound variables.

## End-to-end against the R3 live replays (first 15 min)

Simulator settings:
- presets `window_*_runpod`: `gpu_p2p: true`;
- multipliers 1.10 / 1.11 (case 21);
- R3's measured agent-text acceptance (p 0.853 typical / 0.845 busy);
- KV pool 2,901,760 tokens.

| window | run | requests | TTFT p50 / p90 / p99 s | hit / miss TTFT p90 s | TPOT p50 / p99 ms |
|---|---|---:|---|---|---|
| typical | live R3 | 307 | 0.41 / 2.0 / 11.7 | 1.7 / 10.4 | 15.6 / 58 |
| | sim, RunPod profile | 307 | 0.32 / 2.0 / 9.9 | 1.7 / 13.1 | 15.5 / 77 |
| busy | live R3 | 314 | 0.74 / 23.7 / 59.0 | 23.4 / 46.2 | 23.9 / 81 |
| | sim, RunPod profile | 397 | 5.4 / 20.3 / 37.4 | 19.4 / 35.8 | 26.8 / 108 |

- **Typical:** TTFT p90 and TPOT p50 match. Cache-miss TTFT is 26% high.
- **Busy:** the sim completes 26% more requests and TTFT p90 is 14% low; TPOT p50 is 12% high. The queueing
  shape differs: p50 too high, tail too low.
- The same windows with the no-P2P rows (`*_runpod_nop2p`) show what P2P changes: cache-miss TTFT 16.0 -> 13.8 s
  (typical) and busy TPOT p99 162 -> 93 ms.

## Indexer labels and refit on the R4 case-21 capture (2026-11-09)

**Label rules.** `gen_rules.py`, 191 rules for 45 operations:
- Decode: the dense-blocks indexer's torch kernels → `indexer.torch_bookkeeping`, in the decode-graph index layers
  (concurrent tracks), from the k store / `weights_proj` / FP4 logits / top-k to the next non-torch kernel.
- Prefill: from `weights_proj` through the einsum epilogue to the top-k → `indexer.logits_prefill` (primary track).
- Chains are forward-only. `after` is the *immediate* predecessor's operation, so a backward chain advances one kernel
  per sweep and did not converge in 25 sweeps.

Coverage on case 21 went from 81% to 97.7%; decode unmapped time from 20.7 to 3.3 ms per step.

**Refit** (calibrated on case 21, 1M context length, 8 decoding):
- `DENSE_INDEXER_PASSES = 11`: bookkeeping 24.77 ms measured vs 24.21 ms simulated.
- `indexer.logits_decode` past the widest profiled row (65,536): the grid is priced as `pieces` row blocks of width
  65,536, with keys split across them. 4.65 ms measured vs 4.63 ms simulated; it was 2.42 ms before.
- Prefill `indexer.logits_prefill`: 29.44 ms measured vs 29.83 ms simulated, with no change. The earlier "+25 ms" was
  only that its kernels were unlabeled.

| case | decode meas / sim | prefill meas / sim |
|---|---|---|
| 21 (1M context length) | 78.0 / 75.5 ms (**-3.2%**) | 195.4 / 205.7 ms (+5.3%) |
| 20 (16K context length) | 53.5 / 53.5 ms (-0.1%) | 177.0 / 181.5 ms (+2.6%) |

The remaining case 21 decode gap is the small decode all-reduces: 8.11 ms measured vs 5.93 ms (rank skew).

**R3 windows on the RunPod profile** (first 15 min):

| window | run | requests | TTFT p50 / p90 / p99 s | hit / miss TTFT p90 s | TPOT p50 / p99 ms |
|---|---|---:|---|---|---|
| typical | live | 307 | 0.41 / 2.0 / 11.7 | 1.7 / 10.4 | 15.6 / 58 |
| | sim, before the refit | 307 | 0.32 / 2.0 / 9.9 | 1.7 / 13.1 | 15.5 / 77 |
| | **sim, after** | 307 | 0.29 / 1.9 / 11.7 | 1.6 / 13.1 | 16.5 / 83 |
| busy | live | 314 | 0.74 / 23.7 / 59.0 | 23.4 / 46.2 | 23.9 / 81 |
| | sim, before the refit | 397 | 5.4 / 20.3 / 37.4 | 19.4 / 35.8 | 26.8 / 108 |
| | **sim, after** | 378 | 3.4 / 22.1 / 44.8 | 20.2 / 37.6 | 28.6 / 84 |
