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
