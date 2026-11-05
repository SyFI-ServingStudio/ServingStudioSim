# deepseek_v41_sglang on SM120: per-op split, ordered by measured share

**Evidence.** SGLang v0.5.23 torch-profiler traces (CUDA kernels, TP rank 0) from the 2026-11-05 run on 4x RTX PRO
6000: TP4/EP4, DSpark block 6, `flashinfer_mxfp4` MoE, `dsv4` attention, plus the SM120 indexer fix. The traces are
in airbox-cost-calculator `docs/clients/decycle/sim/kt_bench/results_sglang_4xpro6000/prof_traces/`, git-ignored on
kisnis. Shares are the % of GPU kernel time in each traced window. The profiler was on, so absolute times are
inflated; shares are usable for ordering.

**Caveat.** These are torch-profiler captures, not the alignment pipeline's nsys capture (no `sglang_iteration(N)`
ranges). Launch granularity and shares are measured. The attribution of a kernel to the target model or to the
DSpark draft, and the per-iteration phase, are **inferred, unverified** until the B2 nsys capture.

## Shares

| Op | dec bs1 64K | dec bs8 16K | dec bs16 16K* | extend 1K on 32K | prefill 32K | prefill 128K |
|---|---:|---:|---:|---:|---:|---:|
| TP all-reduce (NCCL ring LL, SHM, no P2P) | 10.5 | 17.8 | 38.4 | 20.8 | 47.4 | 36.1 |
| dense FP8 block GEMM (q/kv-a, wq_b, wo_b, shared expert, indexer wq/wk) | 44.6 | 19.6 | 23.8 | 36.5 | 21.7 | 24.7 |
| torch elementwise / index / copy / reduce | 8.6 | **30.4** | 4.6 | 8.9 | 4.5 | 6.3 |
| routed MoE MXFP4 (CUTLASS grouped GEMM + TRT-LLM helpers) | 15.4 | 15.1 | 15.1 | 15.1 | 12.0 | 14.5 |
| hc / mhc mixing | 4.8 | 2.5 | 6.4 | 4.6 | 5.7 | 6.3 |
| BF16 / other GEMM (CUTLASS, cuBLASLt: lm_head, small linears) | 2.1 | 2.5 | 2.8 | 2.5 | 2.5 | 3.1 |
| V4.1 indexer logits (`_fp4_index_logits`) | 2.0 | 3.9 | 0.0 | 1.1 | 0.1 | 0.0 |
| sparse MLA prefill (`sparse_mla_prefill_mg_dual`) | 0.3 | 0.6 | 4.4 | 1.2 | 2.6 | 4.4 |
| top-k / sort (indexer, sampling) | 2.5 | 2.3 | 0.4 | 1.9 | 0.5 | 0.5 |
| sparse MLA decode + merge (`sparse_mla_sm120`) | 1.8 | 1.6 | 0.0 | 1.3 | 0.1 | 0.0 |
| other collectives (all-gather / broadcast: logits, DSpark sampling) | 1.2 | 1.6 | 0.0 | 0.8 | 0.1 | 0.0 |
| Engram (hash, gather, gate) | 0.2 | 0.4 | 2.4 | 0.7 | 1.1 | 2.4 |
| FP8 per-token-group activation quant | 0.9 | 0.4 | 0.8 | 0.8 | 0.7 | 0.8 |
| RMSNorm / RoPE / k-norm / SiLU-mul | 1.1 | 0.5 | 0.4 | 0.9 | 0.4 | 0.5 |
| KV paging metadata (`_page_*`, `_low_ratio_metadata`) | 0.7 | 0.3 | 0.1 | 0.5 | 0.2 | 0.1 |
| router (`tiny_n_gemm`, `_router_triton_kernel`) | 1.2 | 0.2 | 0.1 | 0.8 | 0.2 | 0.1 |
| `wo_a` partial + reduce | 1.7 | 0.0 | 0.0 | 1.2 | 0.1 | 0.0 |

\* In the bs16 window the 16 prompts lost their cached prefix (`cached_tokens` 0), so it is mostly prefill.

The bs8 elementwise share (30%) is unexplained. Candidates: DSpark verify and acceptance bookkeeping at 8 requests,
or the trace window catching post-prefill cache inserts. **This is the first thing the B2 nsys capture must
resolve**: as a modeled op it would be the third largest.

## Decisions (build order = share)

Kinds are existing `profiling/kernels/*` kinds unless marked **new kind**. "New backend" means a new runner for an
existing kind on SM120.

| # | Op | Verdict | Kind / backend | Rows needed |
|---|---|---|---|---|
| 1 | TP all-reduce | reuse kind, existing runner | `all_reduce` / `nccl` (`comm/nccl.py`) | SM120 4-rank; topology identity P2P vs SHM |
| 2 | dense FP8 block GEMM | reuse kind, new backend | `single_gemm` / `sglang_triton_w8a8_block_fp8` | all TP4 linears; prefill and decode M |
| 3 | elementwise / plumbing | `elementwise` placeholder now; investigate | `elementwise` | after nsys attribution |
| 4 | routed MoE MXFP4 | reuse kind, new backend (one op over the whole `cutlass_fused_moe` call) | `nvfp4_fused_moe` (the kind the vLLM V4.1 arch uses for routed MoE) / `flashinfer_cutlass_mxfp4_sm120` | EP4 shard, routed demand from `token_corpus` |
| 5 | hc / mhc mixing | reuse kinds where args match, else **new kind** for the unfused SGLang form | `hc_gate_mix` (sinkhorn), `hc_combine_norm`, `hc_combine`; post as `mhc_post` / `_mhc_post_split_h` / `sglang_triton`; check against `MhcFusedPostPreRmsNormKernel` | per layer, prefill and decode tokens |
| 6 | BF16 / other GEMM | reuse | `single_gemm` / `torch_cublas` | lm_head and small linears on SM120 |
| 7 | V4.1 indexer logits | reuse kind, new backend | `sparse_indexer_logits_decode/prefill` / `sglang_fp4_index_logits` | index layers |
| 8 | sparse MLA prefill and decode | reuse kinds, new backend | `compressed_sparse_mla_prefill/decode` / `flashinfer_sparse_mla_sm120` | compress ratios 1/2/4/128, candidate consumers |
| 9 | top-k / sort | reuse | `logits_topk` / torch | index top-k sizes |
| 10 | other collectives | reuse `all_gather` / `nccl`; broadcast **new kind** (small) | `comm/nccl.py` | logits all-gather, DSpark sampling |
| 11 | Engram | reuse kind, new backend | `engram_lookup` / `sglang_engram_host_gather` | host-table gather over PCIe |
| 12-17 | act quant, norm/RoPE, paging, router, `wo_a` | `elementwise` placeholders first; promote after Phase 5 if they rank | `fp8_per_token_group_quant`, `rms_norm`, `elementwise` | - |
| - | DSpark draft forward | composed from the same kinds as the target (draft model `DeepseekV4ForCausalLMDSpark`, 2 GB) | - | needs nsys ranges to separate draft from target |

**L1 stop set for the first round:** 1, 2, 4, 5, 7, 8, 11. Together they cover about 85-95% of GPU time in every
window, excluding item 3. Items 6, 9 and 10 reuse existing runners, so they only need rows. Everything else starts
as a placeholder.
