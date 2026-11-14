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

## Findings from reading the SGLang v0.5.23 SM120 paths (runner work, 2026-11-05)

- **Attention switches by query-row count, not by phase.** `flash_mla_with_kvcache_sm120`
  (`kernels/ops/attention/flash_mla_sm120.py:298`):
  - up to 64 rows: FlashInfer `sparse_mla_decode_dsv4` + merge;
  - above 64 rows: `sparse_mla_prefill_mg_dual`.

  A DSpark verify of 16 requests x 7 rows (112) already takes the prefill kernel, which explains the bs16 trace. One
  kind (`paged_sparse_mla`) covers decode, verify and prefill.
- **The KV layout is V4's on SM12x** (`select_dsv4_kv_layout`, `deepseek_v4_memory_pool.py:200`): 584-byte rows,
  256-token SWA pages, compressed pages of `256 // ratio`.
- **Indexer.**
  - Decode: one Triton launch (`fp4_index_logits_decode`) over a `[rows, lmax]` grid, where `lmax` is the
    graph-stable capacity.
  - Prefill: torch, per request; BF16 dequant of the visible index K, then `einsum` + ReLU + head weights. Its cost
    grows with query rows x compressed context. That is likely the trace's BF16 GEMM and top-k share in prefill, and
    it could be large on Decycle's long contexts.
  - Index layers: 2, 8, 14, 20, 24, 28, 32, 36. 32 heads x 128, replicated (not TP-sharded).
  - Candidate-block selection is not modeled yet (`select_candidate_block_ids`, `amax_topk_blocks`,
    `topk_among_blocks`).
- **mHC is unfused.** Two `mix_stats_sinkhorn` + `combine_rms_norm` per layer (before attention and before the
  FFN) and one `mhc_post` after each sublayer. On SM12x, mix-stats runs on a side stream during decode/verify, so the
  worklet must model that overlap. These are separate kinds for that reason.
- **Engram** (layers 1 and 14), per layer:
  - a host-table gather: a `torch.sort` of the ids, then `_engram_gather_kernel` over pinned host memory;
  - `wkv` as an FP8 `single_gemm` (m = tokens, n = 25600, k = 6144);
  - `engram_gate`.

  The 2026-11-05 server ran **without huge pages** (`transparent_hugepage/shmem_enabled = never`), and SGLang warns
  that lookups are then about 10x slower. Lookup rows record the page backing (`host_shared_4k` /
  `host_shared_huge`).
- **The dense FP8 linear includes its activation quant** (one row). The MoE row includes its MXFP8 activation
  quant. Do not add separate quant ops for these.
