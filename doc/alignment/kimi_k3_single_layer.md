# Kimi-K3 Single-Layer Alignment (B10)

This note records the CPU-side alignment and the remaining GPU work for the
synthetic rank-1 Kimi-K3 decoder-layer probes on an NVIDIA B200. The evidence
is the 20-iteration CUDA-graph captures under
`scripts-local/vibesim-analysis-container/kimi_single_layer/align/`, their
parsed kernel sequences, and the branch profile database. It is not a
full-model result: the probe does not exercise the 93-layer scheduler,
pipeline parallelism, or a production request mix.

## Status

B10 keeps the B9 rank-local shape fix and makes the profiling runner call the
same routed FlashInfer ABI as the K3 layer. The rank-1 presets already select
`ep_size=1` and `local_experts=112`, so no preset edit is needed. Production
EP8 continues to use the global 896-wide router. The existing
`mxfp4_fused_moe` cache key already contains `num_experts` and
`per_expert_batches`; no new kernel kind is required.

The rank-1 Rust worklet uses a normalized popularity profile captured from the
synchronized seeded driver histogram at B=128, then performs deterministic
weighted top-k sampling for each batch. This preserves the measured sparse
load shape across the timing-predict sweep while leaving the production EP8
routing law unchanged. The new branch-DB rows are real GPU measurements, not
synthetic timings.

Post-B10 direct-call acceptance points:

| layer | point | measured (us) | simulated (us) | error |
| --- | --- | ---: | ---: | ---: |
| KDA MXFP4 | B=32 / 128 | 176.22 / 417.4 | 178.10 / 382.63 | +1.1% / -8.3% |
| MLA MXFP4 | B=128 | 421.24 | 382.63 | -9.2% |

## 1. MXFP4 Root Cause

The driver captures use `--experts 112 --ep 8`. The generated `run.json`
files for both KDA and MLA therefore contain `num_experts=112`,
`num_experts_per_token=16`, and two shared experts. The `ep` value is metadata
in this single-layer driver; the FlashInfer call receives a 112-wide routing
logit tensor and 112 local experts.

Before B9, the runner and simulator used the production global width even for
the rank-1 probe:

| input | old runner / rank-1 simulator | driver layer call |
| --- | --- | --- |
| `num_experts` | 896 | 112 |
| `num_local_experts` | 112 | 112 |
| `per_expert_batches` | 896 entries; the first local shard has 258 rows at B=128 | 112 entries; all 2,048 top-16 assignments are local |
| `top_k`, routing, activation | 16, DeepSeek-V3 sigmoid, SiTU | same |
| finalize and PDL | `do_finalize=True`, `enable_pdl=True` | same for this B=128 call |
| token tuning | `next_power_of_two(B)` | same production setting |

The old 896 row in the branch database was 342.8956 us at B=128. It priced a
local shard containing 258 routed rows, while the rank-1 layer call routes all
2,048 top-16 assignments through its 112 experts. B9 corrected that width.
B10 additionally fixes the runner's input contract: the layer passes
`(topk_ids, topk_weights)` with int32 IDs and float32 weights, `TopK` routing,
no routing bias, SiTU alpha 4, clamp 25, `do_finalize=True`, PDL enabled, and
`tune_max_num_tokens=next_power_of_two(B)`. The runner no longer creates a
dense routing-logit tensor or enters a separate autotune context.

The synchronized driver dump at B=128 has 112 entries, sum 2,048, and 47
nonzero experts. Its B=32 dump has sum 512 and 40 nonzero experts. The runner
matrix below uses the exact B=128 histogram as a popularity profile for the
simulator's deterministic weighted top-k law; the measured runner A/B uses the
exact per-call IDs from both driver dumps.

### Controlled GPU A/B Matrix

Kernel shorthand: `G1` is
`bmm_MxE4m3_MxE2m1MxE4m3_Fp32_...t128x16x256u2_s3_et128x16`, `G2` is
`bmm_Bfloat16_MxE2m1MxE4m3_Fp32_...t128x16x256u2_s3_et128x16`, `R` is the
routing kernel, and `F` is `finalizeKernelVecLoad`. Values are
`G1 + G2 + R + F = total` in us; the parenthesized value is B=32. The layer
reference is 418 us at B=128 and 176.22 us at B=32.

| candidate | runner as-is | modified runner | delta vs layer reference |
| --- | --- | --- | ---: |
| Routing distribution | routed, uniform/balanced: `329.1+169.1+6.9+10.6=515.7` (`187.9+100.2+6.1+9.7=303.9`); same `G1/G2/R=NoOp/Softmax/F` names | routed, driver histogram: `248.6+124.9+6.7+10.2=390.5` (`107.8+57.5+6.2+9.4=180.9`) | -6.6% / +2.6% |
| Activation and input format | old dense API with driver histogram: `248.7+124.9+14.1+10.4=398.2` (`107.9+57.7+11.1+9.7=186.5`); `R=512,22 SigmoidBias/ScaledSum` | routed API above; `R=256,16 NoOp/Softmax` | -4.7% / +5.8% -> -6.6% / +2.6% |
| Autotune context | standard production call: `t128x16x256u2_s3_et128x16`, 390.5 (180.9) | extra `autotune()`: `t128x32x256_s5_et128x32`, `233.3` (`183.5`) | -44.2% / +4.1%; wrong tactic |
| Graph versus eager | eager routed driver histogram: 390.5 (180.9) | CUDA-graph replay of the same call: 392.4 (180.4) | -6.1% / +2.4% |

The legacy as-is point combining dense routing and balanced rows was 523.5 us
at B=128 (329.2 + 169.2 + 14.5 + 10.5) and 310.9 us at B=32. The layer's
kernel names are the standard `t128x16x256u2_s3_et128x16` names, so the
autotune result is not a valid substitute even though it is faster. Removing
finalize from the routed driver-histogram call gives 381.6 us at B=128, which
also rules out finalize as the source of the original discrepancy.

## 2. Unmapped Attention-Residual Work

Each graph has two `sglang::attn_res_fused_tma_kernel` launches with
`attn_res_block_size=12`. Their mean combined cost is 9.04 us for KDA and
8.86 us for MLA. The total unmapped measured work is 17.04 us/step for KDA
and 13.93 us/step for MLA, leaving approximately 8.00 and 5.07 us of
fill/add/direct-copy CUDA-graph glue respectively.

This remains an explicit measurement floor in the alignment note. It is not
folded into an existing GEMM or generic `elementwise` leaf: the TMA kernel is
a distinct SGLang backend operation and no measured profile row exists for
that backend. Adding an unmeasured elementwise row would make the number look
mapped without improving fidelity. A future dedicated TMA kind can replace
the floor after a GPU measurement.

## 3. Small-M Launch Floor

The KDA B=1 graph is 179.6 us measured versus 141 us simulated, a 38.6 us
gap. The graph has 26 launches in the KDA sequence and 29 in the corresponding
MLA-sized sequence. Several graph-replay launches have a practical 2-3 us
minimum, while the isolated timing rows used by timing-predict can return a
smaller `m=1` cost. This is a launch/replay floor, not evidence that the
large-B MXFP4 row should be multiplied by a fixed layer constant.

No blanket small-M floor is added in B10: the captures do not provide a clean
per-kind decomposition of the replay overhead, and adding it to every leaf
would double-count the unmapped graph glue. The next GPU pass should measure
the B=1 sequence with the same graph and record whether the excess is
concentrated in the existing small-M rows or in a separate floor.
Other worker paths expose `gpu_time_multiplier` for whole-step inter-kernel
overhead, and the analyzer derives that multiplier from GPU-cycle evidence.
That correction is intentionally global; applying it only to B=1 would
overfit the graph and applying it to the MXFP4 leaf would distort B=128.

## Post-Fill Alignment Rerun

After the branch-DB fill, both `alignment timing-predict` and `alignment
analyze` completed successfully for the KDA and MLA YAMLs:

```bash
PATH="/home/yilegu/.cargo/bin:$PATH" uv run --no-sync python -m launcher alignment timing-predict \
  presets/alignment/kimi_k3_single_layer/kda_timing_predict.yaml --build-type release
PATH="/home/yilegu/.cargo/bin:$PATH" uv run --no-sync python -m launcher alignment timing-predict \
  presets/alignment/kimi_k3_single_layer/mla_timing_predict.yaml --build-type release
PATH="/home/yilegu/.cargo/bin:$PATH" uv run --no-sync python -m launcher alignment analyze \
  presets/alignment/kimi_k3_single_layer/kda_analyze.yaml --build-type release
PATH="/home/yilegu/.cargo/bin:$PATH" uv run --no-sync python -m launcher alignment analyze \
  presets/alignment/kimi_k3_single_layer/mla_analyze.yaml --build-type release
```

The reports below are post-fill values from the 20-iteration graph captures.
They are selected-device critical-path means, not sums of all concurrent
kernel residency:

| capture | measured critical path (us/step) | cached simulated (us/step) | error |
| --- | ---: | ---: | ---: |
| KDA B=128, kv=8192 | 553.08 | 534.99 | -3.27% |
| MLA B=128, kv=8192 | 677.48 | 608.47 | -10.19% |

The operation rows show the MXFP4 improvement directly:

| operation | measured mean (us) | simulated critical-path mean (us) | error |
| --- | ---: | ---: | ---: |
| KDA `mxfp4_fused_moe` | 384.40 | 382.63 | -0.46% |
| MLA `mxfp4_fused_moe` | 421.24 | 382.63 | -9.17% |

KDA and MLA share the same measured MXFP4 cache row in this rank-1 probe. The
remaining full-step MLA error is dominated by the attention-side profile and
does not indicate a second MoE launch shape.

## `CostNode::Max` and the Optimality Ladder

`CostNode::Max` still reports the maximum child duration as the critical-path
duration. The alignment attribution pass now distributes that duration across
all positive-time child subtrees in proportion to their folded times. Therefore
the critical-path total is unchanged, while non-winning children are no longer
reported as zero.

The post-fill MLA per-op report confirms the change:

| operation | measured mean (us) | simulated critical-path mean (us) | error |
| --- | ---: | ---: | ---: |
| `mla_cache_append` | 15.558 | 2.745 | -82.36% |
| `output_gate` | 9.810 | 8.577 | -12.56% |
| `kv_a_layernorm` | 0.018 | 0.975 | +5439.78% |

These are visible, nonzero rows rather than the previous `-100%` entries. The
same three names are present in both `optimality_waterfall.json` and
`optimality_batch_locked_waterfall.json`; optimality continues to use the
folded leaf workload. Unit tests cover proportional Max sharing, exact ties,
and root-time preservation.

## Measurement Environment

The GPU fill and direct-call matrix used only B200 index 3 through the branch
profile database:

```bash
export VIBESIM_PROFILE_GPUS=GPU-019267a2-092a-6798-3cc5-57ffc761e004
VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db \
TMPDIR=/raid/tmp/yilegu_k3_tmp
```

Every GPU launch was preceded by a memory check on `nvidia-smi -i 3`; the
branch database was used instead of `profiling/profile.db`.

## B12 Chunked Prefill

B12 adds rank-local chunked-prefill timing for the KDA and MLA layer variants.
This is still layer-level kernel evidence from the eager single-layer driver;
it is not a server alignment. The fill used only GPU 7 and the branch database:

```bash
VIBESIM_PROFILE_GPUS=GPU-c88e489a-0693-2c29-a1e3-30952377f742
VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db
TMPDIR=/raid/tmp_yilegu_k3_tmp
```

### Phase-1 Decision Table

The classification below is from the per-kernel tables in
`profiles/prefill_kda_*.json` and `profiles/prefill_mla_*.json`. Existing dense
GEMM, cache-append, norm, and MXFP4 kinds are reused with prefill-shaped
arguments. The new fused leaves preserve the production callable boundary.

| Profile evidence | KDA decision | MLA decision |
| --- | --- | --- |
| `nvjet_sm100_tss_*` | Reuse `gemm_fp32_output` for the merged front | Reuse `gemm_fp32_output` for the merged front |
| `nvjet_sm100_tst_*` and small BF16 GEMMs | Reuse `single_gemm` at `m=T` for qkvbfg, shared/down, latent/up, and output projections | Reuse `single_gemm` at `m=T` for projections; reuse `batched_gemm` for latent KV BMM |
| `bmm_MxE4m3_*` plus route/finalize/quant support | Reuse histogram-aware `mxfp4_fused_moe` with `top_k=2` and `2T` routed rows | Same reuse; the local 112-expert route remains in the cache key |
| `_causal_conv1d_fwd_kernel` | New `causal_conv1d_prefill` | N/A |
| KDA l2norm, cumsum, recompute, intra/inter solve, and output kernels | New `kda_chunk_prefill` group | N/A |
| `attn_res_fused_tma` | New `k3_attn_res_prefill`, two launches, one valid block | New `k3_attn_res_prefill`, two launches, one valid block |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` and related KDA kernels | Part of `kda_chunk_prefill` | N/A |
| `trtllm_ragged_attention_deepseek` causal pass | N/A | New `mla_prefill_attention`, `causal=true`, `q_len=T`, `kv_len=T` |
| `trtllm_ragged_attention_deepseek` prefix pass | N/A | New `mla_prefill_attention`, `causal=false`, `q_len=T`, `kv_len=prefix_chunk` |
| `create_chunked_prefix_cache_kv_indices` and latent-KV gather | N/A | New `mla_prefix_gather` composite |
| `merge_state` | N/A | New `mla_merge_state` |
| SiTU and the final three-way add | New `k3_situ_and_mul_prefill` and `k3_add3_prefill` | Same new leaves |
| BF16 residual adds, copies, route bookkeeping, and small fills | Fold into the owning attention/MoE boundary or keep as an elementwise placeholder; no independent kind | Same policy |

The new runners call the production SGLang/FlashInfer entry points in the
`sglang_k3_env` worker: `causal_conv1d_fn`, `chunk_kda`,
`trtllm_ragged_attention_deepseek`, `attn_res_fused_tma`, the MLA cache
helpers, `merge_state`, `situ_and_mul`, and `add3`. Decode configurations and
decode preset numbers are unchanged.

### Direct-Call Leaf Evidence

The branch rows are direct-call medians. The eager driver tables expose a
larger resident-layer launch context, so the following is the useful
per-leaf comparison rather than a claim that the two measurement boundaries
are interchangeable. Values are microseconds for KDA B=1, T=16384, fresh
prefix unless stated otherwise.

| Logical leaf or group | Eager profile | Direct-call row used by the predictor |
| --- | ---: | ---: |
| Merged FP32 front GEMM | `nvjet_sm100_tss`: 5817.5 | 2694.3 |
| Four large BF16 projection GEMMs | `nvjet_sm100_tst`, total 6111.4 | 2831.1 |
| MXFP4 routed MoE and physical route tail | 2962.8 | 1449.7 |
| Two attention-residual TMA launches | 1882.6 | 323.9 |
| Causal convolution | 130.6 | 125.0 |
| KDA chunk group | 1245.2 | 1057.0 |
| SiTU | 254.9 | 207.5 |
| `add3` | 647.0 | 130.5 |

The same pattern is visible in MLA: the eager prefix profile has 6144.4 us of
prefix ragged attention versus 3533.2 us for the isolated callable, and the
fresh MLA profile has a 2736.2 us SiTU launch while the direct row is 207.5 us.
These differences persist after force-refreshing the major rows on GPU 7;
they are not missing cache keys.

### Prefill Prediction Comparison

`timing-predict` was run against the four prefill preset/case files with the
branch DB. The supplied `result_prefill_*.json` values are eager `us_step`.

| Layer / point | Supplied eager us_step | Simulated us_step | Error |
| --- | ---: | ---: | ---: |
| KDA `1,16384,pf` | 22132.0 | 8894 | -59.8% |
| KDA `1,16384,pf49152` | 22534.7 | 8972 | -60.2% |
| KDA `4,4096,pf` | 22110.6 | 8537 | -61.4% |
| MLA `1,16384,pf49152` | 29538.0 | 11792 | -60.1% |
| MLA `1,16384,pf` | 20584.9 | 8002 | -61.1% |
| MLA `4,4096,pf` | 17762.8 | 7557 | -57.5% |

These direct-call predictions do not meet the requested 15% eager-layer target.
The discrepancy is concentrated in the resident-layer `nvjet`, fused-MoE,
attention-residual, and prefix-attention launches, not in the prefix-path
selection: the fresh/prefix/batch-4 branches select the expected leaves and
the prefix MLA path has the expected gather, latent BMM, ragged attention, and
merge sequence. A future calibration pass needs a profiler boundary that
retains the full eager layer's weight residency and stream state; applying a
global multiplier to these direct-call rows would also corrupt the unchanged
decode alignment.
