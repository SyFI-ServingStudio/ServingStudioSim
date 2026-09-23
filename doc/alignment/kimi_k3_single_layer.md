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
