# Kimi-K3 Single-Layer Alignment

This bundle records the CPU-side alignment setup for the synthetic B=128,
kv=8192 Kimi-K3 decoder-layer probes. The measured inputs are the forward-only
NSYS artifacts under `scripts-local/.../kimi_single_layer/align/`; the checked-in
YAMLs rebuild timing-predict inputs from those traces and the branch-local K3
profile database. KDA uses the BF16-state split decode worklet. The paged-FP8
MLA prediction subsequently completed and was analyzed with its own label set.

The KDA labels keep the wide fused-QKVG projection and the concurrent
`[f_a|beta]` projection as separate leaves. The side-stream leaf is placed in a
`CostNode::Max` with the wide leaf: its bytes and work remain accounted for,
while its time is removed from the critical path. This is the available
worklet-level fallback because the analyzer has no same-device stream-overlap
node. The follow-on `f_b` projection is part of that alternate stream but is
not a separate Rust leaf. Routing and the two routed expert GEMMs remain folded
into `mxfp4_fused_moe`, while the two `attn_res_fused_tma_kernel` launches plus
CUDA-graph glue remain explicitly unmapped. The resulting numbers are evidence
from one instrumented decoder layer, not full-model alignment: no K3 checkpoint
is available here, and the probe does not exercise the 93-layer scheduler,
pipeline parallelism, or request mix.

## KDA Result

The kernel-align report uses the 20 repeated B=128, kv=8192 decode iterations.
Times below are per-step means from `reports/alignment_iteration_report.json`;
relative error is `(simulated - measured) / measured`.

| operation | measured (us) | simulated (us) | relative error |
| --- | ---: | ---: | ---: |
| `attention.qkvbfg_a_proj` | 21.77 | 64.79 | +197.72% |
| `moe.mxfp4_fused_moe` | 384.40 | 342.90 | -10.79% |
| `attention.kda_recurrent_decode` | 25.50 | 30.11 | +18.08% |
| `moe.shared_gate_up_activation` | 5.18 | 1.92 | -62.69% |
| `moe.merged_front` | 47.69 | 45.76 | -4.04% |
| `attention.o_proj` | 6.95 | 5.60 | -19.41% |
| `attention.kda_conv_decode` | 5.48 | 6.57 | +20.08% |
| `moe.add3` | 3.42 | 2.36 | -29.40% |
| `moe.routed_norm` | 2.18 | 2.59 | +19.46% |
| `attention.kda_gated_norm` | 3.02 | 2.74 | -9.01% |
| `moe.shared_down` | 18.91 | 18.83 | -0.43% |
| `moe.latent_up` | 11.54 | 11.68 | +1.27% |

The measured critical path is 553.08 us/step versus 542.42 us/step simulated.
The duration-weighted signed and absolute errors are -1.9266% and 1.9266%
(11.0616 ms measured versus 10.8485 ms simulated across 20 steps). The
recommended duty-cycle multiplier is 1.00723. Unmapped measured time averages
17.04 us/step, ranging from 16.38 to 18.11 us: the two TMA launches account
for about 9.04 us, and fill/add/direct-copy graph glue accounts for the rest.
The separate torch graph probe reports 623.2 us and 676.8 us of total kernel
residency; those are not substituted for the NSYS critical-path measurements.

The first three cost-model follow-ups, ordered by aggregate absolute error, are:

1. `qkvbfg_a_proj`: the old simulator modeled one serial 7168-to-7692 GEMM,
   while SGLang launches a wide QKVG projection with shape
   `m=128,n=6144,k=7168` (`4 * 12 * 128`) plus an alternate-stream GEMV with
   padded shape `m=128,n=144,k=7168` (`128 + 12`, rounded to 8). The worklet
   now models those as separate leaves in a `CostNode::Max`; the production
   table shows about 21.6 us for the wide kernel and about 21.3 us for the
   side leaf (16.5 us GEMV plus 4.9 us split-K reduction). The branch profile
   database has neither new shape family, so the corrected timing-predict
   cannot run to completion on CPU. It reports 68 missing `m` rows for each
   leaf (136 rows total); no synthetic CPU values were inserted.
2. `mxfp4_fused_moe`: the measured `t128x16x256` MXFP4/BF16 expert pair plus
   routing/finalize totals about 384 us, 10.8% above the `sglang_trtllm_mxfp4`
   row. The runner times one `trtllm_fp4_block_scale_moe` call with
   `do_finalize=True`, so routing and finalize are already included in the
   registered row. No measured expert histogram is present in this alignment
   payload; the simulator's uniform PPM is therefore still an explicit
   assumption rather than a hidden omission.
3. `kda_recurrent_decode`: the BF16-state measured
   `fused_sigmoid_gating_delta_rule_update_kernel` is 18.1% faster than the
   `sglang_triton` recurrent row. The lookup was checked against the probe's
   exact B=128, 12-head, head-dim-128, BF16-state shape and the runner passes
   the production-style `a`, `b`, and `cache_indices` layout. Likewise,
   `kda_conv_decode` uses the exact 4608-channel, kernel-4, BF16-state row.
   These residual errors are therefore kernel-model/cache fidelity gaps, not
   shape mismatches.

## KDA Post-Fix Status

The corrected tree was compiled and inspected, but a complete post-fix KDA
alignment is not claimable on this CPU-only checkout because the branch profile
database lacks the 136 production rows needed by the new leaves. The current
comparison is consequently:

| leaf | measured production evidence (us) | old simulated (us) | post-fix status |
| --- | ---: | ---: | --- |
| `attention.qkvbfg_a_proj` (wide QKVG) | 21.77 | 64.79 | blocked on `m=128,n=6144,k=7168` and the remaining `m` sweep |
| `attention.qkvbfg_a_proj_bfa` (side GEMV + split-K) | about 21.3 | folded into 64.79 | blocked on `m=128,n=144,k=7168` and the remaining `m` sweep |
| `moe.mxfp4_fused_moe` | 384.40 | 342.90 | -10.79%; one-call runner confirmed, uniform PPM retained |
| `attention.kda_recurrent_decode` | 25.50 | 30.11 | +18.08%; exact probe row confirmed |
| `attention.kda_conv_decode` | 5.48 | 6.57 | +20.08%; exact probe row confirmed |

After the missing rows are measured, rerun `kda_timing_predict.yaml` before
applying the new label manifest. The old manifest still contains the former
7168-to-7692 request because the failed preflight deliberately did not replace
it.

## MLA Result

The completed MLA pass covered the same 20 B=128, kv=8192 decode iterations.
The original serial tree measured 677.48 us/step versus 584.64 us simulated,
for a duration-weighted signed/absolute error of -13.7038% / 13.7038%.
Mapped measured coverage was 98.02%, with 13.93 us/step of unmapped TMA and
graph glue. The cache-append runner was also corrected to allocate the
page-planar FP8 backing store as `torch.uint8`, matching the production
UnifiedKVPool byte storage; its existing database row still needs a GPU
refresh.

The structural rerun with the two available `CostNode::Max` fan-outs produced
568.74 us/step simulated versus the same 677.48 us measured, or -16.0507%
duration-weighted error. Because `Max` is currently also the analyzer's
critical-path attribution operator, non-critical leaves report zero simulated
time even though their bytes and work are included in the tree:

| operation | measured (us) | structural simulated (us) | relative error |
| --- | ---: | ---: | ---: |
| `moe.mxfp4_fused_moe` | 421.24 | 342.90 | -18.60% |
| `moe.merged_front` | 39.61 | 45.76 | +15.56% |
| `attention.fused_qkv_a_proj` | 8.21 | 11.55 | +56.56% |
| `attention.mla_cache_append` | 15.56 | 0.00 | hidden by Max attribution |
| `attention.output_gate` | 9.81 | 0.00 | hidden by Max attribution |
| `attention.mla_decode_attention` | 107.32 | 104.47 | -2.66% |
| `moe.shared_gate_up_activation` | 4.96 | 1.92 | -60.84% |
| `attention.o_proj` | 6.53 | 5.60 | -14.23% |
| `moe.latent_up` | 11.40 | 11.68 | +2.51% |
| `moe.shared_down` | 18.79 | 18.83 | +0.19% |

This structural rerun is not an acceptance result: the cache row is stale and
the current analyzer cannot attribute same-device overlap to both branches.
The two largest actionable gaps remain the MXFP4 row (-18.60%) and the cache
append row (15.56 us measured but hidden by the fallback). A GPU refresh plus a
first-class overlap attribution node is required before the MLA <=10% target
can be evaluated fairly.

## Operator Profile Commands

Run the following on the profiling GPU with the branch database. The two KDA
commands below are the probe-critical `m=128` rows; the timing-predict output
lists the other `m` values that must be repeated for a complete sweep. The MLA
command refreshes the corrected page-planar byte-storage runner.

```bash
export VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db

uv run --no-sync python -m launcher kernel-profile run single_gemm \
  --backend sglang_bf16_auto \
  --gpu-name "NVIDIA B200" \
  --db /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db \
  --spec '{"m":128,"n":6144,"k":7168,"dtype":"bf16"}' \
  --json --output-dir /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/refresh_k3_kda_qkvg_m128_n6144

uv run --no-sync python -m launcher kernel-profile run single_gemm \
  --backend sglang_bf16_auto \
  --gpu-name "NVIDIA B200" \
  --db /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db \
  --spec '{"m":128,"n":144,"k":7168,"dtype":"bf16"}' \
  --json --output-dir /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/refresh_k3_kda_bfa_m128_n144

uv run --no-sync python -m launcher kernel-profile run mla_cache_append \
  --backend sglang_cuda \
  --gpu-name "NVIDIA B200" \
  --db /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db \
  --spec '{"num_tokens":128,"kv_lora_rank":512,"rope_dim":64,"block_size":64,"input_dtype":"bf16","kv_dtype":"fp8_e4m3","cache_format":"page_planar_fp8"}' \
  --json --output-dir /raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/profiles/refresh_k3_mla_cache_b128
```
