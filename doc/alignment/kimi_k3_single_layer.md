# Kimi-K3 Single-Layer Alignment

This bundle records the CPU-side alignment setup for the synthetic B=128,
kv=8192 Kimi-K3 decoder-layer probes. The measured inputs are the forward-only
NSYS artifacts under `scripts-local/.../kimi_single_layer/align/`; the checked-in
YAMLs rebuild timing-predict inputs from those traces and the branch-local K3
profile database. KDA uses the BF16-state split decode worklet. The paged-FP8
MLA prediction subsequently completed and was analyzed with its own label set.

The KDA labels charge KDA's concurrent `[f_a|beta]` projection to the fused
`qkvbfg_a_proj` leaf, fold routing and the two routed expert GEMMs into the
`mxfp4_fused_moe` leaf, and leave the two `attn_res_fused_tma_kernel` launches
plus CUDA-graph glue explicitly unmapped. The resulting numbers are evidence
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

1. `qkvbfg_a_proj`: the simulator models one serial 7168-to-7692 GEMM, while
   SGLang launches a wide qkvg projection plus a concurrent `[f_a|beta]` path;
   97.6% of the first-step operation time is hidden by the side stream. This
   is both a production-shape/decomposition mismatch and a CostTree overlap
   mismatch.
2. `mxfp4_fused_moe`: the measured `t128x16x256` MXFP4/BF16 expert pair plus
   routing/finalize totals about 384 us, 10.8% above the `sglang_trtllm_mxfp4`
   row. Check the cache row against the EP8 local-expert shape and the actual
   token-to-expert popularity rather than the simulator's uniform PPM.
3. `kda_recurrent_decode`: the BF16-state measured
   `fused_sigmoid_gating_delta_rule_update_kernel` is 18.1% faster than the
   `sglang_triton` recurrent row. Re-profile or split the cache entry by the
   exact B=128/state/backend shape before changing the recurrent model.

## MLA Result

The completed MLA pass covered the same 20 B=128, kv=8192 decode iterations.
Its measured critical path averaged 677.48 us/step versus 584.64 us
simulated, for a duration-weighted signed/absolute error of -13.7038% / 13.7038%.
Mapped measured coverage was 98.02%, with 13.93 us/step of unmapped TMA and
graph glue. This remains layer-level probe evidence rather than full-model
alignment.
