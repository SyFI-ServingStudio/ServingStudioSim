# Kimi-K3 Single-Layer Alignment (B9)

This note records the CPU-side alignment and the remaining GPU work for the
synthetic rank-1 Kimi-K3 decoder-layer probes on an NVIDIA B200. The evidence
is the 20-iteration CUDA-graph captures under
`scripts-local/vibesim-analysis-container/kimi_single_layer/align/`, their
parsed kernel sequences, and the branch profile database. It is not a
full-model result: the probe does not exercise the 93-layer scheduler,
pipeline parallelism, or a production request mix.

## Status

B9 fixes the shape mismatch between the rank-1 driver and the MXFP4 runner.
The rank-1 presets already select `ep_size=1` and `local_experts=112`, so no
preset edit is needed. The Rust worklet now passes a separate
`routing_experts` value to the fused-MoE kernel: rank 1 uses 112, while
production EP8 continues to use the global 896. The existing
`mxfp4_fused_moe` cache key already contains `num_experts` and
`per_expert_batches`; no new kernel kind is required.

The source fix is post-B9. Numeric post-fix alignment is still waiting for a
GPU fill: the branch database has no 112-expert MXFP4 rows, and each rank-1
timing-predict preset reports 68 missing MXFP4 specs. No synthetic timings
were inserted.

The supplied post-B8 graph-step results remain the high-level reference:

| layer | point | measured (us) | simulated (us) | error |
| --- | --- | ---: | ---: | ---: |
| KDA | B=1 / 32 / 128, L=8k | 179.6 / 310.7 / 556.5 | 141 / 277 / 495 | -21% / -11% / -11% |
| MLA | 128x8k / 1x1M / 16x64k | 680.4 / 345.5 / 357.9 | 569 / 274 / 317 | -16% / -21% / -11% |

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

The old 896 row in the branch database is 342.8956 us at B=128. It prices a
local shard containing 258 routed rows, while the rank-1 layer call routes
2,048 rows through its 112 experts. This is the root cause of the missing
dimension; it is not a missing finalize kernel. B9 makes the Python runner
accept exactly the two valid widths, checks the batch-count vector against the
selected width, and makes the K3 worklet pass the rank-1 width to
`Mxfp4FusedMoeKernelConfig`. The production EP8 test asserts that its width
remains 896. The timing kernel implementation itself already uses
`config.num_experts` in its payload, so it needed no separate code change.

The KDA/MLA spread is a second, distinct fact. The same six measured kernel
rows are assigned to each fused-MoE operation, but their dynamic routed BMM
work differs:

| measured kernel mean over 20 iterations | KDA (us) | MLA (us) |
| --- | ---: | ---: |
| MXFP4 gate/up BMM | 245.26 | 268.92 |
| BF16 down BMM | 119.36 | 132.06 |
| router + quantize + routing + finalize | 19.78 | 20.26 |
| operation total | 384.40 | 421.24 |

The BMM pair accounts for 36.36 us of the 36.84 us operation difference.
The captures show identical kernel shapes and call flags, so PDL and
finalize selection are not the explanation. The layer's seeded random state
produces different per-expert occupancy, which changes the dynamic grouped
BMM work. The alignment payload does not contain that realized 112-entry
histogram, so it cannot identify the individual expert bins responsible. The
simulator continues to use its explicit uniform PPM assumption; the cache key
already has `per_expert_batches` if a future capture supplies layer-specific
histograms. B9 fixes the proven 896-versus-112 mismatch without inventing a
KDA/MLA-specific constant or a new semantic kernel kind.

The cached, pre-fill operation numbers are therefore retained only as a
baseline:

| operation | measured (us) | old cached simulation (us) | old error | B9 status |
| --- | ---: | ---: | ---: | --- |
| KDA `mxfp4_fused_moe` | 384.40 | 342.90 | -10.79% | corrected 112-wide specs pending GPU fill |
| MLA `mxfp4_fused_moe` | 421.24 | 342.90 | -18.60% | corrected 112-wide specs pending GPU fill |

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

No blanket small-M floor is added in B9: the captures do not provide a clean
per-kind decomposition of the replay overhead, and adding it to every leaf
would double-count the unmapped graph glue. The next GPU pass should measure
the B=1 sequence with the same graph and record whether the excess is
concentrated in the existing small-M rows or in a separate floor.
Other worker paths expose `gpu_time_multiplier` for whole-step inter-kernel
overhead, and the analyzer derives that multiplier from GPU-cycle evidence.
That correction is intentionally global; applying it only to B=1 would
overfit the graph and applying it to the MXFP4 leaf would distort B=128.

## CPU Alignment Rerun

The CPU `alignment analyze` pass was rerun successfully for both YAMLs:

```bash
PATH="/home/yilegu/.cargo/bin:$PATH" uv run --no-sync python -m launcher alignment analyze \
  presets/alignment/kimi_k3_single_layer/kda_analyze.yaml --build-type release
PATH="/home/yilegu/.cargo/bin:$PATH" uv run --no-sync python -m launcher alignment analyze \
  presets/alignment/kimi_k3_single_layer/mla_analyze.yaml --build-type release
```

Those reports intentionally consume the cached timing-predict artifacts, so
their numbers are a pre-fill baseline rather than a B9 acceptance result:

| capture | measured critical path (us/step) | cached simulated (us/step) | error |
| --- | ---: | ---: | ---: |
| KDA B=128, kv=8192 | 553.08 | 542.42 | -1.93% |
| MLA B=128, kv=8192 | 677.48 | 568.74 | -16.05% |

The old cached manifests also predate the corrected MXFP4 width (and retain
the already-known stale timing artifacts), so these values must not be read as
post-B9 predictions. Re-run timing-predict after the operator fills the 112
expert rows, then run the two analyze commands above again.

## `CostNode::Max` and the Optimality Ladder

`CostNode::Max` does not hide `mla_cache_append` or `output_gate` from the
optimality ladder. The analyzer's `trace::manifest::fold_mean` visits every
child of `Max` and assigns each a mean-fold weight; `optimality/prepare.rs`
uses that fold. Their bytes and workload therefore remain visible to
optimality.

`Max` does hide non-winning child time from the critical-path view used by the
alignment operation table. `node_time` selects the maximum child, and the
alignment breakdown reports the non-critical `mla_cache_append` and
`output_gate` simulated critical time as zero even though their folded leaf
work remains present. This is a limitation of using `Max` as the worklet's
same-device stream-overlap fallback, not an optimality-ladder omission.

The smallest change is analyzer-side presentation: publish both
`simulated_critical_path_ms` and `simulated_leaf_workload_ms` for each
operation, and let the ladder/UI use the latter when showing non-critical
work. That preserves current analyzer semantics and needs no new kernel row.
If exact stream attribution is later required, add a distinct `Overlap` tree
node with Max wall time and full child workload, then use it in the worklet
instead of overloading `CostNode::Max`.

## GPU Fill Command

Run this exact command on GPU 3. The wrapper fills the branch database; it is
the only required B9 measurement step. This checkout did not run it.

```bash
K3_GPU_INDEX=3 \
VIBESIM_PROFILE_DB=/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/kimi_single_layer/k3_branch_profile.db \
/raid/yilegu/roofline_guided_agent/VibeSimWorkspace/scripts-local/vibesim-analysis-container/run_k3_jit_fill.sh \
  presets/predict_kimi_k3_b200_rank1_layer_kda.json \
  presets/predict_kimi_k3_b200_rank1_layer_mla.json
```

After the fill, rerun the two CPU `alignment analyze` commands above and
replace the cached-baseline tables with the new 112-wide timing-predict
results. If the filled row still leaves a stable KDA/MLA difference, capture
the realized per-expert batches and promote that histogram to an explicit
input rather than adding a layer-name branch.
