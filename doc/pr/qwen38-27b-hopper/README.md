# Qwen3.8-27B-NVFP4 evidence (Hopper: H100 and H200, vLLM TP1)

This is the evidence for running `nvidia/Qwen3.8-27B-NVFP4` on one H100 or one H200. The B200 model of the
same checkpoint is described in [`../qwen38-27b/README.md`](../qwen38-27b/README.md); this document only covers
what Hopper changes.

## Maintained artifacts

| Path | What it is |
|---|---|
| `simulator/src/arch/qwen35_vllm_nvfp4_gdn_dense.rs` | `KernelPath` (SM100 / SM90), selected from the GPU name; the SM90 backends |
| `simulator/src/worklet/vllm_nvfp4_swiglu_mlp_local.rs` | `weight_only`: the MLP without its two NVFP4 activation quants |
| `simulator/src/timing/kernels/paged_attn_batch_{decode,prefill}.rs` | `vllm_fa3` backend: context knots to 1,048,576, wave edges at 132 SMs |
| `simulator/src/timing/kernels/gdn_prefill_post_conv.rs` | `infeasible_mask` for grid cells whose int32 offsets overflow |
| `profiling/runners/gemm/vllm_marlin_nvfp4.py` | Marlin W4A16 NVFP4 linear (`apply_fp4_marlin_linear`) |
| `profiling/runners/gemm/vllm_cutlass_fp8_tensor.py` | Cutlass SM90 per-tensor FP8 linear |
| `profiling/runners/attention/paged_attn_batch_vllm_fa3.py` | vLLM's FLASH_ATTN (FlashAttention 3) paged attention over the FP8 cache |
| `presets/alignment/qwen38_27b_nvfp4_{h100,h200}_tp1/` | The two alignment packs: 4 cases each, SM90 label rules, declared multiplier |
| `doc/pr/qwen38-27b-hopper/alignment_matrix.md` | The generated campaign matrices |
| `doc/pr/qwen38-27b-hopper/campaign_metrics_{h100,h200}.json` | The `alignment-campaign extract` documents the matrices are computed from |

## Deployment modeled

| | H100 80GB HBM3 | H200 |
|---|---|---|
| Captured on | ptc, 2026-11-03 | coriander (Slurm), 2026-11-03 |
| vLLM | ServingStudio fork `e1423a4d`, TP1 | same |
| Server | FP8 KV, FP32 GDN state, prefix caching, `--max-model-len 131072`, chunk 2048 | same |
| `gpu_memory_utilization` | 0.75 (0.85 runs out of memory in start-up profiling on 80 GB) | 0.85 |
| `max_num_seqs` | 256 (the default 1,024 runs out of memory at start-up) | vLLM default |
| KV pool (`Available KV cache memory`) | 36.89 GiB = 39.6103 GB | 95.54 GiB = 102.5853 GB |
| `gpu_time_multiplier` | 1.01958 | 1.02157 |

The cases are B200 pack cases 03 (balanced, c32), 04 (prefill spectrum, c32), 05 (decode spectrum, c64) and 11
(long-prefill ladder, c1), all at chunk 2048.

## How each model feature is modeled

The graph, KV accounting and GDN modeling are the B200 ones. Hopper changes four things, all selected by
`KernelPath::of(gpu_name)` (`Sm90` for H100, H200, H800, H20 and GH200):

| | B200 (SM100) | H100 / H200 (SM90) |
|---|---|---|
| NVFP4 linears (MLP, lm_head) | `flashinfer_cutedsl`, after an NVFP4 activation quant | `vllm_marlin_nvfp4` (Marlin W4A16) on the BF16 activation; the quant steps are dropped |
| FP8 linears (GDN qkvz / out_proj, GQA qkv / o_proj) | `flashinfer_fp8_tensor` | `vllm_cutlass_fp8_tensor` (`cutlass_3x_gemm_sm90_fp8`) |
| Paged attention | `flashinfer_trtllm_gen` | `vllm_fa3`, with the same prefill and decode slots, summed in a mixed step |
| GDN, norms, copies, KV append | Triton / FlashInfer / Inductor | unchanged kinds and backends (FlashInfer's chunked delta rule dispatches its SM90 kernels) |

## Kernels

What vLLM runs on Hopper, from the folded inventories of all 8 captures: 61 distinct forward kernels. Share of
measured kernel time, summed over the four cases:

| Operation | Kernels | H100 | H200 |
|---|---|---:|---:|
| `ffn.gate_up` + `ffn.down` | Marlin, 1-3 launches per GEMM by m | 64.7% | 66.5% |
| `gqa.attention` | `FlashAttnFwdSm90` + `FlashAttnFwdCombine` | 9.2% | 9.4% |
| FP8 projections (4) | `cutlass_3x_gemm_sm90_fp8` | 10.9% | 9.9% |
| `gdn.decode.recurrent` | Triton packed decode | 4.4% | 3.5% |
| `gdn.prefill.chunk_delta_rule` | FlashInfer SM90: the four context-parallel kernels or the fully fused kernel | 2.0% | 2.0% |

Every new row was measured by this branch's runners, which check their output against an fp32 reference. The
Marlin runner checks 256 sampled rows; checking all of them ran out of memory at lm_head m = 65536.

## Labeling

The packs' rules (`label_rules/gen_rules.py`, the same file in both packs) have 50 rules over 38 operations:

- 26 rules are unchanged from B200: GDN plumbing, KV append, qk-RoPE, gated norm.
- 24 are rewritten for SM90 kernel names.
- The three NVFP4 quant operations are removed. Their kernels do not exist on Hopper, and the SM90 cost manifest
  has no slots for them.
- **`gqa.attention`** owns both `gqa.prefill_attention` and `gqa.decode_attention`. vLLM makes one
  `flash_attn_varlen_func` call per layer and step, over prefill and decode rows together; in all 8 captures
  every GQA position is KV append, then one FA3 main kernel, then its combine. The simulator prices the call as
  the two slots summed, and zeroes a slot whose rows are absent, so the label compares the one measured call with
  that sum. In a decode-only step this is one call against one leaf.
- **Many-to-one** applies where one profiled call launches several kernels: FA3 main + combine; the 1-3 Marlin
  launches of one GEMM (CUPTI times every launch of the call); the five SM90 delta-rule kernels.
- Unmapped by design, as on B200: preprocess (input preparation and FA3's `prepare_varlen_num_blocks`), sampling,
  bookkeeping.

On both GPUs, every forward kernel of the four cases maps, `alignment label check` reports no findings, and
the labels reach a fixpoint in 3 sweeps with no rule disagreements.

## Check 1: kernel alignment

There is no Hopper `diverse_100` capture, so Check 1 runs on the four campaign cases. Signed error of the
simulated critical path against the measured one, per iteration, from each case's
`analysis_kernel/reports/alignment_iteration_report.json`:

| GPU | Case | Iterations | Decode median | Mixed median | All median | All total | Coverage | Recommended multiplier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H100 | 03_balanced_anchor_c32 | 185 | -2.57% | -1.03% | -2.41% | -1.42% | 99.74% | 1.0187 |
| H100 | 04_prefill_spectrum_c32 | 74 | – | -1.03% | -1.03% | -0.97% | 99.87% | 1.0168 |
| H100 | 05_decode_spectrum_c64 | 500 | -3.65% | -2.46% | -3.62% | -2.73% | 99.17% | 1.0334 |
| H100 | 11_long_prefill_ladder_c1 | 380 | +1.73% | -0.12% | +1.61% | -0.30% | 99.85% | 1.0154 |
| H200 | 03_balanced_anchor_c32 | 207 | -2.05% | -1.62% | -2.01% | -2.07% | 99.74% | 1.0199 |
| H200 | 04_prefill_spectrum_c32 | 77 | – | -1.46% | -1.46% | -1.56% | 99.88% | 1.0148 |
| H200 | 05_decode_spectrum_c64 | 521 | -3.76% | -2.94% | -3.75% | -2.87% | 99.13% | 1.0339 |
| H200 | 11_long_prefill_ladder_c1 | 380 | +2.61% | -0.85% | +2.37% | -1.45% | 99.86% | 1.0203 |

The largest per-operation deviations, ms summed over the four cases (*measured* vs *simulated*):

| Operation | H100 measured | H100 simulated | Δ | H200 measured | H200 simulated | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `ffn.gate_up` | 33534.1 | 34829.3 | +3.9% | 33890.6 | 34906.6 | +3.0% |
| `ffn.down` | 17270.6 | 16820.2 | -2.6% | 17453.0 | 16651.5 | -4.6% |
| `gqa.attention` | 7220.1 | 6860.3 | -5.0% | 7251.1 | 6785.0 | -6.4% |
| `gdn.qkvz` | 4531.6 | 4130.7 | -8.8% | 4041.8 | 3707.4 | -8.3% |
| `gdn.out_proj` | 2009.5 | 1748.4 | -13.0% | 1870.2 | 1560.1 | -16.6% |
| `gdn.prefill.chunk_delta_rule` | 1577.6 | 1455.2 | -7.8% | 1556.4 | 1417.7 | -8.9% |
| `gqa.qkv_proj` | 1319.9 | 1231.6 | -6.7% | 1176.0 | 1111.8 | -5.5% |
| `gqa.o_proj` | 657.5 | 582.8 | -11.4% | 612.8 | 520.0 | -15.1% |

- **Marlin.** Decode steps are within 1.5% for both GEMMs. In mixed steps gate_up reads +2.0 to +5.5% and down
  -2.0 to -5.9%, which nets to about +1% for the MLP.
- **FP8 projections** read low, more so in mixed steps (qkvz about -10%, out_proj and o_proj -12 to -18%).
- **Attention** is driven by its decode row. See the first item under Known gaps.

## The multiplier

Each case's `recommended_gpu_time_multiplier` is the measured GPU cycle over the measured kernel time, so it does
not depend on the simulator. Each pack declares the pooled value of its four cases: the sum of measured cycle over
the sum of measured kernel time, which is each case's value weighted by its kernel time. That gives H100 1.01958
(per case 1.0154-1.0334) and H200 1.02157 (1.0148-1.0339). It is not fitted to any E2E number. The Qwen3.8-Flash-Next
preset also derives its multiplier from campaign cases (their median, together with diverse_100).

## Simulation

The campaign at the declared multiplier is [`alignment_matrix.md`](alignment_matrix.md). For comparison, the
same runs at 1.0:

| GPU | Case | E2E at 1.0 | E2E at the multiplier | TTFT at 1.0 | TTFT at the multiplier | TPOT at 1.0 | TPOT at the multiplier |
|---|---|---:|---:|---:|---:|---:|---:|
| H100 | 03 | -3.04% | -1.16% | -2.64% | -0.74% | -2.08% | -0.18% |
| H100 | 04 | -1.17% | +0.76% | -5.41% | -3.55% | +1.26% | +3.24% |
| H100 | 05 | -4.56% | -2.69% | -1.83% | +0.10% | -4.78% | -2.92% |
| H100 | 11 | -1.49% | +0.43% | -1.35% | +0.58% | -1.04% | +0.97% |
| H200 | 03 | -3.86% | -1.81% | -2.71% | -0.61% | -3.04% | -0.97% |
| H200 | 04 | -2.83% | -0.74% | -5.27% | -3.23% | -0.86% | +1.27% |
| H200 | 05 | -4.67% | -2.62% | -6.30% | -4.28% | -5.56% | -3.52% |
| H200 | 11 | -2.05% | +0.06% | -2.04% | +0.07% | +2.57% | +4.74% |

The only out-of-tolerance items are case 04's scheduler-structure counts on both GPUs: iterations -1.75% (H100) /
-1.47% (H200) against a 1% bound, with decode batch, prefill tokens and scheduled KV off by the same amount. B200
case 04 shows the same signature (-2.04%), which the B200 evidence attributes to closed-loop requests re-entering a
few steps late through the client and frontend round trip. E2E in case 04 stays within -0.7% to +0.8%.

## Changes that reach other models

- **B200 Qwen3.8-27B:** `KernelPath::Sm100` keeps the B200 graph, backends and attention grids; the `vllm_fa3`
  knots and 132-SM wave edges apply only to that backend. The B200 regression run is in the PR's Test Result.
- **GDN post-conv int32 mask.** vLLM's `fused_gdn_prefill_post_conv` indexes rows as `token * stride + column` in
  int32, and faulted with an illegal address at the 262,144-token knot on H100. Grid cells past `i32::MAX` are now
  infeasible on every GPU:

  | Model | GDN stride | Masked above | Effect |
  |---|---:|---:|---|
  | Qwen3.6-35B-A3B | 8,192 | 262,144 tokens | 262,144 × 8,192 is exactly 2^31, so no cell is masked |
  | Qwen3.8-27B (B200, Hopper) | 10,240 | about 209,715 tokens | the 262,144 cell; B200's measured row there is no longer read |
  | Qwen3.8-Flash-Next (B200) | 10,240 | about 209,715 tokens | the 262,144 cell, if its grid has one |

  Only one GDN prefill launch of more than about 209k tokens reaches these cells. With chunked prefill, a step is at
  most its chunk (2,048-8,192 tokens).
- **Other Hopper models** (DeepSeek-V4-Flash, GLM-5.2, Llama-3.1-8B, the Qwen3 MoEs, Qwen3.6) use other kernel
  kinds and rows. The multiplier is a per-pack value and reaches no other deployment.

## Modeling choices

- **One graph, two kernel paths.** The GPU name selects the path. A new Hopper SKU therefore needs its name in
  `KernelPath::of`, and otherwise falls to SM100, where its rows will be missing.
- **Attention keeps two slots on Hopper.** It reuses the trtllm-gen decode and prefill kinds with an FA3 backend,
  and the label merges them. The first item under Known gaps proposes the alternative.
- **The Marlin row is the whole linear,** however many launches it takes, because the profiler times
  `apply_fp4_marlin_linear` with CUPTI.

## Known gaps and follow-up

- **FA3 decode attention at high concurrency reads low.** In decode-only steps: -22.4% (H100) / -25.1% (H200) at c64
  (case 05), -6.1% / -8.8% at c32 (case 03), +0.4% / +1.4% at c1 (case 11). Per layer on H100 at batch 64 the FA3
  main kernel measures 73-101 µs and the combine 1-2 µs, against 60-78 µs simulated.
  - The cause is not the merged label: a decode-only step compares one call with one leaf.
  - The likely cause is the decode cache key, inherited from trtllm-gen: `(batch, max_kv_len, fill)` over canonical
    batches (one request at the maximum, the rest even). The worklet has the exact `decode_kv_lens`, but collapses
    them to three numbers before the lookup. A c64 step holds about 8 requests near its maximum context, so FA3
    has many more long (request, KV head) blocks than the canonical row of the same key. The error grows with batch
    size and length spread. The "same key, within about 10%" property in the kind's header was measured for
    trtllm-gen on B200.
  - A second contributor is not excluded: in serving, FA3's split count is fixed at CUDA-graph capture.
  - Follow-up: an FA3 SM90 implementation of the varlen makespan projection (`fa4_varlen_projection.rs`: CTAs per
    request, KV head, tile and split over 132 SMs, giving an equivalent context), priced as one call per step, and
    validated by replaying case 05's `decode_kv_lens` through the FA3 runner. That is a cost-model change with a
    re-profile, so it is a separate PR.
- **FP8 projections read 6-18% low.** The labels are unambiguous, so this is about row accuracy. Not investigated;
  on B200 the analogous gap came from the serving process's autotune cache.
- **Marlin at large m:** gate_up reads high and down reads low, partly cancelling. Not investigated.
- **Small-kernel decode biases:** norms read 10-21% low, decode SiLU 21-35% high, quants +3 to +14%, together about
  3% of kernel time. These are the same pattern as on B200.
- **Coverage of the campaign:** 4 of the B200 pack's 20 cases, all at chunk 2048. There is no `diverse_100` capture
  and no 4096 / 8192 chunk, high-concurrency saturation or IO-quadrant case on Hopper.
- **Capture file names:** the 2026-11-03 captures were run under the B200 pack's `run_name_prefix`, so their files
  are named `qwen38_27b_nvfp4_b200_tp1_*`. The packs now name themselves.
- **Simulated-only copy time:** about 40 ms each of `gdn.ba_contiguous` and `gdn.core_output_copy` (about 0.1% of
  simulated time) sit at layer positions where the capture has no copy. Every iteration still pairs.

## Reproduce

From the repository root, with `PACK=presets/alignment/qwen38_27b_nvfp4_h100_tp1` (or `_h200_tp1`), a host
profile for a Hopper machine, and a fresh `OUT`:

```bash
uv run python -m launcher alignment-campaign check --pack "$PACK"
uv run python -m launcher alignment-campaign render --pack "$PACK" --host <host profile> --out-root "$OUT"
for phase in profile_workload profile_nsys timing_predict; do
  uv run python -m launcher alignment-campaign run --pack "$PACK" --out-root "$OUT" --phase "$phase"
done
uv run python -m launcher alignment-campaign label --pack "$PACK" --out-root "$OUT"
for phase in analysis_kernel simulation analysis_e2e; do
  uv run python -m launcher alignment-campaign run --pack "$PACK" --out-root "$OUT" --phase "$phase"
done
uv run python -m launcher alignment-campaign extract --pack "$PACK" --runs "$OUT" --out metrics.json
uv run python -m launcher alignment-campaign compare --pack "$PACK" --measured metrics.json --markdown
```

The profile phases need one Hopper GPU per pass. On an 80 GB H100 the pack's `gpu_memory_utilization` 0.75 and
`max_num_seqs` 256 are required.
