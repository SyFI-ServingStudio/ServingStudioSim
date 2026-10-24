# Qwen3.8-27B-NVFP4 on TensorFold, one GB10: kernel split (draft)

Phase 1 output of `top-add-new-arch` / `top-split-model-into-kernels`. The evidence is one nsys capture of the
TensorFold perf-study build serving the checkpoint with the DFlash2 drafter.

**Capture** (*measured*): `wt-tf-spark/logs/20261023_4_tf27b_nsys/`
- TensorFold `perf-study-nvtx` 845c3623, which is perf-study 6469730a plus default-off NVTX ranges.
- Torch 2.14.1+cu130, nsys 2025.3.2, driver 580.159.03.
- Server: `--context 262144`, reserve 61 GiB. TensorFold allocated the full 262144-token window.
- Workload: four batch-1 greedy requests (prompt 174 / 1552 / 10444 / 49997 tokens, 256 output tokens each), after
  a warm-up outside the capture window.
- 211 decode rounds and 21 prefill chunks; 48.67 s of kernel time.
- Kernels are joined to the innermost `tensorfold.*` range through their launch call, which leaves 16.6 ms
  unattributed: the per-request final lm_head row after prefill.

## Engine facts the arch must follow (source anchors in `notes.md`)

- One request at a time. Eager launches, no CUDA graphs, one stream.
- A decode iteration is one round:
  1. DFlash2 draft: GPU `launch_block`, then a host-side best-first tree of at most 23 nodes.
  2. sync, then the target `tree_forward` over W rows: 24 tree rows, or a copy chain of 24 to 128 rows.
  3. sync, then host sampling and accept.
  4. Commit: GDN replay, KV and conv copies, drafter taps.
  5. sync.
- Prefill runs in even chunks of at most 4096 rows. Prefill activations are bf16. GEMMs quantize separately
  (`quant8`/`quant4`), except the fused MLP.
- The KV cache is bf16, not FP8: 64 KiB per token over 16 layers. GDN state is fp32, 3 MiB per layer.

## Where decode time goes (*measured*, 211 rounds, 22.14 s)

| Range | Host wall | Kernel busy | Note |
|---|---:|---:|---|
| `engine.verify` | 19.81 s | 19.19 s | GPU-bound |
| `engine.draft` | 1.52 s | 1.31 s | `draft.launch` 1.31 s GPU, `draft.tree` 0.91 s host with no kernels |
| `engine.commit` | 0.74 s | 0.63 s | includes `draft.taps` 0.32 s |
| `sampling.d2h` | 0.04 s | 0.01 s | |
| whole round | 22.14 s | 21.14 s | about 1 s of host time on the critical path, mostly the drafter's host tree |

The drafter's host tree build (about 4.3 ms per round) is serialized between the draft and verify syncs. The arch
must carry it as per-round host time; it is not a kernel.

## Per-op decision table (sorted by measured share of all kernel time)

The workload mix weights prefill heavily: the 50k-token prompt alone accounts for most of the 56% prefill share.
Each row's verdict follows the four homes in `top-split-model-into-kernels`. "Reuse kind" means a new
`tensorfold_*` backend on an existing kind with unchanged args semantics.

| # | op | measured kernel(s) | share | verdict | maps to |
|---:|---|---|---:|---|---|
| 1 | prefill linears (GDN qkv/z/out, attention q/k/v/o FP8; MLP down, lm_head NVFP4) | `gemm_kernel` (`gemm_ck`), activations pre-quantized | 29.8% (part) | reuse kind | `single_gemm`, backend `tensorfold_ck`, dtypes `fp8` / `nvfp4` |
| 1b | prefill fused MLP gate\|up + SwiGLU + NVFP4 quant | `gemm_kernel` EPI>0 (`gemm_gu_ck`) | (in #1) | **new kind**: one launch whose output is the quantized activation, so it cannot be split | `gemm_swiglu_quant` (m, n, k, dtype) |
| 2 | prefill causal attention over the bf16 KV cache (head dim 256, 24 q / 4 kv heads) | `pattn_kernel` | 15.7% | reuse kind | `flashinfer_attn_prefill` (prefix_len, append_len), backend `tensorfold_pattn` |
| 3 | verify linears: stacked gate\|up, down, GDN qkv/z/out, attention q/k/v/o, lm_head | `lane_kernel` (split-K from weight shape) | 33.6% total (12.2 + 10.3 + 4.7 + 2.8 + 1.9 + 1.3 + 0.5) | reuse kind | `single_gemm`, backend `tensorfold_lane`, dtypes `fp8` / `nvfp4`; the stacked gate\|up is one GEMM with n = 2 × 17408 |
| 4 | verify tree attention: committed keys plus each row's own tree path | `_shared` + `_tail` + `_merge` | 3.6% | **new kind**: no existing kind attends a tree mask over a committed prefix | `tree_verify_attention` (heads, head_dim, dtypes, committed_len, rows, tree depth) |
| 5 | residual add + RMSNorm | `_add_rmsnorm` | 3.0% | reuse kind | `residual_rms_norm` (m, hidden, dtype), backend `tensorfold_triton` |
| 6 | prefill GDN delta rule | `chain_kernel` | 2.8% | reuse kind | `gdn_chunk_delta_rule`, backend `tensorfold_chain` |
| 7 | drafter W4A16 affine GEMMs (5 layers + taps) | `group_kernel` (`qmm_group.cu`) | 2.3% | reuse kind; dtype `w4a16_affine_g64` to be confirmed | `single_gemm` or `grouped_gemm`, backend `tensorfold_qmm` |
| 8 | activation quant to FP8 per-tensor / NVFP4 | `quant8_kernel`, `quant4_kernel` | 1.9% | reuse kind | `static_fp8_quant`, `nvfp4_quant`, backend `tensorfold_cuda` |
| 9 | GDN conv over the tree path + SiLU + L2 norm + gates | `_gdn_pre` | 1.4% | placeholder | `elementwise` (bytes per token) |
| 10 | GDN gated RMSNorm | `_gated_norm` | 1.1% | reuse kind | `gdn_gated_rms_norm`, backend `tensorfold_triton` |
| 11 | GDN verify recurrence over tree paths | `tree_kernel` | 1.0% | placeholder this round; new kind later if Check 1 ranks it | `elementwise` (state bytes) |
| 12 | attention gate, q/k norm + RoPE, SwiGLU glue | `_gate_mul`, `_attn_prep`, `_swiglu` | 0.8% | placeholder | `elementwise` |
| 13 | in_proj_b/a bf16 pair (one launch) | `b16_kernel`, `b16_prompt_kernel` | 0.7% | reuse kind | `single_gemm` bf16, n = 96 (two n=48 halves in one launch) |
| 14 | commit GDN replay (all 48 layers in one launch) + KV/conv copies | `replay_kernel`, cat / index_select | 0.9% | placeholder | `elementwise` |
| 15 | drafter vocabulary-subset lm_head | `lane_kernel` | 0.5% | reuse kind | row 3's backend |
| 16 | drafter block attention, dconv, prep, topk | `_block_attention`, `_dconv`, … | 0.4% | placeholder | `elementwise` |
| — | sampling top-k/argmax reduce | `reduce_kernel` | 0.02% | framework plumbing | — |

## Proposed L1 build set (stop rule)

1. Rows 1, 3 and 15: `single_gemm` backends `tensorfold_lane` and `tensorfold_ck` (fp8, nvfp4).
2. Row 1b: new kind `gemm_swiglu_quant`.
3. Row 2: `flashinfer_attn_prefill` backend `tensorfold_pattn`.
4. Row 4: new kind `tree_verify_attention`.
5. Rows 5, 6, 7, 8 and 10: backends on existing kinds.

Together these cover about 93% of measured kernel time. Rows 9, 11, 12, 14 and 16 stay `elementwise` placeholders
(about 4.5%) until Phase 5's Check 1 ranks them.

All new backends run in a new `tensorfold_env` profile env (aarch64 image with TensorFold at the perf-study commit,
Torch 2.14.1+cu130), keyed `NVIDIA GB10`.

## Batched serving update (`--parallel 16`, RAG-agent replay; *measured*)

**Capture:** `logs/20261023_5_tf27b_batch/`.
- One server configured as `serve_tf_27b_nvfp4.sh` (07:02): `--parallel 16 --context 131072`, reserve 30 GiB,
  `--checkpoint-slots 32`.
- Closed-loop replay of the 85 RAG-agent conversations at 1/2/4/8/16 lanes, with a 60 s nsys window per level.
- Replayed prompts are within 1-3% of the recorded token counts.

On this path (`MultiDecoder`), every decoding stream's 16-row window is verified in ONE `multi_tree_forward`. Each
iteration runs one prefill step first: at most 1024 rows while anyone decodes, with several queued prompts batched
into one forward.

| N | decoding streams per round | verify rows per round | median ctx | median round | prefill share | verify share | prefix-cache hit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 16 | 31.8k | 106 ms | 8% | 84% | 0.99 |
| 2 | 2 | 32 | 69.6k | 141 ms | 5% | 86% | 0.51 |
| 4 | 4 | 64 | 26.6k | 159 ms | 19% | 70% | 0.85 |
| 8 | 8 | 92 | 21.7k | 204 ms | 35% | 52% | 0.81 |
| 16 | 10-11 | 114 | 31.2k | 685 ms | 63% | 30% | 0.06 |

**N=16 is a cold-start ramp, not steady state.** Each lane's first conversation starts mid-history, so it does a full
cold prefill (about 500k tokens in total), and the 240 s warm-up could not absorb it. As a result every round
carries a 1024-row prefill step, only 10-11 of the 16 streams decode, and the cache hit rate is 6%. The ramp is
valid evidence for the kernel inventory. It must be re-captured with a longer warm-up before it serves as a
steady-state alignment target.

**Kernel-set changes vs batch 1:**
- One new kernel, Triton `_append` (drafter KV append in `add_taps_streams`, 0.5-1%) → `elementwise` placeholder.
- Tree attention (`_shared` + `_tail` + `_merge`) grows to about 17% of kernel time at N=8, from 3.6% at batch 1.
  The `tree_verify_attention` lane moves to the front of batch 2. Its args must express several streams per
  launch: `query_context_pairs` (rows, committed_len) per stream, in the style of `fa4_varlen_attention`.
- Commit GDN `replay_kernel` reaches 3.6% at N=8 (48 layers × S streams in one launch). It moves from placeholder to
  a dedicated kernel in batch 3.

**GEMM M range seen:** verify M = Σ rows ≤ 16·16 = 256 (median 114 at N=16); prefill steps ≤ 1024 rows. Both are
inside the batch-1 lanes' grids.
