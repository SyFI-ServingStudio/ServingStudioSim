# Layers: per-layer alignment

The second milestone, after the kernels of `2_kernels.md` pass. One layer of each type runs on one GPU, with its
kernels back to back, and its time is compared with the layer's target. This checks what the kernel micro-benchmark
cannot: the kernels at the exact sizes one layer calls, the tensors handed from one kernel to the next, and the glue
ops between them. Nothing here needs the pipeline, the scheduler or the cache.

## 1. The workload

The one-request bring-up of `1_contract.md` §2.1: for each L in 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k, one prompt of
L fresh tokens, no prefix, the whole prompt as one microbatch. A layer sees all L tokens in one call.

## 2. Layer types

| Layer type | Layers | What it holds |
|---|---|---|
| `first_kda_dense` | 0 | mHC pre-norm (TileLang), KDA attention, mHC boundary, dense NVFP4 FFN |
| `kda_dense` | 1–2 | mHC boundary, KDA attention, mHC boundary, dense NVFP4 FFN |
| `kda_moe` | the other KDA layers | mHC boundary, KDA attention, mHC boundary, MoE (router, routed experts, shared expert) |
| `dsa_moe` | 3, 7, 11, …, 43 | mHC boundary, DSA indexer, sparse MLA, mHC boundary, MoE |
| `input_head` | stage 0 | embedding, expand to the 4 mHC streams |
| `output_head` | stage 3 | final mHC post, average of the 4 streams, final RMSNorm, `lm_head` on the last token |

Every layer but layer 0 opens with an mHC boundary: the previous block's post fused with this block's pre-norm. The
stages:

| Stage | Layers | Composition |
|---|---|---|
| 0 | 0–11 | `input_head` + `first_kda_dense` + 2 × `kda_dense` + 3 × `dsa_moe` + 6 × `kda_moe` |
| 1 | 12–22 | 2 × `dsa_moe` + 9 × `kda_moe` |
| 2 | 23–33 | 3 × `dsa_moe` + 8 × `kda_moe` |
| 3 | 34–44 | 3 × `dsa_moe` + 8 × `kda_moe` + `output_head` |

## 3. Targets

Time of one layer, in ms:

| Layer type | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|---|
| `first_kda_dense` | 0.523 | 0.949 | 1.852 | 3.671 | 7.433 | 15.080 | 30.112 | 60.165 |
| `kda_dense` | 0.527 | 0.962 | 1.879 | 3.733 | 7.558 | 15.321 | 30.602 | 61.151 |
| `kda_moe` | 1.173 | 1.634 | 2.651 | 4.712 | 8.922 | 17.384 | 33.878 | 66.856 |
| `dsa_moe` | 1.232 | 1.776 | 3.128 | 5.863 | 11.821 | 23.324 | 45.291 | 95.448 |
| `input_head` | 0.014 | 0.023 | 0.044 | 0.090 | 0.185 | 0.377 | 0.761 | 1.530 |
| `output_head` | 0.227 | 0.247 | 0.289 | 0.373 | 0.538 | 0.863 | 1.481 | 2.716 |

Each stage's compute time in the README is the sum of its layers by §2. At 8k, stage 1 is 2 × 5.863 + 9 × 4.712 =
54.1 ms.

**The cost tree of each layer type** is in `data/layers/<layer type>.txt`. It is how a layer's time is built from its
kernels, with every node's time at each prompt length:
- `Sum` adds its children; `Scale{n=k}` runs its child k times (17 small index launches in a DSA layer, 8 prefill
  index launches, 2 gathers).
- `Max{overlap}` and `Parallel{overlap}` take their slowest child over the overlap. In the MoE, the `Parallel` branch
  overlaps the shared expert with the routed experts and applies only at ≤ 256 tokens; from 1k on, the routed experts
  and then the shared expert run one after the other.
- A row of `-` is a branch this workload never takes: decode kernels, and the indexer at 1k and 2k, where the whole
  context fits in the 2,048-token top-k.
- `Scale{n=0}` is an op the chosen kernels do not need: the KDA beta sigmoid, which FlashInfer's KDA prefill applies
  itself because it takes the beta logits.

An excerpt of `dsa_moe.txt`, ms:

```
                                                                     1k       8k      64k     128k
Sum                                                               1.232    5.863   45.291   95.448
  - pp.dsa_moe.attn_mhc_post_pre  [mhc_fused_post_pre_rms_norm]   0.027    0.178    1.387    2.792
  Sum  pp.dsa_moe.dsa                                             0.448    3.910   32.911   71.051
    ...
    - pp.dsa_moe.dsa.indexer.mqa_logits_prefill                       -    0.146    2.374    8.801
    - pp.dsa_moe.dsa.indexer.topk_prefill                             -    0.129    1.523    4.635
    ...
    - pp.dsa_moe.dsa.o_proj  [single_gemm]                        0.101    0.807    6.173   12.193
    Scale{n=8}                                                    0.031    0.140    1.280    2.583
      - pp.dsa_moe.dsa.prefill_glue  [elementwise]                0.004    0.018    0.160    0.323
  - pp.dsa_moe.ffn_mhc_post_pre  [mhc_fused_post_pre_rms_norm]   0.027    0.178    1.387    2.792
  Sum  pp.dsa_moe.moe (MoE)                                       0.732    1.598    9.605   18.812
    ...
```

`data/layer_targets.jsonl` holds one line per (prompt length, stage, layer type):

```json
{"prompt_tokens": 8192, "stage": 1, "layer_type": "dsa_moe", "layers": [15, 19], "count": 2, "layer_ms": 5.863,
 "kernels": [{"position": "pp.dsa_moe.attn_mhc_post_pre", "kind": "mhc_fused_post_pre_rms_norm",
              "backend": "deepgemm_mega_nonshifted", "call": {"num_tokens": 8192}, "launches": 1,
              "time_ms": 0.178, "total_ms": 0.178, "beyond_rows": false}, "..."],
 "moe_per_expert_batches": [2168, "...", 15]}
```

- `kernels` lists every kernel with time in the layer, in the order of the layer's graph (the small index glue of a
  DSA layer comes last): its position, library (kind and backend), the sizes of the call, how many times the layer
  launches it, and its target time per launch and in total at exactly those sizes. The totals add up to `layer_ms`.
  A layer that misses its target can be traced to the kernel that grew.
- `beyond_rows` marks a call larger than every profiled row of its kernel (§6).
- `moe_per_expert_batches` is the routing to force (§4).
- To regenerate the file: `data/layer_targets.py`, after `data/kernel_specs.py` (usage in its docstring).

## 4. How to run one layer

- **Weights.** The layer's real weights from the checkpoint. The layers of one type have the same shapes, so any one
  layer of the type will do.
- **Input.** L tokens of enwik9 text (`1_contract.md` §2.2), embedded and expanded to the 4 mHC streams.
- **Routing.** Force the router's choice so that the 288 experts receive exactly the counts in
  `moe_per_expert_batches`: expert i of the list gets the i-th count, and every token picks 8 distinct experts. The
  targets use these counts for every MoE layer. `profiling/runners/moe/exact_topk.py` builds such an assignment. Run
  each MoE layer a second time with its real router on the same text, and record that time beside the forced one.
- **State.** DSA layers write their KV for the L tokens into the cache, as in serving. KDA layers start from a zero
  state.
- **Launch.** The layer's kernels back to back, with no host gaps: capture the layer in a CUDA graph, or launch it
  the way the serving code will. Do not flush L2 between kernels.
- **Time.** GPU time from the start of the layer's first kernel to the end of its last, from CUPTI or Nsight Systems.
  Warm up, then average at least 10 runs. Keep the per-kernel times of the trace too.
- Alone on the GPU (`1_contract.md` §5).

## 5. Pass rule

A layer type passes at a prompt length when its time is within ±10% of the target.
- **Slower by more than 10%:** an implementation gap. Compare the per-kernel times with `kernels` to find the kernel
  that grew.
- **Faster by more than 10%:** not a failure, but record it.

**Stage sums.** Add the measured layer times by §2 and compare each stage with the README's one-request stage time.
The same ±10% applies.

## 6. Where these targets may be off

- **Past the largest row.** Calls larger than every profiled row are extrapolated: KDA chunked prefill from 32k
  tokens (about 20% of a KDA layer's time), and at 128k most kernels (54% of a DSA layer's time, 92–93% of a KDA
  layer's). The real times at these sizes are the first measurement of them.
- **Sparse MLA past 32,768 rows.** TRTLLM's sparse MLA rejects a 65,536-query launch on B200, and 32,768 is the
  largest measured launch. The target therefore splits at 32,768 rows: 2 launches at 64k (11.7 ms) and 4 at 128k
  (23.4 ms). An implementation that splits elsewhere has a different time.
- **Warm L2.** The kernel rows were measured with a cold L2. In a layer, each kernel finds part of its input in L2,
  so the real layer can be faster, most at short prompts.
- **Glue ops** are priced, not measured (`2_kernels.md` §6).
- **MoE at 128k.** No profiled row reaches 131,072 tokens. The file gives twice the 64k counts, which keeps the
  64k routing proportions.

## 7. What to record

**Every layer run** (`measured/layers.jsonl`), one line per (prompt length, layer type, routing):

```json
{"prompt_tokens": 8192, "layer_type": "dsa_moe", "layer": 15, "routing": "forced", "real_ms": 5.91,
 "target_ms": 5.863, "ratio": 1.008, "runs": 10, "kernels": [{"position": "pp.dsa_moe.attn_mhc_post_pre",
 "gpu_kernels": ["..."], "real_ms": 0.181, "target_ms": 0.178}, "..."]}
```

- `routing` is `forced` or `router` (§4). `ratio` is real ÷ target.
- `kernels` maps the trace's GPU kernels to the positions of `data/layer_targets.jsonl`, each with its real time.
  Kernels that have no position (extra copies, casts) are listed under position `null`.

**Summary** (`measured/layers.md`):
- The ratio of every layer type at every prompt length, forced routing, laid out like the table in §3.
- The same for the router runs, and the forced ÷ router time of each MoE layer type.
- The stage sums of §5 beside the README stage times.
- For every ratio above 1.10: the kernels whose time grew the most, and the suspected cause.
- For the sizes past the largest row (§6): the real per-kernel times, so the targets there can be corrected.

**Gaps** (`issues.md`): every layer type and length above 1.10, with the kernels that grew.
