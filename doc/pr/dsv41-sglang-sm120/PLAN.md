# DeepSeek-V4.1-Flash on SGLang, 4x RTX PRO 6000 (SM120): plan and data-collection checklist

Goal: simulate DeepSeek-V4.1-Flash(-1017) served by SGLang TP4/EP4 with DSpark on 4x RTX PRO 6000 Blackwell Server
Edition (SM120, PCIe), aligned to measured runs, so that the Decycle presets can be predicted (typical and busy day).

What already exists (from the 2026-11-05 measurement, airbox-cost-calculator
`docs/clients/decycle/sim/kt_bench/`, branch `decycle-gpu-benchmarks`):
- **Working server config:** stock SGLang v0.5.23 (Docker `lmsysorg/sglang:v0.5.23-cu130`), with the SM120 patch to
  the V4.1 candidate indexer (`serve_sglang_4xpro6000.sh`).
- **Measured e2e ground truth:** Decycle typical busiest hour and busy hour-13 windows (per-request jsonl and the
  per-batch server log).
- **Fixed-shape torch-profiler traces** on all 4 ranks (1.5 GB, on kisnis only). They give the SM120 kernel
  inventory and op order per layer; see `servingstudio_sm120_gap.md`.
- **NCCL collective sweep** on a VM without GPU P2P.

Decision: a **separate arch, `deepseek_v41_sglang`**, following the GLM-5.2 pattern
(`glm52_sglang_nvfp4_tp_dsa_moe` beside `glm52_vllm_*`). The vLLM graph cannot be reused, because the op structure
differs: hc mixing is unfused inside the layer, the all-reduce is plain NCCL and unfused, `wo_a` is a partial plus a
reduce, Engram is read from a host table, and there is the DSpark draft and sampling chain. The arch keeps a
`KernelPath` (SM120 now, SM100 later) for SGLang on B200.

## Phase A: before renting (no SM120 GPU needed)

1. **Port the alignment instrumentation onto SGLang v0.5.23.**
   - The fork `alignment/profiler/sglang` (`vibesim-alignment`, 449cb04a5, 2026-09-02) has DeepSeek-V4 and DSpark, but
     no V4.1 Engram host table (and probably no candidate indexer).
   - Rebase its alignment commits onto v0.5.23 as `vibesim-alignment-v0.5.23`, and add the SM120 indexer fallback as
     a real code change. The commits provide the `sglang_iteration(N): <phase>` NVTX ranges, request timing,
     expert-load and worker-device records.
   - Ship it as a thin layer over `lmsysorg/sglang:v0.5.23-cu130`: `pip install -e` of the fork's `python/` into the
     image, then `docker commit`. Push the image or keep it as a tarball. **Pre-building it saves about 30 min of
     rental time.**
   - Pin the runtime contract (`python_runtime`) as the GLM-5.2 SGLang pack does.
2. **Write the arch skeleton `deepseek_v41_sglang`** with SM120 backends declared:
   - **reused:** `DeepseekV41ModelCfg`, layer typing / `fold_layers`, `ExpertDemand`, the TP/EP shape math;
   - **new worklets, with the op order from the traces:**
     - `sglang_dsv41_attention_tp`: hc pre, fused q/kv-a FP8 GEMM, RMSNorm, fused k-norm + RoPE, wq_b, RoPE, page
       metadata, `sparse_mla_sm120` decode / `mg_dual` prefill, inverse RoPE, wo_a partial + reduce, wo_b,
       all-reduce;
     - `sglang_dsv41_moe_ep`: hc pre, router `tiny_n_gemm` + Triton top-k, shared expert FP8, routed
       `cutlass_fused_moe` MXFP4 as one op, shared down, all-reduce;
     - Engram from the host table;
     - head: hc collapse, norm, lm_head, logits all-gather;
     - DSpark: the draft forward plus the sampling chain, which is 6 x (`_markov_sample` + all-gather + broadcast);
   - SGLang DSV4 KV accounting: full, swa (cap mode), c4 and c128 pools;
   - presets with `worker: speculative` and `batch_policy: separate-prefill-priority`.

   Then run the Decycle and quadrant presets with `count_missing_*` (no GPU). **That dry run is the exact list of
   L1 rows to fill on the rental.**
3. **Write L1 runners for the SM120 kernels** (`profiling/runners/...`, each with a reference check):

   | Kind / op | Backend to add | Notes |
   |---|---|---|
   | dense FP8 block GEMM | `sglang_triton_w8a8_block_fp8` (+ `per_token_group_quant_8bit_v2`) | all q/kv/o/shared/indexer linears |
   | routed MoE MXFP4 | `flashinfer_cutlass_mxfp4_sm120` (whole `cutlass_fused_moe` call: quant, prefix sums, expand, 2 grouped GEMMs, activation, finalize) | EP4 shard, routed demand |
   | sparse MLA | `flashinfer_sparse_mla_sm120` decode (+ merge) and `sparse_mla_prefill_mg_dual` | V4.1 compress ratios 1/2/4/128, candidate layers |
   | V4.1 indexer (dense blocks) | `sglang_fp4_index_logits` + torch top-k | the patched path |
   | hc / mhc | SGLang Triton / tilelang `_hc_mix_*`, `_hc_combine_norm`, `mhc_post`, `_mhc_post_split_h` | not the vLLM mega kernels |
   | Engram | `sglang_engram_host_gather` (pinned host table over PCIe) | depends on PCIe and host memory |
   | norm / RoPE / act | FlashInfer CuTe RMSNorm, `deepseek_rope`, `fused_k_norm_rope_flashmla`, `silu_mul_clamp` | small |
   | DSpark sampling | `_markov_sample_partial/finish` + collectives | per draft token |
   | collectives | existing `comm/nccl.py`: all_reduce, all_gather, broadcast at 4 ranks | rows need a topology identity (P2P vs SHM) |

   Wherever the shapes allow, make the runners single-GPU. All non-comm kernels run at TP4/EP4 shard shapes on one
   card.
4. Add RTX PRO 6000 Blackwell Server Edition to `gpu/spec.json` (12.0, 96 GB GDDR7, 1.8 TB/s, PCIe Gen5 x16, 600 W).
5. **Campaign preset `presets/alignment/dsv41_flash_rtxpro6000_sglang_tp4_ep4`**, modeled on
   `glm52_nvfp4_b200_sglang_tp4` and the B200 `dsv41_flash_b200_tp4_ep4` labels:
   - passes: `token_corpus` (routes), `workload_metrics` (DSpark acceptance per position), an nsys capture
     (`cuda_profiler_api`, bounded), and e2e;
   - workloads: the B200 capture's quadrant set (128/128, 128/4096, 4096/128, 3072/3072) for comparability;
     `diverse_100` real text for routing and acceptance; and the Decycle session trace for e2e.

## Phase B: rentals

**B1: one RTX PRO 6000, 2-3 h (about $2-5/h).** Runner bring-up and single-GPU L1 fills.
- Use RunPod or Massed Compute 1x. No model download is needed (L1 runners use synthetic tensors).
- Run each new runner's reference check, then JIT-fill every non-comm row from the Phase A dry run.
- Save `profile.db`, runner logs and versions (torch, flashinfer, sgl-kernel, CUDA, driver).

**B2: 4x RTX PRO 6000, 3-4 h ($8.76/h on Massed Compute; a bare-metal box with GPU P2P preferred).** Comm rows,
alignment captures, validation.

| Step | Time | Output |
|---|---|---|
| boot, `nvidia-smi topo -m`, P2P check (`p2pBandwidthLatencyTest` or NCCL `NCCL_P2P_LEVEL` probe) | 10 min | provenance, topology identity |
| model download in the background (`snapshot_download`, `huggingface_hub<0.36`, about 200 MB/s, 45 min) **while** comm fills run | 45 min | 510 GB checkpoint |
| comm L1 fills (`comm/nccl.py`: all_reduce, all_gather, broadcast; the dry-run shapes) | 15-20 min | comm rows keyed by topology |
| start the instrumented server (pre-built image), check `/get_server_info` | 10 min | server config |
| campaign `token_corpus` pass, `diverse_100` | 15 min | routing corpus |
| campaign `workload_metrics` pass with DSpark, real text | 15 min | acceptance per position |
| campaign nsys capture (quadrant, bounded, warmup on) | 20 min | `.nsys-rep` + iteration and request records |
| campaign e2e (quadrant at saturation) | 15 min | e2e ground truth |
| optional: Decycle typical replay with `--chunked-prefill-size 1024`, and DSpark off for A/B | 30 min | TPOT tuning evidence |
| copy everything, delete the instance, verify deletion | 10 min | - |

To save money: if the provider has persistent storage, pre-stage the checkpoint (for example a RunPod network
volume). That removes the 45 min download.

Record per run: engine commit and dirty state, image id, full server command and env, client command, nsys version
and flags, GPU / driver / CUDA, P2P state, NCCL transport line (`via P2P` / `via SHM`).

## Phase C: after the rentals

1. Merge `profile.db` by row: SM120 rows under the RTX PRO 6000 GPU name; don't merge the binary file through git.
2. Run `alignment analyze` kernel-align, which gives `gpu_time_multiplier`, then the simulation and e2e-align.
3. Validate on the Decycle windows measured on 2026-11-05:
   - typical: TTFT p90 2.0 / 14.0 s, TPOT p99 135 ms;
   - busy dense: 19.6 / 34.2 s, 121 ms.
4. Run the Decycle typical and busy presets for 4 and 8 cards; update the client cost comparison.

## Open risks

- Porting the fork onto v0.5.23 (SGLang moves fast). Fallback: stock v0.5.23 plus the nsys capture, with
  iteration boundaries recovered from the scheduler log. That is less precise.
- The SM120 FlashInfer kernels need the exact FlashInfer wheels in the image. Pin `flashinfer-python`, `-cubin` and
  `-jit-cache` from the v0.5.23 image.
- DSpark acceptance depends on the content. Use real text for acceptance and the token corpus; random token ids
  inflated it to 5.2-5.9.
- VM versus bare metal: without P2P, all-reduce is 47% of prefill. Profile the topology you intend to recommend, or
  both.
