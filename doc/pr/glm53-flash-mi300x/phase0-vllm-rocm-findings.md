# Phase 0 findings — vLLM-ROCm support for GLM-5.3-Flash on MI300X

Gating question: can we actually run the `zai-org/GLM-5.3-Flash` checkpoint
(HF arch `Glm5NextForConditionalGeneration`; 320B-total / 18B-active FP8 MoE with
a hybrid KDA linear-attention + KPool-DSA sparse-MLA stack) on vLLM-ROCm on AMD
MI300X (CDNA3, `gfx942`) and capture an nsys profile? This decides go/no-go for
the capture that gates all downstream kernel-backend porting.

All evidence below is from offline inspection: the vendored NVIDIA-side vLLM fork
in this repo, and the pre-built vLLM ROCm image already staged on the cluster
(inspected with `apptainer exec` on the shared login node — no GPU time spent).

## Task 1 — vLLM model support

**Verdict: supported, on both the NVIDIA fork and the cluster ROCm image.**

- **Vendored NVIDIA fork.** `alignment/profiler/vllm` is a git submodule pinned to
  `serendipity-zk/vllm` @ `3f667d7` (branch dated 2026-09-23; the maintained
  "ServingStudio alignment" line, based on the vLLM ~v0.28 series). It is not
  checked out locally, so it was read via the GitHub API. Its model registry
  (`vllm/model_executor/models/registry.py`) registers the arch, but maps it to an
  **out-of-tree plugin namespace**, not an in-tree model file:
  - `Glm5NextForConditionalGeneration` -> `("vllm.models.glm5next", "Glm5NextForConditionalGeneration")`
  - `Glm5NextForCausalLM` -> `("vllm.models.glm5next", "Glm5NextForCausalLM")`
  - `Glm5NextMTPModel` -> `("vllm.models.glm5next", "Glm5NextMTP")`
  - `GlmMoeDsaForCausalLM` -> `("vllm.models.deepseek_v32", "GlmMoeDsaForCausalLM")`

  There is no `glm5next.py` under `vllm/model_executor/models/`; the implementation
  ships as a separate `vllm.models.*` package.

- **Cluster vLLM ROCm image.** The image reports vLLM `0.3.1.dev190+g3df4ae153`
  (this is the fork line's own version numbering, not upstream vLLM 0.x; it tracks
  the same v0.28-era alignment branch), on torch `2.12.0` / HIP `7.2` (ROCm 7.2).
  `current_platform` resolves to `rocm` inside the image. Crucially, the image
  **already bundles the GLM-5.3-Flash plugin** — `vllm.models.glm5next` is present
  and `ModelRegistry.get_supported_archs()` returns:
  `GlmMoeDsaForCausalLM, Glm5NextForCausalLM, Glm5NextForConditionalGeneration, Glm5NextMTPModel`.
  So the exact arch id for this checkpoint is registered and importable on ROCm.

- **Upstream status.** This is a brand-new architecture carried on a private
  alignment fork + out-of-tree plugin, not mainline upstream vLLM. We do not need
  upstream: the cluster image already has the plugin and registers the arch.

## Task 2 — checkpoint availability

**Verdict: absent from the cluster HF cache. Do not download in this phase.**

- The cluster HF caches hold only `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.5-0.8B` (one
  cache) and unrelated project assets; no `GLM-5*` blob or snapshot anywhere under
  `$WORK`.
- Size vs quota: the FP8 weights are ~328 GB. `$WORK` is a 2 TB project quota with
  ~296 GB currently used (~1.4 TB headroom), on a filesystem with ample free space.
  The checkpoint **fits** in quota comfortably; the constraint is the Phase 0 rule
  not to start a multi-hundred-GB transfer, not capacity.

## Task 3 — MI300X kernel reality

**Verdict: a real CDNA3 execution path exists for the hybrid stack; it is not
CUDA/Blackwell-only.** The plugin is explicitly dual-backend. Under
`vllm/models/glm5next/` the image ships `common/`, `nvidia/`, and a dedicated
**`amd/`** subtree, with platform dispatch wired through `current_platform`:

- **KDA linear-attention.** `common/kda.py` dispatches on platform:
  `if current_platform.is_rocm(): from ...amd.ops.third_party.kda ... else ...nvidia...`.
  The AMD KDA kernels (`amd/ops/third_party/kda/kernels.py`, `fused_recurrent.py`)
  are Triton, ported from flash-linear-attention, with explicit `is_amd` autotune
  branches (e.g. `num_warps` 2/4/8/16 on AMD vs 4/8/16/32 on NVIDIA). This is a
  real ROCm-Triton path, not a CUDA stub.
- **KPool-DSA sparse-MLA indexer.** `amd/sparse_indexer.py` imports
  `from vllm._aiter_ops import rocm_aiter_ops` and the `torch.ops._C` custom ops —
  an aiter-backed AMD indexer, alongside `amd/ops/kpool_compress.py`.
- **FP8 fused MoE / GEMM.** The model routes the MoE through vLLM's standard
  `vllm.model_executor.layers.fused_moe`, which on ROCm resolves to vLLM's own
  ROCm fused-MoE path (aiter / Triton). The checkpoint is FP8-E4M3 block-scaled;
  weight loading dequantizes FP8 blocks as needed. This is the engine's portable
  quant path, not the Blackwell DeepGEMM/TRT-LLM-gen path the NVIDIA arch pins.
- **Attention backend.** The multimodal/attention path references
  `AttentionBackendEnum.ROCM_AITER_FA` — an explicit ROCm aiter FlashAttention
  backend.
- **aiter** is installed in the image and its core module (`module_aiter_core.so`)
  loads; the nodes are ROCm 7.2 / HIP 7.2.

Implication for the downstream split: AMD does **not** reproduce the FlashInfer /
TRT-LLM-gen fusion. KDA and the DSA indexer are separate aiter/Triton launches,
and the MoE goes through vLLM's generic ROCm fused-MoE rather than a single
Blackwell fused kernel. The L1 kernel inventory and some fusion boundaries will
differ from the B200 arch — exactly the structural risk Phase 0 was meant to
surface. This is tractable (dedicated AMD code exists) but is more than a backend
swap for the fused-MoE and sparse-MLA boundaries.

**Caveats to confirm at capture time (potential, not proven, blockers):**

- `common/attention.py` imports `fwht128_quant_fp8` unconditionally from the
  `nvidia.ops.kpool_compress` module (not platform-gated), even though an
  `amd/ops/kpool_compress.py` exists. If that specific helper is CUDA-only it would
  fault on the hot attention path; if it is Triton it is fine. Verify at init.
- `common/mtp.py` imports `fused_eh_norm` unconditionally from `nvidia.ops`. This
  is only on the MTP / speculative-decode path, which the target arch runs with MTP
  off, so it is off the hot path — but confirm engine init does not import-fault.

## Task 4 — go/no-go for the nsys capture

The software stack supports the model on MI300X (Task 1 + Task 3). The only
missing ingredient is the weights (Task 2), which Phase 0 explicitly must not
download. Registration and the AMD execution path are already proven offline from
the image, so a GPU job that only runs `--help` or a weightless registry check
would add no evidence. A real forward/nsys capture needs the 328 GB checkpoint.
Separately, the 8×MI300X partition is currently saturated (0 idle nodes, ~30 jobs
queued), so a speculative job would also sit in queue.

**Decision: do NOT submit a GPU job in Phase 0.** No allocation was spent.

### Verdict: **GO-WITH-CAVEATS**

The nsys capture is viable. vLLM-ROCm registers `Glm5NextForConditionalGeneration`
and ships a dedicated AMD (aiter + ROCm-Triton) path for the hybrid KDA + DSA +
FP8-MoE stack, on a ROCm 7.2 image with aiter available. Proceed to the capture
once weights are staged, and treat the two unconditional `nvidia.ops` imports in
`common/attention.py` (`fwht128_quant_fp8`) and `common/mtp.py` (`fused_eh_norm`)
as the first things to watch at engine init.

### Smallest action to unblock

1. Stage `zai-org/GLM-5.3-Flash` (~328 GB FP8) into the `$WORK` HF cache (fits in
   quota). This is the one gating prerequisite.
2. Then submit one short 8×MI300X job (whole-node; TP4/EP4 fits): offline weight
   load + minimal decode of a few tokens under nsys, `HF_HUB_OFFLINE=1`, bounded
   runtime — not a full serving load. If engine init clears the two `nvidia.ops`
   imports above, the capture is unobstructed.
