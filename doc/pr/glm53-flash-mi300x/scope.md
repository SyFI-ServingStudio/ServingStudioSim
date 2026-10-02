# GLM-5.3-Flash → MI300X re-targeting

Goal: profile GLM-5.3-Flash on `mi3008x` (8× AMD MI300X) end to end, re-targeting
the existing arch `glm53_flash_vllm_fp8_kda_dsa_moe` (currently pinned to NVIDIA
B200).

## Framing: this is a kernel-backend port, not a re-profile

ServingStudio Sim answers one question per element — is its timing **measured** (L1
kernel, a real GPU launch benchmarked into `profile.db`) or **composed** (L2 op /
L3 worklet / L4 arch, which only wire measured L1 timings)? Composition is
hardware-independent, so the L2–L4 structure of GLM-5.3-Flash carries over to
MI300X unchanged. What is NVIDIA-locked is the L1 layer — which kernel *backend*
is measured — plus the arch's backend/fabric/GPU pins.

## What carries over unchanged

- **The cost cache / GPU keying.** `profile.db` is keyed
  `(gpu_name, backend, args_hash)` and `gpu_name` travels per-call from each
  kernel's config; the Rust bridge is GPU-agnostic. MI300X rows coexist with B200
  rows — no schema change. Swapping `"NVIDIA B200"` → `"MI300X"` (already in
  `gpu/spec.json`) reroutes lookups.
- **Quantization numerics.** The flash checkpoint is FP8-E4M3; CDNA3 has hardware
  FP8. NVFP4 would be dead on CDNA3, but this arch does not use NVFP4 (it is a
  dormant hazard in the shared `nvfp4_fused_moe` kind, not on the active path).
- **All L2/L3/L4 composition** (ops, worklets, arch wiring math).

## The work — L1 backends (22 kernels, build largest-measured-share first)

The arch pins one backend per kernel (`arch.rs:90–114`). **No ROCm/aiter/HIP
backend exists anywhere in `profiling/` today**, and `BackendSupport.allows()`
rejects a backend whose `gpus` set omits MI300X, so every kernel needs at least a
new MI300X registration.

### Tier A — hard NVIDIA/Blackwell locks (new AMD backend from scratch)
Includes the two hottest kernels, so the expensive work is front-loaded.
- `dsa_sparse_mla_attention` — FlashInfer TRT-LLM-gen FP8 sparse MLA
- `nvfp4_fused_moe` @ `fp8_block_sm100` — TRT-LLM Blackwell fused MoE (routed hot path)
- `all_reduce_fusion` — FlashInfer MNNVL collective, 91×/iter (needs RCCL/xGMI);
  arch also hardcodes `Fabric::Nvlink`
- `deepgemm` FP8 GEMM + the two DeepGEMM `dsa_*_mqa_logits_*` indexer kernels (sm100)
- `vllm_tilelang` mHC norm kernels (`mhc_pre_rms_norm`, `mhc_fused_post_pre_rms_norm`)

### Tier B — Triton/PyTorch, portable in principle (new MI300X reg + ROCm re-profile)
`single_gemm`, `gemm_fp32_output`, `batched_gemm`, `q_kv_rms_norm`, `rms_norm`,
`gdn_causal_conv_{prefill,decode}`, `kda_chunk_prefill`, `kda_recurrent_decode`,
`dsa_sparse_index_remap`, `dsa_topk_prefill`, `dsa_persistent_topk_decode`,
`mla_cache_append`, `fp8_per_token_group_quant`, `elementwise`.

Note: the FP8 scale layout (`ue8m0_packed_int32`) is DeepGEMM/Blackwell-oriented;
an AMD GEMM/fused path uses a different scale layout, so even the "portable"
quant/GEMM kernels are not a pure re-profile.

## L4 arch change
New arch variant (do not mutate the B200 one): swap the six `*_BACKENDS` consts to
AMD equivalents, `Fabric::Nvlink` → Infinity Fabric/xGMI, `gpu_name` → MI300X,
keep TP4/EP4 (fits one 8× MI300X node). Then `impl-wire-new-arch` for the
predictor + deployment dispatch arms.

## Phase 0 is the gating prerequisite
Before any backend work: capture a real vLLM-ROCm run of GLM-5.3-Flash on
`mi3008x` (nsys, profile-only). The measured kernel inventory — not B200 source —
decides which AMD kernels exist and how they fuse. **Structural risk:** AMD's
stack may fuse the hybrid KDA+DSA+MoE path differently than FlashInfer/TRT-LLM, or
vLLM-ROCm may not implement `Glm5NextForConditionalGeneration` at all. If fusion
boundaries differ, the kernel *split* changes (new L1/L2 boundaries), a bigger job
than swapping backends. The capture is cheap (one GPU profile, reused in Phase 5)
and resolves this before committing to backend work.

## Sequencing
0. Confirm vLLM-ROCm support + capture (gates everything).
1. Explore/split against the capture; produce the per-op decision table ordered by
   measured share.
2. Build L1 backends largest-share-first → L2/L3 (mostly unchanged) → L4 variant.
3. Wire the variant (selectable + predictable; `timing-predict` green).
4. `model.work` label already exists for GLM-5.3-Flash; verify it covers the
   MI300X deployment map.
5. Validate: offline `timing-predict`, then Check-1 kernel-alignment vs the Phase 0
   capture.
