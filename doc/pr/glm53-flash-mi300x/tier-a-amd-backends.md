# Tier-A AMD backends for GLM-5.3-Flash on MI300X

Design and de-risk note for the two "Tier-A" L1 kernels of the
`glm53_flash_vllm_fp8_kda_dsa_moe` arch — the two that have no portable
torch/Triton path and need AMD-native backends written from scratch. Every
other kernel on the arch reuses the proven torch/Triton-on-ROCm pattern (see
the landed `rms_norm` and `kda_recurrent_decode` `torch_rocm` backends); these
two do not, and they dominate iteration time, so they are the long pole.

This is a scope/de-risk document for a later implementation agent. It does not
implement the backends.

## Terms

- **vLLM-ROCm plugin (`glm5next`)** — the model's vLLM implementation,
  `vllm.models.glm5next`, with a platform split: a `nvidia/` subtree and an
  `amd/` subtree, selected at runtime by `vllm.platforms.current_platform.is_rocm()`.
- **AITER** — AMD's inference kernel library (the ROCm analog of the
  CUDA/CUTLASS/FlashInfer stack). vLLM routes to it when
  `VLLM_ROCM_USE_AITER=1`.
- **rocprofv3 / rocpd** — the ROCm kernel tracer and its SQLite output. The
  repo has no `nsys` on AMD; kernel-only timing comes from rocpd kernel-dispatch
  durations via `Timer.rocprof`. See `profiling/profilers/rocprof_kernel_profiler.py`.
- **B200 pin** — the NVIDIA backend name the arch currently names for this
  kernel kind (a documented placeholder on the MI300X path until these AMD
  backends land).

## The measurement harness these backends plug into

Both backends follow the exact mechanism already proven by the landed
`rms_norm` / `kda_recurrent_decode` ROCm backends. Per kernel:

1. **Register a `KernelProfilerSpec`** in the kind's `profiling/kernels/*.py`
   with a new AMD backend name, `BackendSupport(compute=…, gpus=frozenset({"MI300X"}))`,
   and `subprocess_env="vllm_rocm_env"` (the apptainer/singularity vLLM-ROCm SIF
   registered in `profiling/exec/env.py`). MI300X-gating keeps the NVIDIA B200
   rows byte-identical; `profile.db` is keyed `(gpu_name, backend, args_hash)`,
   so MI300X rows coexist.
2. **Write a runner** `profiling/runners/<area>/<kind>_<backend>.py` exposing
   `build_<kind>_kernel(spec…) -> {"torch", "kernel", "warmup", "rep", "kernel_name", …}`
   and `profile_<kind>_<backend>(…) -> ComputeMetrics`. The `build_` function
   constructs the timed callable and its fixed-seed inputs; the `profile_`
   function runs a correctness check, then times kernel-only via
   `measure_registered_via_rocprofv3(...)`.
3. **Add the `(kind, backend)` builder** to `_BUILDERS` in
   `profiling/profilers/rocprof_run.py` so the under-tracer driver replays the
   identical kernel from the same spec.

Timing modes (from `rocprof_kernel_profiler.py`):

- **Single stable-named dispatch** → pass `kernel_name_contains="<substr>"`; the
  parser keeps only matching dispatches and drops `warmup`.
- **Compound call (several dispatches per launch)** → pass
  `kernel_name_contains=None, fold_per_launch=True`, and build operands
  host→device with `.to()` so setup emits **no** kernel dispatches; the fold
  then sums each launch's constant dispatch set (this is how KDA is timed).

Both Tier-A kernels **measure standalone, no engine needed** — the same as every
existing backend of these two kinds. Synthetic operands (selected indices for
attention, `per_expert_batches` for MoE) are constructed exactly as the B200
runners do; we do not stand up a vLLM engine to time one kernel.

---

## Kernel 1 — `dsa_sparse_mla_attention` (DSA sparse MLA attention)

Attention where each query head attends only to the DSA-indexer-selected cache
tokens. B200 pin: `flashinfer_trtllm_fp8` (`SPARSE_ATTN_BACKENDS`), FP8 E4M3
query + paged latent cache, `selected_k = 2176`, `rope_dim = 0`.
Args schema: `profiling/kernels/dsa_sparse_mla_attention.py::DsaSparseMlaAttentionArgs`.

### AMD callable to wrap

On ROCm the `glm5next` plugin's sparse-MLA / DSA path dispatches into its `amd/`
subtree (`sparse_indexer.py` has `if is_rocm(): from .amd.sparse_indexer`), which
calls AITER. The Phase-0 import probe confirmed the AMD indexer/attention routes
to **`rocm_aiter_sparse_attn_indexer`** (the indexer, a separate kind) and
**`rocm_aiter_mla_sparse`** (the sparse-MLA attention — this kernel). The KDA
precedent wrapped the plugin's own AMD callable
(`vllm.models.glm5next.amd.ops.third_party.kda.fused_recurrent_kda`) rather than
raw AITER, and this backend should wrap the plugin's AMD sparse-MLA entry point
the same way, so we measure exactly what the engine calls.

- **To wrap:** the plugin AMD sparse-MLA attention callable (dispatch name
  `rocm_aiter_mla_sparse`), underneath which sits an AITER MLA kernel
  (candidate: `aiter.mla` / `aiter.ops.mla.mla_decode_fwd` family, sparse
  variant). **Exact module path + signature are an OPEN QUESTION** — pin them
  with an inspection script that mirrors `kda/_inspect_amd_kda.py` from the
  cluster job tree: import `vllm.models.glm5next.amd`, introspect the sparse-MLA
  symbol, print `inspect.signature`, and read the `common/…` call site to copy
  the exact kwargs.

### KernelArgs → callable argument mapping (to confirm against the signature)

| Arg | Maps to |
|---|---|
| `num_queries` | query rows (decode: one per request) |
| `num_cache_tokens` | paged latent-cache length |
| `num_heads` / `num_kv_heads` | MLA query heads on-rank / 1 shared latent "KV head" |
| `selected_k` = 2176 | selected index slots per query (page-table width) |
| `latent_dim` = 512 (`kv_lora`) + `rope_dim` | compressed KV width; AMD cache layout may require `rope_dim` split out |
| `value_dim` | output width per head |
| `softmax_scale` | score multiplier |
| `q_dtype` / `cache_dtype` | FP8_E4M3 (see dtype note) |
| `index_dtype`, `valid_counts`, `index_distribution`, `cache_layout` | synthetic selected-index construction (as in the B200 runner) |

### dtype / FP8 scale-layout notes

MI300X (CDNA3) has hardware FP8 E4M3, and the flash checkpoint's FP8 maps to it
(notes.md). The scale-**layout** difference matters: the Blackwell/DeepGEMM path
packs per-128 block scales as **UE8M0 in a transposed/packed-int32** layout;
AITER expects plain FP32 per-tensor or per-token scales in a straightforward
(non-packed) layout. Because the runner **builds synthetic operands directly in
the AMD layout**, there is no cross-layout conversion at measurement time — the
conversion only exists conceptually, and the runner constructs AMD-native FP8
tensors + FP32 scales from scratch (the B200 runner likewise builds its own
layout). The real risk is dtype **support**, not conversion: if the AITER sparse
MLA kernel on MI300X supports only BF16 KV (vLLM's H200 analog is
`vllm_flashmla_bf16`), the MI300X backend's `compute` set must be `BF16`, not
`FP8_E4M3`, and the arch's `cache_dtype` on this path changes accordingly.

### Harness plug

- Register under kind `dsa_sparse_mla_attention`, new backend (e.g.
  `rocm_aiter_mla_sparse`), `gpus={MI300X}`, `compute={FP8_E4M3}` **or** `{BF16}`
  pending the dtype answer, `subprocess_env="vllm_rocm_env"`. Standalone (no
  engine). If the AITER call is a single fused dispatch, time with a
  `kernel_name_contains` substring of the aiter MLA kernel symbol; if it fans out
  into a fixed dispatch set, use `fold_per_launch=True` with host-built operands.

### Effort & risks

- **Estimated effort: 8–12 hours** (pin callable + signature via inspection;
  build synthetic FP8/BF16 paged latent cache + selected indices in AMD layout;
  correctness check vs the existing torch reference; rocprof plug + a CPU
  registration test).
- **Top 3 risks:**
  1. **FP8 unsupported** — AITER sparse MLA may accept only BF16 KV on CDNA3,
     forcing a BF16 backend and a dtype change on the arch's DSA path.
  2. **Signature/cache-layout mismatch** — the AMD paged sparse-MLA cache layout
     and `rope_dim` handling may differ from `flashinfer_trtllm_fp8` (which uses
     `rope_dim = 0` with its own layout); the synthetic cache must match what the
     AITER kernel reads, or the correctness check fails.
  3. **Dispatch fan-out** — if one logical call issues a variable (not constant)
     number of dispatches, `fold_per_launch` cannot split launches and timing
     needs a stable kernel-name filter instead; the symbol name must be found
     first.

---

## Kernel 2 — fused-MoE grouped GEMM (`nvfp4_fused_moe` @ fp8_block)

The routed MoE as one fused call: routing + gate-up GEMM + SwiGLU + down GEMM
(+ combine). B200 pin: `flashinfer_trtllm_fp8_block_sm100` (`FUSED_MOE_BACKENDS`),
registered under kind `nvfp4_fused_moe` with `weight_format = fp8_e4m3`,
`group_size = 128` (128×128 weight blocks, per-token-group-128 activations),
`routing_method = deepseek_v3`. Args schema:
`profiling/kernels/nvfp4_fused_moe.py::Nvfp4FusedMoeArgs`; B200 runner:
`profiling/runners/moe/fp8_block_fused_moe.py`.

### AMD callable to wrap

The GLM-5.3 MoE layer is a plain vLLM `FusedMoE` (not glm5next-custom), so on
ROCm with AITER enabled it dispatches through vLLM's standard ROCm AITER MoE
path:

- **To wrap:**
  `vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe.rocm_aiter_fused_experts`
  invoked with `use_fp8_w8a8=True` and `block_shape=[128, 128]`, which selects
  the AITER FP8 block-scale kernel **`rocm_aiter_fmoe_fp8_blockscale_g1u1`**
  (added in vLLM's AITER Fused-MoE V1 support). This is the direct ROCm analog
  of the B200 runner wrapping `flashinfer.fused_moe.trtllm_fp8_block_scale_moe`.

### KernelArgs → callable argument mapping

`Nvfp4FusedMoeArgs` is reused unchanged (register a new MI300X backend under the
same kind). The runner builds, per `per_expert_batches`:

| Arg | Maps to |
|---|---|
| `num_tokens`, `hidden_size`=4096, `intermediate_size`=2048 | activation / expert GEMM shapes |
| `num_experts`=288, `num_local_experts`=72 | `w1`/`w2` stacked per local expert |
| `top_k`=8 | `topk_ids` / `topk_weights` width |
| `input_dtype`=BF16 | unquantized activation + output precision |
| `weight_format`=fp8_e4m3, `group_size`=128 | `use_fp8_w8a8=True`, `block_shape=[128,128]` |
| `routing_method`=deepseek_v3, `n_group`, `topk_group`, `routed_scaling_*` | forced router logits → exact top-k ids (as B200 runner does) |
| `per_expert_batches` | synthetic token-to-expert assignment; drives `topk_ids` so each expert gets its count |

### dtype / FP8 scale-layout notes

The B200 runner shuffles weights into FlashInfer **W31 BlockMajorK** layout
(`swap_w13_to_w31` + `_shuffle_deepseek_fp8_moe_weights`) with 128×128 FP32 block
scales and per-token-group-128 FP32 activation scales transposed to `[H/128, T]`.
**AITER expects a different weight pre-shuffle and plain 128×128 FP32 block
scales**, not the FlashInfer BlockMajorK / UE8M0-packed layout. The conversion
lives in the AMD runner's operand construction: build FP8 weights, run AITER's
own weight-prep (`aiter.shuffle_weight` / the prep inside
`rocm_aiter_fused_experts`), and pass plain FP32 block scales — mirroring how the
B200 runner builds FlashInfer-layout operands. No tensor crosses from one layout
to the other at runtime; each backend builds its own.

### AITER path required — and the fallback hazard (central MoE question)

- **`VLLM_ROCM_USE_AITER=1` is required** (master switch; the cluster capture
  scripts set it). `VLLM_ROCM_USE_AITER_MOE` defaults to True **when the master
  is on**, so no extra flag is strictly needed — but the prior diagnosis found
  vLLM silently falling back to the **Triton FP8-MoE** path in some
  configurations. The runner must therefore (a) set both envs explicitly before
  importing vLLM, and (b) **assert the AITER path was actually selected** by
  checking the captured dispatch carries the aiter `fmoe`/`g1u1` kernel symbol,
  not a Triton `fused_moe` kernel name. Timing the wrong path would silently
  measure Triton and mislabel it AITER.
- **Which path is production?** The capture runs with AITER on, so **if AITER has
  an FP8-block MoE kernel for these dims (288 experts, 72 local, top-8, hidden
  4096, inter 2048, group 128), AITER is the one to measure.** If AITER lacks
  FP8-block coverage for this shape and the engine really falls back to Triton,
  then the Triton FP8-MoE path IS production and should be the measured backend
  instead. **This must be resolved from the Phase-0 / reduced-depth capture's
  kernel-dispatch table before implementation** (OPEN QUESTION).

### token_corpus pass — NOT required for this backend

The L1 fused-MoE backend measures a **given** `per_expert_batches` tuple passed
in its args; it does not read a `token_corpus_file` (the B200
`fp8_block_fused_moe` runner reads `args["per_expert_batches"]` directly and
builds routing from it). The grouped-GEMM cost does depend on routing skew, but
that skew enters at the **arch/worklet** level through `ExpertDemand`
(`simulator/src/timing/expert_demand.rs`, consumed by
`simulator/src/worklet/nvfp4_moe_local.rs`), which is what the top-add-new-arch
"second `token_corpus` pass" feeds. That routing-evidence pass is
backend-agnostic, already exists for B200, and is owned by the arch/alignment
agents — it is **not** new work for this AMD backend. In short: the AMD MoE
backend needs no `token_corpus`; the arch's MoE composition needs realistic
`per_expert_batches` (unchanged from B200).

### Harness plug

- Register under kind `nvfp4_fused_moe`, new backend (e.g. `rocm_aiter_fp8_block`),
  `gpus={MI300X}`, `compute={FP8_E4M3}`, `subprocess_env="vllm_rocm_env"`.
  Standalone (no engine). The fused call is a compound of several dispatches
  (routing, FC1, activation, FC2, finalize); time with `fold_per_launch=True`
  and host-built operands, matching how B200 counts overlapping launches once —
  **but** keep a stable aiter kernel-name filter available for the
  path-selection assertion above.

### Effort & risks

- **Estimated effort: 10–14 hours** (weight FP8 quant + AITER pre-shuffle; FP8
  block-scale activation/weight-scale operands; forced-logits exact top-k with a
  realistic `per_expert_batches`; AITER-vs-Triton selection assertion;
  correctness check; rocprof plug; CPU registration test). **+4–6 hours if the
  production path turns out to be Triton** (different operand prep, a second
  backend).
- **Top 3 risks:**
  1. **Silent Triton fallback** — measuring Triton while labeling it AITER, or
     discovering AITER has no FP8-block kernel for these dims so production is
     actually Triton. Gates which backend to build.
  2. **Weight pre-shuffle / block-scale layout** — getting AITER's weight layout
     and 128×128 FP32 block-scale arrangement wrong fails the correctness check;
     the aiter prep API differs from vLLM's FlashInfer prep used on B200.
  3. **Autotune / warmup noise** — AITER/CK MoE kernels autotune per process and
     autotune selections are not cached; without enough warmup the rocprofv3
     capture may include a tuning launch. Build warmup to cover tuning, as the
     B200 runner does (3 warm-up calls after autotune).

---

## OPEN QUESTIONS

1. **Sparse-MLA exact callable + signature.** The precise module path and
   `inspect.signature` of the plugin AMD sparse-MLA attention entry point
   (dispatch `rocm_aiter_mla_sparse`) and the underlying AITER MLA callable.
   Resolve with an inspection script mirroring `kda/_inspect_amd_kda.py` on a
   devel slice.
2. **Sparse-MLA FP8 vs BF16.** Does AITER sparse MLA on MI300X accept FP8 E4M3
   KV, or only BF16? Sets the backend's `compute` set and the arch's DSA
   `cache_dtype`.
3. **Sparse-MLA cache layout + rope.** Does the AMD paged latent cache layout and
   `rope_dim` handling match the `flashinfer_trtllm_fp8` convention
   (`rope_dim = 0`, `selected_k ∈ {2048, 2176}`), or need a different synthetic
   cache?
4. **MoE: AITER FP8-block coverage for these dims.** Does
   `rocm_aiter_fmoe_fp8_blockscale_g1u1` support 288 experts / 72 local / top-8 /
   hidden 4096 / inter 2048 / group 128 — or does the engine fall back to Triton?
   Decide from the Phase-0 / reduced capture dispatch table. This picks the MoE
   backend to build.
5. **MoE: exact AITER weight-prep + block-scale API.** The precise
   `aiter.shuffle_weight` / `rocm_aiter_fused_experts` pre-shuffle and the FP8
   128×128 block-scale tensor shapes AITER expects.
6. **MoE: path-selection assertion signal.** The aiter kernel symbol substring
   (and the Triton one) used to prove which path ran, for the runner's assertion.
7. **rocprofv3 in-process capture.** `Timer.rocprof`'s in-process collector is
   not yet GPU-verified; both backends rely on the
   `measure_registered_via_rocprofv3` subprocess path (which is wired). Confirm
   on the first real MI300X run that the dispatch fold is exact for each compound
   call (constant dispatches per launch).
8. **all_reduce / collective.** Out of scope here but adjacent: these two kernels
   are measured standalone; their composed cost still rides the arch's AMD
   Infinity-Fabric all-reduce, a separate Tier-A item.
