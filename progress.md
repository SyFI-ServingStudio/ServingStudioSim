# DeepSeek V4 Clean L1 Rebuild Progress

This worktree rebuilds the retained DeepSeek V4 L1 support from `master`
instead of copying the previous branch. The goal is a small, source-grounded
implementation with no vendored native kernels, duplicate kinds, or tests that
only restate implementation details.

## Scope And Baseline

- Worktree: `wt-deepseek-l1-clean`
- Branch: `deepseek-l1-clean`
- Baseline: `master` at `4496493`
- Current phase: clean L1-L4 implementation, canonical profile fill, kernel
  alignment, and predictive E2E alignment are complete; final scoped
  validation and change-set audit remain.
- Validation-only smokes use databases under `$TMPDIR`. The alignment phase now
  also fills this worktree's shared `profiling/profile.db`; it is marked
  `skip-worktree`, so successful rows must be copied explicitly into the
  eventual integration checkout rather than inferred from `git status`.

## Frozen Decisions

### Clean L3/L4 composition (2026-08-24)

- DeepSeek uses the ordinary `hp_unified` worker for unbounded/soft-budget runs
  and the generic `chunked_prefill` recipe when a hard prompt-token cap is
  requested. No EqualShard worker, equal request-shape admission rule, trace
  rewrite, observed-rank replay, or model-specific scheduler was introduced.
- `chunked_prefill` reserves one request's full KV footprint and carries its
  continuation outside the pending policy. A 16,384-token prompt under the
  8,192 cap is rendered as `(prefix, append) = (0,8192)` then `(8192,8192)`;
  first-token completion occurs only after the second iteration.
- The execution adapter carries the exact token count of each existing
  attention-DP partition for every multi-partition model. DeepSeek consumes
  that ragged vector for EP collectives; no model-specific worker hook exists.
- Both DeepSeek selectors use the shared Qwen/GLM expert-popularity loader.
  That master-owned loader canonicalizes each layer by descending EP-rank load
  and descending within-rank expert load before aggregating; DeepSeek adds no
  second popularity formula. L4 then selects one contiguous 64-entry canonical
  critical shard for Marlin cost. Popularity never changes all-gather or
  reduce-scatter bytes.
- The model has seven unique layer bodies. Layers 5--42 are represented as
  `Scale(19, C128 + C4)`, not copied per layer or expanded by rank/chunk.
- `deepseek_v4_vllm` preserves source parallel regions; the explicit
  `deepseek_v4_vllm_serial_streams` counterfactual replaces those Max nodes
  with Sum without changing kernel inputs or profile identities.
- Each semantic operation has one fixed slot. Ragged request loops and
  multi-launch public callables are timed inside that slot.
- The terminal `mhc_post -> MTP copy -> hc_head -> RMSNorm` sequence is one
  source-owned compound L1. The local shared+routed BF16 add reuses the existing
  `elementwise:torch` contract.

### Production code is called, not copied

- Profile the exact framework/library callable when one exists.
- Keep correctness construction and checking outside the timed closure.
- Do not vendor Marlin, CUTLASS, FlashMLA, CUDA, Triton, or CuteDSL source into
  `profiling/runners/`.
- A pinned submodule or library API owns the native implementation.

### Tests protect observable failures

- Keep tests for independently derived results, real argument forwarding,
  invalid-input rejection, launch boundaries when stable, and public profile
  behavior.
- Do not add per-kernel tests that merely repeat registry metadata or rebuild
  the implementation formula.
- Do not make ordinary unit tests build a profile DB or invoke a GPU runner.
  Public CLI/profile validation is a separate smoke using a temporary DB.

### Keep code proportional to the wrapper

- Set an expected line budget before implementing each kernel.
- A simple library wrapper should normally be tens to low hundreds of lines,
  including validation and focused tests.
- Exceeding that range triggers a boundary/reuse review before more code is
  added.

### Prove that a new kind cannot reuse an existing kernel

- Before creating a kind, compare its semantic contract, physical callable,
  Args identity, and launch boundary with the nearest existing kinds.
- A different name or model is not evidence for a new kind.
- If the operation has the same contract and only the implementation changes,
  add a backend to the existing kind. A backend must not change the meaning or
  fields of that kind's Args schema.
- If source inspection does not settle timing reuse, run matched-shape A/B
  measurements with identical logical inputs and outputs. Record both relative
  and absolute differences.
- Create the new kind only after this evidence shows that reuse would measure a
  different operation or violate the required fidelity.

### Semantic slots do not expand with runtime capacity

- One modeled operation gets one semantic slot.
- Do not materialize maximum request/chunk capacity as permanent CostTree
  leaves.
- A source-faithful fixed sequence may be timed by one semantic kind only when
  the launches are inseparable parts of that operation. Distinct production
  operations remain distinct kinds.

### Workload dimensions must earn their schema cost

- Do not propagate expert popularity or another upstream detail through every
  downstream kernel automatically.
- Add a cache-key dimension only when controlled same-shape measurements show
  a material absolute timing or cache-fidelity benefit.
- Include DB compatibility, grid growth, migration cost, and runtime query
  complexity in that decision.

### Specialized environments execute their own interpreter

- Adding another environment's `site-packages` to `PYTHONPATH` does not execute
  its `.pth` initialization and can leave native extensions unresolved.
- The candidate `vllm_env` change executes the pinned checkout's interpreter.
  It passed public smokes for both `moe_align_block_size` and the existing
  `moe_fused_topk`, but remains a shared-infra change that must be audited and
  made self-contained before commit.

## Kernel Decisions

### `moe_align_block_size` — retain, four-field schema

Status: implementation and independent validation complete.

- Production callable: `vllm._custom_ops.moe_align_block_size`.
- Public cache identity remains compatible:
  `(num_tokens, num_experts, top_k, block_size)`.
- `expert_counts` is not part of the schema.
- The runner constructs deterministic balanced cyclic routing for the compact
  four-field identity and checks padded route count, expert block owners, and
  routed assignment membership outside timing.
- Existing H200 rows showed normal popularity variants changing this small
  kernel by roughly 0.0001--0.0016 ms per call. That does not justify breaking
  the existing Qwen table contract. Isolated near-2x samples require controlled
  repeats and are not treated as proof of popularity sensitivity.
- The old Torch timing backend and duplicated Torch reference test suite are
  removed. The retained backend calls the production library operation.

### `tensor_zero_fill` — discard as a duplicate kind

Status: decision frozen; no new Python or Rust kind will be added.

- The old implementation timed a preallocated contiguous `output.zero_()`.
- Existing `elementwise:torch` already implements the same PyTorch
  TensorIterator zero-fill boundary when `input_size_bytes == 0`.
- DeepSeek worklets must reuse:
  `ElementwiseKernelConfig { input_bytes_per_token: 0,
  output_bytes_per_token: BF16 output bytes per token }`.
- This preserves one semantic slot without duplicating a Python schema, runner,
  Rust cache, split-threshold helpers, or tests.

### `moe_sum` — next retained production kind

Status: Python implementation and independent validation complete.

- Production callable: `vllm._custom_ops.moe_sum(input, output)`.
- Native source is `csrc/moe/moe_align_sum_kernels.cu`. Top-k 2/3/4 launch the
  dedicated `moe_sum_kernel`; DeepSeek top-k 6 takes the default
  `at::sum_out` path and launches a `reduce_kernel`. The old DeepSeek CUPTI
  filter was therefore correct.
- This remains distinct from the generic `elementwise:torch` curve: that runner
  profiles a byte-shaped `uint8 amax`, while production DeepSeek performs a
  BF16 sum reduction with FP32 accumulation semantics.
- It must also remain a distinct kernel kind rather than an `elementwise`
  backend. Backend selection may choose an implementation of one frozen Args
  contract; it must not change a byte-level fan-in operation into a shaped BF16
  reduction with additional `top_k`, `hidden_dim`, and `dtype` identity.
- Fresh matched-byte H200 measurements confirm that neither generic curve can
  substitute for the production operation. At T=128/1024/8192,
  `elementwise:torch` was 29.0%/77.6%/86.0% slower than production `moe_sum`,
  while `elementwise:triton` was 44.2%/48.0%/49.1% faster. All comparisons used
  identical logical input and output byte counts.
- Planned identity is the physical input shape and dtype; exact supported
  `top_k`, hidden size, dtype, and launch constraints must be frozen from source
  before implementation.
- Current kernel-specific size is 200 lines: 35 registration, 16 independent
  reference, 118 production runner, and 31 behavioral tests. The shared Args
  addition and registry barrel add approximately 10 more lines.
- Independent public-entry validation passed on H200 at
  `T=128, top_k=6, hidden_dim=4096, dtype=bf16`: `0.0063221 ms`, one row in a
  fresh `$TMPDIR` database, `count-missing=0`, query status `ok`, and SQLite
  integrity `ok`. The validator independently recomputed 2,621,440 FLOPs and
  7,340,032 logical bytes and matched the runner metrics.

### `clamped_swiglu` — retain, production Inductor callable

Status: Python implementation and independent validation complete.

- Production call path: DeepSeek's Marlin expert path invokes
  `vllm.model_executor.layers.fused_moe.utils.swiglu_limit_func` after FC1.
- The operation cannot reuse ordinary `silu_and_mul`: it clamps the gate only
  above 10 and clamps the up half to `[-10, 10]` before SiLU and multiply.
- The callable is `torch.compile` decorated and becomes one fused Triton launch
  after setup compilation. Compilation, warmup, correctness, and synchronization
  remain outside the timed closure.
- Public identity is `(num_rows, hidden_dim, dtype)`; the production backend is
  fail-closed to DeepSeek's H200/BF16/hidden-dim-2048 path.
- Current kernel-specific size is 194 lines: 35 registration, 21 independent
  reference, 113 production runner, and 25 behavioral tests.
- Independent public-entry validation passed on H200 at
  `rows=128, hidden_dim=2048, dtype=bf16`: `0.0037586 ms`, one row in a fresh
  `$TMPDIR` database, `count-missing=0`, query status `ok`, and SQLite
  integrity `ok`. Scoped Ruff and all three focused tests passed.

### `mxfp4_marlin_moe_gemm` — retain, public vLLM Marlin callable

Status: minimal implementation and independent H200 validation complete.

- Production callable: `vllm._custom_ops.moe_wna16_marlin_gemm` from the
  pinned vLLM checkout. Weight preparation uses vLLM's public
  `rand_marlin_weight_mxfp4_like`/Marlin helpers outside timing.
- This cannot reuse `grouped_gemm`: the latter's frozen schema describes a
  logically dense grouped GEMM, while Marlin consumes packed MXFP4/UE8M0
  weights plus block-aligned routing metadata, `top_k`, and optional router
  weights. Adding it as a backend would change both Args meaning and launch
  inputs.
- The clean public identity is the native launch's physical work:
  `(m, n, k, dtype, input_top_k, block_size_m, mul_topk_weights,
  per_group_batches)`. `fc_role`, `routing_top_k`, and `num_global_experts` do
  not enter the native callable. `num_local_experts` is exactly
  `len(per_group_batches)` and is not duplicated.
- FC1 and FC2 remain distinguishable without a role string: FC1 is
  `(n=4096,k=4096,input_top_k=6,mul=false)`; FC2 is
  `(n=4096,k=2048,input_top_k=1,mul=true)`.
- Two public smokes failed before the timed Marlin callable with
  `cudaErrorUnsupportedPtxVersion`. CPU construction removed unrelated Torch
  setup kernels, but the second run proved the failure is environmental:
  `gptq_marlin_repack` succeeds and synchronizes, a same-shape pure Torch H2D
  succeeds, while the old native-extension context followed by H2D fails.
- Python source resolves from this clean checkout, but the editable wheel's
  `_C.abi3.so` and `_moe_C.abi3.so` resolve from the old `wt-deepseek` tree.
  The host driver is 570.211.01; the successful source alignment used the
  unpacked official `cuda-compat-13-3` 610.57.04 driver libraries. The clean
  `vllm_env` now prepends an environment-owned `lib/cuda-compat` link through
  the existing `ProfileEnv.additional_library_paths` contract.
- The first compat-enabled retry loaded `libcuda.so.610.57.04`, passed public
  Marlin correctness, and reached CUPTI. It then exposed a separate profiler
  bug: the filter expected a demangled name while CUPTI returned the mangled
  `marlin_moe_wna166Marlin...` symbol. Because the timed callable contains only
  that production launch, timing now uses `kernel_name=None` and a fresh public
  smoke must pass before the kind is considered validated.
- The old 814-line private runner, 218-line reference, 796-line runner test,
  222-line native-loader test, and vendored CUDA/CUTLASS tree are discarded.
  The clean implementation currently uses 355 lines total: 35 registration,
  274 public wrapper, and 46 focused behavioral tests. The 24-line increase is
  isolated CPU operand construction that avoids the incompatible GPU setup
  kernels; growth beyond this accepted bound requires review.
- The final public H200 smoke passed at `m=128, n=4096, k=4096`, top-k 6,
  block 16, and 64 local experts: `0.2408471 ms`, `106.9965 TFLOP/s`, and
  `0.1888836 J`. A fresh temporary DB moved from one missing row to a successful
  query and passed integrity checking. Three structural CUPTI runs each
  captured exactly one mangled `marlin_moe_wna166Marlin...` launch.

### Softplus-sqrt routing — merge two old kinds into one physical kind

Status: Python implementation and independent H200 validation complete.

- Pinned vLLM routes both hash and learned selection through the same public
  callable, `vllm._custom_ops.topk_hash_softplus_sqrt`, which launches
  `_moe_C::topk_softplus_sqrt` once. Hash mode supplies `input_tokens` and a
  hash table; learned mode supplies correction bias and performs top-k.
- This cannot be a backend of existing `moe_fused_topk`: that kind's frozen
  operation is softmax/top-k and its Args cannot express sqrt-softplus scoring
  or hash-table selection. A backend would silently change operation meaning.
- The old branch incorrectly materialized the two modes as separate kinds,
  each with Torch and vLLM timing implementations plus duplicated tests. The
  clean design will use one `moe_topk_softplus_sqrt` kind with selection mode
  in Args and one production `vllm_cuda` backend.
- The clean implementation is 250 lines total: 35 registration, 184 production
  wrapper, and 31 behavioral tests. Scoped Ruff and four CPU tests pass.
- One fresh public batch profiled both modes successfully. Learned routing was
  `0.0053902 ms`; hash routing was `0.0053073 ms`. CUPTI captured exactly one
  `topkGatingSoftplusSqrt` production launch per invocation in both modes; both
  cache rows query successfully and the temporary SQLite database is intact.
- Validation found and fixed a metrics-only error: hash routing does not write
  `token_expert_indices`, so its logical-byte numerator now counts the accessed
  hash-table entries instead of that absent output. Timing, energy, and cache
  identity were unaffected.

### `gemm_fp32_output` — retain, exact production `torch.mm`

Status: minimal Python implementation and independent H200 validation complete.

- Production evidence is explicit in pinned vLLM: both outer and indexer
  compressor projections call `torch.mm(hidden_states, weight.T,
  out_dtype=torch.float32)`. Current DeepSeek shapes use `k=4096` and
  `n in {512,1024,2048}`; the MoE router adds `n=256` through the same BF16 to
  FP32 cuBLAS boundary.
- This cannot be a `single_gemm` backend. That kind's frozen `dtype` field
  describes both compute/output identity and its logical bytes assume a
  two-byte output. Reusing it would silently replace BF16 output with FP32 and
  select a different cuBLAS/NvJet launch contract.
- The clean kind has one `torch_cublas` backend and times the exact allocating
  `torch.mm(..., out_dtype=torch.float32)` form. It does not retain the old
  semantic timing backend, custom torch-profiler fallback, or preallocated
  `out=` call, because none of those are the production source boundary.
- Current kernel-specific size is 202 lines: 35 registration, 128 production
  runner, and 39 behavioral tests. Scoped Ruff and five focused tests pass.
- A fresh public H200 batch passed for `m=128,k=4096`: `n=256` measured
  `0.0064743 ms` and `n=2048` measured `0.0107364 ms`; both rows query and the
  temporary DB is intact. Structural CUPTI capture showed the small `n=256`
  call is a stable NvJet split-K plus reduction pair, while `n=2048` is one
  NvJet launch. Complete-call timing therefore preserves both valid paths.

### MHC pre and fused post/pre — retain two public compound operations

Status: minimal Python implementation and independent H200 validation complete.

- They cannot reuse `rms_norm`: standalone pre first performs TF32 HC prenorm
  GEMM, mix reductions, sigmoid/Sinkhorn transforms, residual collapse, and
  fused RMSNorm; fused post/pre additionally applies the preceding post-mix and
  has a distinct small-token fused kernel path.
- They remain two kinds because pinned vLLM exposes and invokes two different
  public operations, `mhc_pre_tilelang` and `mhc_fused_post_pre_tilelang`, with
  different inputs, outputs, and launch sequences. Their identical four-field
  shape identity reuses one `MhcRmsNormArgs` schema rather than duplicating it.
- Each runner times the complete public function with `kernel_name=None` so all
  inseparable internal DeepGEMM/TileLang launches count. It does not copy the
  old internal launch orchestration, custom profiler fallback, Torch timing
  backend, or hundreds of implementation-restatement tests.
- Both kinds together use 391 production/registration/helper lines plus 15
  focused invalid-identity test lines, versus 4,209 old runner/test lines.
- The checkout-local CUDA-13-compatible environment makes the public support
  gate select the intended DeepGEMM path. Matched same-process CUPTI A/B then
  kept old/manual versus clean/public differences within 2.3% for T=8 and
  T=128. The earlier T=128 +93--125% result came from a missing compat driver:
  `is_deep_gemm_supported=False` selected the slower TileLang fallback.
- Current default cold-L2 public measurements are pre `0.009322/0.010666 ms`
  and fused post/pre `0.010667/0.014938 ms` at T=8/128. CUPTI confirms pre is
  always two launches; fused post/pre is the source-real two-launch small-token
  path at T=8 and three-launch decomposed path at T=128. One semantic slot owns
  the complete production operation without exposing its internals.

## Discarded Old-Branch Patterns

- Vendored `mxfp4_marlin_native` CUDA/CUTLASS source.
- `expert_counts` in `moe_align_block_size`.
- A dedicated `tensor_zero_fill` Python/Rust kind.
- Permanent per-capacity prefill/indexer chunk slots.
- Torch timing implementations that reproduce a production library kernel.
- Per-kernel registry/facade tests already covered by shared infrastructure.
- `reduce_scatter` and `moe_ep_all_gather` from the old EqualShard deployment.
  They were introduced for the P0-invalid worker that forced DP4, FIFO, and
  four-request equal shapes. The clean L1 rebuild does not preserve kernels for
  a discarded deployment; a correct worker or fresh source trace must prove a
  production call before either kind can return.

## Validation Status

- `moe_align_block_size` four-field public smoke passed on H200 at
  `T=128, E=256, K=6, block=16`: `0.0056186 ms`, one row in a fresh `$TMPDIR`
  database, `count-missing=0`, query status `ok`, and SQLite integrity `ok`.
  Scoped Ruff and all six focused tests passed.
- `moe_sum` public smoke passed on H200 at `T=128, K=6, H=4096, bf16`:
  `0.0063221 ms`; scoped Ruff and all four focused tests passed.
- No Rust wiring has been started in this clean branch.

### Clean smoke versus old profile DB

The old DB is a regression signal, not ground truth: the compared rows carry
`verified=0`, several old runners timed a different callable form, and some
clean schemas intentionally removed non-physical identity fields.  Exact or
closest-shape comparisons currently show:

- `moe_sum` T128: `0.0063751 -> 0.0063221 ms` (-0.8%).
- `moe_align_block_size` T128/E256/K6/B16: `0.0061210 -> 0.0056186 ms`
  (-8.2%); the old row additionally encoded the discarded routing layout.
- learned routing's historical +13.8% is mainly a measurement-policy mismatch:
  old used warm L2 and clean used cold L2; matched cold penalty is 9--11%.
  Hash routing's historical +11.4% is mainly input locality: sequential token
  IDs are about 8--9% faster than the dispersed production-like construction,
  while L2 clearing changes it by roughly 0%.
- `clamped_swiglu` matched old/clean public calls differ by less than 1% in the
  current environment. The historical `0.0025933 -> 0.0037586 ms` delta is
  compiler/runtime provenance, not a changed callable; the clean historical
  row is also much closer to the available NSYS capture.
- `gemm_fp32_output` T128: N256 `0.0068923 -> 0.0064743 ms` (-6.1%), N2048
  `0.0085915 -> 0.0107364 ms` (+25.0%). The clean runner uses the exact
  allocating production `torch.mm`, while the old runner used preallocated
  `out=`; the difference is expected to be investigated, not silently accepted.
- MHC matched old/manual and clean/public CUPTI differs by at most 2.3%. The
  former large T128 delta was entirely the missing-driver fallback described
  above, so it is closed rather than accepted as kernel drift.
- Exact Marlin `[12] * 64` CUPTI is `0.2391781 ms` in the old cu128 runner and
  `0.2367448 ms` in the clean cu130 public runner (-1.0%). The different
  interpreter/driver prevents calling this a strict same-process A/B, but the
  callable, shape, routing, and timing seam match.

Therefore the completed structural/correctness validation must not be reported
as full numerical DB equivalence.  Major deltas need matched-environment A/B
before profile-grid migration.

### Retained DeepSeek attention kinds not yet rebuilt

- `deepseek_v4_sparse_mla_decode`: clean Python implementation and exact-schema
  public H200 validation are complete. Its 13 fields preserve exact per-row
  SWA/extra valid-count tuples and explicit `extra_index_capacity`; aggregate
  totals alone cannot identify the FlashMLA planner because it reads every
  row's lengths. C128 capacity derives from runtime max context (512 at 65K, up
  to 8192 at 1M), not from compression ratio alone. Model/dtype identity remains
  explicit and the backend fail-closes to the supported V4 FlashMLA shape.
- One CUDA-graph replay remains one semantic slot, while CUPTI sums every
  internal production launch. Planned graphs always contain planner + splitKV +
  combine; reused graphs contain splitKV + combine. The corrected per-token
  patterned oracle detects wrong indices, row boundaries, valid lengths, or
  SWA/extra crossover. Eight exact-schema points pass public correctness and
  query, including C128 capacities 512 and 8192. Two C4 planned inputs with the
  same aggregate total but different row distributions occupy distinct keys
  and measure `0.075679/0.076727 ms`. The temporary DB has no missing rows and
  passes integrity checking.
- `deepseek_v4_packed_cache_gather`: rebuilt and validated through the public
  H200 CuteDSL dispatcher; details are recorded below.
- `deepseek_v4_sparse_attn_compress_store`: retained; C4 is one launch and C128
  is one public compound operation containing two ordered launches.
- `deepseek_v4_sparse_mla_prefill`: rebuilt and independently validated; see
  the completion record below.
- `deepseek_v4_indexer_q_rope_quant`: rebuilt; CPU/registry checks pass and its
  H200 validation is waiting for an idle GPU.
- `deepseek_v4_indexer_mqa_logits_prefill` and
  `deepseek_v4_indexer_topk_prefill`: rebuilt around one shared source-exact
  ragged workload splitter. CPU/registry checks pass; independent H200 public
  validation is pending.
- `deepseek_v4_fused_inv_rope_fp8_quant`: validated through the public entry
  after rebuilding the checkout-local vLLM environment. Exact Torch FP8 output
  and scale checks pass. T=1/128 measure `0.0020984/0.0073857 ms`; each replay
  contains exactly one `_fused_inv_rope_fp8_quant_per_head` launch. Both rows
  query from the fresh temporary DB and SQLite integrity is `ok`.

## Next Steps

1. Continue retained attention kinds from production source, keeping each
   public compound operation in one semantic slot.
2. Keep auditing measurement policy and input locality separately from callable
   changes before migrating old DB rows.
3. Start Rust wiring only for validated retained kinds, reusing existing
   kernels wherever the production boundary is identical.
## 2026-08-24 — `deepseek_v4_packed_cache_gather` complete

- Added the production `vllm_deepseek_v4_cutedsl` backend with exact ragged
  sequence/gather tuples, workspace offset, block-table width, physical block
  size, and the fixed DeepSeek V4 packed-cache identity.
- The runner calls the public `dequantize_and_gather_k_cache` dispatcher and
  fails closed unless it selects the H200 CuteDSL implementation. Patterned
  cache rows, reversed physical pages, and untouched workspace regions are
  checked exactly before timing.
- Production block strides are padded to 32 bytes (`1184` for block size 2 and
  `37376` for block size 64). This fixed the real CuteDSL compile failure from
  the naive contiguous block-2 stride (`1168`).
- Fresh temporary-DB H200 validation passed for block-64 ragged suffix gather
  (`0.00256107796 ms`) and block-2 full gather (`0.00255784596 ms`). Both are
  exactly one `DequantGatherKCacheKernel` launch per logical call; public query,
  missing-count, and SQLite integrity checks all passed. The shared profile DB
  was not changed.

## 2026-08-24 — compressor store and sparse prefill complete

- `deepseek_v4_sparse_attn_compress_store` now keys exact row positions,
  request ownership, and state block-table width. It calls the same public
  `compress_norm_rope_store_cutedsl` operation as `DeepseekCompressor.forward`;
  outer projection and `save_partial_states` remain outside this boundary.
- C4 mixed partial/full topology measures `0.01059522602 ms` and is exactly one
  fused launch. C128 boundary topology measures `0.00566115436 ms` and is
  exactly two ordered launches (compress, then norm/RoPE/store) inside the same
  semantic sample. Independent Torch checks and poisoned sentinels pass for
  both, with production page strides 37440/1728. Public query, missing-count,
  and temporary-DB integrity checks pass; the shared DB was not changed.
- `deepseek_v4_sparse_mla_prefill` has only the production FlashMLA backend.
  One L1 row owns the full request batch and sums the public model loop's
  `ceil(num_requests / 4)` physical launches; no physical chunk becomes a
  CostTree slot. Runtime `max_model_len` remains explicit, so C128 capacity is
  512 at 65K rather than being confused with the checkpoint's 8192 at 1M.
- Fresh H200 results are C1 `0.0082527242 ms`, ragged C4/5-request
  `0.02271294468 ms` (two launches), and C128/65K `0.02134662074 ms`.
  Sampled independent Torch correctness, exact launch counts, public query,
  missing-count, and SQLite integrity all pass.

## 2026-08-24 — indexer clean rebuild in progress

- `deepseek_v4_indexer_q_rope_quant` now has one production-only
  `vllm_cutedsl_fp8` backend. The public callable, explicit model/storage
  identity, runtime context, independent Torch RoPE/FP8 reference, and the
  source dispatch boundary at 511/512 tokens are covered by focused tests.
- Registry/facades, scoped Ruff, and focused CPU tests pass. Independent H200
  public profiling and CUPTI specialization validation are running.
- MQA-logits and top-k prefill remain next; their physical sub-chunks will be
  summed inside one semantic sample rather than expanded into fixed slots.

## 2026-08-24 — indexer prefill Python implementation complete

- MQA logits and top-k each expose one production backend and one cache row for
  the complete ordered `query_context_pairs` workload. There is no Torch timing
  backend and no fixed physical-chunk slot array.
- A shared helper mirrors vLLM's request-greedy then query-slice launcher. It
  derives exact C4 causal row starts/ends, the 512 MiB FP32 logits limit, and the
  physical gather-workspace cap. An 8192-query/1M-context request becomes 16
  launches inside one semantic call.
- MQA calls public `vllm.utils.deep_gemm.fp8_fp4_mqa_logits`; top-k calls public
  `vllm._custom_ops.top_k_per_row_prefill`. Both perform sampled independent
  Torch correctness before `Timer.cupti(kernel_name=None)`.
- Focused behavioral tests (ragged spans, source chunking, nested public args,
  padded logits stride, and production argument order) plus the indexer-Q tests
  pass 9/9. Scoped Ruff, registry listing, and `git diff --check` pass. H200
  public/CUPTI validation is delegated and still pending.

## Expert-popularity audit

- DeepSeek and Qwen use the same simulator-side popularity conversion:
  schema-v2 `counts_by_layer` enters `resolve_routing_source`, then the shared
  layerwise EP canonicalization introduced before the DeepSeek work.
- Their capture sources differ. Qwen's older evidence came from the optional
  EPLB load buffer; DeepSeek records raw logical router assignments because its
  fused backend did not reliably update that buffer. The DeepSeek trial freezes
  EPLB movement (`step_interval` beyond the run, zero redundant experts), so
  logical IDs still map to fixed contiguous EP shards.
- Required follow-up: make the raw-routing extractor fail closed unless the
  run proves placement remained fixed. Raw logical popularity must not be used
  after EPLB remaps experts, because it would no longer represent physical
  per-rank grouped-GEMM load.
- Separate P0 in the old Rust L1 wiring: DeepSeek squash `8ec6b05` replaced the
  shared `RoutingDistribution::to_per_expert_counts` with
  `marlin_per_expert_counts` for Marlin GEMM and the Marlin align path, while
  fixed-block alignment used another `balanced_full_local_counts` special case.
  Its own comment says this preserved imported old DB keys after the shared
  Qwen active-set policy changed. That is cache compatibility, not production
  evidence. Clean Rust wiring must restore the shared routing projection for
  Marlin GEMM and keep `moe_align_block_size` on its four-field physical schema;
  old rows will be reprofiled rather than dictating runtime semantics.

## 2026-08-24 — clean Rust L1 wiring checkpoint

- `mxfp4_marlin_moe_gemm` now uses the shared
  `RoutingDistribution::to_per_expert_counts`; the old DeepSeek-only routing
  projection is absent. Its three tests cover production block transitions,
  active-set routing, and FC1/FC2 physical workload differences.
- Six compact 1D kinds are wired: `moe_sum`, `clamped_swiglu`,
  `moe_topk_softplus_sqrt`, `gemm_fp32_output`, and both MHC compound kinds.
  MHC shares one Config/Input implementation without sharing its KIND/table.
  Independent validation passed 394 timing tests, six kernel-query grids, and
  scoped formatting; no profile DB was touched.
- The two one-dimensional DeepSeek Q kernels retain the real 511/512/513
  specialization boundary. Compressor/store retains exact row positions and
  request IDs in SlotInput while fitting only launched-token and active-boundary
  coordinates; one C4 or ordered two-launch C128 operation remains one L1 kind.
- Current ten new Rust kernel files total 1,066 lines. Ordinary wrappers are
  36--121 lines; Marlin is 219 and ragged compressor/store is 173. Tests protect
  physical boundaries or semantic workload identity rather than registry rows
  or DB construction.

## 2026-08-24 — clean Rust L1 registry complete

- Every clean DeepSeek Python kind now has a Rust `KernelSpec`; the exact
  registry diff is eight DeepSeek Python kinds and eight wired Rust kinds, with
  no missing entry.
- Sparse prefill and both indexer-prefill leaves retain the complete ordered
  request batch in one typed `SlotInput`. Profiling runners own request-greedy
  and memory-driven physical splitting; Rust does not allocate permanent
  `chunk_00...chunk_N` slots. Packed gather retains one slot per actual public
  gather call, and sparse decode retains one slot per CUDA-graph replay.
- The 15 new Rust kernel files, including MoE/MHC and attention, total 1,877
  lines. Shared MHC and indexer-prefill structures avoid duplicate Config/Input
  code. Each file has zero or one focused behavior test except Marlin, whose
  three tests protect three independent production boundaries.
- `cargo test --offline --lib timing` completed with 401 passed and 2 ignored;
  an offline simulator build passed; `git diff --check` passed. Independent
  kernel-query/schema review is still running.
# 2026-08-24 — kernel-query ragged/re-axis correction

- Found an existing Analyzer assumption that cache axes and physical Input JSON
  fields are interchangeable. It fails for ragged DeepSeek Inputs and gives the
  wrong semantics for re-axis caches.
- Fix direction: preserve `eval` for physical cache-fidelity queries; add
  `eval_coords` for declared-grid inspection. Analyzer expands the numeric cache
  axes, queries those coordinates directly, and labels points with coordinate
  names. No kernel timing, profile row, CostTree, or slot changes.
- This makes the current cache approximation visible; it does not claim that a
  low-dimensional projection preserves every ragged distribution. That remains
  a separate cache-fidelity check.
# 2026-08-24: close the two remaining common-attention L1 gaps

- Production source and captured NSYS both prove two real launches that the old
  model did not represent faithfully: `_fused_q_kv_rmsnorm_kernel` was absent,
  while `fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel{ReducedGrid}` used a
  generic elementwise proxy.
- Added one semantic kind per production callable:
  `deepseek_v4_fused_q_kv_rmsnorm` and
  `deepseek_v4_qnorm_rope_kv_insert`. Neither callable is decomposed into
  component slots.
- QNorm/RoPE/KV insert preserves DP padding as
  `(num_tokens, insert_fraction)` and keeps the source 1023/1024/1025 dispatch
  boundary. Both token axes are explicitly sorted and include decode sizes.
- Scoped Ruff, focused pytest (2 passed), Rust focused tests (8 passed), build,
  registry listing, and kernel-query grid smoke pass. GPU public-entry/CUPTI
  validation remains pending because all eight H200s were busy; no shared DB was
  changed.

# 2026-08-24: unified DeepSeek attention worklet

- Added one `DeepseekV4AttentionLocalWorklet` instead of separate prefill and
  decode copies. A mixed iteration evaluates MHC and all common projections on
  the merged padded token count exactly once; prefill and decode sparse/indexer
  callables remain separate fixed leaves.
- The worklet derives exact compressor row positions/request ownership and
  ragged decode valid-count vectors from `(Q, context)` prefill pairs plus
  resident decode KV lengths. `num_insert_tokens` must equal the phase rows and
  may be smaller than `num_tokens` only for DP padding.
- C1 has no compressor/indexer branches. C4 has one compressor branch and one
  indexer branch. C128 has one compressor branch. Runtime chunking remains
  inside each L1 callable; there are no `chunk_N` or rank-specific slots.
- `serialize_streams` changes only branch composition (`Sum` versus `Max`), not
  the leaf inventory or inputs. The serial alignment architecture can therefore
  use the same production kernel boundaries without duplicating the worklet.
- New worklet size is 1,024 lines including two behavior tests, versus 6,996
  lines across the old seven DeepSeek worklet files. Focused tests are 2/2;
  timing regression is 402 passed / 2 ignored; scoped rustfmt and diff-check pass.

## 2026-08-24 — attention source-fidelity P0 closure

- Mixed compressor rows now follow vLLM's decode-first physical layout. Runtime
  `max_model_len` is fail-closed to `1..=1_048_576`; C128 decode derives its
  compressed capacity from that runtime value instead of the C4 top-k cap.
- The compressor tail remains one semantic slot, but now measures its complete
  production launch sequence: `save_partial_states` followed by compress/norm/
  RoPE/store. Main head-512 uses the public CuteDSL path and 584-byte cache row;
  the C4 indexer head-128 uses the public Triton path and 132-byte cache row.
  Both reuse the same kind/schema and fail closed on backend-specific identity.
- The non-serial CostTree now has the two source barriers: projection fanout,
  fused Q/KV RMSNorm, then the Q/cache/indexer fanout. The indexer itself joins
  its Q path with its compressor tail before logits/top-k. Serial mode changes
  only each fanout's `Max` to `Sum`.
- Padding-only ranks still charge padded projection rows but zero both compressor
  tails because there are no actual insert rows. Current focused validation:
  Python 5/5 and Rust DeepSeek 12/12; full timing regression and H200 compound
  launch validation remain pending.

## 2026-08-24 — alignment cache preparation

- Fixed two L4 backend-selection regressions found by the real cache report:
  shared-expert activation now selects `clamped_swiglu:vllm_inductor`, and
  indexer decode logits selects
  `dsa_paged_mqa_logits_decode:vllm_deepgemm_fp8`.
- Reused existing rows only for four kernels whose registry, runner source,
  timing policy, and SQLite schema are byte-for-byte identical to the prior
  DeepSeek worktree: `single_gemm`, `fp8_per_token_group_quant`, `elementwise`,
  and `dsa_paged_mqa_logits_decode`. The pre-import DB backup is
  `$TMPDIR/deepseek-clean-profile-before-shared-import.5XQIVq.db`; integrity is
  `ok`. MHC, router, Marlin, and all new DeepSeek kinds deliberately remain
  subject to clean profiling.
- The alignment simulation enables the ordinary launcher cache-build phase so
  unique missing specs are profiled in batches before DES execution. This is a
  cache operation only; it does not alter trace shapes or simulator policy.
- Merged 25 already-validated rows from fresh temporary DBs produced by the
  current clean public entrypoints. They cover representative inverse-RoPE,
  indexer-Q, compressor/store, sparse decode/prefill, packed gather, FP32 GEMM,
  Marlin, and clamped-SwiGLU shapes; every source run had its own correctness,
  query, and integrity evidence. The pre-merge backup is
  `$TMPDIR/deepseek-clean-profile-before-fresh-merge.hgsu8m.db`. These are not
  historical rows copied from the old worktree.
- Removed an accidental dense-axis Cartesian explosion from compressor/store:
  total rows and active boundary rows now use power-of-two anchors through
  8,192, including the exact zero-active boundary. The feasible grid is 119
  rows/config instead of 1,595. This follows the compound's two smooth work
  terms and is not trace-specific; acceptance requires off-grid public truth
  probes after the grid is filled.

## 2026-08-24 — clean alignment cache-build checkpoint

- Filled the complete 68-point `mhc_pre_rms_norm:vllm_tilelang` H200 sweep
  through the public profiling entrypoint. All rows query successfully and the
  shared clean DB passes `PRAGMA integrity_check`.
- Full CPU validation is green: Rust library tests are 902 passed / 6 ignored;
  the explicit new Python L1 suite is 83 passed. The only Rust warning is an
  unchanged pre-existing helper in `timing/routing.rs`.
- The next official `build-cache-only` attempt reached the first uncached
  `deepseek_v4_fused_q_kv_rmsnorm:vllm_triton` point after loading the already
  complete MHC, quantization, and GEMM caches. The H200 pool was taken by an
  external workload before that JIT batch acquired a device, so the row stayed
  missing and model construction failed closed. This is a GPU-admission race,
  not a callable, correctness, or schema failure.
- Private tmux now waits for a genuinely idle H200, profiles the full 68-point
  fused Q/KV RMSNorm sweep through the public entrypoint, and then retries the
  official launcher cache build. No idle guard is bypassed and no failed row is
  written.
- The fused Q/KV RMSNorm sweep subsequently completed: 68/68 rows cover
  `num_tokens=1..65536`, with `time_ms=0.001742..0.156964`; the DB integrity
  check remains `ok`.
- The next JIT batch exposed a real environment-registration bug in the qnorm/
  RoPE/KV-insert runner. The target op is owned by production
  `vllm._C_stable_libtorch`, but the standalone runner had not imported that
  extension, so `torch.ops._C` lacked the callable. Added the same explicit
  extension import used by vLLM's CUDA platform initialization. A fresh public
  H200 T1/insert0 run now passes correctness, CUPTI timing, energy, and DB
  persistence (`time_ms=0.00277467476`, missing count zero). The official cache
  build has resumed from the remaining qnorm sweep points.
- The complete qnorm/RoPE/KV-insert canonical sweep is now covered: 339 Rust
  grid rows query successfully. The public batch also wrote six valid but
  unintended off-grid fractions at the 1023/1025 dispatch boundary; they are
  recorded for exact cleanup after the active cache build, rather than mutating
  the DB concurrently.
- The next sparse-MLA-prefill miss exposed two schema-validation bugs before the
  production callable: Python mislabeled the complete physical index topology
  as `evenly_spread` while Rust/L3 correctly emitted
  `request_local_topk_plus_swa`, and it compared the Rust/Python softmax-scale
  expressions by exact floating-point equality. The runner now preserves the
  full topology identity and checks the scale with a strict `1e-15` absolute
  tolerance; index construction and FlashMLA launch behavior are unchanged.
  Scoped Ruff, focused pytest (4 passed), and diff-check pass. A fresh public
  H200 C1 spec then passed correctness, CUPTI timing, energy, persistence, and
  DB integrity at `0.00779739838 ms`; the official cache build resumed.
- The official Rust payload then revealed that L3 still labeled sparse-prefill
  with the upstream packed FP8 KV cache. Production first gathers/dequantizes
  that cache into a flat BF16 workspace and passes the BF16 workspace to
  `flash_mla_sparse_fwd`; this L1 measures the latter callable. The worklet now
  sends `cache_dtype=bf16` and
  `cache_layout=request_slot_major_flat_mqa_bf16_d512` for this leaf only.
  Other packed-cache leaves remain FP8. Focused Rust worklet tests are 4/4 and
  scoped rustfmt/diff-check pass; the cache build was restarted with the exact
  public row already present.
- The public C1 sparse-prefill canonical batch subsequently completed in one
  376-spec submission: 376/376 results are `ok`, post-run missing is zero, the
  profile artifact is complete, and DB integrity remains `ok`. The C4 and C128
  batches use the identical Rust canonical-pair generator with only the
  ratio-specific config identity changed. Both also completed as one public
  376-spec submission with 376/376 `ok` and post-run missing zero. Thus all
  three production ratios have complete canonical sparse-prefill grids.
- After the profiler workers exited, removed exactly the six accidental qnorm
  off-grid rows at T=1023/1025 that the earlier shell generator had produced.
  The table is back to its 339 canonical rows and integrity is `ok`; the
  recoverable pre-delete copy is
  `$TMPDIR/deepseek-clean-profile-before-qnorm-offgrid-cleanup.20260824T211643Z.db`.
- Sparse-decode profiling exposed a support-axis error rather than a kernel
  timing error. The Rust sweep had reused the 8,192-token batch axis for decode,
  but the captured vLLM recipe independently sets `max_num_seqs=256` and
  `max_num_batched_tokens=8192`. A reused graph at batch 8,192 produced an
  unspecified CUDA launch failure and poisoned that worker context. The Rust
  axis and public runner now fail closed above 256 decode rows; focused Python
  tests are 14/14 and the Rust kernel test passes.
- The corrected sparse-decode grid contains 378 specs across C1/C4/C128 and
  planned/reused modes. One complete public JIT-fill submission returned
  378/378 `ok`; post-run query also returns 378/378 with missing count zero.
  Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_sparse_decode_canonical_256`.
  After all workers exited, removed exactly 111 rows from the obsolete
  batch-size-greater-than-256 axis. The table now contains 386 rows including
  valid off-grid truth points, no row exceeds batch 256, and integrity is `ok`.
  The recoverable pre-delete copy is
  `$TMPDIR/deepseek-clean-profile-before-sparse-decode-cap-cleanup.20260824T2148Z.db`.
- The next cache miss exposed the same domain mismatch in inverse-RoPE FP8
  quantization: the public runner and attention worklet are bounded by the
  8,192-token serving batch, while the Rust sweep had inherited the generic
  65,536-token axis. The Rust axis now stops at 8,192 and still preserves the
  production 511/512/513 coarsening boundary. Focused Python tests are 7/7 and
  the Rust boundary test passes.
- The corrected inverse-RoPE grid has 57 points. One complete public JIT-fill
  submission returned 57/57 `ok`; post-run query is also 57/57 with missing
  count zero, and the table spans exactly 1..8,192 with integrity `ok`.
  Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_fused_inv_rope_fp8_quant_canonical_8192`.
- Filled the complete 68-point `mhc_fused_post_pre_rms_norm:vllm_tilelang`
  sweep through one public JIT-fill submission. All 68 results are `ok`, the
  post-run query has missing count zero, the table spans 1..65,536 tokens, and
  DB integrity is `ok`. This is the complete production compound callable, not
  a manual sum of its small/large dispatch kernels. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_mhc_fused_post_pre_canonical`.
- The official cache builder then filled the complete 68-point
  `gemm_fp32_output:torch_cublas` router-gate grid without intervention.
- Filled the complete 15-point EP4 ragged
  `moe_ep_all_gather:vllm_pynccl` grid through one public four-GPU submission.
  All 15 results are `ok`, including zero-row ranks and the imbalanced
  `[4096,1366,1365,1365]` topology; post-run missing is zero and DB integrity
  is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_moe_ep_all_gather_canonical`.
- Filled the complete 68-point `clamped_swiglu:vllm_inductor` sweep through one
  public seven-worker JIT-fill submission. All 68 results are `ok`, post-run
  missing is zero, the table spans 1..65,536 routed rows, and DB integrity is
  `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_clamped_swiglu_canonical`.
- Filled both 68-point learned/hash specializations of
  `moe_topk_softplus_sqrt:vllm_cuda`. The first public batch passed 135/136;
  the only failure was learned T=65,536, where one row had two exactly equal
  corrected scores and CUDA/Torch returned the same expert set in opposite tie
  order. The independent oracle now compares the selected set and recomputes
  weights in the production-returned order; the timed callable is unchanged.
  Scoped Ruff and focused pytest (4 passed) are green. The missing-only public
  retry passed, so the final 136/136 query has missing count zero, both modes
  span 1..65,536, and DB integrity is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_moe_topk_softplus_sqrt_canonical`
  and the tie-fix retry sibling.
- Filled the complete 18-token x 5-runtime-block-size
  `moe_align_block_size:vllm_cuda` grid in one public submission. All 90
  canonical DeepSeek rows are `ok` and post-run missing is zero for block sizes
  8/16/32/48/64. Existing additional block-16 rows are intentionally preserved
  because this semantic kind is also shared by Qwen; DB integrity is `ok`.
  Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_moe_align_block_size_canonical`.
- Reproduced the exact Rust popularity path to generate the Marlin grid:
  layerwise EP-rank canonicalization, f32-to-ppm normalization, critical
  contiguous 64-expert shard, and active-set Hamilton allocation. Its first
  FC1 payload exactly matches the cache builder's `m=1`, block-8,
  `[1,1,0...]` key. One public `mxfp4_marlin_moe_gemm:vllm_marlin`
  submission then filled FC1 78/78 and FC2 78/78 canonical rows with all 156
  results `ok`; post-run missing is zero and DB integrity is `ok`. Artifacts
  are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_mxfp4_marlin_moe_gemm_canonical`.
- Filled the complete 68-point `moe_sum:vllm_cuda` grid through one public
  submission. The axis is gathered tokens; the runner itself owns the top-6
  reduction from `[tokens,6,4096]` to `[tokens,4096]`. All 68 results are
  `ok`, post-run missing is zero, the table spans 1..65,536, and DB integrity
  is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_moe_sum_canonical`.
- Filled the complete 15-point EP4 ragged
  `moe_ep_reduce_scatter:vllm_pynccl` grid through one public four-GPU
  submission. It reuses exactly the all-gather topology grid, including
  zero-row ranks and imbalanced partitions, while measuring only the BF16
  hidden combine payload. All 15 results are `ok`, post-run missing is zero,
  and DB integrity is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_moe_ep_reduce_scatter_canonical`.
- The first canonical main-compressor submission passed 86/238 points and
  exposed a real sweep-topology bug at 128 or more rows: every inactive row
  reused position zero, so rows belonging to the same request concurrently
  overwrote one `save_partial_states` slot. The canonical generator now gives
  each request strictly increasing positions: active rows remain on
  compression boundaries and inactive rows use the corresponding non-boundary
  ordinal. This changes only synthetic sweep payloads, not production worklet
  inputs or slot structure. The Rust regression test checks the observable
  uniqueness/order invariant; focused Rust and Python tests pass.
- Regenerated the C4/C128 main-compressor grids with the corrected topology and
  ran a missing-only public submission. The final canonical query is 238/238
  `ok` with missing count zero and DB integrity `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_sparse_attn_compress_store_main_canonical`
  and the `..._canonical_topology_fix` retry sibling. Each profile row remains
  one semantic compound leaf: `save_partial_states` followed by the public
  compressor/store callable.
- Filled the complete 119-point C4 indexer-compressor grid through one public
  `vllm_deepseek_v4_triton` submission. All results are `ok`, post-run missing
  is zero, and DB integrity is `ok`. This is the indexer compound leaf with
  `save_partial_states` followed by the production Triton compressor/store;
  artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_sparse_attn_compress_store_indexer_canonical`.
- Source audit of the pinned vLLM indexer found that both prefill
  `fp8_fp4_mqa_logits` and decode `fp8_fp4_paged_mqa_logits` are called with
  `clean_logits=False`. The clean worklet had incorrectly configured both as
  true while the public runners and existing DSA contracts correctly required
  false. Changed only those two L3 config values; no L1 schema or timed callable
  changed. Scoped formatting/checks and the four focused attention-worklet
  tests pass, and corrected MQA canonical endpoints pass public schema coercion
  plus runner validation.
- Filled the complete 18-point
  `deepseek_v4_indexer_q_rope_quant:vllm_cutedsl_fp8` sweep through one public
  submission. All results are `ok`, post-run missing is zero, the grid spans
  1..8,192 and retains the 511/512/513 production coarsening boundary, and DB
  integrity is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_indexer_q_rope_quant_canonical`.
- Filled the complete 311-point
  `deepseek_v4_indexer_mqa_logits_prefill:vllm_deepgemm_fp8` feasible grid
  after correcting the L3 production identity to `clean_logits=false`. All
  results are `ok`, post-run missing is zero, the table contains exactly the
  311 canonical false-valued rows, and DB integrity is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_indexer_mqa_logits_prefill_canonical`.
- Filled the matching complete 311-point
  `deepseek_v4_indexer_topk_prefill:vllm_cuda` feasible grid. All results are
  `ok`, post-run missing is zero, the table contains 311 canonical rows, and DB
  integrity is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_indexer_topk_prefill_canonical`.
- The next official cache build passed all newly filled compressor and indexer
  prefill kinds, then correctly rejected reuse of the existing 1M-context
  `dsa_paged_mqa_logits_decode` rows. Those old rows use the synthetic
  `uniform/unique_scattered/page_planar_fp8_fp32_scale` identity, whereas the
  DeepSeek indexer decode path requests the production
  `max_ragged/request_contiguous/fp8_e4m3_ue8m0` identity. The shape axes match,
  but the physical cache topology does not, so the old timings must remain
  separate rather than being relabeled.
- Added a distinct `deepseek_v4_indexer_mqa_logits_decode` kind instead of
  relabeling the GLM DSA rows.  The Python runner uses the pinned public
  `fp8_fp4_paged_mqa_logits` callable with max-ragged context lengths,
  request-contiguous pages, and `clean_logits=false`.  Its first public H200
  smoke exposed an important layout detail hidden by the tensor shape: each
  `[block,64,1,132]` page stores all FP8 key bytes first and all FP32 scale
  bytes second; the 576-byte allocator alignment only pads the stride between
  pages.  After correcting the independent Torch oracle and allocation to that
  page-planar layout, the public smoke passed at 0.003395 ms, with one DB row
  and SQLite integrity `ok`.
- The Rust kind stays compact by reusing only the established physical
  18-by-21 `(batch_size, context_len)` grid, 32-GiB infeasible mask, and
  `Cache2DLinear(Product)` machinery.  It keeps its own KIND/profile table and
  the DeepSeek worklet now resolves this kind instead of the GLM one.  Timing
  tests pass 400/400 with 2 ignored, the worklet tests pass 4/4, and
  `kernel-query grid` reports the expected two axes.  The 374 feasible
  production-identity rows were filled through the public CLI: 374/374 are
  `ok`, post-run missing is zero, and DB integrity is `ok`. Artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_indexer_mqa_logits_decode_canonical`.
- Added a distinct `deepseek_v4_indexer_topk_decode` kind for production
  `top_k=512` and `max_ragged` lengths instead of reusing the GLM DSA
  `top_k=2048/uniform` table. The runner calls the pinned public
  `torch.ops._C.persistent_topk`, keeps its 1-MiB workspace and all setup plus
  the independent Torch value-set oracle outside timing, and profiles the
  complete public call with `Timer.cupti(kernel_name=None)`. The Rust wrapper
  reuses only the generic schema/cache machinery under its own KIND and a
  compact 9-by-21 batch/context grid; full timing tests pass 401/401 with 2
  ignored and kernel-query reports the expected axes.
- The first canonical top-K profile attempts exposed a synthetic-input issue,
  not a shape or capacity issue. Million-point `linspace`, then uniform
  `[0,1]`, over-concentrated candidates in the persistent kernel's radix and
  coarse-histogram buckets. A fixed-seed Gaussian row matches the pinned vLLM
  tests and production-logit distribution while remaining deterministic; the
  32K radix boundary and a B64/262K large case both pass the exact Torch
  selected-value oracle. The missing-only retries completed the full 189/189
  canonical grid, post-run missing is zero, the table spans B1..256 and
  context 0..1,048,576, and DB integrity is `ok`. Artifacts are the
  `profile_indexer_topk_decode_canonical` directory plus its two retry siblings.
- The official cache build then passed every attention, MoE, and mixed
  C1/C4/C128 body and stopped only at the final
  `deepseek_v4_terminal_mhc_head` table. Its initial public profile failed
  before any timed launch because standalone construction of vLLM `RMSNorm`
  lacked the engine-owned `set_current_vllm_config` context. The runner now
  supplies a default `VllmConfig` only while constructing the module, outside
  the timed closure; the production dispatch and compound launch sequence are
  unchanged. A direct T=1 replay passed at 0.022929 ms, then the public retry
  filled all 68/68 canonical rows spanning 1..65,536 tokens. Post-run missing
  is zero and DB integrity is `ok`; artifacts are in
  `logs/20260824_1_deepseek_v4_clean_alignment/profile_terminal_mhc_head_canonical_retry`.

## Fresh Alignment Result (2026-08-25)

- The final cache probe reports `0 / 23556` missing specs across 252 kernels.
  Timing-predict completed all 192 captured iterations without changing the
  captured workload or injecting observed rank/ingress metadata.
- Kernel alignment artifacts are under
  `logs/20260824_1_deepseek_v4_clean_alignment/analysis_kernel`. The derived
  GPU-time multiplier is `1.0664830501667724`. Per-iteration absolute relative
  error has mean 4.973%, p50 3.545%, and p90 11.045%. Aggregate decode time is
  3021.157 ms measured versus 2989.901 ms modelled (-1.035%); mixed time is
  17972.738 ms versus 17913.719 ms (-0.328%). Measured-duration mapping coverage
  is 97.557% and simulated-work mapping coverage is 98.731%.
- Operation-level residuals do not all have the same sign. The close aggregate
  path includes cancellation, notably under-modelled GEMM/Marlin work and
  over-modelled packed insert, fused RMSNorm, compressor, and prefill-indexer
  work. Therefore the result supports the total critical path, not a claim that
  every isolated kernel is already within 5%.
- The first predictive E2E run used the kernel-derived multiplier and completed
  128/128 requests. Throughput was 1051.504 requests/s measured versus 1048.892
  modelled (-0.248%), but E2E mean was initially +8.94% because the generic
  replay scheduler stamped only the first of several already-due requests when
  one completion opened multiple concurrency slots.
- Fixed that generic frontend clock bug without changing scheduling policy,
  trace contents, kernel inputs, or model cost. All already-due requests drained
  during one capacity-opening event now receive that event's timestamp. The
  max-concurrency invariant returned from 82 to exactly 64 requests at time
  zero; the focused regression and all 42 frontend tests pass.
- The post-fix predictive artifacts are under
  `analysis_e2e_aligned_frontend_fix`. Throughput remains -0.248%, proving the
  patch changes clock accounting rather than GPU work. E2E mean is now -2.139%
  (p50 +7.156%, p90 -1.353%); client TTFT mean is -8.245% and p50 -4.700%;
  server TTFT mean is +1.746%. TPOT mean remains +11.38%, with p50 -14.39% and
  p90 +59.62%; this distribution residual is disclosed rather than fitted with
  observed DP-rank or engine-ingress conditioning.
- Alignment infrastructure now joins physical kernels across ranks by semantic
  operation/category ordinal, accepts byte-identical trace copies, and matches
  workload iterations relative to each source's iteration origin. These fixes
  preserve physical source IDs and reject ambiguous matches.

### Final scoped validation

- Analyzer alignment tests: 65 passed.
- Alignment launcher tests: 41 passed.
- Generic replay frontend tests: 42 passed, including the multi-slot deferred
  timestamp regression.
- Clean DeepSeek and retained L1 Python behavioral tests: 89 passed.
- DeepSeek Rust arch/worklet/timing filter: 17 passed.
- Full Rust timing suite: 401 passed, 2 ignored internal microbenchmarks.
- Scoped Ruff and scoped `rustfmt --check` pass; `git diff --check` passes.
- `profiling/profile.db` remains `skip-worktree`; SQLite integrity is `ok`.
