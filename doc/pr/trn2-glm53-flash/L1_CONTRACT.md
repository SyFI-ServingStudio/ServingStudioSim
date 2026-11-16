# Neuron L1 contract for GLM-5.3-Flash FP8 on Trainium2 (LNC2)

The single source of truth for the Neuron L1 kinds of `glm53_flash_neuron_fp8`: Python kind name =
Rust `KernelSpec::KIND`; the args set = the Python `KernelArgs` fields = the Rust `ArgsPayload`
keys minus `backend` (Rust emits them in any order).

## Conventions

- DB key `gpu_name` = `"AWS Trainium2 LNC2"` (catalog `Trainium2-LNC2` in `gpu/spec.json`: one LNC2
  logical core = 2 physical NeuronCore-v3, 24 GiB HBM bank, 158 dense BF16 TFLOPS).
- Every backend is `neuron_<impl>`, `<impl>` the bench row's `impl` (the engine op module's
  `IMPLS` key). Registered impls: `profiling/kernels/neuron_common.py::GLM_IMPLS`. A Neuron
  backend has `device_family = "neuron"`, architecture `Trainium2`; it never runs on CUDA and a
  CUDA backend never on Neuron (`profiling/db/registry.py::BackendSupport.allows_gpu`).
- **Rows are imported, never profiled here.** The runner (`profiling/runners/neuron/imported.py`)
  refuses a live run with the importer command; `--no-gpu` runs fail on a missing row instead.
- Time = NRT device time of ONE compiled region per call (`glmflash/neuron/runtime.device_bench`
  `total_median_ms`), from the glm5-trn2 bench scripts. Host launch and dispatch are not in any
  row: the arch's pipelined launch model owns them.
- All shapes are **per rank**: local heads (`64 / attn_tp`), local experts (`288 / ep`), the vocab
  shard (`ceil(154880 / attn_tp)`). Rows never carry `tp` / `ep`.
- `dtype` = activation dtype `"bf16"`; `weight_dtype` = how the weights sit in HBM: `"bf16"`
  (attention, mHC, embedding, lm_head stay BF16 in the FP8 checkpoint) or `"fp8_e4m3"` (dense FFN,
  shared and routed experts: FP8 in HBM, upcast to BF16 in-graph; Trainium FP8 is IEEE e4m3, the
  engine re-encodes the checkpoint's e4m3fn as w/2 with a doubled block scale).

## Grids (Rust `simulator/src/timing/kernels/neuron_grid.rs`)

| name | values |
|---|---|
| `TOKENS` | 1, 2, 4, ..., 8192 |
| `DECODE` | 1, 2, 4, ..., 256 |
| `CHUNK` | 128, 256, 512, 1024, 2048, 4096, 8192 |
| `DSA_PREFILL_ROWS` | = `CHUNK` |
| `DSA_CONTEXT` | 2048, 4096, 8192, 16384, 32768 |
| `FFN_ROWS` | 8, 16, ..., 65536 (world-gathered rows) |
| `HEAD_ROWS` | 1, 2, ..., 1024 |
| `COMM_BYTES` | 16 KiB, 256 KiB, 1 MiB, 4 MiB, 16 MiB |

1D kinds use `Cache1DLinear` (linear interpolation, linear extrapolation past the grid). DSA
kinds use `Cache2DLinear(Weighted)`; prefill masks cells with rows > context.

## Kinds

Model geometry (`model/config/glm53_flash.json` = `zai-org/GLM-5.3-Flash` FP8): H 4096; KDA 64 heads
x 128, short conv 4; MLA (NoPE) 64 heads x 256, q_lora 1536, kv_lora 512; indexer 32 heads x 128,
top-2048 (512 pools of 4 + tail); MoE 288 experts top-8, inter 2048, 1 shared; dense FFN 12288
(layers 0-2); mHC 4 streams, Sinkhorn 20; vocab 154880.

| kind | shape args | config args (in order) | grid | bench source (`logs/bench/`) | impls |
|---|---|---|---|---|---|
| `kda_prefill` | num_tokens | hidden_size, num_heads, head_dim, conv_kernel_size, dtype, weight_dtype | CHUNK | `kda.jsonl` kind `kda_prefill` (`tokens`) | torch, torch_barrier, torch_nobarrier |
| `kda_decode` | batch_size | as kda_prefill | DECODE | `kda.jsonl` kind `kda_decode` (`batch`) | torch, torch_nobarrier |
| `dsa_prefill` | num_tokens, context_len | hidden_size, num_heads, kv_lora_rank, head_dim, index_heads, index_head_dim, index_topk, dtype, weight_dtype | DSA_PREFILL_ROWS x DSA_CONTEXT (rows <= context) | `dsa.jsonl` kind `dsa_prefill` (`tokens`, `context`) | radix, topk |
| `dsa_decode` | batch_size, context_len | as dsa_prefill | DECODE x DSA_CONTEXT | `dsa.jsonl` kind `dsa_decode` (`batch`, `context`) | radix, topk |
| `moe_partial` | num_tokens | hidden_size, num_experts, top_k, local_experts, intermediate_size, shared_intermediate_size, dtype, weight_dtype | FFN_ROWS | `moe.jsonl` kind `moe_partial` (`tokens` = gathered rows M, `ep`) | auto, dense_local, cte_hybrid |
| `dense_ffn_partial` | num_tokens | hidden_size, intermediate_size, dtype, weight_dtype | FFN_ROWS | `moe.jsonl` kind `dense_ffn_partial` | dense |
| `moe_router` | num_tokens | hidden_size, num_experts, top_k, dtype | FFN_ROWS | `moe.jsonl` kind `router` | fp32 |
| `mhc_pre` | num_tokens | hidden_size, hc_mult, sinkhorn_iters, dtype | TOKENS | `mhc_mhc_pre.jsonl` (`bench_mhc_head.py`) | torch_bf16, torch_fp32 |
| `mhc_post` | num_tokens | hidden_size, hc_mult, dtype | TOKENS | `mhc_mhc_post.jsonl` | torch |
| `lm_head` | num_tokens | hidden_size, vocab_size, hc_mult, dtype | HEAD_ROWS | `mhc_lm_head.jsonl` (`tp` = shards) | torch |
| `embed` | num_tokens | hidden_size, vocab_size, hc_mult, dtype | TOKENS | `mhc_embed.jsonl` | torch |
| `all_gather` | message_size_bytes | num_gpus, dtype, fabric | COMM_BYTES | trn2-flashnext rows (below) | cc |
| `reduce_scatter` | message_size_bytes | num_gpus, dtype, fabric | COMM_BYTES | trn2-flashnext rows (below) | cc |

What each region holds (the engine's `engine.py` region bodies, without their collectives):

- `kda_*`: the whole KDA block of one rank: q/k/v/f/g/b projections of its `64 / tp` heads (f_a,
  g_a replicated), short conv, chunked delta rule (prefill: one chunk of T tokens with incoming
  state) or the recurrent step (decode: B rows, own state each), gated norm, o_proj partial.
- `dsa_*`: the whole DSA block: q_a / kv_a (replicated) + norms, the replicated indexer and its
  top-k selection over the context bucket `[:C]`, latent / indexer-key / gate cache writes, MLA over
  the rank's `64 / tp` heads, o_proj partial. Prefill is one chunk of T at the end of the context
  bucket (`start = C - T`); decode is B rows at context C.
- `moe_partial`: router + the rank's 9 local experts (EP32) + its 1/ep shard of the shared expert
  over all M world-gathered rows; the critical (most loaded) EP rank under real routing.
  `auto` is the engine's `ffn_auto` (dense_local up to M <= 1024 at EP32, cte_hybrid above); the
  arch default lists `[dense_local, cte_hybrid]`, i.e. the measured best per shape.
- `mhc_pre`: `hc_pre` (fp32 RMS of the flattened streams, mixing projection, Sinkhorn) + the
  module's RMSNorm, on the rank's own `R / tp` rows. `mhc_post`: `hc_post`.
- `lm_head`: `final_hidden` (stream mean + norm) + vocab-parallel logits + local argmax on the
  sampled rows. `embed`: `embed_partial` + the expand to 4 streams on the group's rows.

## Collectives

`all_gather` (message = per-rank input shard) and `reduce_scatter` (message = per-rank full
input), args `num_gpus, message_size_bytes, dtype, fabric` as on the trn2-flashnext branch.
`fabric` names how the group's (contiguous) ranks sit on chips (Rust `common::Fabric`):
`neuron_c4` = every logical core of each chip (TP4, TP8 = 2 chips, TP16 = 4 chips, the 32-rank
world), `neuron_c2` = 2 cores of one chip (TP2), `neuron_c1` = one core per chip.

Rows: the trn2-flashnext branch's imported in-graph rows (`flashnext/neuron/sim_l1/collectives.py`
v2 sweep: device-time slope of 33 vs 1 chained collectives in one region, barrier before every
call, median of 3), kept verbatim in `tools/neuron-bench-import/data/trn2_flashnext_collectives.jsonl`
(all_gather / reduce_scatter over 2 / 4 / 8 / 16 / 32 ranks at every `COMM_BYTES`). They are
measured on the Qwen3.8-Flash-Next campaign's runtime (same image, same functional collectives)
and transfer as they are; they are **not** re-measured for GLM.

## Import and the derived grid fill

`tools/neuron-bench-import/import_bench.py --fill-grid` (re-runnable):

1. imports every clean bench row (`device_ms > 0`, no `error`); `backend_version` =
   `imported:<file>:<line>`, `verified = 1`;
2. fills every grid cell the arch can ask for at `attn_tp` in {1, 2, 4, 8, 16}, world 32, that no
   measured row covers, with `analytic(cell) x r` (`analytic.py`: a Trainium2 roofline per region,
   `grid.py`): r interpolated (log-log) from the same series' measured/analytic ratios
   (`derived:interp(n=..)`), else the kind's median ratio (`derived:kind-median(n=..)`), else 1
   (`derived:analytic`). `verified = 0`. A later import replaces a derived row with the measured
   one of the same key;
3. writes the measured / derived counts per (kind, backend) to `l1_coverage.json`.

Predictions that depend on derived rows are labeled as such in `CONFIG_SEARCH.md`.

## Not in any row

The TP and world collectives inside a region are separate leaves (above). Host time (region
dispatch, per-forward input assembly / H2D / id readback, the step's scheduling) is the arch's
pipelined launch model, its constants fitted on the seed engine (`ALIGNMENT.md`), plus the
prefill-forward device factor. First-execution stalls of a new shape
and sampling (non-greedy) are not modeled.
