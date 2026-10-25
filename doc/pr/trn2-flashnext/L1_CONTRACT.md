# Neuron L1 contract for Qwen3.8-Flash-Next BF16 on Trainium2 (LNC2)

This is the single source of truth for the Neuron L1 kinds: Python table = Rust `KernelSpec::KIND`,
args field order = Python `KernelArgs` field order = Rust `ArgsPayload` fields (Rust payload keys are
emitted in any order; the set must equal the Python fields plus `backend`).

Conventions:

- DB key `gpu_name` = `"AWS Trainium2 LNC2"` (catalog `Trainium2-LNC2`).
- Every Neuron backend is `neuron_<impl>`, `<impl>` = the op module's `IMPLS` key (`profiling/kernels/neuron_common.py::neuron_spec`).
- Time = NRT system-trace device time (`flashnext.neuron.sim_l1.measure`), median over 30 calls, summed over the
  NEFFs one call launches. Host launch overhead is the separate kind `region_launch_overhead`.
- All shapes are **per rank** (local head counts, local experts, local vocab rows). The arch derives them from
  the parallel layout; rows never carry `tp`/`ep`.
- `dtype` fields are `"bf16"` (DType.BF16) unless noted.
- Grids (Rust `sweep_grid`), named here and reused below:
  - `TOKENS`  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
  - `DECODE`  = [1, 2, 4, 8, 16, 32, 64, 128, 256]  (decode/verify requests)
  - `CHUNK`   = [16, 128, 256, 512, 1024, 2048, 4096, 8192]  (one prefill request chunk)
  - `QSA_PREFILL_ROWS` = [128, 512, 2048, 8192], `QSA_PRIOR` = [0, 32768]
  - `QSA_DECODE_ROWS` = [1, 8, 64, 256], `QSA_CONTEXT` = [4096, 32768, 131072]
  - `COMM_BYTES` = [16384, 262144, 1048576, 4194304, 16777216]
  - `REGIONS` = [1, 2, 4, 8, 16, 32, 64]
- Model geometry (from `/scratch/models/Qwen3.8-Flash-Next/config.json`): H=2560; GDN 16 K heads / 48 V heads,
  dim 128, conv 4; QSA 24 Q / 2 KV heads, dim 256, indexer 4 heads x 128, budget 2048 tokens, compress 4;
  MoE 512 experts top-10, inter 640, shared inter 640; HC 4 streams, low rank 320; PLE 16 heads x 160, table
  320,001,536 rows; vocab 248320.
- Per-rank geometry by tp: GDN `num_key_heads = 16/tp`, `num_value_heads = 48/tp` (tp 1,2,4,8,16).
  QSA `num_q_heads = 24/tp`, `num_kv_heads = 2 if tp == 1 else 1` (tp 1,2,4,8; no tp16).
  MoE `local_experts = 512/ep`, `intermediate_size = 640/tp` (moe tp), shared `intermediate_size = 640/split`.

## Shared kinds reused with a Neuron backend

Each gets a Neuron grid branch in Rust: when **every** backend in the config starts with `neuron_`, the grid
below replaces the GPU grid (the GPU grid stays unchanged for every other config).

| kind | Python args (existing) | Neuron backends | Neuron grid | used for |
|---|---|---|---|---|
| `grouped_rms_norm` | num_tokens, hidden_size, num_groups, hidden_dtype | neuron_torch | num_tokens TOKENS | HC entry norm (hidden 2560, groups 4) |
| `hc_combine` | num_tokens, hidden_size, hc_mult, hidden_dtype | neuron_torch | num_tokens TOKENS | standalone HC combine |
| `ple_ngram_ids` | num_tokens, num_reqs, ngram_context_len, heads_per_ngram | neuron_torch | num_tokens TOKENS x num_reqs [1, 8, 64, 256], mask num_reqs > num_tokens | PLE hash ids (ctx 2, heads 8) |
| `ple_gate` | num_tokens, hidden_size, hc_mult, dtype | neuron_torch | num_tokens TOKENS | PLE gate |
| `ple_short_conv` | batch_size, query_len, channels, kernel_size, dilation, mode, dtype | neuron_torch | decode: batch DECODE x query_len [1]; spec: batch DECODE x query_len [2, 3, 4]; prefill: batch [1] x query_len TOKENS | PLE conv (channels = hc_mult x hidden = 10240, k 4, dilation 3) |
| `embedding_row_gather` | num_tokens, num_heads, head_dim, table_rows, residency, weight_dtype | neuron_torch | num_tokens TOKENS | PLE local gather (16 x 160, rows = 320001536/shards, residency "device") and token embed (1 x 2560, rows = 248320/tp) |
| `gemm_fp32_output` | m, n, k, input_dtype | neuron_torch | m TOKENS | lm_head logits (n = 248320/tp, k = 2560) |
| `single_gemm` | m, n, k, dtype | neuron_torch | m TOKENS | MTP fc (n 2560, k 5120) |
| `all_reduce` | num_gpus, message_size_bytes, dtype, fabric | neuron_cc | message_size_bytes COMM_BYTES | in-graph all-reduce |

`fabric` for Neuron collectives (Rust `common::Fabric` gains three variants, serde names):
`neuron_c1` (one rank per chip, chips strided), `neuron_c2` (2 ranks per chip), `neuron_c4` (4 ranks = every
logical core of each chip). A group is `num_gpus` ranks; chips = num_gpus / cores_per_chip.

## New kinds

GDN common args (after the shape args, in this order): `hidden_size, num_key_heads, num_value_heads, head_dim,
conv_kernel_size, dtype`. Shape args come first.

| kind | shape args | backends | grid | notes |
|---|---|---|---|---|
| `gdn_prefill_mixer` | num_tokens | neuron_torch, neuron_nki | num_tokens CHUNK | whole mixer, one request chunk, pool gather/scatter in region |
| `gdn_decode_mixer` | batch_size, query_len | neuron_torch, neuron_nki | batch_size DECODE (query_len is Config) | query_len 1 = decode, 1+k = MTP verify; bench `gdn_decode_mixer` (L=1) + `gdn_verify_mixer` (`spec_tokens` = L) |
| `gdn_in_proj` | num_tokens | neuron_torch | TOKENS | |
| `gdn_conv1d_prefill` | num_tokens | neuron_torch, neuron_nki | CHUNK | |
| `gdn_conv1d_decode` | batch_size, query_len | neuron_torch, neuron_nki | DECODE | |
| `gdn_split_gating` | num_tokens | neuron_torch | CHUNK | |
| `gdn_prefill_core` | num_tokens | neuron_torch, neuron_nki | CHUNK | |
| `gdn_decode_core` | batch_size, query_len | neuron_torch, neuron_nki | DECODE | |
| `gdn_output` | num_tokens | neuron_torch | TOKENS | |
| `gdn_state_io` | batch_size | neuron_torch | DECODE | decode state gather + scatter |

QSA common args (after shape args): `hidden_size, num_q_heads, num_kv_heads, head_dim, index_heads,
index_head_dim, index_topk_tokens, compress_ratio, dtype`. Every QSA kind has `mode` ("prefill" | "decode") as
its FIRST arg (a Config field in Rust). Shape args: `num_rows` (prefill: tokens of the one request chunk;
decode: requests, one row each) and, for context-dependent kinds, `context_len` (prefill: prior + num_rows,
the total visible context of the chunk's last row; decode: the KV length including the new token).
Rust grids: prefill rows QSA_PREFILL_ROWS x prior QSA_PRIOR (Input stays `{num_rows, context_len}`;
`cache_coords` re-axes prefill to `(num_rows, context_len - num_rows)` and `enumerate` emits
`context_len = prior + num_rows`); decode rows QSA_DECODE_ROWS x context QSA_CONTEXT. Both modes use
`Cache2DLinear(Extrapolation::Weighted)` (`cache_kind` cannot see `mode`); on the two-point prior axis Weighted
fits no slope and holds the boundary, i.e. behaves as Clamp (pinned by `neuron_grid::qsa_cache_tests`). Row-only kinds: prefill QSA_PREFILL_ROWS, decode QSA_DECODE_ROWS.

| kind | shape args | backends | notes |
|---|---|---|---|
| `qsa_mixer2` | num_rows, context_len | neuron_torch_sparse, neuron_nki | two regions (front: project..select, back: attention+o_proj), select = nki; bench `qsa_mixer2_{prefill,decode}` with `select_impl == "nki"` |
| `qsa_project` | num_rows | neuron_torch | |
| `qsa_indexer_project` | num_rows | neuron_torch | bench `indexer_project` |
| `qsa_cache_write` | num_rows | neuron_torch | |
| `qsa_indexer_compress` | num_rows | neuron_torch | bench `indexer_compress` |
| `qsa_indexer_scores` | num_rows, context_len | neuron_torch, neuron_torch_bf16 | bench `indexer_scores` |
| `qsa_indexer_select` | num_rows, context_len | neuron_torch, neuron_nki | bench `indexer_select` |
| `qsa_attention` | num_rows, context_len | neuron_torch_dense, neuron_torch_sparse, neuron_nki | |
| `qsa_output` | num_rows | neuron_torch | |

MoE (shape arg `num_tokens`, grid TOKENS for all):

| kind | args (full order) | backends | notes |
|---|---|---|---|
| `moe_router` | num_tokens, hidden_size, num_experts, top_k, dtype | neuron_torch, neuron_torch_topk_first, neuron_torch_bf16, neuron_nki_topk | |
| `moe_routed_experts` | num_tokens, hidden_size, num_experts, top_k, local_experts, intermediate_size, dtype | neuron_auto, neuron_nki, neuron_nki_tkg, neuron_nki_tkg_all, neuron_nki_cte, neuron_nki_cte_cf2, neuron_torch_gather_w, neuron_torch_dense, neuron_torch_gather, neuron_torch_onehot | critical-path EP rank, real layer-0 routing (bench `routing == "real"`) |
| `moe_shared_expert` | num_tokens, hidden_size, intermediate_size, dtype | neuron_torch, neuron_nki | intermediate = 640/split |
| `moe_block` | num_tokens, hidden_size, num_experts, top_k, local_experts, intermediate_size, shared_intermediate_size, dtype | neuron_auto, neuron_nki | router torch_topk_first, shared torch; bench impl `torch_topk_first/<routed>/torch` -> `neuron_<routed>`; real routing |

Rust: `moe_routed_experts` declares an `infeasible_mask` when its config has exactly one backend whose kernel
cannot run part of TOKENS: nki_tkg T > 128, torch_gather_w T > 16, torch_dense T > 256, nki_tkg_all T > 512,
nki_cte / nki_cte_cf2 / torch_onehot T < 128.

HC (shape `num_tokens`, grid TOKENS, args `num_tokens, hidden_size, hc_mult, lowrank, dtype`):
`hc_mix` (neuron_torch, neuron_torch_cat), `hc_mix_final` (neuron_torch; final mixer, no inject),
`hc_combine_mix` (neuron_torch, neuron_torch_cat; combine of the previous block fused with the next mix).

PLE (shape `num_tokens`, grid TOKENS):

| kind | args | backends | notes |
|---|---|---|---|
| `ple_project` | num_tokens, embed_dim, hidden_size, hc_mult, dtype | neuron_torch | embed_dim = 16*160 = 2560 |
| `ple_residual` | num_tokens, hidden_size, hc_mult, dtype | neuron_torch | |
| `ple_post` | num_tokens, embed_dim, hidden_size, hc_mult, conv_kernel_size, dilation, dtype | neuron_torch | fused project+gate+conv+residual (post all-reduce) |
| `ple_block` | num_tokens, num_heads, head_dim, table_rows, hidden_size, hc_mult, dtype | neuron_torch | whole block with the rank's local table slice, all-reduce omitted |

Head (grid TOKENS on num_tokens):

| kind | args | backends |
|---|---|---|
| `vocab_argmax` | num_tokens, vocab_size, dtype | neuron_torch, neuron_nki (vocab_size = local rows) |
| `lm_head_argmax` | num_tokens, hidden_size, vocab_size, dtype | neuron_torch, neuron_nki |

Collectives (args `num_gpus, message_size_bytes, dtype, fabric`, like `all_reduce`; grid COMM_BYTES; COMM metric
family; backend `neuron_cc`): `all_gather` (message = per-rank input shard), `reduce_scatter` (message =
per-rank full input), `all_to_all` (message = per-rank buffer). Bench source
`logs/bench/collectives_ingraph.jsonl` `per_cc_device_ms` (drop rows with error or non-positive value).

Launch: `region_launch_overhead` args `num_regions, dtype`; backend `neuron_torch`; grid REGIONS; time =
pipelined wall per step of `num_regions` back-to-back trivial compiled regions (host launch + inter-region
gap). Seed: `logs/probe_overhead.json` (`trivial_add` = 1 region, `chain_k_regions` k=8 at tokens=1).

## Implementation notes (deviations settled during the build)

- Multi-region rows: on the lite route every NEFF's system-trace `model_name` is None, so
  `runtime.device_exec_ms` merges a two-region call's NEFFs and `total_median_ms` is the median over all
  executions, not the per-call sum. The bench's `qsa_mixer2_*` rows are therefore not imported;
  `sim_l1/qsa.py` sums consecutive executions per call itself and `qsa_mixer2` is live-profiled.
- PLE ops keyed by `num_tokens` alone use one layout rule (builder and importer): `num_tokens <= 256` is a
  decode step (N requests x 1 token), larger is one prefill chunk (1 x N).
- `ple_short_conv` mode `spec` times the same conv over `query_len` rows (the op has no rollback path).
- `moe_block` splits the shared expert over all ep x tp ranks (`shared_intermediate_size = 640/(ep*tp)`).
- `region_launch_overhead` is wall time of N distinct trivial regions back to back (minimum of two runs on a
  loaded host); re-measure on a quiet host.
- Collectives: multi-chip groups come from `flashnext/neuron/sim_l1/collectives.py` (v2 sweep, chip locks held)
  and are imported; the Sim's `neuron_cc` runner measures one-chip groups only. one-core-per-chip (`neuron_c1`)
  all_to_all fails on this runtime at every size ("Failed to schedule neff execution"), so it has no rows.
