---
name: dev-calculate-kv-cache-capacity
description: "Use when the user asks for KV cache size, attention-state size, bytes per token, max cached tokens, max concurrent requests, or KV capacity of a model or deployment. For a model ServingStudio Sim already has an arch for, read the numbers from the simulator (`launcher <preset> --kv-report`, a run's `raw/kv_cache.json`, `simulator cost-trees`' `kv_cache`) instead of calculating by hand. For a model without an arch, derive its cache part by part (per-token, sliding-window, per-request) from config/public sources and vLLM's allocator, in the `KvCacheLayout` form its arch will declare, and show every step."
---

# Dev Calculate KV Cache Capacity

First check whether the model has a ServingStudio Sim arch (an arch `type` in
`uv run python -m launcher list-params`, a file in `simulator/src/arch/`). That
decides the route.

## Existing arch: read the simulator, do not calculate

The answer has two halves, and each comes from a different block of the config:

- **arch**: the layout (each part's growth, bytes per entry, padding, copies,
  prefix reuse; the engine block and fixed bytes), the GPUs one KV partition
  spans (`attention_shards`), and `max_model_len`;
- **worker**: the KV budget (`attn_gpu_memory_gb`), `kv_overhead: vllm|none`,
  prefix caching and its resume points, the tokens in flight
  (`kv_max_in_flight_tokens`, or 2 × `max_batch_tokens`), and any units the
  worker sets aside (PD decode).

| Question | Read it from |
| --- | --- |
| Max concurrency / KV tokens of a deployment | `uv run python -m launcher <preset> --kv-report` (`--kv-report-json` for the machine form). No profile.db, GPU or run. |
| A what-if (budget, context, overhead mode, chunk size) | the same, with `--override`, e.g. `pools.main.groups.0.worker.attn_gpu_memory_gb=100`, `pools.main.groups.0.arch.max_model_len=131072`, `pools.main.groups.0.worker.kv_overhead=none` |
| What a finished run used | `<log_dir>/raw/kv_cache.json` (fields: `simulator/src/log/README.md`) |
| The arch's layout alone | `simulator cost-trees <blocks.json>` (a JSON list of `{gpu, arch}`): each build's `kv_cache {attention_shards, layout}`. Bytes are over one partition; divide by `attention_shards` for one GPU. |

Quote the printed numbers and the per-part split ("One request at max_model_len
N in vLLM's estimate, per GPU"); do not re-derive them. The report prints the concurrency under both
`kv_overhead` modes, so the engine's tax is the gap between the two.

When a printed number disagrees with a vLLM log ("Maximum concurrency for N
tokens per request", "GPU KV cache size"), the arch's `kv_cache_layout` is wrong
or missing a part: fix it with `impl-compose-arch` instead of correcting the
answer by hand.

## New model: derive the layout

The derivation is the arch's future `kv_cache_layout`, so write it in that
form. Show every step with substituted values. Mind the cache dtype, which is
often not the weight dtype.

1. **Budget.** Prefer vLLM's logged "Available KV cache memory" from a capture.
   Without one, use the per-GPU HBM (`operate-gpu-spec`) times
   `gpu_memory_utilization`, minus the weights (mind their dtype), activations
   and CUDA graphs. Say which reserves you assumed.
2. **Parts.** Classify each tensor that persists between decode steps into one
   part, listing its layers and its bytes per entry on one GPU:
   - per token: GQA K/V (`max(num_kv_heads, tp)` heads, since replicated heads
     are still stored), MLA latent plus RoPE key, a DSA indexer key or a
     compressed cache (one entry per `tokens_per_entry` tokens);
   - sliding window: SWA layers, a short-convolution window;
   - per request: GDN/KDA/Mamba recurrent and conv state, ring buffers.

   Note whether a part is split across ranks (KV heads) or held whole on each
   rank (MLA latent, recurrent state).
3. **vLLM's overhead on top.** Add page padding (hybrid models unify page
   sizes), the block size, and extra state copies (one more per speculative
   block). With prefix caching on, `mamba_cache_mode: align` adds one more
   copy (the previous step's state) to vLLM's max-concurrency estimate only: a
   running request holds `1 + num_speculative_blocks` and frees that block the
   step after crossing a block boundary. Count the null block as well.
4. **One request at `max_model_len` L, and capacity.** Charge each part by its
   growth:
   - per token: L rounded up to whole blocks;
   - sliding window: `min(L, W − 1 + in-flight)`, where in-flight is
     2 × `max_num_batched_tokens` under async scheduling;
   - per request: bytes × copies.

   vLLM's "Maximum concurrency for L tokens per request" is the budget over this
   one-request charge (with the align copy); the requests it actually runs at
   once are bounded by the held charge (without it). Its "GPU KV cache size: N
   tokens" is a token pool only when every part grows per token.
5. **Hand off.** Implement the parts as the arch's `kv_cache_layout`
   (`impl-compose-arch`), pin it against the capture's logged concurrency, and
   check `launcher <preset> --kv-report` reproduces this derivation. From then
   on, the model takes the first route.

## Evidence rules

- Use `top-explore-models` first when the model repo/config is not resolved.
- Treat `config.json`, the README/model card, official blog, technical report,
  paper and release notes as primary evidence. For what vLLM allocates, read the
  model's vLLM implementation and its attention backend's cache spec.
- Use `dev-lookup-transformers-model` when public sources do not prove the
  persistent tensors or when code contradicts them.
- Do not guess for MLA, DSA, GDN, quantized, compressed or custom caches. If a
  shape is unresolved, report the field and the source that would settle it.

## Report back

- Existing arch: the command you ran, the printed report (or its relevant
  lines), and which arch and worker values drove the result.
- New model:
  - the budget and its reserve assumptions;
  - each part with its formula, substituted values and source;
  - vLLM's overhead terms;
  - the one-request charge and the resulting concurrency or token capacity;
  - caveats (sharding, mixed layer types, sliding windows, quantization
    metadata, speculative decoding, unresolved shapes).
