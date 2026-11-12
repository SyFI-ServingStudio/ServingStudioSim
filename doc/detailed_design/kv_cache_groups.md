# KV cache groups — decision record

Status: implemented 2026-10-11 (branch `kv-cache-groups`). The design now lives
in `L4.md` (KV cache layout) and `L5.md` (§3 store, §3.1 KV accounting, §3.2
resume positions and the soft tier, §4 chunk alignment, §7–§11). This file
keeps only what was decided and the engine evidence behind it.

## Problem

Each arch described its persistent state with one per-token byte count (plus
an optional per-request recurrent state), and two stores, one for full
attention and one for hybrids, worked in tokens of it. That counted sliding windows at full length
(Inkling capacity about 3.9x too small at 131072 tokens per request), left out
DeepSeek-V4.1's SWA and compressor rings, ignored GQA head replication when TP
exceeds `num_kv_heads`, and gave the prefix cache no per-cache-type hit rule.
The hybrid store also lacked the PD and AFD capabilities.

## Decisions

- The arch declares its state as `KvCacheLayout`: `KvCachePart`s growing
  `PerToken { tokens_per_entry }`, `SlidingWindow { window_tokens }`, or
  `PerRequest`, with bytes summed over one KV partition's ranks. The contract
  method is required; the scalar methods are gone.
- The simulator models structure, not an allocator: no block ids, hashing,
  layer grouping, page unification, or 16-token block rounding. Engine terms
  that move capacity are declared separately (`PartOverhead`, `EngineOverhead`)
  and charged only under `kv_overhead: vllm` (default); `none` charges model
  state only, an upper bound whose gap to `vllm` is the engine's overhead.
- One store, `KvCacheStore`, counts KV units (gcd of entry bytes) through
  `KvAccounting`: held while running, needed to admit, kept by a retained entry.
  A layout of one per-token part counts plain tokens.
- Prefix entries resume only at chosen positions (`prefix_resume_points`,
  default `prompt_end` under `vllm`, `response_end` under `none`;
  `prefix_resume_interval` adds a series). A hit is the largest position at or
  below the declared prefix. `prefix_cache_event.full_hit_tokens` logs what the
  positions cost.
- Live checkpoints join retained prefixes in the soft tier: evictable, never in
  admission or `kv_usage`.
- SGLang (separate SWA and mamba pools, decode reservation) is out of scope; it
  would be another `KvOverhead` value.

## Where the code differs from the proposal

- Names: `KvClass`/`KvExtent`/`granule` became `KvCachePart`/`KvGrowth`/`entry`;
  `KvCharge` became `KvAccounting`; `engine | ideal` became
  `kv_overhead: vllm | none`; `prefix_retention` became `prefix_resume_points`
  (`prompt_end`, `response_end`, `prompt_and_response_end`, `every_block`).
- No default `kv_cache_layout()`; every arch migrated at once.
- Without a declared engine block nothing is block-rounded, so the proposal's
  example hits read 8500 / 7999 rather than 8496 / 7984 (L5.md §3.2).
- `run_meta.json` v6 `kv_pools` records `capacity_units`, `bytes_per_unit`,
  and `kv_overhead`, not capacity under both modes.

## vLLM evidence (fork 892da082)

- Grouping: specs bucketed by type and split into equal layer-count groups;
  `num_blocks = available // max_group_bytes`. GLM-5.3-Flash uses
  `kv_cache_utils.py` `_get_kv_cache_groups_glm5_next`; DeepSeek-V4/V4.1 use
  packed groups where every group's block costs the widest page.
- Logged `GPU KV cache size: N tokens` is concurrency at `max_model_len`:
  `num_blocks / Σ_g blocks_g(max_model_len) × max_model_len`.
- Admission: `scheduler_reserve_full_isl` gates a waiting request on its whole
  prompt, SWA capped at `cdiv(W - 1 + max_in_flight, bs) + 1` blocks with
  `max_in_flight = 2 × max_num_batched_tokens` under async scheduling. No
  future-decode reservation; recompute preemption of the newest request.
- Prefix: with `prefix_cache_retention_interval = 0`, SWA tails and mamba states
  are kept only at the replay boundary `num_prompt - 1`
  (`kv_cache_coordinator.get_replay_boundaries`, `reachable_block_mask`); align
  mode splits chunks on blocks (`Scheduler._mamba_block_aligned_split`).
- Reuse pattern depends on the chat template: Inkling, GLM-5.3, MiMo-V2.6,
  MiniMax-M3 and Qwen3.8 keep earlier thinking in history (a next turn extends
  the previous context); Qwen3 to 3.6, GLM-5.2 and Nemotron-3 drop it.

## Capacity evidence

| Model | Deployment | Logged | Status |
|---|---|---|---|
| DeepSeek-V4-Flash | DP4, 1M, 82.36 GiB | 8.19x async / 11.84x sync (current allocator) | test |
| DeepSeek-V4.1-Flash | TP4, 128k, 86.14 GiB | 404.89x | test |
| GLM-5.2 NVFP4 | TP4, 131k / 8k, no MTP and MTP5 | 7.14x, 6.05x, 5.84x, 32.18x and their token counts | test |
| GLM-5.3-Flash | TP4, prefix caching, block 2176 | 405.85 / 75.13 / 19.23 / 13.15 / 3.13x | test |
| Inkling-NVFP4 | TP8, 73.85 GiB, 8k to 512k | 56.00 / 45.6 / 37.61 / 27.9 / 18.3x | test |
| Inkling-Small-NVFP4 | TP4, 106.11 GiB | 69.6 / 45.25x at 8k / 128k | test |
| Qwen3.6 | TP1, 9216 | 367.67 / 237.17x; 294.13x with prefix caching | test |
| MiMo-V2.6-Flash | TP4, 128k, 110.52 GiB | 118.76x | reproduced by hand; arch unmerged |

Every test asserts within 0.5% (`kv_cache_layout_reproduces_*` in each arch).
The logs come from the fork the alignment captures pin (892da082) or one with
the same allocator. Two corrections came out of them:

- GLM-5.2 / 5.3 NVFP4 gives an index key only to the 21 layers that compute
  top-k, plus the MTP layer. Logs before about 2026-09-08 (an older vLLM)
  show one on all 78 layers; the old 341,120-token test pinned that. SGLang
  still allocates all 78, so its arch keeps that layout.
- vLLM sizes the pool from the smallest rank's free memory. A log line
  printed by one rank can overstate it (GLM-5.2's TP0 printed 37.84 GiB for a
  pool sized from 34.78).

Dynamic usage targets (per-iteration `GPU KV cache usage`): GLM-5.3 dflash2 pack
cases 05/07/08, GLM-5.2 MTP5 cases 08 and 15, GLM-5.2-FP8 DP8. The SWA models
ran with stats off; rebuild their occupancy from `*_metrics.jsonl`
`decode_kv_lens`.
