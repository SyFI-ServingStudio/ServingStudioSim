# `ssengine.yaml`: SSEngine as one preset

`ssengine.yaml` is the simulator configuration of SSEngine as built (pp-impl `lanes/release`, commit 831a02a): what
`ssengine serve` runs (`ssengine/serve/launch.py` `design_config` + `DESIGN_TIERS`), with the tier sizes of the M6
evaluation's full setting (`docs/results/eval_m6.md`). Each mechanism is its own key. An ablation is this preset with
the key(s) in the last column changed, nothing else.

Paths below are relative to `pp-impl/lanes/release`. "未验证" marks a value whose engine source was not found or that
the simulator models differently; the right column says how to check it.

## Keys and their engine source

| Key | Value | SSEngine behavior and source | Ablation |
| --- | --- | --- | --- |
| `deployment: pp`, `pp_size: 4`, `layer_partition` | 4 stages, [12, 11, 11, 11] | `ssengine/config.py` `LAYER_SPLIT = [12, 11, 11, 11]`, one B200 per stage | – |
| arch keys, `backends_file` | as the other presets (README.md) | the engine's kernels; see `backends_ssengine.yaml` | – |
| `external_decode` | `true` | README L14: "Prefill and one output token only … the prefill side of a deployment whose decode runs elsewhere" | `false` |
| `max_batch_tokens` | 32,768 | `SCHEDULER_STEPS["good"]`: `mb_tokens=32768` | – |
| `pending_order`, `srpt` | `shortest-prefill-first`, `true` | `good`: `order="srpt"` (started and waiting prompts by remaining compute; `engine/scheduler.py` docstring) | `fifo`, `false` |
| `load_budget_low_tokens`, `load_budget_backlog_lo_tokens`, `load_budget_backlog_hi_tokens` | 4,096 / 1e6 / 3e6 | `good`: `load_budget=True`, `load_budget_low=4096`, `backlog_lo=1_000_000`, `backlog_hi=3_000_000` | remove the three |
| `microbatch_split`, `min_microbatch_tokens` | `even`, 512 | `good`: `even_split=True`, `even_min=512`, `depth=MAX_IN_FLIGHT=4` | `greedy`, 0 |
| `force_schedule_after_ms` | 60,000 | `good`: `force_after_ms=60_000.0` | 0 |
| `force_schedule_hits_first(_when_at_risk)` | absent (off) | `engine/scheduler.py` docstring: "a request whose wait since it arrived has reached the bound goes first", with no hit/miss split named. 未验证 in code: read `Scheduler.form` | `true` |
| `long_prefill_token_threshold` | 0 | `SchedulerConfig` has no long-prompt cap | – |
| `prefill_chunk_alignment` | `plain` | chunks of `min(remaining, budget)`; 未验证 in code (docstring "Budget" only): read `Scheduler.form` | – |
| `attn_gpu_memory_gb` | 100.18 | eval_m6.md: hbmonly = 61.26M tokens = the simulator's pool at 100.18 GB in 8,576-token blocks (7,143 blocks). With 128-token pages the engine profiles its free memory (`model/stage.py` `_profile_blocks`); whether that pool is still 100.18 GB is 未验证: compare a pages run's `Ready.num_blocks` x 128 x 1,635 B | – |
| `kv_block_tokens` | 128 | `config.py` `PAGE_TOKENS = 128` (HBM KV in pages; eval_m6.md "HBM KV in 128-token pages", lane-m6-pack 8d50066) | absent (8,576-token blocks, the eval engine a78dc6f) |
| `kv_part_padding` | `[kda=66969600, kpool_tail=0]` | `docs/design/kept_states.md` "compact": a state is c = 64 x R = 5 = 320 pages of every KV tensor on a stage, the kpool tails inside it. A page on the binding 3-DSA stage is 128 x 1,635 B = 209,280 B, so 320 pages = 66,969,600 B (notes.md unit 2) | absent (vLLM padding: kda 5 blocks + kpool tail 1 block) |
| `prefix_resume_points` | `context_end` | a kept context keeps its end state (`kept_states.md` k1/k2); needed by `recurrent_state_checkpoints` > 1 | – |
| `recurrent_state_checkpoints` | 2 | `config.py` `RunConfig.kept_states = 2` (k2: also the state its round resumed from). Equivalence of the engine's k2 (copy-on-resume, `drop_stale`) and the simulator's `Latest{k: 2}` is 未验证: compare per-session kept states on a short session trace | 1 |
| `dram_tier_gb` | 50 | `serve/launch.py` `--dram-gb` help: "the evaluation's full setting: 50" (the `TierConfig` default is 150) | 0 (with `dram_tier_writes: every-context`, `ssd_tier_writes: every-context`) |
| `dram_tier_gb_per_s` | 50 | `TierConfig.dram_read_gb_s = 50.0` | – |
| `dram_tier_writes` | `hbm-evictions-exclusive` | `DESIGN_TIERS["dram_writes"]`; `TierConfig` comment: a DRAM read releases the entry, HBM writes a context when it evicts it | `hbm-evictions` (inclusive victim) |
| `ssd_tier_gb` | 128 | `TierConfig.ssd_gb = 128.0` ("128 GB per GPU (user, 2026-10-16)"); `--ssd-gb` help "(128)" | 0 (with `ssd_tier_writes: every-context`) |
| `ssd_tier_gb_per_s` | 10 | `TierConfig.ssd_read_gb_s = 10.0` | – |
| `ssd_tier_writes` | `dram-evictions` | `TierConfig` comment: "the victim chain HBM -> DRAM -> SSD … a tier takes only what the level above evicts"; SSD stays inclusive | `every-context` |
| `prefix_tier_balanced_load` | `true` | the server's slot accounting: a slot is one GPU's share of one KV block on all stages (balanced storage; `TierConfig.slot_bytes` comment, `kvpool/manager.py`) | `false` |
| `prefix_tier_loading_area` | 0.1 | `DESIGN_TIERS["loading_area"] = 0.1` | 0 (the engine then uses its 1 s read-wait bound: also set `prefix_tier_max_read_wait_ms: 1000`) |
| `prefix_tier_max_read_wait_ms` | absent (0) | `engine/scheduler.py` docstring, staged loading: "The read-wait bound is not used". The simulator rejects the bound with a loading area | – |
| `prefix_tier_warm_start` | `false` | the evaluation starts empty (handoff `7_offloading.md` §5 "Start state") | – |
| `max_concurrency`, trace | 300, `sessions_2ps_v3.csv` | eval_m6.md: `benchmarks/run.py` sched_good_nN, trace `data/tracelab-v3/m6/sessions_2ps_v3.csv` | – |

## What the simulator does not model, or models differently

- **Tier slot granularity (未验证 impact).** The server stores a context in slots of one 8,576-token block on all
  stages plus 3 state slots of 12.86 MB per GPU (`TierConfig.slot_bytes`, `state_slots = 3`, `kept_states.md` "Cost").
  The simulator charges tiers by the byte: tokens x balanced bytes per token plus the state's bytes.
- **Write-ahead (not modeled).** `TierConfig.hbm_write_ahead` / `dram_write_ahead = 0.9`: a level writes its LRU tail
  ahead of eviction past 90% occupancy. Exclusive DRAM writes from HBM only at eviction ("no write-ahead, as the
  simulator"); whether DRAM -> SSD still writes ahead under exclusive DRAM is 未验证 (read `kvpool/writeback.py`). The simulator writes at eviction and charges no write time
  (eval_m6.md item 5: "the sim charges no write cost").
- **CUDA graphs (未验证 impact).** `RunConfig.cuda_graphs`: microbatches up to 8,192 tokens replay a graph per
  power-of-2 bucket (padded to it). The preset sets no `cudagraph_capture_sizes`, so the simulator times every
  microbatch eagerly at its own size.
- **Null block.** The simulator's engine overhead is one null block of 128 tokens (204 KiB). The engine's reserved
  pages are 未验证 (`FakeConfig.num_blocks` comment counts one null block of 8,576 tokens before pages).
- **External decode and the kept states.** The engine has no hand-back: a round adds one state, and a tier read moves
  only the state the round resumes from (`kept_states.md`). The simulator's HBM take keeps survivors with one added
  state, but its tier reads use `1 + external_decode` = 2 added states, so with k = 2 no earlier state survives a
  tier read. On `sessions_2ps_v3.csv` every round has one output, so `external_decode: false` changes only this
  (notes.md unit 7 gives the measured effect). Not fixed.
- **Host costs.** Plan forming, dispatch and admission host time (scheduler.md "Low load": ~4 ms before a GPU start)
  are not in the simulator.

## Evaluation engine vs this engine

eval_m6.md's measured runs used commit a78dc6f: 8,576-token blocks, the KDA state padded to whole blocks, one kept
state. The engine as built adds 128-token pages, the compact state and k2. To compare against those measured rows,
change `kv_block_tokens` (absent), `kv_part_padding` (absent) and `recurrent_state_checkpoints` (1) together.

## Anchors (2026-11-14, nothing tuned)

Each anchor run is this preset with only the anchor's workload, memory and named tier mode changed. Scripts and the
full item-by-item tables: `logs/20261114_9_pp_u7_equiv/anchors/` (`gen.py`, `analyze.py`, `report.md`,
`sched_check_blocks.md`; logs are not committed). T = the v2-realistic targets the engine was built against
(`pp-impl/third_party/handoff_v2_realistic/data`, simulator outputs of wt-pp-hitrule 5153041a); real = eval_m6.md's
measured rows (engine a78dc6f).

| Anchor | This preset | Target | Deviation and where it comes from |
| --- | --- | --- | --- |
| Scheduling check, `good` | 58 microbatches, all 58 chunk for chunk; 128k TTFT 1,875.1 ms | 58; 1,862.8 ms | +0.66% worst TTFT. Every microbatch's compute is 0.0-0.9% above T (41 of 58 above); the order is exact. The per-kernel source of the 0.9% (profile.db rows vs T's 20261016 profile.db.new) is 未验证: compare the two runs' cost_log kernel times |
| Force check, `good` | 32k TTFT 60.90 s; 1k p50 / p99 17.93 / 33.45 s; 1,300 microbatches | 61.02 s; 17.73 / 33.18 s; 1,303 | -0.2% / +1.1% / +0.8%; 3 fewer microbatches |
| Sessions good_hbm N = 400 (v2 trace) | 24.26 r/s, TTFT p99 60.7 s | 25.06 r/s, p99 25.5 s (engine measured 24.76) | -3.2% r/s. The eval-engine KV form (`kv_block_tokens`, `kv_part_padding`, `recurrent_state_checkpoints` back to the defaults) gives 25.12 (+0.2%): the gap is this engine's KV form, mostly the second kept state (below) |
| Sessions h12d19_ssd128 N = 500 (v2 trace) | 28.61 r/s | 27.69 r/s | +3.3%. T used every-context writes without a loading area; with T's tier mode and the eval-engine KV form: 27.68 (-0.1%) |
| Tier bring-up (DRAM / SSD, balanced / own stage, P = 32k-512k) | read times 1.72-89.63 ms | identical to 0.01 ms | round-1 TTFT +0.1-0.3%, round 0 +0.8-0.9% (the same per-microbatch compute offset) |
| eval_m6 full N = 300 / 400 | 34.15 / 39.65 r/s | real 33.70 / 40.51 | +1.3% / -2.1% |
| eval_m6 tiers N = 300 / 400 | 33.66 / 35.47 | real 33.45 / 37.66 | +0.6% / -5.8% |
| eval_m6 ssdheavy N = 300 / 400 | 33.70 / 34.69 | real 33.76 / 36.71 | -0.2% / -5.5% |
| eval_m6 excl tiers N = 300 / 400 | 33.68 / 35.42 | real 32.88 / 33.10 | +2.4% / +7.0%: the simulator charges no write cost (eval_m6.md item 5) |
| eval_m6 hbmonly N = 300 / 400 | 25.49 / 3.56 | real 27.24 / 12.97 | -6.4% / -72.5% (collapse, recompute/new 7.9) |

Localized by changing one key at N = 400 (report.md §6): `recurrent_state_checkpoints: 1` alone gives hbmonly
22.95 r/s (recompute/new 0.69), tiers 37.66 (= real), ssdheavy 36.95 (real 36.71); `kv_block_tokens` absent or
`kv_part_padding` absent alone change little (hbmonly 3.11 / 2.81). So the N = 400 deviations, the hbmonly collapse
included, come from the second kept state per context in HBM. The real rows are the eval engine a78dc6f, which kept
one state; the engine as built keeps two, and no measured run of it at hbmonly N = 400 exists here (the pages run
8475, hbmonly N = 400, 16.16 r/s, is lane-m6-pack 8d50066; whether it kept two states is 未验证: check that
commit's `RunConfig.kept_states`). The equivalence of the engine's k2 and `Latest{k: 2}` in HBM is 未验证 (above).

`external_decode: false` gives per-request identical results in all 18 tiered/untiered pairs run (sessions v2 and
eval_m6): on these traces every round resumes at its context's end, where a tier read moves no earlier state under
either assumption (`StateRetention::survivors`), so the tier-read state asymmetry does not act.
