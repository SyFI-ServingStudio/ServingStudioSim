# `b200_pp4_prefill.yaml`: SSEngine, GLM-5.3-Flash NVFP4 prefill on B200 PP4

`b200_pp4_prefill.yaml` is the simulator configuration of SSEngine as built (pp-impl `lanes/release`, commit 831a02a): what
`ssengine serve` runs (`ssengine/serve/launch.py` `design_config` + `DESIGN_TIERS`), with the tier sizes of the M6
evaluation's full setting (`docs/results/eval_m6.md`), except that it keeps one state per context, as the measured
evaluation engine (a78dc6f) did, where the release keeps two (`recurrent_state_checkpoints` below). Each mechanism is
its own key. An ablation is this preset with the key(s) in the last column changed, nothing else.

Paths below are relative to `pp-impl/lanes/release`. "未验证" marks a value whose engine source was not found or that
the simulator models differently; the right column says how to check it.

## Keys and their engine source

| Key | Value | SSEngine behavior and source | Ablation |
| --- | --- | --- | --- |
| `deployment: pp`, `pp_size: 4`, `layer_partition` | 4 stages, [12, 11, 11, 11] | `ssengine/config.py` `LAYER_SPLIT = [12, 11, 11, 11]`, one B200 per stage | – |
| arch keys | no MLA layout copies, indexer logits cap 512 MB, measured per-token MoE routes (TP4/EP4 capture), as `presets/glm53_flash_prefill/` | the engine's model path | – |
| `backends_file` | `b200_pp4_prefill.backends.yaml` | the engine's kernels; the roles that differ from the vLLM defaults are listed below | – |
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
| `prefix_resume_points` | `context_end` | a kept context keeps its end state (`kept_states.md` k1/k2). The simulator requires it only above one kept state (`kv_accounting.rs`: "recurrent_state_checkpoints > 1 … need prefix_resume_points: context_end"), but it also sets where a round resumes at k = 1: absent, it follows `kv_overhead: vllm` to `prompt_end`, the prompt minus one floored to a 128-token page, so a next round would not resume at the context's end. Kept | – |
| `recurrent_state_checkpoints` | 1 | the measured evaluation engine (eval_m6.md "Real code: commit a78dc6f") kept one state per context. The release's `config.py` `RunConfig.kept_states = 2` (k2: also the state its round resumed from) is the variant: no measured run of it exists here, and whether the simulator's `Latest{k: 2}` charges HBM as the engine's k2 (copy-on-resume, `drop_stale`) does is 未验证: compare per-session kept states on a short session trace | 2 (the release's k2) |
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

## Backends that differ from the vLLM defaults

`b200_pp4_prefill.backends.yaml` sets all 111 kernel roles of the PP arch. Against the defaults the launcher emits
for this preset (`uv run python -m launcher presets/tuned_engine/GLM-5.3-Flash-NVFP4/b200_pp4_prefill.yaml --no-gpu
--emit-backends FILE`, which ignores `backends_file`), 14 roles differ, the same on every layer kind that has them:

| Role (layer kinds) | vLLM default | This preset |
| --- | --- | --- |
| `kda.chunk_prefill` (first_kda_dense, kda_dense, kda_moe) | `vllm_triton` | `flashinfer_cute_persistent` |
| `kda.short_conv_prefill` (first_kda_dense, kda_dense, kda_moe) | `vllm_triton` | `dao_channellast` |
| `attn_mhc_post_pre` (kda_dense, kda_moe, dsa_moe) | `vllm_tilelang` | `deepgemm_mega_nonshifted` |
| `ffn_mhc_post_pre` (first_kda_dense, kda_dense, kda_moe, dsa_moe) | `vllm_tilelang` | `deepgemm_mega_nonshifted` |
| `dsa.indexer.topk_prefill` (dsa_moe) | `vllm_cuda` | `[deep_select, vllm_cuda]` (the faster per launch) |

The other 97 roles are at their defaults (`first_kda_dense.attn_mhc_pre` stays `vllm_tilelang`, the only option the
emitted skeleton lists for it).

## Scope, and where the simulation is not trusted

- Model and hardware: GLM-5.3-Flash NVFP4 on one B200 PP4 pipeline, layers [12, 11, 11, 11].
- Workload: prefill only (`external_decode`: a round completes at its first token), coding-agent multi-round
  sessions (TraceLab v3 M6, `sessions_2ps_v3.csv`) and the bring-up traces below. Decode, other session shapes and
  other models are outside what the anchors cover.
- Not trusted: HBM only (no DRAM or SSD tier) at high concurrency. At N = 400 the simulator gives 22.95 r/s
  against the engine's measured 12.97 (+76.9%) with recompute/new 0.692 against 1.757; at N = 300, 30.44 against
  27.24 (+11.7%) (`logs/20261114_9_pp_u7_equiv/anchors/report.md` §6, `k1`). The cause is 未验证.
- Not trusted: exclusive DRAM without a loading area (`excl tiers`), where the simulator charges no tier write time
  (eval_m6.md item 5); at k = 2 it read +7.0% at N = 400 (report.md §5), at this preset 未跑.
- Tiered settings (full, tiers, ssdheavy) at N = 300 / 400 are within -2.0..+1.4% of the measured rows (anchors
  below).

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
  only the state the round resumes from (`kept_states.md`). The simulator's round under `external_decode` adds two
  (its pass's end and the handed-back decode's end), and an HBM take and a tier read both keep the states that
  survive those two, so with k = 2 a round carries no earlier state either way. At this preset's k = 1 no earlier
  state is kept, so the count does not act.
- **Host costs.** Plan forming, dispatch and admission host time (scheduler.md "Low load": ~4 ms before a GPU start)
  are not in the simulator.

## Evaluation engine vs this engine

eval_m6.md's measured runs used commit a78dc6f: 8,576-token blocks, the KDA state padded to whole blocks, one kept
state. The engine as built adds 128-token pages, the compact state and k2; this preset takes the first two and keeps
one state. To reproduce the eval engine's KV form exactly, also remove `kv_block_tokens` and `kv_part_padding`
(the `evaleng` runs below); for the release's k2, set `recurrent_state_checkpoints: 2`.

## Anchors (nothing tuned)

Each anchor run is this preset with only the anchor's workload, memory and named tier mode changed (rendered from
it at its earlier path, `presets/glm53_flash_prefill/ssengine.yaml`, with the same keys). T = the
v2-realistic targets the engine was built against (`pp-impl/third_party/handoff_v2_realistic/data`, simulator outputs
of wt-pp-hitrule 5153041a); real = eval_m6.md's measured rows (engine a78dc6f). Scripts and full tables (logs are not
committed):

- scheduling, force, tier bring-up and v2 sessions at this preset (k = 1): `logs/20261114_10_pp_u8_preset/`
  (`gen.py`, `analyze.py`, `report.md`, `sched_check_blocks.md`; 2026-11-16);
- eval_m6 at this preset: the `k1` runs of `logs/20261114_9_pp_u7_equiv/anchors/` (report.md §6: the 2026-11-14
  preset with only `recurrent_state_checkpoints: 1`, i.e. this preset); the k = 2 column and `evaleng` come from
  the same report (§3, §5, §6). eval_m6 deviations are computed from report.md's two-decimal r/s.

The k = 2 numbers predate the unified state count (an HBM take then counted one added state under
`external_decode`, a tier read two); the current code carries one state fewer per running round there, so they are
not what k = 2 gives now (未跑 at the current code). The k = 1 numbers are unaffected: one kept state carries no
earlier state, and the unit-8 regression's `external_decode` k = 1 runs are byte-identical across the change
(`logs/20261114_10_pp_u8_preset/regress/compare.txt`).

| Anchor | This preset (k = 1) | Target | Deviation and where it comes from |
| --- | --- | --- | --- |
| Scheduling check, `good` | 58 microbatches, all 58 chunk for chunk; 128k TTFT 1,875.1 ms | 58; 1,862.8 ms | +0.66% worst TTFT. Every microbatch's compute is 0.0-0.9% above T (41 of 58 above); the order is exact. The per-kernel source of the 0.9% (profile.db rows vs T's 20261016 profile.db.new) is 未验证: compare the two runs' cost_log kernel times. k = 2 gives the same request table byte for byte |
| Force check, `good` | 32k TTFT 60.90 s; 1k p50 / p99 17.93 / 33.45 s; 1,300 microbatches | 61.02 s; 17.73 / 33.18 s; 1,303 | -0.2% / +1.1% / +0.8%; 3 fewer microbatches. k = 2: byte-identical request table |
| Sessions good_hbm N = 400 (v2 trace) | 25.18 r/s, TTFT p99 18.6 s | 25.06 r/s, p99 25.5 s (engine measured 24.76) | +0.5% r/s. k = 2: 24.26 (-3.2%); `evaleng`: 25.12 (+0.2%) |
| Sessions h12d19_ssd128 N = 500 (v2 trace) | 27.98 r/s | 27.69 r/s | +1.0%. With T's own tier mode (every-context writes, no loading area): 27.76 (+0.2%). k = 2: 28.61 (+3.3%) |
| Tier bring-up (DRAM / SSD, balanced / own stage, P = 32k-512k) | read times 1.72-89.63 ms | identical to 0.01 ms | round-1 TTFT +0.1-0.3%, round 0 +0.8-0.9% (the same per-microbatch compute offset). k = 2: byte-identical request tables |
| eval_m6 full N = 300 / 400 | 34.18 / 39.71 r/s | real 33.70 / 40.51 | +1.4% / -2.0% (k = 2: 34.15 / 39.65) |
| eval_m6 tiers N = 300 / 400 | 33.80 / 37.66 | real 33.45 / 37.66 | +1.0% / 0.0% (k = 2: 33.66 / 35.47) |
| eval_m6 ssdheavy N = 300 / 400 | 33.76 / 36.95 | real 33.76 / 36.71 | 0.0% / +0.7% (k = 2: 33.70 / 34.69) |
| eval_m6 excl tiers N = 300 / 400 | 未跑 | real 32.88 / 33.10 | k = 2: 33.68 / 35.42 (+2.4% / +7.0%): the simulator charges no write cost (eval_m6.md item 5) |
| eval_m6 hbmonly N = 300 / 400 | 30.44 / 22.95 (recompute/new 0.212 / 0.692) | real 27.24 / 12.97 (0.397 / 1.757) | +11.7% / +76.9%. k = 2: 25.49 / 3.56 (recompute/new 7.931 at N = 400, a collapse); `evaleng`: 29.82 / 19.59 |

The second kept state is what moves the N = 400 rows: in the 2026-11-14 localization (report.md §6, one key changed
at a time from the k = 2 preset) `recurrent_state_checkpoints` alone takes hbmonly N = 400 from 3.56 to 22.95 r/s and
tiers from 35.47 to 37.66, while `kv_block_tokens` absent or `kv_part_padding` absent alone change little (hbmonly
3.11 / 2.81). At k = 1 the tiered rows sit within -2.0..+1.4% of real; hbmonly overshoots real (+11.7% / +76.9%,
less recompute than measured), whose cause is 未验证. No measured run of the release's k2 exists here (the pages run
8475, hbmonly N = 400, 16.16 r/s, is lane-m6-pack 8d50066; whether it kept two states is 未验证: check that commit's
`RunConfig.kept_states`).

`external_decode: false` gave per-request identical results in all 18 tiered/untiered pairs run at k = 2 (sessions
v2 and eval_m6, 2026-11-14, when an HBM take still counted one added state): on these traces every round resumes
at its context's end, where a tier read moves no earlier state under either count (`StateRetention::survivors`). At k = 1 no earlier state is kept at all; the
`ed_off` pairs were not rerun at k = 1 (未跑).
