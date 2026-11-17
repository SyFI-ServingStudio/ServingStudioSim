# GLM-5.3-Flash NVFP4 prefill on B200 PP4, as SSEngine runs it

One B200 PP4 pipeline (layers split [12, 11, 11, 11]) serving prefill only: decode runs elsewhere
(`external_decode`), so a round completes at its first token and its context is handed back. Every preset here
uses the same engine choices over the PP arch's vLLM defaults:

- arch: no MLA layout copies, indexer logits cap 512 MB, measured per-token MoE routes (TP4/EP4 capture);
- backends: `../tuned_engine/GLM-5.3-Flash-NVFP4/b200_pp4_prefill.backends.yaml`, shared with the SSEngine
  preset (FlashInfer persistent KDA prefill, Dao channel-last short conv, DeepGEMM
  mega mHC, top-k the faster of DeepSelect and vLLM);
- scheduler (`good`): even microbatch split (floor 512 tokens), shortest remaining prefill first across started
  and queued prompts, load-following budget (4,096 tokens up to 1M tokens of backlog, max_batch_tokens 32,768 at
  3M), any request waiting 60 s scheduled first;
- tiers: HBM -> DRAM -> SSD victim chain (DRAM exclusive of HBM, SSD written with what DRAM evicts), a staged
  loading area of 10% of HBM (`st10`). Staged reads take no read-wait bound (the engine does not use it either,
  `engine/scheduler.py`); the bound applies only to the unstaged reads of `check_tier_read.yaml`.

The SSEngine preset itself (128-token pages, the compact KDA state, one kept state, every mechanism its own key)
is `presets/tuned_engine/GLM-5.3-Flash-NVFP4/b200_pp4_prefill.yaml`; `b200_pp4_prefill.md` beside it gives each
key's engine source, the key an ablation changes, and the comparison with the measured engine. The presets here keep
the simulator's default KV accounting (`kv_overhead: vllm`: 8,576-token blocks, the KDA state padded to whole
blocks, one kept state), as in the evaluation (eval_m6.md, engine a78dc6f).

KV budget in HBM is 100.18 GB per GPU (149.84 GB minus the largest stage's NVFP4 weights) unless a preset says
otherwise. Run any preset from the repository root:
`uv run python -m launcher presets/glm53_flash_prefill/<preset> --no-gpu`
(`launcher timing-predict` for `iteration_predict.json`).

| Preset | What it runs |
| --- | --- |
| `iteration_predict.json` | One iteration of fresh prefill, 1k-32k tokens, and the four 32k chunks of a 128k prompt |
| `check_single_request.yaml` | One request alone, 1k-128k tokens, five per length |
| `check_saturated.yaml` | A batch of same-length requests all at t = 0, 1k-128k tokens |
| `check_prefix_hit.yaml` | Bring-up sessions whose later rounds hit the HBM prefix cache |
| `check_scheduler_order.yaml` | Bring-up requests that show the scheduler's order |
| `check_force_schedule.yaml` | Bring-up load where one request waits past the 60 s force threshold |
| `check_tier_read.yaml` | Two-round sessions whose round 1 reads from HBM, DRAM or SSD |
| `sessions_scheduler.yaml` | v3 M6 sessions, HBM only: the scheduler ladder plain -> good at N = 100-700 |
| `sessions_concurrency.yaml` | v3 M6 sessions, full tiers, `good`: N = 100-700 |
| `sessions_tiers.yaml` | v3 M6 sessions at N = 100, 300, 400: full / tiers / ssdheavy / hbmonly |
| `sessions_ssd_size.yaml` | v3 M6 sessions at N = 300, 400: SSD 32-256 GB under the full and tiers settings |
| `v3_m6_full.yaml` | v3 M6 sessions, N = 300, full setting (HBM 100.18 + DRAM 50 + SSD 128 GB) |
| `v3_m6_dram128_ssd8t.yaml` | v3 M6 sessions, N = 300, HBM 100.18 + DRAM 128 GB + SSD 8 TB |

The `check_*` presets read bring-up CSVs from `bringup/` (from the v2 handoff's `data/bringup/`). They are not
committed; generate them in place before running one of those presets:
`uv run python presets/glm53_flash_prefill/bringup/make_bringup.py` (deterministic; the script's docstring describes
each file). `tests/test_glm53_flash_prefill_presets.py` regenerates them itself. The v3 session
traces (`trace/sessions_2ps_v3.csv`, `trace/closed_c2000_v3.csv`) are not committed; the session trace's manifest and
the closed-loop generator are under `traces/`. The 8 h closed-loop preset that read `trace/closed_c2000_v3.csv` is the
tuned preset `presets/tuned_engine/GLM-5.3-Flash-NVFP4/b200_pp4_prefill.yaml` (its manifest is beside it).
