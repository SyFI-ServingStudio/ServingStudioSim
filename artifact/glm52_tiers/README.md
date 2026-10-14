# GLM-5.2 prefix tier study

Presets, inputs and analysis scripts behind `report.html`: GLM-5.2 NVFP4 on 8 B200 serving closed-loop TraceLab
monotonic sessions with a three-tier prefix store (HBM, DRAM, SSD), DP8EP8 against PP8. GLM-5.3 (not Flash) uses
the same target graph (`simulator/src/arch/glm53_vllm_nvfp4_dsa_moe_dflash2.rs`), and these runs are prefill only
with decode elsewhere, so the results also stand for GLM-5.3.

Nothing in this folder is a run output. Paths are relative to the repository root, and every preset runs from there:

```bash
artifact/glm52_tiers/gen_traces.sh            # once; writes logs/glm52_tiers_traces/
uv run python -m launcher artifact/glm52_tiers/presets/<preset>.yaml --no-gpu
```

## Code and data

| Item | Value |
| --- | --- |
| Closed-loop runs | branch `glm52-tiers`: `pp-study` `42895fea` plus the commit that gives the chunked-prefill worker prefix tiers for any arch (each run's `git_snapshot/diff.patch` is that change). Tiers off is bit-identical to `42895fea` (`20261009_1_bitcheck_base` vs `20261009_2_bitcheck_after`). |
| Burst run | `main` `6c8e98a7` (PR #85); no tiers, so it needs none of this branch. |
| profile.db | origin/main's, sha256 `cbb0fa72…`. pp-study's local DB lacks the 55 GLM-5.2 `dsa_sparse_index_remap` rows at `num_queries` 16384; main has them. |
| Scheduler | pp-study's: shortest-prefill-first, SRPT, load budget per DP rank (low 4096, backlog 1M–3M), tier read wait ≤ 1 s with pass-over, force-schedule after 60 s, warm start, sticky sessions. PP8 adds balanced tier load and even microbatches. |
| Routing, DP | vLLM c32_long popularity (`hf://…/glm52_nvfp4/vllm/c32_long/capture/20260824/popularity.json`). |
| Routing, PP8 | SGLang diverse_100 popularity (`hf://…/glm52_nvfp4/sglang/diverse_100/capture/20260829/popularity.json`, sha256 `177b2642…`, the file the runs read from a local copy). c32_long lacks 66 EP1 `nvfp4_fused_moe` rows, so PP8 and DP8 MoE times differ slightly. |
| KV budget | DP8 70.56 GB per rank (1,277,698 tokens) from the public DP preset; PP8 100.08 GB per stage (14,135,593 tokens per pipeline) by the same rule. Neither is measured. |

## Traces

`trace/tracelab_preserving.csv` (untracked, sha256 `9049e764…`) → `trace/session_decode_wait.py --decode-tok-s 80` →
`mono_sessions_d80.csv` (sha256 `2a5fb715…`) → `trace/session_closed_loop.py --concurrency c --after 20c --seed 0` →
`closed_c{c}_s0.csv`. `gen_traces.sh` reproduces the files the runs used:

| c | sha256 |
| ---: | --- |
| 100 | `a3ecacfa6d3960113711d3e136bebf7051f989029199f02c82c79a117f665510` |
| 200 | `2c7328c75baa47d59733fcf1eb017c0aa87a69c0f4a4414ba032f3e424cad307` |
| 250 | `3e7ca3b35da77f234e4def74170938ff24e3dc5079ba5131367f8175a136fbed` |
| 375 | `7ed40c07997b6ee0ce2def745b77403f302094550a4aa754e5a4e53967bf4374` |
| 500 | `e1246887832708c4ccf921292b1d29f767c474f2c0756ea110eddbf2576937e8` |
| 750 | `41e1b7ebc398e9c87320d26cd0254409578e73f269557f2f7c3e5bf975c97f23` |
| 1000 | `8920a36375ef6f050aa06ba3f20a7990e310e9187f0ab831de233ed9c1c0ed68` |
| 1125 | `84dff32b0e2a82b2d0b1884f8e56cf53b48751aaedaa6991ce1900ffde25f245` |
| 1250 | `a9386ed10a59ac6bd672443d505a985d5512bf74bb40f1db0b4aa5631ed3c73d` |
| 1375 | `3e1a532cc85e8c7bb3964aab7334914b1c363a27b91d984084874b3bd9b484dc` |
| 1500 | `d2a0f04a2941476d8c89361df6cfb60509c6e2e5e5654c0bbb79fa87cbe5c309` |
| 2000 | `fb67c7dd2a348043273f5e765b80c02ea965f65373ee8a53a68ec4e67367e7de` |
| 2500 | `c290d95521930e16bde16a62caa707fa97622a85cdb081882cba141f0ddf78a2` |
| 3000 | `59da61f14f746a0a821ac9e5343c25130680c3e44e91f5b2a9e9c4de0aa20901` |
| 4000 | `e8fe37ae1c9763215506bd25fd7648ab083478b7c5dc733fba39890433789e1f` |

The burst trace `presets/prefix100k_new8192_128.csv` (128 requests at t=0, 100,000-token prefix, 8,192 fresh, output 1)
is small enough to keep here.

## Presets

Each file is the experiment's `preset.yaml` as run, with trace paths pointed at `logs/glm52_tiers_traces/` and the
PP8 popularity file at its dataset URL. `log_dir` is unchanged.

| Preset | What it runs | Runs |
| --- | --- | ---: |
| `20261009_0_glm52_dp8_tiers` | DP8EP8, DRAM 150 GB + SSD 8 TB at 10 GB/s, c100–c1000, 8 h | 7 |
| `20261009_1_bitcheck_base`, `_2_bitcheck_after` | DP4/DP8 100k-prefix burst, tiers off, before/after the code change | 2 + 2 |
| `20261009_3_glm52_dp8_infdram` | DP8, DRAM 1e6 GB at 50 GB/s, no SSD, c500–c4000 | 8 |
| `20261009_4_glm52_dp8_dram_size` | DP8, DRAM 500 / 750 GB + SSD, c750 / c1000 | 4 |
| `20261009_5_glm52_dp8_budget` | DP8, DRAM 750 GB, load budget low 4k / 8k / 16k / none × c750 / c1000 / c1500 | 12 |
| `20261009_6_glm52_pp8_tiers` | PP8, DRAM 150 / 750 GB + SSD, c500–c2000 | 10 |
| `20261009_7_glm52_pp8_peak` | PP8, DRAM 750 GB, c1125 / c1250 / c1375 | 3 |
| `20261009_glm52_dp_prefix100k_burst` | DP4EP4 / DP8EP8 burst, run on `main` | 2 |

## Scripts

| Script | Use |
| --- | --- |
| `scripts/steady.py <experiment dir> 2 7 [out.json]` | rounds/s, prefix hit and TTFT p50/p90/p99 over arrivals in [2 h, 7 h) |
| `scripts/tier_rates.py <run dir>` | per-GPU DRAM/SSD reads, writes, evictions and end-of-run occupancy |
| `scripts/hbm_split.py <run dir> 2 7` | DP8 HBM per rank: running, read holds, retained prefixes, free |
| `scripts/hbm_need.py <run dir> 2 7` | unevictable HBM (running + read holds) as a share of the pool, mean and percentiles |
| `scripts/lru_need.py <run dir>` | DRAM size a DP rank needs, by LRU replay of its tier log |

`hbm_split.py`, `hbm_need.py` and `lru_need.py` read `raw/kv_snapshot` and the `prefix_tiers_w*.csv` tier logs that these
runs wrote; check a rerun keeps them before deleting bulk output.

## Results

| Config | Peak rounds/s | TTFT p50 / p90 / p99 at peak | Per-GPU tier reads at peak | Bound |
| --- | ---: | --- | --- | --- |
| DP8, DRAM 150 GB + SSD | 21.3 (c750) | 1.7 / 24 / 65 s | DRAM 9.6, SSD 9.6 GB/s | SSD bandwidth |
| DP8, DRAM 750 GB + SSD | 28.9 (c1000) | 2.6 / 21 / 65 s | DRAM 29.0, SSD 0.7 GB/s | compute, rank imbalance, HBM floor |
| PP8, DRAM 150 GB + SSD | 26.5 (c750) | 0.50 / 2.6 / 62 s | DRAM 6.3, SSD 10.0 GB/s | SSD bandwidth |
| PP8, DRAM 750 GB + SSD | 42.0 (c1125); 40.9 at c1000 | 1.7 / 4.3 / 24 s; 0.64 / 1.7 / 6.2 s | DRAM 38.9, SSD 1.2 GB/s | collapses past c1125 |

`report.html` has the curves, the TTFT–throughput trade-off, DRAM sizing, the load-budget sweep, the HBM floor, cost
against online providers, and the caveats. Single seed per point.
