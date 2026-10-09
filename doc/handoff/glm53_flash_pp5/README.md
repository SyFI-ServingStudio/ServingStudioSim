# Handoff: GLM-5.3-Flash NVFP4, PP5 on B200, coding-agent sessions

Simulated target for a real serving system to chase (ServingStudio Sim, study branch `pp-sched-tiers`, 2026-10-09;
see §12). Everything below runs from this directory: presets, kernel map, trace build and timing-predict config.
Use it with the skill `skills/top-compose-real-framework-from-sim/SKILL.md`. **The simulator is the reference; the real
system chases it.** When a real measurement disagrees, first ask whether an exogenous fact (§1) changed. If it did not,
the gap belongs to the real system and is fixed there. Every number below is a simulation output, labeled with its run.

**Final target:** one PP5 pipeline (5 × B200, layers [9,9,9,9,9]) using prefix KV in HBM + host DRAM + local SSD.
- **Throughput point:** 103.9 rounds/s (20.8 per GPU) at TTFT p50 0.88 s / p99 7.0 s.
- **Low-latency point:** 400 sessions/GPU, 84.1 rounds/s at p50 0.17 s / p99 1.35 s.
- Peak is 105.8 rounds/s (21.2 per GPU), past the knee (p99 62 s).
- PP5 beats PP8 by 6% per GPU and DP8EP8 by 1.35× per GPU (§3.2).

**Build it in six milestones**, in order. Each has a sim target and a check.
- Get as close to the target as you can, then move on.
- Record every remaining gap in an issues log: what (leaf, layer, stage or metric), measured vs sim, and the
  suspected cause.
- A gap is not a blocker.

| Milestone | What works at the end | Sim target to match |
|---|---|---|
| **M1 Kernels** (§4) | every kernel of the selected backend list runs in the real stack at its DB time | DB rows; per-leaf ms of the M2/M3 cases |
| **M2 Layers** (§5) | each of the three layer types runs its kernels in the sim's order at the sim's time | per-layer ms for 8k/16k/32k × prefix 0–512k |
| **M3 Whole model on PP5** (§6) | correct outputs through 5 stages; every stage's time and the inter-stage send match, for one microbatch end to end | per-stage ms + send ms for the same cases |
| **M4 Scheduling** (§7) | the §7 scheduler serves the closed-loop workload with HBM prefix reuse only | `pp5_16min_hbm.yaml`, c500 (§2.1) |
| **M5 Prefix cache** (§8) | exact prefix reuse, retention and KV hand-back from decode | HBM hit share of `pp5_16min_hbm.yaml` (98.7% at c500) |
| **M6 Loading and offloading** (§9) | DRAM + SSD tiers, async loads, read-wait bound, balanced storage | `pp5_16min.yaml` (§2.1), then §3 |

Contents: 0 setup and terms · 1 contract · 2 workload and the 16-min benchmark · 3 final target numbers · 4–9 milestones M1–M6 · 10 optimality · 11 caveats ·
12 provenance. Companion docs: [`kernels.md`](kernels.md) (every leaf, DB, runners)
and [`scheduler.md`](scheduler.md) (every scheduler mechanism with `file:line`).
`$R` below is the 8 h reference run at c3000, `logs/glm53_flash_pp5/runs_8h/c3000`, which `pp5_8h.yaml` writes with
every analyzer report. The numbers quoted from it are in this document; rerun it only to inspect raw logs.

**Terms.** KDA: Kimi Delta Attention, the linear-attention layer with a recurrent state and a short conv. DSA:
DeepSeek Sparse Attention, MLA attention over the pools a lightning indexer selects (§4.1). MLA: multi-head latent
attention. mHC: manifold-constrained hyper-connections, the 4-wide residual stream with fused post/pre RMS-norm
boundaries. kpool: the indexer's per-pool key cache. FWHT: fast Walsh–Hadamard transform. MTP: multi-token prediction
(off here). CUPTI: the CUDA profiling interface used for kernel time. SRPT: shortest remaining processing time. SPF:
shortest prefill first. fastk: the study's kernel path, `backends.yaml`. A **round** is one request of a coding-agent
session; **cN** is a closed loop of N concurrent sessions, and rounds/s counts completed rounds.

## 0. Setup

- **Simulator (CPU only, no GPU):** a ServingStudioSim checkout of this branch with submodules
  (`git submodule update --init --recursive`), a Rust toolchain, `uv`, then `just sync` (CLAUDE.md). The routing corpus
  is an `hf://` URI and is fetched once from Hugging Face, so the first run needs network.
- **Source trace:** `trace/tracelab_preserving.csv` is not in git. Build it as `trace/README.md` §Regenerating
  describes (TraceLab v0.0.1 release asset, a TraceLab clone, then `tracegen` from the req-frontend submodule). Its
  sha256 is the first line of `traces.sha256`.
- **Then, from the ServingStudioSim root:**
  ```bash
  doc/handoff/glm53_flash_pp5/make_traces.sh                    # 2 min, 3 GB
  uv run python -m launcher timing-predict doc/handoff/glm53_flash_pp5/predict_pp5.json --no-gpu --analyzer-essential-only
  uv run python -m launcher doc/handoff/glm53_flash_pp5/pp5_16min_hbm.yaml doc/handoff/glm53_flash_pp5/pp5_16min.yaml --no-gpu --no-plot
  uv run python doc/handoff/glm53_flash_pp5/window_metrics.py logs/glm53_flash_pp5/runs_16min/c2656/raw/request_slo.parquet
  ```
  `--no-gpu` makes a missing profile row an error instead of a GPU profiling job; at this commit none is missing.
  `pp5_8h.yaml` is optional (five 8 h runs, about 1.5 GB of logs each); it gives `$R` and §3.1.
- **Real system:** M1 onward needs 5 B200 in one NVLink domain (M1 and M2 can start on one B200).
- **Benchmark client:** `alignment/load_generator/req-frontend` `session_runner` replays the same CSVs with input format
  `text-generation-session-execution-v2`: it submits round i, awaits its response, waits `tool_wait_after_ms`, then
  submits round i + 1. Use `replay.arrival_mode: saturated` with `replay.max_concurrency: N` (a session holds its slot across tool waits), and 1 output token per round (decode
  runs elsewhere, §2). Score its per-round log with `window_metrics.py`'s rule: rounds arriving in the window that
  complete; TTFT = first token − arrival.

**Which sim target each milestone matches.** M4–M6 use the 16-min runs of §2.1; the sim finishes one load in seconds.
- **M1–M3:** timing-predict cases (§5.2, §6.3); no workload.
- **M4 and M5:** `pp5_16min_hbm.yaml`, the full §7 scheduler with HBM only (no DRAM/SSD
  tiers). Target load c500: 20.4 rounds/s (4.1 per GPU), TTFT p50 0.11 / p90 0.17 / p99 0.74 s, HBM hit 98.7%.
- **M6:** `pp5_16min.yaml`, the full configuration (§2.1 table).
- Command, from the ServingStudioSim root, after `doc/handoff/glm53_flash_pp5/make_traces.sh` (§2):
  `uv run python -m launcher doc/handoff/glm53_flash_pp5/<preset>.yaml --no-plot`.
- To match a real system that lacks a feature, change only that feature in a copy of the preset and rerun.

One attributable change per cycle (skill: Tick → Tock → Probe).

---

## 1. Comparison contract

Classify every field before any sim/real comparison (skill §Step 1).

| Class | Fields |
|---|---|
| **Exogenous: must match** | Model GLM-5.3-Flash, NVFP4 weights (`model/config/glm53_flash_nvfp4.json`, `fp8: false`), MTP/spec decode off. 5 × NVIDIA B200 (180 GB) in one NVLink domain. Workload, arrival process and metric definitions of §2. Per-GPU host DRAM 150 GB at 50 GB/s and local SSD 8 TB at 10 GB/s for prefix KV; for the 5-GPU node that is 750 GB DRAM, 40 TB SSD and 50 GB/s aggregate SSD read. `max_model_len` 1,048,576. |
| **Simulated target design: chased by the real system** | The §3.1 preset (`pp5_8h.yaml`; its 16-min form is `pp5_16min.yaml`). PP5 with layer partition [9,9,9,9,9]. The scheduler of §7 (SRPT + shortest-prefill-first, load-following microbatch budget up to 32,768 tokens, even split, 60 s force-schedule, 1 s tier read-wait bound with pass-over). Prefix tiers as write-through LRU with balanced per-stage loads (§9.1). The kernels of §4.1. Plain chunking (no Mamba-align chunk ends). |
| **Real-engine limitations: diagnostic gaps, never copied into the sim** | Whatever the real engine does differently today: slot caps, KV pool size below the §6.4 budget, CUDA-graph coverage, host/Python overhead, unbalanced tier reads, missing async KV loads. |

The real system implements all of it:
- **Basics:** PP size and layer assignment as configuration, so PP4 or PP8 needs no code change (§6.1); chunked
  prefill; at most PP (here 5) microbatches in flight; async KV loads that hold their HBM blocks; a fixed
  KDA state charge per request.
- **The policies that set the target:** SRPT, shortest-prefill-first, the load-following budget with even split,
  the 60 s force-schedule, the 1 s read-wait bound with pass-over, and balanced tier storage.

**Model and correctness reference.**
- **Checkpoint:** `nvidia/GLM-5.3-Flash-NVFP4`, snapshot `da920bb0b9f4a06727223a349e55468e38352348`. The sim's config `model/config/glm53_flash_nvfp4.json` is this checkpoint's.
- **Correctness gate: vLLM** serving the same checkpoint without PP, for example TP4 + EP on 4 B200 with
  `--enable-expert-parallel --kv-cache-dtype fp8_e4m3`, as in `presets/alignment/glm53_flash_fp8_b200_tp4_ep4/campaign.yaml` (that preset serves the FP8
  checkpoint; use the same flags with the NVFP4 checkpoint).
  Compare greedy outputs (or logits) with and without a cached prefix, and log every mismatch.
- **Forward-math reference:** the vLLM fork's model code `vllm/models/glm5next/` in `ServingStudioSim/alignment/profiler/vllm`
  (HEAD `892da0822f`). Read it for the exact math of each op; the real system does not have to be built on vLLM.
- **Reusing kernels is encouraged**, including CUDA/Triton kernels taken from the vLLM tree (vendor them).

## 2. Workload

- **Corpus:** TraceLab coding-agent sessions, materialized from the public syfi dataset under the `monotonic` context
  policy: `trace/tracelab_preserving.csv` (`trace/README.md` §"Session-wise traces"; `trace/tracelab_preserving.manifest.json`).
  4,281 sessions, 357,161 rounds. Per round: context 153.7k tokens on average (54.90B prompt tokens / rounds), of which
  152.0k are the session's own prefix, so about 1.75k fresh tokens per round; output 523 tokens on average. Planned
  prefix-hit rate 98.86%.
- **Decode is outside the system** (`external_decode: true`): a round completes at its first token. Decode time at
  80 tokens/s, `(output − 1) / 80` s, is added to the round's tool wait (`trace/session_decode_wait.py --decode-tok-s
  80` → `mono_sessions_d80.csv`, sha256 `2a5fb715…`). The round's context, with
  every output token but the last, goes back to the prefix cache.
- **Closed loop at fixed session concurrency N** (`arrival_mode: saturated`, `max_concurrency: N`):
  `trace/session_closed_loop.py mono_sessions_d80.csv --concurrency N --after 20N --seed 0` → `closed_cN.csv`.
- **Build them all** with `make_traces.sh` (about 2 min, 3 GB under `logs/glm53_flash_pp5/traces/`). It starts
  from `trace/tracelab_preserving.csv` (`trace/README.md` §Regenerating) and checks every file against
  `traces.sha256`.
  - The first N sessions start in the renewal equilibrium: drawn by duration, joined mid-life, each with a 1-token
    placeholder round 0.
  - The joining round's context is read from the slowest tier (`prefix_tier_warm_start`).
  - Each finished session lets in a uniformly drawn whole session.
  - A real benchmark must replay the same file in the same order, with the same rule.
- **Duration and window:** 8 h runs; metrics over rounds that arrive in [1 h, 8 h) and complete.
  - Throughput = completed rounds in the window ÷ 7 h.
  - TTFT = first-token time − arrival.
  - Session concurrency per GPU = N ÷ GPUs.
- **Routing:** measured per-token expert routes (`routing: corpus`), never uniform. Corpus:
  `hf://datasets/UW-SyFI/servingstudio-workload@c3f5ecaa…/glm53_flash_fp8/vllm/balanced_c32/capture/20260924/manifest.json`.

**Benchmark client: what a real replay must do.**
- **CSV columns:** `request_id` (`session_X_round_Y`), `session_id`, `round_idx`, `arrival_time_ms`, `prefix_len`,
  `input_len`, `output_len`, `tool_wait_after_ms`.
  - `prefix_len` is the round's declared prefix: the previous round's `prefix_len + input_len + output_len`. The
    joining round of a joined session declares its whole mid-life context.
  - `input_len` is the fresh prompt; `output_len` counts the first token.
- **Replay rule (closed loop of N):**
  - The first N sessions in file order start at t = 0. When a session finishes its last round, the next session in
    file order starts.
  - Within a session, round k+1 arrives at first token(k) + `tool_wait_after_ms`(k). The decode time is already in the
    wait.
  - `arrival_time_ms` is always 0; ignore it.
  - Round 0 of a joined session is a real 1-token, 1-output request. Its long wait spreads the joins.
- **Token synthesis:** deterministic per session. A round's prompt is the previous round's prompt and outputs
  (`prefix_len` tokens), followed by `input_len` new tokens. Any deterministic generator works, as long as every
  prefix is token-identical to what the cache holds.
- **KV hand-back emulation.** Decode runs elsewhere, so the system never computes the output tokens. When a round
  completes, allocate and mark as valid the KV of its first `output_len − 1` output tokens without computing them,
  retain it with the context, and write it through to the tiers. The next round then hits on all of its declared
  prefix except the last output token, which it computes with its fresh prompt.
- **Warm-start SSD seeding (M6 only).** The joining round of each of the first N sessions reads its context from the
  slowest tier. Pre-stage those contexts on SSD as synthetic KV before the run: for c2656 over 16 min that is
  1,775 sessions and 310M tokens. In the HBM-only presets (`warm_start: false`) the joining round recomputes its
  context instead.
- **Tier writes** are real work in the real system, but the sim does not price them (no write bandwidth, no write
  queue). Record their cost as a gap if it shows.

### 2.1 Short benchmark: 16 minutes (use this for every check)

**Run 16 minutes and measure the rounds that arrive in [5, 15) min.** The first 5 minutes are warm-up: HBM and the
tiers fill, and throughput reads up to 12% high.

- **Traces:** `closed16m_cN.csv`, 2–13 MB each (`make_traces.sh`).
  - Full configuration: c2000, c2500, c2656, c3000 (sha256 prefixes `f48a739983a4`, `d8f23e1e5d6d`,
    `ed51b6eda564`, `bebc1346367c`).
  - HBM only: c500, c600, c650, c750, c1000, c1250, c1500 (c500 `50d024d2869f`, c600 `a0d2c1b16fc9`).
  - They are cut from the 8 h closed-loop files by `cut_trace.py --minutes 16 --sessions 2N`. It keeps each
    session's rounds that can arrive before 16 min, plus one more, so no session ends and frees its slot early.
- **Sim presets:** `pp5_16min.yaml` is the §3.1 preset at 16 min; `pp5_16min_hbm.yaml` is the same with DRAM and
  SSD off. Rounds/s and TTFT below are over rounds arriving in [5, 15) min, from `raw/request_slo.parquet`;
  `window_metrics.py` prints each row. The HBM preset also runs c2000 and c2656, deep in collapse.
- **Checked:** the 16-min runs give the same [5, 15) numbers as the 8 h runs' first 16 minutes (identical rounds/s
  and p50/p90; p99 within 0.2 s).

Targets for the short benchmark (PP5, full configuration). Use these, not the 8 h numbers, when comparing a short run.

| Sessions (per GPU) | Rounds/s (per GPU) | TTFT p50 / p90 / p99 (s) | 8 h steady state, for reference |
|---|---|---|---|
| 2000 (400) | 85.1 (17.0) | 0.18 / 0.32 / 1.62 | 84.1; 0.17 / 0.30 / 1.35 |
| 2500 (500) | 98.5 (19.7) | 0.65 / 1.17 / 3.81 | 100.6; 0.57 / 1.20 / 3.67 |
| **2656 (531)** | **101.7 (20.3)** | **0.90 / 1.69 / 4.78** | 103.9; 0.88 / 1.86 / 7.03 |
| 3000 (600) | 108.1 (21.6) | 1.01 / 2.22 / 46.4 | 105.8; 1.23 / 2.74 / 61.9 |

- Throughput and p50 from a short run are within about 2% and 0.1 s of the steady state.
- p99 is not: the long-prompt tail builds over hours. Compare p99 only with the short-run row.

HBM only (`pp5_16min_hbm.yaml`, the M4/M5 target):

| Sessions (per GPU) | Rounds/s (per GPU) | TTFT p50 / p90 / p99 (s) | HBM prefix hit |
|---|---|---|---|
| **500 (100)** | **20.4 (4.1)** | **0.11 / 0.17 / 0.74** | 98.7% |
| 600 (120) | 22.2 (4.4) | 0.15 / 0.69 / 60.1 | 97.2% |
| 650 (130) and above | collapses: under 1 round/s, p50 over 5 min | | |

- c500 is the target. c600 is at the knee: p99 sits at the 60 s force-schedule bound.
- From c650 the queue outlives HBM residency, every round misses and recomputes its whole context, and the run never
  recovers. A real system there should collapse the same way; it is not an operating point.
- HBM hit = Σ `prefix_cache_hit_tokens` ÷ Σ `declared_prefix_tokens` over the window's rounds, from
  `logs/glm53_flash_pp5/runs_16min_hbm/cN/raw/request_slo.parquet`.

## 3. Final target numbers (simulation)

### 3.1 PP5, fastk kernels, the recommended configuration

Preset `pp5_8h.yaml` (8 h, arrivals read over [1 h, 8 h), every analyzer subject).

| Sessions (per GPU) | Rounds/s (per GPU) | TTFT p50 / p90 / p99 (s) | Mean stage-0 microbatch (tokens) | Stage busy (mean) |
|---|---|---|---|---|
| 2000 (400) | 84.1 (16.8) | **0.17** / 0.30 / 1.35 | 4,032 | 92.8% |
| 2500 (500) | 100.6 (20.1) | 0.57 / 1.20 / 3.67 | 12,308 | 94.5% |
| **2656 (531)** | **103.9 (20.8)** | **0.88 / 1.86 / 7.03** | 19,020 | 94.6% |
| 3000 (600) | 105.8 (21.2) | 1.23 / 2.74 / 61.9 | 28,174 | 94.6% |
| 3500 (700) | 104.9 (21.0) | 1.28 / 11.5 / 62.1 | 31,432 | 94.0% |

- At light load the budget stays at 4k tokens, so microbatches are small and p50 drops to 0.17 s at 79% of the peak
  throughput (81% of the throughput point).
  This is the low-latency point; c2656 is the throughput point.
- Past 600 sessions/GPU p99 sits at the 60 s force-schedule bound: overload, not an operating point.
- Per-stage busy at c3000: 90.5 / 94.1 / 94.1 / **99.5** / 94.7%. Stage 3 is the bottleneck (3 DSA + 6 KDA layers).
- Computed prefill: 182.6k tokens/s for the whole pipeline at c3000 (`summary.json`, whole 8 h).

### 3.2 Against PP8 and DP8EP8

Same workload, kernels and tiers: PP5's peak is 21.2 rounds/s per GPU, against 20.9 for PP4 (split [12,11,11,11];
20.5 with the default [11,11,12,11]), 19.9 for PP8 and 15.7 for DP8EP8. These are the study's 8 h sweeps; only PP5's
preset is included here (§12).

| Design | Peak rounds/s/GPU | Best point with TTFT p50 / p99 under ~1.2 / 5 s |
|---|---|---|
| **PP5** (5 B200) | **21.2** | **20.7** at 531 sessions/GPU, chunk 32k fixed (1.19 / 4.5 s) |
| PP8 (8 B200) | 19.9 | 19.7 (1.17 / 3.6 s) |
| DP8EP8, shortest-prefill-first (8 B200) | 15.0 (8k chunk) / 15.7 (16k) | 12.7 at 312 sessions/GPU (0.41 / 5.7 s) |

- PP5's stages are better balanced than PP8's (busy 94.6% vs 89.0%).
- DP's attention for one request runs on one rank, and EP keeps the ranks in lockstep. A short prompt waits for the
  busiest rank's chunk, so DP's p50 never falls below about 0.4 s.
- DP loses throughput past its peak (15.7 → 13.0 from 438 to 562 sessions/GPU) because the 1 s tier read-wait bound
  exists only on the pipeline head. That is a known gap in the sim's DP worker, not a DP property.

---

## 4. M1: every kernel works, at the DB's performance

**Goal.** Each kernel the sim prices runs in the real stack, with the same callable and layout, at the time the
profile DB records for it. Nothing above the kernel is needed yet.

### 4.1 Which kernel at each position

| Position | Kernel | DB kind / backend | Profile runner (`profiling/runners/…`) | Env |
|---|---|---|---|---|
| mHC boundary (2 per layer) | DeepGEMM `mega_mhc` | `mhc_fused_post_pre_rms_norm` / `deepgemm_mega_nonshifted` | `mhc/mhc_fused_post_pre_rms_norm_deepgemm_mega_nonshifted.py:148` | fork |
| Layer 0 first mHC pre | TileLang mHC pre | `mhc_pre_rms_norm` / `vllm_tilelang` | `mhc/mhc_pre_rms_norm_vllm_tilelang.py:62` | vllm |
| DSA `fused_qkv_a`, `q_b_proj`, `o_proj`; indexer `wq_b`, `wk_weights`, `kpool_gate_score` | BF16 `F.linear` (cuBLAS) | `single_gemm` / `torch_linear_vllm` | `gemm/torch.py:75` | vllm |
| DSA q/kv latent norm | `fused_q_kv_rmsnorm` (Triton) | `q_kv_rms_norm` / `vllm_triton` | `attention/q_kv_rms_norm_vllm_triton.py:39` | vllm |
| Indexer `head_weights` | `torch.mm`, FP32 out | `gemm_fp32_output` / `torch_cublas` | `gemm/gemm_fp32_output_torch_cublas.py:94` | vllm |
| Indexer logits | DeepGEMM `fp8_mqa_logits` (chunked, ≤ 512 MiB logits) | `dsa_mqa_logits_prefill` / `deepgemm_fp8` | `attention/dsa_mqa_logits_prefill.py:516` | vllm |
| Indexer top-k (512 pools) | faster of DeepSelect `topk` and `topKPerRowPrefill` (CUDA) | `dsa_topk_prefill` / `deep_select`, `vllm_cuda` | `attention/dsa_topk_prefill.py:316`, `:548` | fork, vllm |
| MLA q absorb (W_UK), V up (W_UV) | `torch.bmm` | `batched_gemm` / `torch_mla_q_absorb_no_rope`, `torch_mla_v_up_unpadded` | `gemm/batched_gemm.py:463`, `:479` | vllm |
| MLA cache write | `concat_and_cache_mla` (CUDA, BF16 → FP8) | `mla_cache_append` / `vllm_cuda` | `attention/mla_cache_append.py:375` | vllm |
| Top-k ids to cache slots | `triton_convert_req_index_to_global_index` (Triton) | `dsa_sparse_index_remap` / `vllm_triton` | `attention/dsa_sparse_index_remap.py:940` | vllm |
| Sparse MLA attention | FlashInfer trtllm MLA, sparse, FP8 | `dsa_sparse_mla_attention` / `flashinfer_trtllm_fp8` | `attention/dsa_sparse_mla_attention.py:873` | vllm |
| KDA `in_proj`, `f_b_proj`, `g_b_proj`, `o_proj` | BF16 `F.linear` | `single_gemm` / `torch_linear_vllm` | `gemm/torch.py:75` | vllm |
| KDA short conv | Dao-AILab `causal-conv1d`, channel-last, one launch per sequence | `gdn_causal_conv_prefill` / `dao_channellast` | `attention/gdn_causal_conv_prefill_dao_channellast.py:305` | conv |
| KDA recurrence | FlashInfer `recurrent_kda`, `cute-dsl-persistent` | `kda_chunk_prefill` / `flashinfer_cute_persistent` | `attention/kda_chunk_prefill_flashinfer.py:322` | kda |
| MoE router gate (the sim prices two launches; one is enough) | `torch.mm`, FP32 out | `gemm_fp32_output` / `torch_cublas` | `gemm/gemm_fp32_output_torch_cublas.py:94` | vllm |
| MoE input quant, dense-FFN input quant | `scaled_fp4_quant` (CUDA) | `nvfp4_quant` / `vllm_cuda` | `elementwise/nvfp4_quant.py:68` | vllm |
| Routed experts (288, top-8) | FlashInfer `trtllm_fp4_block_scale_moe` | `nvfp4_fused_moe` / `flashinfer_trtllm_sm100` | `moe/nvfp4_fused_moe.py:501` | vllm |
| Shared expert `gate_up`, `down` (serial after routed) | BF16 `F.linear` | `single_gemm` / `torch_linear_vllm` | `gemm/torch.py:75` | vllm |
| Dense FFN (layers 0–2) `gate_up`, `down` | FlashInfer `mm_fp4`, cute-dsl, NVFP4 | `single_gemm` / `flashinfer_cutedsl` | `gemm/flashinfer_fp4.py:140` | vllm |
| Final mHC post (stage 4) | TileLang mHC | `mhc_fused_post_pre_rms_norm` / `vllm_tilelang` | `mhc/mhc_fused_post_pre_rms_norm_vllm_tilelang.py:84` | vllm |
| Final norm (stage 4) | `rms_norm` (CUDA) | `rms_norm` / `vllm_cuda` | `norm/rms_norm_vllm_cuda.py:33` | vllm |
| `lm_head` (stage 4) | BF16 `F.linear` | `single_gemm` / `torch_linear_vllm` | `gemm/torch.py:75` | vllm |
| Glue (see the note below) | **stand-in only**: a Triton copy sized by the bytes the real op moves | `elementwise` / `triton` | `elementwise/triton.py:123` | project |

The sim selects the KDA recurrence, short conv, mHC boundary and top-k rows above through the run config's native
per-role `backends:` override: [`backends.yaml`](backends.yaml), which every preset here loads. It is the
`--emit-backends` skeleton with 14 `stage/pp.*` roles changed.

The kernels are standalone library kernels (FlashInfer, DeepGEMM, DeepSelect, cuBLAS, causal-conv1d, and CUDA/Triton
kernels whose source sits in the vLLM tree). Use the kernel itself; no serving framework is assumed. Reuse them: the
vLLM-tree kernels can be vendored as they are. Backend and env names below only identify the DB rows and the
profiling setup.

Envs (details in `profiling/README.md`):
- **vllm** (`vllm_env`): the profiling container built from the vLLM fork at commit `3f667d7`; the vLLM-tree kernels
  above come from that commit. Some rows carry a NULL `backend_version` in `_profile_run`; their env is still this one.
- **fork** (`vllm_upstream_fork_env`): `alignment/profiler/vllm` and its `.venv`.
- **kda**: `~/profile_envs/flashinfer_kda`, built by `profiling/exec/flashinfer_kda_env.sh`.
- **conv**: the project interpreter plus `~/profile_envs/causal_conv1d` (causal-conv1d 1.7.0).
- **project**: the ServingStudioSim `uv` env.

Each runner builds the kernel's inputs, including the corpus expert histogram, the causal-tail valid counts and the
paged FP8 cache, then times the public call with CUPTI. Use it as the template for calling the kernel the same way.

**How the DB times a kernel, so time yours the same way.** `time_ms` is the CUPTI GPU-active time of one logical
call (all its launches) on the runner's inputs, with L2 displaced before each timed call (`Timer.cupti`). Time the
real kernel the same way: same inputs, cold L2, CUPTI kernel time, not wall time.

**Glue must be implemented for real.** The `elementwise` / `triton` rows are not real kernels. Each is a synthetic
Triton copy of a fixed number of bytes per token, so the sim prices it at memory bandwidth. Read each byte count as a
**time budget**, not as the op's real traffic:
- Most budgets are the op's tensor bytes in and out. `q_fwht_quant` (Hadamard-128 + quant) gets 5× its tensor bytes,
  which matches a measured kernel at 2,048 tokens (`worklet/glm53_dsa_attn_local.rs:85-89`).
- `glue` (×17 launches) and `prefill_glue` (×8) in DSA attention are index and mask plumbing around MLA, sized from a
  captured serving trace. A real system that folds them into its kernels and spends nothing there is fine.

The real system needs a real kernel for each, fused where it can be, within its budget. Together they are about 4.4%
of busy time.

| Layer | Glue to implement |
|---|---|
| DSA indexer | `k_norm`, `q_fwht_quant` (FWHT + FP8 quant of q), `weight_scale`, `kpool_prefill_write`, `kpool_tail_seed`, `prefill_gather` (×2), `expand_pools` |
| DSA attention | `q_fp8_quant`; `glue` (×17) and `prefill_glue` (×8), plumbing around MLA (64 and 8,704 B/token each) |
| KDA | `state_gather` and `state_scatter` (recurrent state in and out of the cache), `gated_norm` |
| MoE | `input_glue`, `combine_glue`; shared expert `act_and_mul` |
| Dense FFN | `act_and_mul` |
| Stage ends | `embedding`, `hc_expand` (stage 0); `hc_contract_mean` (stage 4) |

Bytes per token of each are in the kernels doc §1.4 (`static key`), and their per-case ms are in `iter_breakdown.json`.

**The DSA indexer selects pools, not tokens** (`index_kpool: 4`, `index_topk: 2048` in the model config).
- Keys are compressed into pools of 4 tokens. A query at position `pos` scores the `(pos + 1) / 4` complete pools
  before it (indexer logits), and top-k keeps the best 512 pools.
- The partial tail pool (the query's last 0–3 tokens) is always selected (`index_kpool_always_select_tail`).
- `expand_pools` turns the pools into token slots: at most 512 × 4 + 3 = 2,051, padded to `selected_k` = 2,176.
  Sparse MLA attends over those slots; a query with a short context has fewer valid slots (`valid_counts`).

**Which top-k kernel.** The sim takes the faster of the two per call; build both, or the one that covers your rows.
- DeepSelect wins on long single rows, about 65k keys and up.
- `topKPerRowPrefill` wins on packed rows of many short-to-medium requests, which are most calls: 85% at the
  operating point (§4.5).

Not executed in this deployment (do not implement or time them):
- **Decode kernels**: sparse MLA decode, paged MQA logits decode, persistent top-k decode, conv update and recurrent
  KDA decode. The system is prefill-only; decode runs elsewhere.
- **MLA layout copies** `q_concat` and `output_copy`: off (`mla_layout_copies: false`). The kernels take their
  inputs without these extra copies.
- **KDA beta sigmoid glue** (`prefill_glue`, Scale 0): the KDA kernel takes beta logits and applies the sigmoid
  itself.
- **The overlapped shared-expert copy**: the shared expert overlaps the routed experts only at ≤ 256 tokens, which
  never happens here. It runs serially after them.
- **Collectives**: each stage is TP1/EP1, so there is no all-reduce or all-to-all. The stage-to-stage send is M3.

### 4.2 Which DB, and its schema

- `profiling/profile.db` at this commit (the same on main since PRs #82 and #85) holds every row the runs read:
  - the fastk rows: `flashinfer_cute_persistent` 1,024, `dao_channellast` 432, `deepgemm_mega_nonshifted` 88 and
    `deep_select` 412 (plus 1,536 `flashkda` rows, the alternative KDA prefill that fastk does not pick);
  - 412 `vllm_cuda` top-k rows re-profiled on unsorted scores;
  - the remap and glue rows.
- Sparse-MLA `tflops` count valid slots only. Older copies of the DB hold larger derived values for the same rows,
  for example 1,600.8 against 1,547.7. **Compare kernels on `time_ms`.**

Schema:
- One table per kernel kind, keyed `UNIQUE(gpu_name, backend, args_hash)`. The spec is the args columns; there is no
  JSON spec column.
- Metric columns: `time_ms` (CUPTI GPU-active time of one logical call), `tflops`, `memory_bandwidth_gbps`, and
  `run_key` → `_profile_run` (profiler git hash, CUDA, driver, backend version).
- List arguments are encoded as text:
  - `nvfp4_fused_moe.per_expert_batches` is the per-expert row histogram, sorted descending.
  - `dsa_sparse_mla_attention.valid_counts` is RLE: `c:1..N@2176` is a causal ramp capped at 2176.
- FLOP formulas are in the kind DOCs (`profiling/kernels/<kind>.py`):
  - GEMM `2mnk`;
  - fused MoE `2·Σrows·3·H·I`;
  - KDA `2·T·H·(3D² + 4·64·D)`;
  - sparse MLA `2·H·Σvalid·(D + value_dim)`.
- The sim never reads a single row. It interpolates each kernel's grid, so an operating-point time lies between grid
  rows (axes in `simulator/src/timing/kernels/<kind>.rs`).

### 4.3 How to match a real kernel

1. Take the leaf's static key from any PP5 run's `raw/cost_manifest/worker_stage_3.json` (`slots[].kernel_config`),
   for example `logs/glm53_flash_pp5/runs_16min/c3000` from `pp5_16min.yaml`; every load and duration writes the same
   manifests. Take its dynamic shape from `slot_input` in the timing-predict cost log (§5.2),
   `logs/glm53_flash_pp5/predict/raw/cost_log/worker_predict_0.parquet`: one row per case, in `predict_cases.json`
   order, with `slot_input` and `slot_time_ms` per slot. The last case is an operating-point microbatch of the c3000
   run (12 sequences, 21,301 tokens).
   The sim's time for that leaf, interpolated over the grid, is its ms in `payloads/iter_breakdown.json`. Compare
   against that number.
2. To see the measured rows around it, query the DB through the read-only CLI. `query` returns a grid row only when
   the spec is exactly on the grid; it does not interpolate.
   ```bash
   uv run python -m launcher kernel-profile query single_gemm --backend torch_linear_vllm \
     --gpu-name "NVIDIA B200" --spec '{"m":32768,"n":24896,"k":4096,"dtype":"bf16"}' --json
   # time_ms 4.878, tflops 1370.0
   ```
3. Time the real engine's kernel with CUPTI at the same shape. Use the backend's runner as the template for inputs and
   layout (`profiling/runners/…`; table in the kernels doc §4). The runner builds the corpus-shaped expert histogram,
   the causal-tail valid counts and the paged FP8 cache.
4. A gap above the band is a real-engine gap; fix the engine.
   Use the timing method of §4.1 (CUPTI kernel time, cold L2). If the real callable or layout differs from the
   runner, re-profile it (skill `operate-profile-existing-kernel`; run it on a GPU host or through your cluster's scheduler) before comparing.

Spot checks. These are rows read straight from the DB: run the same kernel at the same shape on a B200 and you
should see about this time. They are a quick first test before the per-leaf comparison of §4.6.

| Position (§4.1) | Shape | DB time (ms) | Achieved |
|---|---|---|---|
| Routed experts | 16,384 tokens / 32,768 tokens | 1.931 / 3.453 | 3,416 / 3,821 TFLOP/s |
| KDA recurrence | 32,768 tokens, 2 sequences of 16,384 | 2.622 | 131 TFLOP/s |
| Sparse MLA attention | 32,768 queries over their own 32,768-token context (causal), at most 2,176 selected slots each | 5.838 | 1,548 TFLOP/s |
| Indexer logits | 8,192 queries × 65,536 keys | 2.070 | 1,996 TFLOP/s |
| Indexer top-k | 8,192 rows × 16,384 keys: `topKPerRowPrefill` / DeepSelect | 0.240 / 0.462 | — |
| KDA `in_proj` | m 16,384 / 32,768 (n 24,896, k 4,096) | 2.430 / 4.878 | 1,375 / 1,370 TFLOP/s |
| mHC boundary | 32,768 tokens | 0.685 | 5,498 GB/s |
| KDA short conv | one sequence of 16,384 tokens | 0.357 | — |
| Dense FFN `gate_up` | m 32,768 (n 24,576, k 4,096) | 1.540 | 4,284 TFLOP/s |

To query any other shape, use the CLI of step 2; the kernels doc §3.4 has the SQL.

Row caveats:
- B200 rows profiled before 2026-09-25 may carry a warm-L2 bias.
- Triton elementwise rows: per-process autotune makes the same spec read up to 57% apart. Match them as a band.
- The routing corpus is a GLM-5.3-Flash **FP8** serving capture (`balanced_c32`, 2026-09-24), reused for the NVFP4
  expert histograms.
- `dsa_mqa_logits_prefill` B200 rows are kept from an older kernel. They are 5–15% slower than today's kernel, so
  the real kernel may beat the sim.
- KDA rows use anchor-tuned autotune configs, not one per shape. Do the same; do not tune per shape.
- The KDA chunk-prefill grid stops at 32,768 tokens per step, so 64k microbatches are extrapolated. This only
  matters for the 64k rows of §7.2.

### 4.4 Which kernels matter most (time share at the operating point, c3000)

| Group | All 5 stages | Stage 3 |
|---|---|---|
| KDA projections (BF16 GEMMs) | 26.4% | 22.2% |
| Routed MoE (NVFP4 fused MoE + quant) | 17.8% | 18.2% |
| KDA core (chunk prefill, conv, state, norm) | 14.3% | 12.0% |
| DSA sparse MLA (attention, remap, cache append, absorb, v_up) | 10.3% | 13.3% |
| DSA indexer (logits, top-k, its GEMMs) | 8.2% | 10.6% |
| mHC boundaries | 7.3% | 6.9% |
| DSA projections and glue | 7.1% | 9.2% |
| Shared expert (BF16) | 5.9% | 6.0% |
| Dense FFN (stage 0 only), router, MoE glue, embedding/head | 2.7% | 1.7% |
| Communication | 0 | 0 |

- By kind over all stages, `single_gemm` (BF16 projections) is the largest at 39.5%, then `nvfp4_fused_moe` at 17.5%.
- The single largest leaf is KDA `in_proj`: 15.9% of stage 3.
- Work in this order. The BF16 projections are the first place a real system can lose. They sit at 1,360–1,460 TFLOP/s against a grid
  peak of 1,490–1,640.

### 4.5 Throughput the kernels reach at the operating point (p50, from the full run)

| Leaf | Achieved | Grid peak |
|---|---|---|
| routed `fused_moe` | 3,721 TFLOP/s | 4,082 |
| sparse MLA prefill (valid-slot FLOPs) | 1,548 TFLOP/s | 1,564 |
| indexer logits | 1,877 TFLOP/s | 2,055 |
| KDA `in_proj` / `o_proj` | 1,418 / 1,365 TFLOP/s | 1,591 / 1,488 |
| DSA `o_proj` / `q_b_proj` / `fused_qkv_a` | 1,391 / 1,395 / 1,427 TFLOP/s | 1,512 / 1,519 / 1,639 |
| KDA chunk prefill | 132.5 TFLOP/s (1,048 GB/s) | 138.4 |
| mHC fused | 5,498 GB/s | 5,508 |
| indexer top-k | 1,987 GB/s | 6,306 (deep_select, 1M-key single row) |
| dense FFN `gate_up` (NVFP4) | 4,557 TFLOP/s | 5,614 |

Top-k picks `topKPerRowPrefill` (`vllm_cuda`) for 8,960 calls and DeepSelect for 1,555.

### 4.6 Check

- Every position of §4.1 runs the listed kernel in the real stack.
- Compare its CUPTI time with `time_ms` at the §4.3 reference shapes and at the leaves of the M2/M3 cases
  (`iter_breakdown.json`, §5.2). Get it as close as you can.
- Log each gap per leaf with its cause. A real kernel that is faster than the DB is a candidate to re-profile into
  the DB.

---

## 5. M2: each layer matches the sim's speed

**Goal.** Compose the M1 kernels into the three layer types, in the sim's launch order, and match each layer's time.

### 5.1 Layer structure: follow the cost tree

The sim's per-stage CostTree lists every layer's kernels in launch order, with their shapes. Build each layer type
(DSA + MoE, KDA + MoE, KDA + dense FFN) to match it.
- **Readable tree:** kernels doc §1.3 has the stage-3 tree. §1.4 has every leaf with its kernel, static shape and the
  input that varies per iteration.
- **Machine-readable tree:** `raw/cost_manifest/worker_stage_{0..4}.json` of any PP5 run (§4.3), with `nodes`, `node_labels` and
  `slots[].kernel_config`. How the nodes combine: `simulator/src/timing/COST_TREE.md`.
- **Per-case time of every node:** `logs/glm53_flash_pp5/predict/reports/iter_breakdown.ans` (text) and
  `payloads/iter_breakdown.json` (§5.2). How to read the JSON:
  - `iterations[i]` is case i of `predict_cases.json`; `nodes` is the tree in pre-order, with `depth` giving the nesting.
  - Depth 1 is a stage ("unified pipeline stage k of 5"), depth 2 a layer group such as `kda_moe x7 (Scale 7)`.
    A group collects the stage's layers of one type; the real launch order is by layer index (§6.1).
  - On a leaf, `slot` indexes the cost manifest's `slots[]` (its kernel and static key), `ms` is one launch and
    `total_ms` is `ms` times every enclosing Scale count. The text view shows `total_ms`.
- **Source:** `simulator/src/arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs` (stages) on top of
  `glm53_flash_vllm_fp8_kda_dsa_moe.rs` (layers). The worklets `simulator/src/worklet/glm53_*` give the launch order
  in their module docs.
- **Precision:** NVFP4 for the routed experts and dense FFN; BF16 for the projections, routers, shared experts and
  `lm_head`; FP8 KV cache.

### 5.2 Per-layer target

From timing-predict on single requests: one request is one microbatch, with no chunking and no other requests. The
same cases give the per-stage targets of M3 (§6.3).

- Each stage runs its CostTree once on the case. Same arch, fastk backends and corpus routing as the reference run.
- Config: `predict_pp5.json`. Cases: `predict_cases.json`, labelled in `predict_case_labels.json` (documentation only;
  no tool reads it).
- Output: `payloads/iter_breakdown.json`, which has per-stage, per-layer-group and per-leaf ms for every case.
- Command:
  `uv run python -m launcher timing-predict doc/handoff/glm53_flash_pp5/predict_pp5.json --no-gpu --analyzer-essential-only`
  (CPU only, seconds; output in `logs/glm53_flash_pp5/predict`).
  It needed no new profile rows (dry-run: 0 / 7,648 specs missing).
- A case is `[prefix, new tokens]`. The prefix is already in the KV cache.

One layer (layer group ÷ count, from stages 1–3), ms:

| New tokens | Prefix | DSA + MoE layer | KDA + MoE layer | KDA + dense FFN layer (stage 0) |
|---|---|---|---|---|
| 8,192 | 0 | 5.85 | 4.71 | 3.7 |
| 8,192 | 512k | 11.7 | 4.71 | 3.7 |
| 16,384 | 0 | 11.8 | 8.93 | 7.5 |
| 16,384 | 512k | 22.6 | 8.93 | 7.5 |
| 32,768 | 0 | 23.3 | 17.4 | 15.3 |
| 32,768 | 512k | 45.4 | 17.4 | 15.3 |

How to read the tables:
- **The prefix only costs on the DSA layers.** KDA carries a fixed-size recurrent state, so a KDA layer's time does not
  change with the prefix. The DSA layer grows through the indexer (logits over all prefix pools, then top-k) and the
  sparse MLA attention (up to 2,176 selected slots per query).
- That is why stage 3, with three DSA layers, falls further behind as the prefix grows: 1.03× the stage-1 time at
  prefix 0, 1.13× at 512k.
- Time is close to linear in new tokens: 8k → 32k is 3.8× on stage 3 at prefix 0.
- The two batched rows of §6.3 (16 × 2,048 and 32 × 1,024 over 128k each) bridge to the operating point. Many
  short chunks over a 128k prefix cost about the same as one 32k request over a 128k prefix.
- Stage 0 is cheaper than stages 1–2 because its three dense-FFN layers cost less than MoE layers. Stage 4 adds
  0.4–0.9 ms for the final mHC, the norm and `lm_head`.
- Per-leaf ms for any case is in `iter_breakdown.json`, keyed by the same leaf names as the cost tree (§5.1).
  Compare the real kernels leaf by leaf, then per layer group, then per stage.
- The glue leaves (§4.1) are bandwidth estimates. A real layer with unfused glue will likely miss here first.

### 5.3 Check

- Time one layer of each type in the real stack, alone, at the six rows of §5.2, with the real checkpoint weights and
  the case's prefix already in the KV cache.
- Report two numbers per layer: the CUPTI kernel sum (compare it with the sim) and the wall time (the gap between
  them is launch and host overhead, which the sim does not model; §11).
- Each should land on its ms, and its leaves should sum the same way as `iter_breakdown.json`.
- A layer slower than the sum of its M1 kernels means launch gaps, extra copies or a different fusion. Find the
  extra kernels in the trace and remove them; do not change the sim.

---

## 6. M3: the whole model on PP5 (correctness, per-stage time, inter-stage send)

**Goal.** Wire the 45 layers into 5 stages and run one microbatch end to end. The outputs must be correct, each
stage's time must match, and the inter-stage send must work at the expected cost. No scheduling yet: one microbatch
at a time, nothing else in flight.

### 6.1 Layer partition

- 45 layers. 11 are DSA (sparse MLA + indexer): layers 3, 7, …, 43, every 4th from 3. The other 34 are KDA
  (linear attention). Layers 0–2 have a dense FFN and 3–44 have MoE. Every layer carries the 4-wide mHC residual
  stream.

  | Stage | Layers | DSA layers | KDA layers | KV B/token | Extras |
  |---|---|---|---|---|---|
  | 0 | 0–8 | 3, 7 | 7 | 1,090 | embedding, hc_expand, dense FFN on 0–2 |
  | 1 | 9–17 | 11, 15 | 7 | 1,090 | |
  | 2 | 18–26 | 19, 23 | 7 | 1,090 | |
  | 3 | 27–35 | 27, 31, 35 | 6 | **1,635** | the bottleneck, 99.5% busy |
  | 4 | 36–44 | 39, 43 | 7 | 1,090 | final mHC post, hc_contract, norm, lm_head |

- Split [9,9,9,9,9]: 9 consecutive layers per stage.
  - A search over every contiguous split found it min-max optimal. The study's static estimate is 97.1% balance
    against PP8's 91.0%; a per-layer fit at 8,192 tokens gives 97.7% (scheduler.md §7).
  - Measured stage busy is 94.6% mean against 99.5% max (95.1%).
- Do not rebalance layers further. The study tried it and it is a dead end.

**Build it for any PP size.** PP5 is the target, but the stage count and the layer assignment must be configuration,
not code, so that PP4 or PP8 is only a different assignment:
- A stage is a contiguous layer range. Its layer types, its KV bytes per token (545 B per DSA layer it holds; KDA
  layers add only the per-request state) and its extras (embedding on the first stage, final mHC, norm and
  `lm_head` on the last) follow from the range. Nothing may assume 5 stages or 9 layers.
- Keep at least one DSA layer per stage. The sim's KV model stores the KDA state pages inside the DSA KV tensors and
  rejects a stage without one (§6.4).
- What follows from the assignment: the pool size (set by the stage with the most DSA layers), the stage-to-stage
  send count (PP − 1 per microbatch), at most PP microbatches in flight, the budget split over PP, and the
  balanced tier load (§9.1).
- The sim takes the same two knobs: `pp_size` and `layer_partition` (per-stage layer counts summing to 45; empty =
  the default split, remainder layers to the stages before the last). Default splits: PP4 [11,11,12,11], PP8
  [5,5,6,6,6,6,6,5]. The PP8 numbers come from `pp5_8h.yaml` with only `pp_size: 8` and
  `attn_gpu_memory_gb: 123.43` changed.

### 6.2 What crosses a stage boundary

- **Close the mHC at the boundary.** Each stage finishes its last layer's mHC post. It then sends the full 4-wide
  residual stream, and the next stage starts with its first layer's mHC pre. Nothing else is pending across the
  boundary (`arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:23-32`).
- **Where the sim prices the boundary mHC.** Inside a stage, one layer's mHC post and the next layer's mHC pre run as
  one fused leaf (`*_mhc_post_pre`). The sim keeps that fusion across the boundary: stage k's last post is priced in
  stage k+1's first fused leaf (stages 1–4 start with `attn_mhc_post_pre`, stage 0 with a plain `attn_mhc_pre`). A
  real stage k that runs its last post itself moves one standalone post, priced as fused minus pre (≈ 0.06 ms at
  8k, 0.24 ms at 32k), from stage k+1 to stage k (`arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:28-32`). Compare stage times with that shift in mind; the sum over stages is the same.
- **Size:** `hc_mult × hidden × bf16` = 4 × 4096 × 2 = 32,768 B/token, 1.07 GB for a 32k microbatch (`:716-720`).
- **Cost:** one NVLink point-to-point send per hop, priced on the profiled B200 `p2p_intra` curve
  (`deployment/pp.rs:342-356`).
- **Overlap:** a stage receives the next microbatch while it computes the current one. At the operating point the
  sends add no busy time (communication bucket 0).

### 6.3 Per-stage target for one microbatch

Stage time, ms (stage 3 is the bottleneck):

| New tokens | Prefix | Stage 0 | Stage 1 | Stage 2 | **Stage 3** | Stage 4 | Sum of stages |
|---|---|---|---|---|---|---|---|
| 8,192 | 0 | 41.8 | 44.7 | 44.7 | **45.9** | 45.1 | 222.2 |
| 8,192 | 32k | 42.3 | 45.2 | 45.2 | **46.6** | 45.6 | 224.9 |
| 8,192 | 128k | 44.6 | 47.5 | 47.5 | **50.1** | 47.9 | 237.5 |
| 8,192 | 512k | 53.5 | 56.4 | 56.4 | **63.4** | 56.7 | 286.3 |
| 16,384 | 0 | 82.1 | 86.1 | 86.1 | **89.0** | 86.6 | 429.9 |
| 16,384 | 32k | 82.1 | 86.2 | 86.2 | **89.1** | 86.7 | 430.2 |
| 16,384 | 128k | 86.6 | 90.6 | 90.6 | **95.8** | 91.2 | 454.9 |
| 16,384 | 512k | 103.5 | 107.5 | 107.5 | **121.2** | 108.1 | 547.8 |
| 32,768 | 0 | 162.3 | 168.3 | 168.3 | **174.3** | 169.2 | 842.4 |
| 32,768 | 32k | 163.0 | 169.1 | 169.1 | **175.4** | 169.9 | 846.5 |
| 32,768 | 128k | 172.4 | 178.4 | 178.4 | **189.4** | 179.3 | 897.8 |
| 32,768 | 512k | 206.4 | 212.5 | 212.5 | **240.5** | 213.4 | 1,085.3 |
| 16 × 2,048 | 128k each | 166.2 | 172.3 | 172.3 | **183.5** | 173.2 | 867.5 |
| 32 × 1,024 | 128k each | 167.6 | 173.6 | 173.6 | **185.0** | 174.5 | 874.4 |
| operating-point microbatch (below) | mixed | 103.6 | 108.2 | 108.2 | **112.9** | 108.9 | 541.7 |

- **Operating-point microbatch:** a real stage-3 microbatch from the full c3000 run (iteration 98405): 21,301 new tokens
  over 12 sequences. As `[prefix, new tokens]`: [23757, 1], [87782, 7483], [46892, 13235], [129831, 1], [91873, 47],
  [89546, 49], [188930, 51], [228269, 70], [215446, 46], [210122, 47], [154563, 180], [299798, 91].
  - timing-predict gives stage 3 112.9 ms, which equals what the full run recorded for it.
  - Stage 3 splits as DSA layers 47.0 ms and KDA layers 65.9 ms.
  - Typical of the load: two sequences carry most new tokens, and ten carry tens of tokens over long prefixes.
- **Not included: the stage-to-stage send.** It moves 32,768 B/token: 0.27 GB for 8k (0.48 ms on the B200
  `p2p_intra` NCCL row); 0.54 / 1.07 GB for 16k / 32k (about 0.96 / 1.93 ms at the largest row's 557 GB/s, since the
  grid stops at 268 MB).

So one microbatch's end-to-end time is the sum of its 5 stage times plus 4 sends. For a fresh request:
- 8k: 222.2 + 4 × 0.48 ≈ 224 ms;
- 16k: 429.9 + 4 × 0.96 ≈ 434 ms;
- 32k: 842.4 + 4 × 1.93 ≈ 850 ms.

### 6.4 HBM budget and KV pool

- The device budget is 149.84 GB per B200: the weights + KV that vLLM allocated at `--gpu-memory-utilization 0.85`
  in a measured TP4 FP8 run (81.00 GB weights + 68.84 GB KV per GPU). This is measured, not 0.85 × the nominal
  180 GB; CUDA context, activations and the allocator take the rest.
- Per-stage NVFP4 weights for [9,9,9,9,9] are 28.74 / 39.61 / 39.61 / 39.58 / 40.88 GB (GB = 1e9 B). They were counted
  from the checkpoint's safetensors headers: routed experts and the layer 0–2 dense MLP at 0.5 B + one E4M3 scale per
  16 values, everything else BF16 (FP32 stays FP32), MTP layers excluded. The vision tower and embedding are on stage 0;
  the final norm and lm_head are on stage 4.
- 149.84 − 40.88 (the largest stage) leaves **108.96 GB per GPU for KV** (`attn_gpu_memory_gb`). PP8's [5,5,6,6,6,6,6,5]
  has a largest stage of 26.41 GB, so 123.43 GB.
  - Activation and workspace peaks are not subtracted: 32k-token microbatches, and the 512 MiB indexer logits chunk.
  - So a real system will probably have fewer blocks. Record a shortfall as a real-engine gap to minimize.
  - A larger pool than the sim's is not a violation. Record it as a beneficial gap: it raises the HBM hit rate, so
    M4–M6 may beat the target.
- One block pool for the pipeline, sized by the most constrained stage: 66,626,944 tokens. A block is 8,576 tokens,
  which is 134 pages of 64 tokens: the block holds one KDA state per KDA group and the matching attention pages, so
  all groups allocate in step. KDA recurrent state costs 42,880 tokens (5 blocks) per request (log line "hybrid
  pipeline head"). A real allocator may use other page sizes, as long as the token capacity and state charge match.
- Prefix reuse is token-exact; chunk ends are not rounded to blocks (`prefill_chunk_alignment: plain`).

### 6.5 Check

- **Correctness.** Greedy outputs (or logits) of the PP5 run match the vLLM gate of §1 on the same checkpoint, for
  prompts with and without a cached prefix. Log every mismatch with its prompt and first diverging token.
- **Per-stage time.** Each stage's CUPTI kernel sum for each §6.3 case matches its row, allowing for the boundary mHC
  shift of §6.2. Report its wall time beside it.
- **Send.** The send is 32,768 B/token at the expected time. For one microbatch with nothing else in flight it adds
  4 sends to the end-to-end time.
- The KV pool comes up at the §6.4 size. Record a smaller pool as a real-engine gap and a larger one as a beneficial gap.

---

## 7. M4: scheduling, admission and chunk management

**Goal.** Serve the closed-loop workload (§2) with the full §7 scheduler and HBM prefix reuse only. Target:
`pp5_16min_hbm.yaml` at c500 (§2.1): 20.4 rounds/s, TTFT p50 0.11 / p90 0.17 / p99 0.74 s.

### 7.1 Mechanisms

Full mechanism-by-mechanism map with formulas, `file:line` and defaults:
[`scheduler.md`](scheduler.md). It was checked against the study code (scheduler unchanged on this branch), and its
inferences are marked. Paths are under `simulator/src/`.

| Mechanism | What the real system must do | Code |
|---|---|---|
| Microbatch cadence | Form microbatch k+1 only when stage 0 has finished k and fewer than 5 are in flight. Do not queue batches ahead, because SRPT and the budget must see the latest state. | `worker/workers/pipeline/pipeline_head_worker.rs:264-372` |
| Progress commit | Advance computed tokens at schedule time, so a prompt's next chunk rides the next microbatch while the previous one is on a later stage. | `worker/admission/pipelined_chunked_prefill_admission.rs:676-778` |
| Order | See the pseudocode below. | `…/pipelined_chunked_prefill_admission.rs:339-564`, `worker/admission/policy/shortest_job_first.rs:13-33` |
| Chunk | `min(remaining, budget_left)`, with no Mamba-align chunk ends (`plain`). | `worker/admission/chunked_prefill_admission.rs:1014-1040` |
| Budget | `B = min(32768, LB(R), clamp(ceil(R/5), 512, 32768))` with `LB(R) = 4096 + round(28672 · clamp((R − 1M)/2M, 0, 1))`. R = started prompts' remaining prefill + queued and overdue fresh tokens + in-flight prefill. Requests out for a tier read are not in R. | `…/pipelined_chunked_prefill_admission.rs:130-143, 286-332` |
| Admission gate | Reserve the whole footprint once: `context after prefill + 1 + 42,880 state tokens`. Stop at the first that does not fit; no skip-ahead, no preemption. | `worker/kv/hybrid_gdn.rs:108-132` |
| Force schedule | Requests waiting ≥ 60 s since their round's arrival move to a FIFO served before everything else. Overdue started prompts sort first. | `worker/admission/overdue.rs` |
| Completion | First token at the exit of stage 4 (lm_head included). Retain the context in HBM, and write it through to DRAM and SSD. | `…/pipelined_chunked_prefill_admission.rs:746-776` |

Microbatch formation (`form_prefill_srpt`), run once per microbatch with the budget B:

```text
started = prompts with a chunk already scheduled, sorted by
          (not overdue, overdue ? arrival time : remaining prefill tokens)
if the overdue FIFO is not empty:
    admit_fresh(budget, overdue_only)                # overdue (>= 60 s) requests first
for s in started:
    if s is not overdue:
        admit_fresh(budget, no_longer_than = remaining(s))   # SRPT: shorter queued prompts go first
    chunk = min(remaining(s), budget); schedule it; budget -= chunk
admit_fresh(budget)                                  # the rest of the budget
newly admitted prompts that did not finish join the end of `started`

admit_fresh(budget, ...): while budget > 0, take the next of
    1. a landed tier read;  2. the overdue FIFO;
    3. the head of the shortest-prefill-first queue, keyed
       fresh + declared - max(HBM-resident, tier hit), frozen at enqueue;
       stop at the first head longer than `no_longer_than`.
    A request whose tier read is past the read-wait bound is passed over (§9.2).
    Each admitted request reserves its footprint (Admission gate) and takes
    a chunk of min(its prefill, budget).
```

The config doc at `worker/config.rs:682-687` is stale: past the read-wait bound the code passes the request over; it
does not stop admission.

### 7.2 Microbatch size: what to use

**Use the load-following budget, not a fixed chunk size.**
- The budget is 4k tokens while the prefill backlog is at most 1M tokens. It rises linearly to 32k at 3M.
- Split it evenly over the 5 in-flight microbatches, at least 512 tokens each.
- The formula is in §7.1 (Budget).

Why (the study's chunk-size sweep):
- **Light load needs small microbatches.** At 400 sessions/GPU, p50 is 0.17 s, against 0.66 s with a fixed 32k chunk.
- **Heavy load needs 32k.** 32k is the best fixed size for throughput: against the 21.2 peak, fixed 8k gives
  8% less and 16k 3% less. 64k loses too, because the stages unbalance (busy falls to 83%). The budget reaches 32k when the backlog is
  large, so it keeps the 32k peak (21.2 rounds/s per GPU).
- **Variant for a p99 SLO:** an 8k floor with the ramp at 0.5M–1.5M. It gives p99 ≤ 6 s up to 20.9 rounds/s per GPU
  (the preset: 20.1), at 1% below the peak. Pick it if p99 matters more than p50.

### 7.3 Operating point the scheduler produces (full config, c3000)

| Quantity | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| stage-3 microbatch time (ms) | 148.6 | 154 | 181 | 193 | 262 |
| tokens per microbatch | 27,263 | 28,369 | 32,768 | 32,768 | 32,768 |
| sequences per microbatch | 16.8 | 17 | 27 | 47 | 600 |
| cached prefix per sequence (tokens) | 148,930 | 117,852 | 258,618 | 834,315 | 999,698 |
| new tokens per sequence per chunk | 1,621 | 237 | 4,636 | 18,071 | 32,768 |

### 7.4 Check

- On the 16-min HBM-only benchmark (§2.1) at c500: rounds/s and TTFT p50/p90/p99 against `pp5_16min_hbm.yaml`.
- Run c600 too: the real system should sit at the same knee (p99 near the 60 s bound) and not collapse earlier.
- Microbatch size and sequences per microbatch: compare their distributions with the HBM-only run's own cost log
  (`logs/glm53_flash_pp5/runs_16min_hbm/c500/raw/cost_log/worker_stage_3.parquet`); §7.3 shows the full-configuration shape at c3000.

---

## 8. M5: the prefix cache system

**Goal.** HBM prefix reuse is exact and complete, so only the tiers are left between M5 and the final target.
Mechanisms in detail: scheduler doc §4 and §6.

- **Exact reuse.** Prefix reuse is token-exact; chunk ends are not rounded to blocks
  (`prefill_chunk_alignment: plain`).
  - The hybrid store uses `with_exact_prefix_reuse` (`worker/kv/hybrid_gdn.rs:678-686`).
  - Each retained context keeps one KDA state per KDA group, at its last token.
- **Retention at completion.** A finished prefill retains its post-prefill context in HBM
  (`release_retaining_prefix`, scheduler doc §4e). Retained contexts are evictable LRU.
- **KDA state charge.** 42,880 tokens (5 blocks) per request, reserved with the request's footprint (§6.4).
- **Decode hands KV back** (`external_decode`, scheduler doc §6b).
  - The decode side returns the round's output KV, so the next round reuses `declared + prompt + outputs − 1`.
  - Without the hand-back, every round recomputes its previous outputs: 187M output tokens against 625M fresh input
    tokens, about +30% prefill.
- **Sessions are sticky** to their pipeline, so its HBM and tiers hold the session's context.
- **The queue key uses residency.** Shortest-prefill-first sorts by `fresh + declared − max(HBM-resident, tier hit)`,
  so a resident prefix makes a request short (§7.1).

Check:
- On `pp5_16min_hbm.yaml` at c500, the HBM prefix hit is 98.7% of declared prefix tokens (Σ `prefix_cache_hit_tokens`
  ÷ Σ `declared_prefix_tokens` in `logs/glm53_flash_pp5/runs_16min_hbm/c500/raw/request_slo.parquet`, §2.1). Measure the real system the same way.
- No round recomputes previous outputs beyond the one last output token (§2, KV hand-back emulation).

---

## 9. M6: loading and offloading (DRAM + SSD tiers)

**Goal.** Add the host DRAM and local SSD tiers with async loads that hold their HBM blocks, the read-wait
bound and balanced (striped) storage. This reaches the §3 final numbers. Mechanisms in detail: scheduler
doc §5.

### 9.1 Tier model and storage balance

- **The tiers carry most of the context.** PP5 at c2656 over 8 h (`logs/glm53_flash_pp5/runs_8h/c2656/prefix_tiers_w0.json` from `pp5_8h.yaml`):
  - of 456.2B declared prefix tokens, 31.2% hit HBM, 34.4% are loaded from DRAM and 34.4% from SSD;
  - 99.999% are resident at admission, so nothing is recomputed;
  - pipeline-wide that is 5.45M tokens/s from DRAM and 5.44M tokens/s from SSD.
  - A real system without host/SSD KV offload computes about 3× the prefill and cannot reach §3.
- **Each tier is per GPU, write-through LRU.** DRAM 150 GB at 50 GB/s, SSD 8 TB at 10 GB/s. Reads queue FIFO per tier.
- **Async loads hold HBM.** A read starts when admission reaches the request in its queue. At that moment it reserves
  the request's whole HBM footprint, and it holds it while the read is queued and in flight, until the request is
  admitted. Landed reads are admitted first.
- **Read-wait bound with pass-over.** A read starts only if its tier would begin it within 1 s. Otherwise
  the request is skipped and holds nothing.
  - Without the bound, queued reads pin HBM, more rounds miss, and PP collapses past its knee: PP5 c2656 falls to
    80 rounds/s.
  - Blocking at the head instead of passing over loses: 84 → 64 rounds/s.
- **PP5 numbers.** Each stage owns its layers' KV: [1090, 1090, 1090, 1635, 1090] B/token.
  - At c3000 the balanced sim reads SSD at 7.76 GB/s per GPU on average, 78% of 10 GB/s, so SSD read bandwidth is the
    binding tier resource.
  - Unbalanced, stage 3 would need about 10.6 GB/s and saturate.
  - Unbalanced, stage 3's 150 GB DRAM would also hold only 91.7M tokens against 125M on the others. A partial miss on
    one stage is a full recompute.
- **Balance the per-stage tier load (`prefix_tier_balanced_load`).**
  - A stage with more DSA layers stores and reads more bytes per token. Its SSD link then caps the whole pipeline,
    because every stage must finish its read before the request runs.
  - PP8 example: the 2-DSA-layer stage holds 1,090 B/token, against a pipeline mean of 750 B (5,995 B/token in all).
    Its 10 GB/s SSD allows 9.2M tokens/s for the pipeline; DP8EP8 gets 13.3M.
  - The sim's balanced mode sizes and reads every stage's tiers at the mean bytes/token.
  - No layer split can equalize 11 DSA layers over 5 or 8 stages. A real system must **spread the storage, not the
    layers**; do not keep each stage's KV isolated in its own fixed slice. Options (scheduler doc §5e):
    1. **Capacity.** Partition the node's DRAM and SSD by bytes per token (stage 3 gets 3/11, the others 2/11 each),
       so every stage holds the same token count. Make one eviction decision per session across all ranks, so the
       per-stage LRUs never diverge.
    2. **SSD bandwidth.** Stripe the NVMe across drives and weight the per-rank I/O queues by bytes, so stage 3 gets
       1.36× the mean rank's bandwidth.
    3. **DRAM → HBM.** It goes over each GPU's own PCIe link. Either accept 1.36× on DRAM loads (DRAM is at 15%
       utilization) or forward part of stage 3's slice through a neighbor GPU over NVLink.
    4. **Read-wait estimate.** Unless (2) holds, it must use the heaviest rank's queue.
  - Measured on PP8 with SSD at 10 GB/s and unbounded reads: balanced loads raised the peak from 102.7 to 113.9 rounds/s
    (the study's tier-model v3 vs v4; logs not included).

### 9.2 Read path in the scheduler

| Mechanism | What the real system must do | Code |
|---|---|---|
| Tier read | Look up when admission reaches the request, not at arrival. Read only what HBM lacks, plus one state. Hold the footprint in HBM until admission. Landed reads go first. | `worker/admission/prefix_fetch.rs:381-498` |
| Read-wait bound | If the tier queue would start the read more than 1 s from now, skip the request: allocate nothing, keep its queue place, continue. | `prefix_fetch.rs:345-354, 444-449` |
| Blocked | If the read is within the bound but the request's footprint does not fit in HBM beside the running requests and the other reads' holds, stop admitting for this microbatch. | `prefix_fetch.rs:292-295, 450-451` |

### 9.3 Check

- On the 16-min benchmark (§2.1), rounds/s and TTFT at c2000–c3000 match its table.
- Tier shares of declared prefix tokens over the whole 16-min c2656 run: HBM 33.1%, DRAM 33.9%, SSD 33.1% (of which
  2.1% is the warm-start seed) (`logs/glm53_flash_pp5/runs_16min/c2656/prefix_tiers_w0.json`).
- SSD reads at c2656 average about 6.1 GB/s per GPU over the run (4.92B tokens × 1,199 B/token ÷ 960 s).
- No stage's tier queue is longer than the others' (balance).
- Past the knee, throughput stays flat (no collapse) thanks to the read-wait bound.

---

## 10. Optimality: how far the target is from the hardware

- **necessary_ratio 0.351** at PP5 c3000 (0.347 at c2656): the model's necessary work at B200 peak is 35% of the
  GPU time held (`$R/reports/optimality_report.json`, from `pp5_8h.yaml`).
- Most of the rest is the hardware gap, real kernels' grid peaks against the B200 peak (34.5%), and work the kernels
  execute above the necessary work (15.9%, mostly sparse MLA re-reading selected KV per query). Batching is 8.9%,
  idle 5.4%, imbalance and communication 0.
- So the target is not near the hardware limit; a real kernel that beats its DB row is a gain, not an error (§4.6).

## 11. Caveats and known gaps

- **The scheduler model is not validated against a real PP system.**
  - Per-stage time is the sum of profiled per-kernel rows; scheduling is the sim's own model.
  - The first real measurements of M3 (§6) are the alignment.
  - Use the operate-run-alignment skill to compare them stage by stage.
- **Decode is not on these GPUs** (`external_decode`). Its time is folded into the tool waits. A real deployment
  needs separate decode capacity, or the comparison breaks.
- **SRPT starves long prompts past the knee.**
  - The 60 s force-schedule bound releases them, so past 600 sessions/GPU, p99 reads about 62 s.
  - Below the knee, the largest prompts (742k–954k tokens) run 2–4× slower than they would alone (the study's scheduler trials).
- **The sim has no host/CPU overhead model.** Kernels sit back to back inside a stage. Gaps the real engine shows
  between kernels are real-engine limitations: fix them with CUDA graphs and async scheduling, never by slowing the
  sim.
- **Benchmark hygiene:**
  - Benchmark passes are CPU-contention sensitive. A contended run read −16% E2E, so run them alone and record
    `uptime`.
  - Compare iterations by cycle, not by elapsed time.

## 12. Provenance

- **Code:** ServingStudioSim branch `pp-sched-tiers` (draft PR #84), the commit that adds this directory. It is a
  study branch, not merged to main. The kernel path and the fixes it builds on are on main (PRs #82 and #85), and
  this branch adds the PP scheduler, the prefix tiers and decode-elsewhere sessions. The req-frontend submodule is
  `7f6aa9d` (uw-syfi/request-factory, branch `session-validate-single-dedup`; `.gitmodules` still names an older branch,
  the pinned commit is what counts).
- **Files here:**
  - `pp5_16min.yaml` and `pp5_16min_hbm.yaml` (§2.1);
  - `pp5_8h.yaml` (§3.1, `$R`);
  - `backends.yaml` (§4.1);
  - `predict_pp5.json` with `predict_cases.json` and `predict_case_labels.json` (§5.2);
  - `make_traces.sh`, `cut_trace.py` and `traces.sha256` (§2).
  - `window_metrics.py`: the window rule of §2 and §2.1 (rounds/s, TTFT percentiles, prefix hit).
- **Check:** at this commit the 16-minute presets and the timing-predict config reproduce the study's runs byte for
  byte: request_slo, cost logs, KV snapshots, prefix events, cost manifests and `iter_breakdown`.
- **Not included:** the study's other sweeps (PP8, PP4, DP8EP8, chunk sizes, scheduler trials, tier-model versions)
  and their raw logs. Their conclusions and numbers are in §3.2, §6.1, §7 and §9; rerun a copy of `pp5_8h.yaml` with
  the stated change to regenerate one.
