# GLM-5.3-Flash NVFP4 · PP4 on B200 · sim-to-real handoff v1

**Frozen design** (2026-10-13): ServingStudioSim branch `pp-main`, commit `7bb224a6`. The targets come from the
simulator at that commit; the real system chases them. A design change makes a new version (`_v2`); this one stays
as it is.

## Design in one sentence

One four-stage pipeline (4 × B200, layers split [12,11,11,11]) serves the prefill of coding-agent sessions on
GLM-5.3-Flash NVFP4. It keeps every session's context in HBM, host DRAM and local SSD, and forms microbatches
shortest-remaining-first under a budget that follows the prefill backlog. On a 20-minute benchmark of 2 new sessions/s
with at most 500 active, it computes 118.8k new prompt tokens/s (33.9 rounds/s) at TTFT p50 / p90 / p99 0.16 / 0.44 /
4.94 s.

## Design

**Model.**
- Checkpoint `nvidia/GLM-5.3-Flash-NVFP4` (snapshot `da920bb0`); MTP and speculative decoding off.
- 45 layers, hidden size 4096:
  - 11 DSA layers (sparse MLA + indexer) at 3, 7, …, 43;
  - 34 KDA layers (linear attention);
  - dense FFN on layers 0–2, MoE on 3–44 (288 routed experts, top-8, one shared expert).
- NVFP4 for the dense FFN and the routed experts. Attention, routers, the shared expert and `lm_head` stay BF16. The
  KV cache is FP8.

**Workload.**
- TraceLab coding-agent sessions, from a corpus of 4,281 sessions. New sessions arrive at 2 per second for 20 minutes
  (2,457 sessions), and at most 500 are active at once; a later one waits for a slot. Every run starts with empty
  caches.
- A round's prompt is the previous round's prompt, its output, then new tokens. The rounds served in the 20 minutes
  average 74k prompt tokens, 3.5k of them new: 95% of each prompt is the session's own prefix.
- Decode runs on other GPUs. These 4 GPUs compute each round's prefill and its first output token. The next round
  arrives after the decode time (80 tokens/s) plus the tool wait. The other output tokens come back as new tokens of
  the next round, so a standard OpenAI-style client can replay the workload.

**GPU partitioning.** PP4 on 4 B200 in one NVLink domain. Each stage holds whole layers, with no tensor or expert
parallelism inside it. The split is [12,11,11,11]:

| Stage | Layers | DSA layers | KV B/token | Notes |
|---|---|---|---|---|
| 0 | 0–11 | 3, 7, 11 | 1,635 | embedding, dense FFN on 0–2; the busiest stage (91.7% at 500 sessions) |
| 1 | 12–22 | 15, 19 | 1,090 | |
| 2 | 23–33 | 23, 27, 31 | 1,635 | |
| 3 | 34–44 | 35, 39, 43 | 1,635 | final mHC, norm, lm_head |

- **Why this split.** It balances the stages best: over the one-request prompts below (1k–128k), the mean stage
  compute is 95.4% of the slowest stage's. Against vLLM's default split [11,11,12,11], it also leaves more HBM for
  KV, because the largest stage's weights are 49.66 GB instead of 52.80 GB, and it serves 4–5% more tokens/s in the
  uniform-length tests below.
- **Configurable split.** The stage count and the layer split are configuration, so another split or PP size needs
  no code change.
- **KV pool.** KV per GPU is 100.18 GB. One block pool serves the pipeline, sized by the stages with 3 DSA layers:
  61.3M tokens.
- **Between stages.** Each stage sends the 4-wide residual stream to the next: 4 × 4096 × 2 B = 32 KiB per token. The
  send overlaps the next stage's compute of the previous microbatch.

**Core kernels.** Standalone library kernels; kernels from the vLLM tree can be vendored as they are.
- mHC (manifold-constrained hyper-connections, the 4-stream residual described under "Layer structure") boundary:
  DeepGEMM `mega_mhc` (fused post + pre RMSNorm); TileLang mHC for layer 0's first pre and the final post.
- DSA indexer: DeepGEMM `fp8_mqa_logits`, logits chunked to at most 512 MiB; top-k with the faster of DeepSelect
  `topk` and vLLM's `topKPerRowPrefill`.
- Sparse MLA: FlashInfer trtllm sparse MLA over the FP8 KV cache, written by `concat_and_cache_mla`. One launch
  takes at most 32,768 query rows on B200, so a longer microbatch runs as several launches in row order.
- KDA: FlashInfer `recurrent_kda` (cute-dsl persistent); short conv with Dao-AILab `causal-conv1d`, channel-last.
- Routed experts: FlashInfer `trtllm_fp4_block_scale_moe` (NVFP4). Dense FFN: FlashInfer `mm_fp4` (cute-dsl).
- Other projections, the shared expert and `lm_head`: BF16 cuBLAS GEMMs.
- Inside a stage, kernels run back to back with no host gaps (CUDA graphs, async scheduling).

**Layer structure: what is unusual.**
- **mHC residual stream.** The residual is 4 copies of the hidden state. Attention and FFN each open with an mHC
  boundary: the previous block's mHC post fused with the next block's pre-RMSNorm. Stage 0 expands the embedding
  into the 4 streams; the last stage runs the final mHC post and averages the streams before the final norm.
- **DSA layers.** The indexer scores the context in pools of 4 tokens, keeps the top 512 pools (2,048 tokens) and expands them to
  tokens; sparse MLA attends only to those. These are the only layers with per-token KV: 545 B/token each.
- **KDA layers.** Linear attention with a fixed-size recurrent state per request and no per-token KV. Each request
  reserves its state once in the KV pool: 6 blocks of 8,576 tokens, 51,456 tokens' worth (`1_contract.md` §1).
- **MoE.** At these batch sizes the shared expert runs after the routed experts, not overlapped with them.

**Scheduler.** It runs on stage 0:
- **In flight.** At most 4 microbatches. The next one forms only when stage 0 is free.
- **Progress commit.** A prompt's progress is committed when its chunk is scheduled, so its next chunk can ride the
  next microbatch.
- **Order.** Started prompts go shortest-remaining-first. Queued prompts go shortest-prefill-first, and a queued
  prompt may enter ahead of a longer started one.
- **Budget.** The smaller of a load budget and an even split. The load budget follows the backlog: 4,096 tokens up
  to a backlog of 1,000,000 tokens, rising to 32,768 at 3,000,000. The even split is the backlog ÷ 4, kept between
  512 and 32,768, so the 4 microbatches in flight share the backlog (`6_scheduler.md` §2).
- **Admission.** Each request reserves its whole footprint once. There is no preemption.
- **Force-schedule.** A request waiting 60 s is served before everything else.

**Prefix cache.**
- **Tiers.** Each GPU has HBM, plus host DRAM (150 GB, 50 GB/s) and local SSD (256 GB, 10 GB/s), all write-through
  LRU. In the 20-minute benchmark the active contexts still fit in HBM (97% of prefix reads at 500 sessions); the
  cache-tier benchmark limits HBM and DRAM so that DRAM and SSD carry their share within minutes.
- **Reads.** A tier read is asynchronous and holds its HBM blocks. A request whose read would wait more than 1 s for
  its tier is passed over: it starts no read and stays queued.
- **Writes.** Tier writes run in the background and never stall compute.
- **Balanced storage.** Each stage holds different layers, so its tier traffic differs. Spread it evenly across the
  stages: shared capacity, SSD striping, bandwidth allocated by bytes. vLLM does not do this today.

## Target numbers

The first two tables bring up the pipeline before any scheduling or cache work: requests of one prompt length (1k to
128k tokens), one output token, no prefix reuse.

**Intermediate: one request, the whole prompt in one microbatch.** Compute time per stage and TTFT, in ms. TTFT adds the three
stage-to-stage sends: 1.4 ms at 8k and 22.6 ms at 128k.

| Prompt | Stage 0 | Stage 1 | Stage 2 | Stage 3 | TTFT |
|---|---|---|---|---|---|
| 1k | 12.3 | 13.0 | 13.1 | 13.3 | 51.9 |
| 2k | 18.0 | 18.3 | 18.4 | 18.6 | 73.7 |
| 4k | 30.9 | 30.1 | 30.6 | 30.9 | 123.3 |
| 8k | 57.1 | 54.1 | 55.3 | 55.7 | 223.6 |
| 16k | 111.7 | 103.9 | 106.8 | 107.4 | 432.7 |
| 32k | 220.4 | 203.1 | 209.0 | 209.9 | 848.1 |
| 64k | 431.2 | 395.5 | 406.9 | 408.4 | 1,653.3 |
| 128k | 871.5 | 792.6 | 821.2 | 823.9 | 3,331.8 |

**Intermediate: saturated, uniform length, 8k chunks.** All requests arrive at once. A plain scheduler fills each microbatch to
8,192 tokens in arrival order and cuts longer prompts into 8k chunks. Tokens/s over the middle 80% of the run.

| Prompt | 1k | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|---|
| Tokens/s (pipeline) | 149.3k | 149.9k | 147.5k | 143.4k | 143.4k | 143.0k | 141.8k | 138.4k |
| Requests/s | 145.8 | 73.2 | 36.0 | 17.5 | 8.75 | 4.36 | 2.16 | 1.06 |

Stage 0 is busy 99–100% of the time and the others 93–98%.

**Final: sessions, 20 minutes** (`1_contract.md` §2.2). Throughput is per pipeline: rounds/s and new prompt tokens/s
(the sum of the rounds' new tokens). TTFT is in seconds. Rounds count when their first token comes in [0, 20 min).
The SLO is p99 ≤ 6 s.

| Point | Max sessions | Rounds/s | New tokens/s | TTFT p50 / p90 / p99 | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|
| Low latency | 400 | 27.8 | 96.1k | 0.14 / 0.31 / 1.64 | 80% / 78% / 80% / 81% |
| **SLO** | **500** | **33.9** | **118.8k** | **0.16 / 0.44 / 4.94** | 92% / 88% / 91% / 91% |

**Final: cache tiers, 20 minutes** (`1_contract.md` §2.3). The same workload with the KV pool and DRAM limited.
Throughput stays within 0.4% of the full-memory runs. Prefix read share over the rounds that arrive in [10, 20 min).

| Memory (KV pool / DRAM per GPU) | Max sessions | New tokens/s | TTFT p50 / p90 / p99 | Prefix reads HBM / DRAM / SSD |
|---|---|---|---|---|
| 15.3M tokens / 37.5 GB | 400 | 95.9k | 0.15 / 0.32 / 1.63 | 49% / 35% / 16% |
| 15.3M tokens / 37.5 GB | **500** | **118.5k** | **0.19 / 0.50 / 5.28** | 37% / 36% / 27% |
| 7.6M tokens / 18.75 GB | 400 | 95.8k | 0.17 / 0.34 / 1.68 | 22% / 27% / 51% |
| 7.6M tokens / 18.75 GB | 500 | 118.4k | 0.21 / 0.57 / 5.58 | 10% / 29% / 61% |

The targets are not a hardware ceiling: a faster kernel or schedule is welcome.

## Files

| File | Content |
|---|---|
| `1_contract.md` | What the real setup must match (model, hardware, workload, metrics), the correctness check, and how to replay the benchmark |
| `data/` | The 20-minute session trace and its manifest, `make_trace.py` that builds it, `metrics.py` that scores a run; the kernel rows `kernel_specs.jsonl`, the per-layer targets `layer_targets.jsonl`, each layer type's cost tree in `layers/`, the single-iteration timeline `iteration_targets.json` and saturated goldens `saturated_targets.json`, the prefix and tier bring-ups `prefix_targets.json` and `tier_targets.json`, the scheduling and force checks `sched_check_targets.json` and `force_check_targets.json`, the session targets `session_targets.json` and the microbatch fit `microbatch_fit.json`, with the scripts that export them; the bring-up traces in `bringup/` |
| `2_kernels.md` | The kernel at each position, its profile rows (`data/kernel_specs.jsonl`), how to micro-benchmark it against them, and the correctness check |
| `3_layers.md` | Per-layer alignment: each layer type's time against its target, at each prompt length |
| `4_iteration.md` | The whole model on 4 GPUs: one request (per-stage, per-send and TTFT targets) and saturated same-length requests (throughput and TTFT goldens) |
| `5_prefix_cache.md` | Sessions with the prefix cache in HBM and the plain chunked-prefill scheduler: a prefix bring-up and the 20-minute session targets |
| `6_scheduler.md` | The design's scheduler, built in four steps (shortest first, load budget, even split, force-schedule): a scheduling check, a force check and the session targets of each step |
| `7_offloading.md` | Prefix KV in DRAM and SSD, built in two steps (DRAM with balanced storage, then SSD): a tier bring-up and the session targets of each step at full memory and with the memory limits; the final targets |
| `measured/` | The real system's measurements, as each file's last section lists them; written by the receiver |
| `issues.md` | The real system's gap log; written by the receiver |

Each file ends with what to record. From `2_kernels.md` on, the section before it says where the targets may be off.
