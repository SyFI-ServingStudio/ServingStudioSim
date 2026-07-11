# Alignment findings & journal — Llama3-8B dense, 1×H200

What the cost-model validation loop found on its first milestone, **and the dead
ends we hit getting there**. `README.md` explains *how the loop works*; this file
records *what it measured* and *which approaches failed and why* — so the next
person does not re-walk the same walls.

Status note (2026-07-10): the transitional `alignment/compare.py` and
`alignment/sim_costlog.py` have been removed. Current comparison, cost-tree
folding, statistics, and plotting live in the top-level analyzer's
`alignment-iteration` and `alignment-e2e` subjects. Historical filenames and
numbers below are retained only as an investigation record.

Workload/hardware for every number below: Llama3-8B dense, 1× H200, TP=1, CUDA
graphs **ON**, 64 concurrent requests, pure batch-64 decode, context ~600.

---

## 2026-07-10 — current pipeline capture/parse smoke passes

After moving the measured-run controller to `alignment/runner.py`, typing the
TraceLab workload frontend, and making indexed iteration markers canonical, a
fresh end-to-end profiling smoke succeeded on H200 GPU 1. The preserved setup
and artifacts are in `logs/20260710_01_alignment_profile_smoke`.

The run used 64 distinct synthetic-token prompts (`input_len=512`,
`output_len=96`), CUDA graphs through size 2048, and the established targeted
capture trigger `vllm_iteration(30): forward@*`. It produced 89 indexed forward
ranges and 32,547 CUDA kernel rows; 31,801 kernel rows had a non-null
`graphNodeId`. Every target decode iteration 34–58 owned 353 kernels through
runtime correlation IDs, and `python -m alignment parse` scanned 8,825 owned
rows with a 9.604 ms mean interval-union busy time.

The six paired, unindexed `gpu_model_runner: <phase>` aliases were removed from
the fork; none appear in the fresh trace. Unpaired output/bookkeeping markers
remain because they carry separate detail. The current parser deliberately
accepts only inline indexed iteration markers; historical and stock traces need
their original parser version.

This pass selected the FLASHINFER attention backend (5.272 ms/iteration in the
parser's attention bucket), whereas the valid numerical comparison below used
FlashAttention. Therefore this result proves the current capture/export/parse
mechanics, not numerical continuity with the older gap table.

---

## TL;DR — the valid result

Once the workload and the accounting were both fixed, VibeSim predicts a real
production decode step to within a few percent:

| comparison | sim | nsys (real) | ratio |
|---|---|---|---|
| full decode step **incl. logits** | 6.71 ms | 6.47 ms | **1.04** (sim over +4%) |
| forward window only (no logits) | 6.45 ms¹ | 6.20 ms | 1.04 |
| real decode **wall** / step (fork `gpu_time`) | — | 6.36 ms | — |

¹ sim forward-only = total − lm_head = 6.713 − 0.259.

The remaining gap is **not** a throughput blind spot — real decode is GPU-bound
(6.36 ms wall vs 6.20 ms kernel-busy ⇒ ~2.5% idle). The residual is a per-op
**GEMM over-prediction (+13% on the four forward GEMMs)**, discussed below.

---

## The method that works (recipe)

Capturing real **CUDA-graph replay** per-op time needs the ref skill
`ref/.codex/skills/profile-vllm-nsys` recipe, not an external nsys wrapper:

1. **FORK vLLM** (`alignment/profiler/vllm/.venv`) — it emits *inline* NVTX
   `vllm_iteration(N): forward` scopes. Stock vLLM only emits the *registered
   string* `gpu_model_runner: forward`, which the nsys trigger cannot match.
2. **NVTX-targeted capture** — `--capture-range=nvtx --capture-range-end=none
   --nvtx-capture='vllm_iteration(N): forward@*'` + `--cuda-graph-trace=node`.
   Do **not** full-capture (see failure #2/#4).
3. **CUDA graphs stay ON** (no `--enforce-eager`). Per-op busy time is
   graph-invariant, and eager changes the *kernel set* (see failure #1).
4. **Dedup dual NVTX scopes** in `nsys_parse.load_ranges` (see failure #7).
5. **Distinct prompt per request** in the since-retired synthetic driver (see
   failure #9).

That historical command has been retired; its artifacts remain under
`logs/align_llama3_8b_1gpu`. Current runs use the explicit `python -m launcher
alignment sim`, `profile`, `timing-predict`, and `analyze` stages. Both inputs
may be YAML or JSON. In the recorded run, the trigger fired at iter 30 and the
analyze window was iters 34–58 (pure decode, verified via `metrics.jsonl`:
`prefill_tokens==0`, `num_generation_requests==64`).

---

## Failed trials & dead ends

Each of these produced a *plausible-looking but wrong* number, or no number at
all. They are the reason the recipe above is shaped the way it is.

### 1. `--enforce-eager` — cannot match production timing
Forcing eager was an unsanctioned methodology change (reverted after pushback).
Eager and CUDA-graph run **different kernels**: eager runs `vllm::rms_norm` /
`act_and_mul`; the graph path runs fused `triton_red_fused_*` / `triton_poi_*`.
You cannot align a cost model to a kernel set the production server never runs.
**Decision: keep CUDA graphs on.**

### 2. External full-capture nsys + cudagraph → all-zeros gap table
`nsys profile --cuda-graph-trace=node` records per-node kernels only at graph
**capture** (startup), not during **replay** (serving). A full capture over the
serving window therefore saw **zero CUDA activity** in the decode iterations →
every bucket came back 0. This is *the* reason external full-capture fails with
graphs on; it is fixed only by the NVTX-targeted capture in the recipe.

### 3. Stock vLLM + nvtx trigger → trigger never fires, no kernel table
Stock vLLM registers its NVTX label as a *string id* (`textId → StringIds`),
emitting `gpu_model_runner: forward`. The nsys `--nvtx-capture` trigger matches
**inline** range text only, never registered strings, so capture never armed →
the export had **no `CUPTI_ACTIVITY_KIND_KERNEL` table** (raw `sqlite3` error).
This is why the FORK (inline `vllm_iteration(N)` text) is mandatory.

### 4. Full-run capture truncates late CUDA rows ("activity stops at ~65 s")
Even where full capture produced kernels, the CUDA kernel rows were dropped
partway through a long run — the skill-documented "CUDA activity stops at ~65s"
truncation. NVTX-targeted capture (a short armed window) sidesteps it.

### 5. Reverse-engineering nsys flush flags instead of running the reference
Time was lost hand-tuning nsys export/flush flags to force replay kernels out.
The reference harness (`profile-vllm-nsys`) already encodes the working recipe.
**Lesson: run the reference, do not recreate it.**

### 6. ray-nsight — captures replay nodes but the config omits NVTX
`--ray-workers-use-nsight` *does* capture graph-replay nodes, and we wired
`nsys_capture.find_ray_nsight_trace`. But vLLM's ray nsight `runtime_env` is
`t: cuda,cudnn,cublas` — **no nvtx** — so there is no forward range to attribute
kernels to, and the trace lands at `$TMPDIR/ray/session_*/logs/nsight/`
(`TMPDIR=/m-coriander/coriander/tmp` here, **not** `/tmp/ray`). Investigated,
kept as a code path, **not used**.

### 7. Dual-scope double-counting → nsys inflated ~1.48×
The historical fork emitted **both** `vllm_iteration(N): forward` (indexed) **and**
`gpu_model_runner: forward` (index-less) around the same forward. Counting both
gave ~220 ranges for 110 iterations → every kernel counted ~1.48× → nsys total
**8.99 ms**, `calls/it` 47 (should be 32). The previous parser fixed those
artifacts by dropping index-less ranges when indexed ones existed, producing
`calls/it` 32 and nsys 6.07. The current fork emits only the indexed phase
family, and the current parser no longer carries the historical dedup path.

### 8. lm_head accounting asymmetry + a wrong "L2-residency" story
The first "dense_gemm +16–19%" over-prediction was partly an **apples-to-oranges
bucket**: sim's `dense_gemm` includes `lm_head`, but the nsys **forward** window
excludes it (lm_head runs in the *sample* phase, after the forward NVTX range).
Separately, an early explanation that a "single weight stays L2-resident" made
the sim's cold GEMM look slow was **wrong** — real decode reads 32 distinct
weights cold every step. Both are corrected below.

### 9. Identical-prompt workload → cascade attention artifact
The biggest trap. 64 **identical** prompts → 98.3% prefix-cache hit → vLLM
switches to **cascade / shared-prefix attention** (the shared prefix KV is
attended **once**, not per sequence). Consequences, all wrong:
- Real attention looked artificially cheap (**1.515 ms**), so the sim appeared to
  *over*-predict attention by +12%.
- Prefill collapsed to ~2 steps and the schedule stalled, faking **17 ms wall /
  11 ms GPU idle / "2.7× throughput over-optimism"**.

Fix in that historical driver: `_synthetic_prompt(input_len, seed)` built a
random `w<0..999>`
word stream, unique per request (token *values* don't affect kernel timing).
With distinct prompts: real attention **1.856 ms** (sim now *under* −7%), and
decode is GPU-bound (6.36 wall vs 6.20 busy). See memory
`alignment-distinct-prompts`.

---

## Valid gap table (distinct prompts, forward window, iters 34–58)

| bucket | sim ms | nsys ms | ratio | read |
|---|---|---|---|---|
| dense_gemm (excl. logits) | 4.436² | 3.936 | 1.13 | sim over +13% |
| attention | 1.731 | 1.856 | 1.07 | sim under −7% |
| norm | 0.184 | 0.267 | 1.45 | sim under −31% |
| elementwise_other | 0.103 | 0.140 | 1.36 | sim under −26% |
| **forward total** | **6.45** | **6.20** | **1.04** | sim over +4% |

² sim forward GEMMs only (qkv+o+up_gate+down); the raw `gap_table.json` shows
`dense_gemm` 4.694 because it folds in lm_head — corrected in the next table.

Real attention kernels are `FlashAttnFwdSm90 (int)2` (1.68) + `FlashAttnFwdCombine`
(0.09, legitimate **split-KV**, not cascade) + `merge_attn_states`.

---

## Per-GEMM breakdown — "consider the logits"

Matching lm_head on both sides (nsys lm_head = `nvjet_...512x64...coopB_TNN`,
24 calls ≈ 1/iter, sample phase):

| GEMM | shape (M,K,N) | sim ms | nsys ms | ratio | read |
|---|---|---|---|---|---|
| qkv_proj | 64, 4096, 6144 | 0.613 | 0.545 | 1.13 | sim over +13% |
| o_proj | 64, 4096, 4096 | 0.436 | 0.370 | 1.18 | sim over +18% |
| up_gate_proj | 64, 4096, 28672 | 2.092 | 1.944 | 1.08 | sim over +8% |
| down_proj | 64, 14336, 4096 | 1.295 | 1.077 | 1.20 | sim over +20% |
| **forward GEMM subtotal** | | **4.436** | **3.936** | **1.13** | sim over +13% |
| lm_head (logits) | 64, 4096, 128256 | 0.259 | 0.272 | **0.95** | sim **under −5%** |
| **all GEMM (fwd + logits)** | | **4.694** | **4.208** | **1.12** | sim over +12% |

**Two conclusions the logits force:**
- The reported `dense_gemm` "4.694 vs 3.937 (+19%)" is an accounting artifact of
  comparing sim-with-logits against nsys-without-logits. Apples-to-apples, the
  four forward GEMMs are **+13%**, and **lm_head is essentially spot-on (−5%)** —
  the single biggest GEMM is the *least* mispredicted, so the over-prediction is
  a small/skinny-GEMM effect, not a size-scaling error.
- Folding logits back into the total, sim 6.71 vs nsys **6.47** = **+4%**
  end-to-end (the earlier "+8%" was itself inflated by the missing logits).

**Root-cause direction (open):** sim `single_gemm` = `torch.mm` (backends=torch,
layout NN vs vLLM's cuBLAS `nvjet_*_TN*` verified ≈equal) profiled with
`Timer.cupti` = **cold L2, no warmup**. Real back-to-back serving GEMMs run ~13%
faster than that isolated cold measurement, worst on the skinny qkv/o/down.

---

## Open items

- **GEMM cold-vs-warm (+13%)** — the one real per-op gap. Quantify the cold-L2
  penalty of the isolated `single_gemm` profile vs back-to-back serving and add a
  warm correction. (`todo_low_batch_grouped_gemm` is the MoE analogue.)
- **norm / elementwise (−30%)** — sim under-predicts fused `triton_red_fused_*`
  rms_norm and the silu/mul activation; the sim's `rms_norm`/`elementwise` costs
  are byte-placeholder-ish and miss real fused-kernel launch cost.
- **Comparison 口径（已迁移）** — logits/sample-phase must be represented through
  the current user-owned kernel mapping and top-level analyzer, not a local
  hard-coded `dense_gemm` bucket.
