# Handoff: the simulated PP5 prefill scheduler, mechanism by mechanism

Source of truth: ServingStudioSim branch `pp-sched-tiers` (draft PR #84), the commit that carries this directory.
Every `file:line` below is relative to `simulator/src/` unless it says otherwise. Claims were
checked against the code. **[inference]** marks reasoning not verified in code.

## 0. The design being matched

Preset: `pp5_8h.yaml` in this directory, member `load=c3000` (8 h). `pp5_16min.yaml` is the same design over
16 minutes.

- `deployment: pp`, one replica, 5 B200 GPUs, one GPU per stage.
- arch `glm53_flash_vllm_nvfp4_pp_kda_dsa_moe`, `model/config/glm53_flash_nvfp4.json`, `pp_size: 5`.
  - `layer_partition` is not set, so the split is [9,9,9,9,9], 9 consecutive layers per stage.
  - `max_model_len` 1048576; `routing: corpus` (measured per-token routes).
  - cudagraph capture sizes go up to 2048; larger batches run eager.
  - Kernel path `fastk` (`backends.yaml`): KDA `flashinfer_cute_persistent`, mHC `deepgemm_mega_nonshifted`,
    conv `dao_channellast`, top-k the faster of `deep_select` / `vllm_cuda`. Arch knobs `mla_layout_copies: false`,
    indexer logits chunk 512 MiB.
- worker `pipeline_chunked_prefill`:
  - Batching: `max_batch_tokens 32768`, `prefill_chunk_alignment plain`, `microbatch_split even`,
    `min_microbatch_tokens 512`, `long_prefill_token_threshold 0`.
  - Ordering: `pending_order shortest-prefill-first`, `srpt true`, `force_schedule_after_ms 60000`.
  - Load budget: `load_budget_low_tokens 4096`, `load_budget_backlog_lo_tokens 1_000_000`,
    `load_budget_backlog_hi_tokens 3_000_000`.
  - Memory and decode: `attn_gpu_memory_gb 108.96`, `external_decode true`.
  - Prefix tiers: `dram_tier_gb 150` at 50 GB/s, `ssd_tier_gb 8000` at 10 GB/s, `prefix_tier_warm_start true`,
    `prefix_tier_max_read_wait_ms 1000`, `prefix_tier_balanced_load true`.
- Workload: closed loop of 3000 concurrent sessions (`arrival_mode: saturated`, `max_concurrency 3000`).
  - Trace: `logs/glm53_flash_pp5/traces/closed_c3000.csv`, built from `mono_sessions_d80.csv` by `make_traces.sh`.
  - Decode time is folded into the tool waits at 80 tok/s (`trace/session_decode_wait.py`).
  - Sessions start in renewal equilibrium (`trace/session_closed_loop.py`).
- HBM 108.96 GB = the device budget 149.84 GB (85% of device memory) minus the largest PP5 stage's NVFP4 weights,
  40.88 GB (the study's per-stage weight footprint count).
- Layer balance is 97.1% on PP5 vs 91.0% on PP8 (see section 7).
- Result (8 h, c3000, arrivals over [1 h, 8 h)):
  - 105.8 rounds/s, which is 21.2 per GPU.
  - TTFT p50 1.23 s, p90 2.74 s, p99 61.9 s. The p99 is the 60 s force-schedule bound being hit.
  - Mean microbatch 28.2k tokens; stage busy 94.6%.
  - Tier log (`$R/prefix_tiers_w0.json`, `$R` = `logs/glm53_flash_pp5/runs_8h/c3000`): declared prefix tokens were found 23.1% in HBM,
    37.3% in DRAM and 39.6% on SSD.

**Stage boundary.** Each stage finishes its last layer's mHC post, then sends the full 4-wide residual stream
(32,768 B/token); nothing else is pending across a stage boundary (`arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:24-32`).

---

## 1. Microbatch formation (stage 0)

### 1a. Cadence and the in-flight cap

**What it does.** Stage 0, the head, forms a new microbatch only when both hold:
- stage 0 is idle (`computing.is_none()`);
- `in_flight.len() < depth`, with depth = `pp_size` = 5. `in_flight` counts every formed microbatch that has not
  left the last stage, including the one stage 0 is computing.

The microbatch starts at `state_changed_at`, the exact time of the latest event the formation depends on: stage 0
finishing, an exit from the last stage, a request arrival, or a tier read landing. It does not wait for the simulator
tick.

Batch k+1 forms only after stage 0 finishes batch k; nothing is formed ahead into a queue. So batch k+1 sees every
arrival, tier landing and exit up to that moment, and SRPT and the load budget react to that late state.

Exits arrive in formation order, because every stage is FIFO. In each tick the head processes exits first, which
releases KV, then lands finished tier reads, then completes stage 0's compute, then forms the next microbatch.

**Code.**
- `worker/workers/pipeline/pipeline_head_worker.rs:264-312` (`tick_inner`; the cap check is at 295-303).
- Exits: `pipeline_head_worker.rs:314-340`.
- Launch: `pipeline_head_worker.rs:344-372`.

### 1b. Progress commit and completion

**What it does.**
- At formation, the chunk's prefill progress is committed: `complete_prefill_chunk` runs inside
  `commit_microbatch`. A prompt's next chunk can therefore ride the next microbatch while its previous chunk is still
  on a later stage.
- Tokens are emitted at the last-stage exit.
- `in_flight_prefill_tokens` is incremented at commit and decremented at exit.

**Code.** `worker/admission/pipelined_chunked_prefill_admission.rs:676-705` (commit) and `:707-778` (exit).

### 1c. Order of work inside one microbatch

`form_microbatch` (`pipelined_chunked_prefill_admission.rs:611-674`) runs these steps:

1. Land finished tier reads. Promote overdue requests (section 3).
2. Schedule decodes: one token each, for requests whose previous step has exited. With `external_decode` there are
   none, because every request finishes at its first token.
3. Compute the prefill budget `B` (section 2).
4. With `srpt: true`, run `form_prefill_srpt` (`:496-564`):
   1. **Overdue fresh prompts first.** Call `admit_fresh_prompts(..., overdue_only=true)` (`:526-528`).
   2. **Sort started prompts** (those with unscheduled prefill left) by `(not_overdue, key)`:
      - an overdue started prompt has key = arrival time, so overdue ones come first, oldest first;
      - every other started prompt has key = remaining prefill tokens (`:508-525`).
   3. **For each started prompt S in that order:**
      - If S is not overdue, first admit fresh prompts, with the bound "remaining prefill ≤ S's remaining". The bound
        applies only to the pending-order source. Landed reads and overdue requests are admitted unbounded, and the
        bounded loop stops at the first pending candidate that is longer than S.
      - Then give S its chunk (`:529-557`).
   4. Admit fresh prompts with no bound until the budget is spent (`:558-560`).
   5. Prompts admitted in this formation and not finished join `started_prefills` after the older ones.
5. Without srpt, started prompts go first in start order (`:653-671`), then fresh prompts.

**Code.** SRPT is switched on by `with_srpt` (`:271-277`).

**Effect.** The study's scheduler ablation (section 8) shows SRPT is the head-of-line fix: the started-first order parks short
prompts behind a long prompt's chunks.

### 1d. Fresh-prompt admission, per candidate

`admit_fresh_prompts`, `pipelined_chunked_prefill_admission.rs:339-467`.

**Candidate source, in priority order:**
1. landed tier reads (FIFO by landing);
2. the overdue queue;
3. the pending-order head. `refresh_head` first re-asks for the head's resident prefix (`:351-380`). For SPF this is
   a no-op: `policy/shortest_job_first.rs` does not override `refresh_head`, so the key stays frozen at enqueue.

**Tier gate** (`prefix_fetch.rs::at_head`, section 5), one of four outcomes:
- `Admit`: continue below.
- `Read`: the read started, and the request leaves the queue.
- `PassedOver`: take the request out for now and put it back after the loop.
- `Blocked`: stop admitting.

**Admission steps, for a candidate that reaches `Admit`:**
1. Resolve the prefill against HBM: `remaining = declared_prefix − resident_prefix + fresh_prompt`
   (`worker/kv/mod.rs:37-111`). A fully cached prompt still computes one token.
2. Compute `chunk = next_chunk_tokens(...)` (`worker/admission/chunked_prefill_admission.rs:1014-1040`).
   - With `plain` alignment and no threshold, `chunk = min(remaining, budget_left)`.
   - With `long_prefill_token_threshold t > 0`, `chunk = min(remaining, t, budget_left)`.
   - With `checkpoint` alignment, non-final chunks end on 8576-token block boundaries. The preset uses `plain`.
3. KV gate: `footprint = post_prefill_context + remaining_output + state_tokens`. If
   `!fits(footprint)`, **break**, with no skip-ahead (`:417-424`).
4. Reserve the whole footprint once, mark the request admitted, stamp it `Prefill`, and schedule the chunk.
5. If the chunk did not finish the prompt, push the request to `started_prefills`.

Section 4 covers the HBM accounting and the alignment.

### 1e. `pending_order: shortest-prefill-first`

**What it does.** The queue is a heap keyed on
`work = fresh_prompt_tokens + (declared_prefix_tokens − rank_tokens)`, with ties broken FIFO.
- `rank_tokens = max(HBM-resident prefix, tier hit tokens)`, frozen at enqueue
  (`pipelined_chunked_prefill_admission.rs:592-602`, `prefix_fetch.rs:358-375`).
- A context sitting in DRAM or on SSD therefore ranks as already cached.

**Code.** `worker/admission/policy/shortest_job_first.rs:13-33` (`JobSize::PrefillTokens`) and `policy/mod.rs:73-75`.

**Config.** Default `fifo` (`worker/config.rs:627`, `default_pipeline_pending_order` at `:293`).

### 1f. `microbatch_split: even` and `min_microbatch_tokens`

**What it does.** The prefill budget is capped at
`clamp(ceil(R / depth), min_microbatch_tokens, max_batch_tokens)`, where R is the round backlog defined in section 2.
- R counts in-flight prefill tokens, so the slice size stays steady through a round. One P-token prompt into an empty
  pipeline runs as `depth` slices of P/depth, not as a shrinking series.
- The split never waits: a microbatch takes `min(pending, target)` and starts.
- The floor 512 only sizes the microbatch and never delays it.

**Code.**
- `pipelined_chunked_prefill_admission.rs:286-298` and `:325-331`.
- Config: `worker/config.rs:611` (default `greedy`) and `:616` (default 0); validation at `config.rs:89-114`.

**Test.** `pipeline_head_worker.rs:991-1004`: a 3000-token prompt at depth 4 runs as [750, 750, 750, 750].

### 1g. `long_prefill_token_threshold`

**What it does.** Caps one request's chunk per microbatch, on top of the budget.

**Code.** `chunked_prefill_admission.rs:1022-1025`.

**Config.** Default 0 = off (`config.rs:622`). The preset uses 0.

---

## 2. Load-following budget

**What it does.** Let the backlog be

```
R = Σ_started remaining_prefill_tokens (HBM-resolved; includes recomputed prefix misses)
  + Σ_queued  fresh_prompt_tokens       (pending-order policy; the trace's new tokens only)
  + Σ_overdue fresh_prompt_tokens
  + in_flight_prefill_tokens            (committed microbatches not yet exited)
```

Requests that are out of the queue for a tier read, reading or landed, are **not** counted. A queued request's missing
declared prefix is not counted either.

The budget is

```
rise = clamp((R − lo) / (hi − lo), 0, 1)
LB(R) = low + round((max_batch_tokens − low) · rise)
B = min(max_batch_tokens − decodes, LB(R), clamp(ceil(R/5), 512, 32768))
```

with `low` 4096, `lo` 1M, `hi` 3M and `max_batch_tokens` 32768.

B only lowers the per-microbatch budget. `max_batch_tokens` 32768 stays the hard cap, so activation and workspace
sizing must assume a 32,768-token microbatch.

Worked values of B:

| R (tokens) | B (tokens) |
|---|---|
| 1,000 | 512 |
| 5,000 | 1,000 |
| 20,480 to 1M | 4,096 |
| 1.5M | 11,264 |
| 2M | 18,432 |
| ≥ 3M | 32,768 |

**Code.**
- `LoadBudget::target` at `pipelined_chunked_prefill_admission.rs:130-143`.
- `prefill_budget` at `:302-332`.
- `with_load_budget` at `:248-269`.
- Config: `worker/config.rs:636-642`. All three default to 0, which means off. Validation is at
  `deployment/pp.rs:296-313`.

**Test.** `pipeline_head_worker.rs:1110-1145`.

**History.** The study replaced a stage-0 busy-fraction signal with this backlog signal.

---

## 3. `force_schedule_after_ms`

**What it does.** An `OverdueQueue` records every accepted request's arrival time, in arrival order.
- **Promotion.** At each formation, `promote` moves every queued request with `now − arrival ≥ 60 s` out of the
  pending order into a FIFO overdue queue, oldest arrival first.
  - The arrival is the round's own arrival time, not the session start.
  - A request that already left the pending order is not promoted. This covers one that was admitted and one that is
    out for a tier read.
- **Service.** Overdue fresh prompts are served before landed reads (the `overdue_only` pass), before started prompts
  and before the pending order, with no shortest-first bound.
- **Started prompts.** A started prompt that is overdue sorts ahead of every other started prompt, oldest arrival
  first (section 1c).
- Overdue tokens count in the backlog R.
- An overdue request still goes through the tier gate: it can start a read or be passed over.

**Code.**
- `worker/admission/overdue.rs`: whole file; `promote` at `:65-83`, `is_overdue` at `:59-61`.
- Wiring: `pipelined_chunked_prefill_admission.rs:205-220` and `:619-621`.
- Recipe: `build_pipeline_head_worker.rs:290-292`.

**Config.** `config.rs:701`, default 0 = off.

**Test.** `pipeline_head_worker.rs:1187-1267`.

**Effect.** It bounds the SRPT starvation tail. The p99 of 61.9 s at c3000 is the share of requests that aged to the
bound.

---

## 4. KV and HBM accounting for the pipeline

### 4a. One block pool sized by the most constrained stage

**What it does.** Every stage holds the same requests' tokens, so there is one shared pool, sized by the stage with
the fewest blocks: the stage with the most KV bytes per token. The capacity in tokens is

```
capacity = (attn_kv_bytes / (kv_bytes_per_token_max · block_tokens) − 1) · block_tokens
```

The −1 is a reserved null block.

For this preset:
- `kv_bytes_per_token_max` = 1635 B (stage 3, which has 3 DSA layers).
- `block_tokens` = 8576.
- That gives 7770 − 1 = 7769 blocks, which is **66,626,944 tokens**. The run's stdout log confirms it:
  `kv_capacity_tokens=66626944 block_tokens=8576 state_tokens_per_request=42880`.

**Code.**
- `PipelineHybridState::capacity_tokens` at `worker/workers/pipeline/build_pipeline_head_worker.rs:138-162`.
- Recipe at `:173-313`.
- `pipeline_kv_bytes_per_token` = max over stages, at `arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:692-699`.
- Wiring at `deployment/pp.rs:174-195`.

**Caveat [inference].** `attn_gpu_memory_gb` is device budget minus weights only. Activation and workspace peaks, for
example at 32k-token prefill microbatches and the 512 MiB indexer logits chunk, are not subtracted. A real memory
profile will leave fewer blocks.

### 4b. Per-request fixed state: `state_tokens_per_request`

**What it does.**
- A request holds `G + 1` fixed blocks: one per KDA cache group (its live recurrent state) plus one kpool-tail scratch
  block.
- G = `max(ceil(kda/dsa))` over the whole model and every stage. For PP5, stages [0..4] have KDA = [7, 7, 7, 6, 7]
  and DSA = [2, 2, 2, 3, 2], so G = 4.
- Fixed state = 5 × 8576 = **42,880 tokens** per request.
- The admission footprint is `post_prefill_context + remaining_output + 42880`, reserved once at admission and
  released at completion. There is no incremental allocation and no preemption. This reserve-once rule is the
  simulator's choice.
- Block grouping: one block pool, one attention group for the 11 DSA layers, one kpool-tail group, and G KDA groups
  (arch module doc `arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:34-40`).

**Code.**
- `state_blocks_per_request` at `arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:710-714`.
- `kda_group_count` at `:186-208`.
- Footprint and fits: `worker/kv/hybrid_gdn.rs:108-132`.

### 4c. Block size 8576

**What it does.**
- The block size is the smallest multiple of 128 tokens whose MLA page (512 B/token × tokens) holds one KDA layer's
  state.
- KDA state at TP1:
  - SSM: 64 heads × 128 × 128 × fp32 = 4,194,304 B;
  - conv: (kernel 4 − 1) × channels × bf16;
  - a total of about 4.34 MB.
- The block works out to 67 × 128 = 8576. The KDA page is padded to 8576 × 512 = 4,390,912 B.

**Code.**
- `hybrid_block_tokens` at `arch/glm53_flash_vllm_fp8_kda_dsa_moe.rs:1363-1375`.
- KDA state at `:1357-1361` and `worklet/glm53_kda_attn_local.rs:155,206`.

**Note.** The conv channel count (3 × 64 × 128) is read from the code, not from a run log. The 8576 result is confirmed by the run
log.

### 4d. KV bytes per token per stage

**What it does.**
- A DSA layer costs `kv_lora_rank + (index_head_dim + 4)/index_kpool = 512 + 132/4 = 545 B/token`: the fp8 MLA latent
  with no rope part, plus the kpool index (128 B fp8 key and 4 B scale per 4-token pool).
- KDA layers have no per-token KV: their pages live inside the DSA tensors.
- A stage's bytes per token = its DSA layer count × 545.
  - PP5 [9,9,9,9,9] gives [1090, 1090, 1090, **1635**, 1090] B/token.
  - The total is 11 × 545 = 5995 B/token.
- A stage with KDA layers but no DSA layer is rejected.

**Code.**
- `dsa_layer_bytes_per_token` at `arch/glm53_flash_vllm_fp8_kda_dsa_moe.rs:1349-1355`.
- Stage bytes at `arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:438-441`.
- `kda_group_count` rejects a stage with KDA and no DSA at `:196-204`.

### 4e. HBM prefix cache: retain and evict

**What it does.**
- Mode `Opportunistic`, policy `Lru`, no retained ceiling (`deployment/pp.rs:285-292`). Finished-session KV may fill
  every token of physical slack. Active, promised and held KV always win.
- Entries are session-scoped. The lookup is destructive: ownership moves to the admitted request.
- With `plain` alignment the hybrid store uses `with_exact_prefix_reuse` (`worker/kv/hybrid_gdn.rs:678-686`). Each retained
  entry is charged its context plus one state (42,880 tokens) and resumes at any token.
- With block-aligned `checkpoint` mode, hits floor to 8576 and one state is kept per block.
- A tier read's hold trims the prefix cache to the remaining slack (`hybrid_gdn.rs:408-427`).
- At completion `release_retaining_prefix` (`hybrid_gdn.rs:452-...`) retains `post_prefill_context`.

**Code.** `worker/kv/shared/prefix_cache.rs:1-80` (modes and policies). The alignment modes are described at
`config.rs:590-597`.

**To match.** Keep the recurrent state at each retained context's end, and allow resume at any token. A cache that
keeps state only at block boundaries gets floor-to-8576 hits and block-aligned chunks instead.

---

## 5. DRAM and SSD prefix tiers

**Configuration in the preset.** DRAM is 150 GB per GPU at 50 GB/s per GPU. SSD is 8000 GB per GPU at 10 GB/s per
GPU.

**Defaults** (`config.rs:656-667`; `prefix_tier_specs` at `config.rs:300-322`): tier capacity 0 = off, DRAM 50 GB/s,
SSD 10 GB/s.

### 5a. Store: write-through LRU per tier, per GPU

**What it does.**
- Every finished context is stored in every tier. With `external_decode`, the stored context is
  `declared_prefix + prompt + (outputs − 1)`.
- Each tier is an independent LRU over sessions. It evicts on its own, and an entry is charged `tokens + state_tokens`.
- A tier never holds an older context than a faster tier.
- The lookup picks the tier with the longest hit, breaking ties toward the fastest tier.
- Writes cost nothing. Their bandwidth is not modeled, only counted. A store counts as written only the tokens the
  session's previous entry in that tier did not already hold.

**Code.**
- `worker/kv/shared/prefix_tiers.rs`:
  - module doc at `:1-20`;
  - `PrefixTier::store` at `:90-112`;
  - `PrefixTiers::store` at `:227-232`;
  - `lookup` at `:168-179`.
- `pipeline_head_worker.rs:227-258` (`on_request_complete`) calls `store_session_context`.

**Units.**
- Tokens of `tier_kv_bytes_per_token`: capacity = GB × 1e9 / bytes per token (`prefix_tiers.rs:130-151`).
- With balanced load (1199 B/token): DRAM holds 125,104,253 tokens and SSD holds 6,672,226,855 tokens. Both match
  `prefix_tiers_w0.json`.

**Measured averages over the 8 h, per GPU at 1199 B/token, c3000:**

| Traffic | Tokens | Average rate |
|---|---|---|
| DRAM writes | 193.2 B | 8.0 GB/s |
| SSD writes | 6.67 B | 0.28 GB/s |
| DRAM reads | 175.3 B | 7.3 GB/s (15% of 50) |
| SSD reads | 186.4 B | 7.76 GB/s (**78% of 10**) |

**SSD read bandwidth is the binding tier resource.** A real system must also pay the DRAM write traffic, which the
simulator does not cost.

### 5b. Read at admission

**What it does.** `at_head` (`worker/admission/prefix_fetch.rs:381-475`) decides each session request when admission
reaches it, not at arrival:
1. **Already landed.** Release the read hold and `restore_prefix` the hit into HBM, then return `Admit`. The
   admission takes over the blocks.
2. **Restored earlier and still resident in HBM.** Return `Admit`. If it was evicted in the meantime, it is re-read.
3. **No tier hit larger than HBM's resident prefix.** Return `Admit`.
4. **A read is needed:**
   1. If the tier's queue wait (`channel_free − now`) exceeds `max_read_wait`, return **`PassedOver`**.
   2. Else, if the whole footprint does not fit, return **`Blocked`**. The admission loop breaks.
   3. Else hold the footprint in HBM (`hold_for_read`), queue the read, and return **`Read`**. The request leaves the
      queue.

**Read time.** Reads queue FIFO on one channel per tier (`prefix_tiers.rs:186-210`):

```
start = max(channel_free, now)
ready = start + (hit_tokens − hbm_resident_tokens + state_tokens) · tier_bytes_per_token / tier_bw
```

The read copies the entry into every faster tier. The entry stays in its own tier, because tiers are inclusive.

**Landing.** `land` (`prefix_fetch.rs:480-498`) moves done reads to a per-partition landed FIFO.
`land_prefix_reads` runs every head tick and at formation. The head wakes at the next landing
(`pipeline_head_worker.rs:374-384`).

**Landed reads first.** `admit_fresh_prompts` takes landed requests before the overdue queue and the pending order,
with no SRPT bound (`prefix_fetch.rs:311-315`).

**Reads hold HBM.** From the read's start until admission, the request's whole footprint is held. Holds count in
`fits` (`hybrid_gdn.rs:124-132`, `ledger.partition_held`).

**Known dynamics.** Tier reads hold HBM until they land (vLLM async-load semantics). Without a read bound, PP5 collapsed at
c2656 to 80 r/s. This was a metastable loop: read-wait grows, holds grow, the HBM prefix share falls, and reads grow.

### 5c. Read-wait bound and pass-over

**What it does.**
- `prefix_tier_max_read_wait_ms` = 1000. A read starts only if its tier's channel would begin it within 1 s.
- Past that, the request is **passed over**. It holds nothing and is put back where it was:
  - into the pending order, keeping its frozen SPF key;
  - or to the front of the overdue queue, newest first so order is restored.
- Admission then continues with the requests behind it (`pipelined_chunked_prefill_admission.rs:388-392`, `:454-465`).
- 0 = unbounded.

**Code.** `prefix_fetch.rs:345-354` and `:444-449`. Config: `config.rs:688`, default 0.

**History.** The first version (`AtHead::Blocked => break`) stopped all admission and lost throughput (64 r/s at
c4000). Pass-over fixed it.

**Stale doc.** `worker/config.rs:682-687` still says "admission stops and the request waits". The code passes over.

**To match.** Track the bytes queued per tier per rank, and estimate wait = queued bytes / bandwidth. Do not start the
read if the wait is over 1 s: leave the request queued, hold nothing, and continue admission.

### 5d. Warm start (benchmark artifact, not a serving mechanism)

**What it does.** Suppose a session's declared prefix exceeds every context the run stored for that session by more
than one token. Then it is treated as a pre-run context, already sitting in the **slowest** tier (SSD):
- it is seeded there and read from there;
- the hit is floored to the hit quantum (1 with `plain`).

This exists because the closed-loop trace joins sessions mid-life (placeholder round 0).

**Code.** `prefix_fetch.rs:107-138`, especially `:116-125`; seeding at `:148-164`; `PrefixTiers::seed` at
`prefix_tiers.rs:217-224`.

**Config.** `config.rs:681`, default false.

**To match in a real benchmark [inference].** Pre-populate the SSD tier with the joining sessions' contexts before
the run, or run long enough that the start-up transient is excluded. The simulator reads arrivals in [1 h, 8 h).

### 5e. `prefix_tier_balanced_load`

**What it does.** Without this flag, every tier is sized and read in the most loaded stage's bytes per token, which is
1635 B on stage 3. That is right when the slowest stage bounds a load, because a load completes only when every stage
has its slice.

With `true`, both tier capacity and read time use the pipeline mean: `ceil(total / depth) = ceil(5995 / 5) = 1199 B`.
HBM is unchanged and still uses 1635.

**Code.**
- `tier_kv_bytes_per_token` at `deployment/pp.rs:234-249`; the test at `:408-426` shows PP8 at 1090 → 750.
- `PipelineLayout::tier_kv_bytes_per_token` at `pipeline_head_worker.rs:65-68`.
- `head_prefix_tiers` at `build_pipeline_head_worker.rs:315-335`.

**Config.** `config.rs:695`, default false. The config doc calls it a what-if: the layer split leaves the stages
unequal.

**What a real system faces.** With PP, each rank stores and loads only its own layers' KV, into its own host DRAM or
SSD slice. A request counts as loaded only when every rank has its slice.

The per-rank byte imbalance is structural:
- PP5 per-token bytes are [1090, 1090, 1090, 1635, 1090] B.
- No contiguous 45-layer split puts 11 DSA layers evenly on 5 stages; the best possible is [2, 2, 2, 2, 3].
- So one rank always moves 1635 / 1199 = **1.36× the mean**.

Consequences at the measured c3000 load **[inference, arithmetic from the tier log]**:
- **SSD reads.** The mean rank needs about 7.8 GB/s of SSD read. Stage 3 would need about 10.6 GB/s, more than the
  10 GB/s per-GPU budget, so an unbalanced real system saturates on stage 3's SSD.
- **DRAM capacity.** A 150 GB per-GPU DRAM slice holds 91.7M tokens on stage 3 against 125M on the others. The session
  then misses on stage 3 first, and a partial miss on one stage means a full recompute.

**What to build so loads are even across stages [inference]:**
1. **Capacity: per-stage tier capacity proportional to bytes per token.** Host DRAM and NVMe are node-shared.
   Partition the node's DRAM and SSD so stage s gets `bytes_s / Σ bytes` of the pool: stage 3 gets 3/11, the others
   2/11 each. Every stage then holds the same token count, which equals the balanced capacity (5 × 150 GB / 5995 B =
   125M tokens). Use one eviction decision per session across all ranks. A scheduler-side LRU directory, with each
   rank storing or dropping its slice together, keeps the per-stage LRUs from diverging.
2. **SSD bandwidth: weight it by bytes.** With a node-level NVMe array striped across drives, and per-rank I/O
   queues weighted by bytes per token, stage 3 gets about 1.36× the mean rank's bandwidth. Each load's wall time is
   then (total bytes / 5) / (per-GPU share), which is what the simulator assumes. GPUDirect Storage or
   pinned-bounce reads must not serialize on one rank.
3. **DRAM → HBM goes over each GPU's own PCIe link**, roughly 50 GB/s each, so per-rank read time cannot be rebalanced
   by allocation alone. Options:
   - accept 1.36× on DRAM loads; DRAM is at 15% utilization, so this is likely fine;
   - or route part of stage 3's slice through a neighbor GPU's PCIe and forward it over NVLink.
4. **Read-wait estimation (5c) must use the per-rank max wait,** the heavy rank's queue, unless (2) holds.
5. **The alternative is to rebalance layers.** A partition with stage 3 at 2 DSA layers is impossible with 11 DSA
   layers. Moving a DSA layer changes compute balance (section 7).

---

## 6. Session stickiness, external decode, completion

### 6a. Placement and stickiness

**What it does.**
- `PpStagePoolController::admit` sends a session's later rounds to the replica of its first round
  (`session_replica`).
- New sessions are placed by `least-queued` (the default), `round-robin` or `least-work-ahead`.
- With `replicas: 1` placement does nothing, but stickiness is what makes the HBM and tier contexts reachable.

**Code.** `orchestrator/impls/pp.rs:131-161` and `:163-172`.

**To match.** For more than one replica, use a session-affinity router: same session to the same PP replica.

### 6b. `external_decode`

**What it does.**
- On arrival the head saves `target_output_tokens` and sets it to 1. The KV footprint therefore covers
  `context + 1 + state` (`pipeline_head_worker.rs:212-223`).
- The request completes at its first token.
- `on_request_complete` (`pipeline_head_worker.rs:225-258`) does two things:
  - if outputs > 1, calls `restore_prefix` to put the **whole** context `declared + prompt + (outputs − 1)` back into
    HBM's retained cache, "as a decode instance would hand it back";
  - writes the same context through to the tiers.
- The last output token's KV is not stored: the next round computes it.

**Code.** Config `config.rs:704` (no doc comment of its own; its description sits in the doc above
`prefix_tier_warm_start` at `:668-680`).

**Decode time.** It is not simulated. `trace/session_decode_wait.py` adds `(output_len − 1) / 80 tok/s` to each
round's tool wait. In the closed loop, a session's next round arrives at this round's first token + decode wait +
tool wait.

**To match [inference].**
- Run a P/D split: the prefill instance produces one token per request, and decode runs elsewhere.
- The decoded tokens' KV must get back to the prefill side's HBM or tiers. For example, the decode instance can write
  it to the shared prefix tier, keyed by session.
- Without that hand-back, every round recomputes its previous outputs. In the source trace outputs are 187M tokens
  against 625M input tokens, so that is roughly +30% prefill work.
- The simulator also writes the output KV for free.

### 6c. Completion and TTFT

**What it does.**
- `record_first_token` is stamped at the microbatch's **exit from the last stage**: the end of the last stage's
  compute, which includes the lm_head (`pipelined_chunked_prefill_admission.rs:746-776`).
- KV is then released with the prefix retained.
- There is no sampling, detokenize or network cost after that point.

**Test.** `pipeline_head_worker.rs:538-569`.

**To match.** Measure TTFT from the round's arrival at the scheduler to the first token leaving the last PP rank.

---

## 7. Activation transfer and layer partition

### 7a. Activation bytes per token

**What it does.** `hc_mult × hidden × bf16 = 4 × 4096 × 2 = 32,768 B/token`: each hop carries the 4-wide mHC residual
stream. A 32k-token microbatch moves 1.07 GB per hop.

**Code.**
- `activation_bytes_per_token` at `arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:716-720`.
- `ACTIVATION_DTYPE = Bf16` at `glm53_flash_vllm_fp8_kda_dsa_moe.rs:102`.
- Microbatch bytes at `pipeline_head_worker.rs:356-366`.

### 7b. Transfer cost model

**What it does.** Each hop is one `submit_transfer` on the shared `GpuCluster`:
- cost from the **profiled `p2p_intra` NCCL NVLink curve** (`deployment/pp.rs:342-356`);
- the send and receive legs share one start time `max(now, send_free, recv_free)`; the slower side bounds the end
  (`worker/gpu_cluster.rs:1-55`; `submit_transfer` at `:360-412`, the `max()` start at `:381`).

**Follower stages double-buffer** (`worker/workers/pipeline/pipeline_stage_worker.rs:130-152`):

```
pull_start(k)    = max(ready_at(k), compute_start(k−1))
pull_end(k)      = submit_transfer(pull_start, prev_gid, own_gid, bytes)
compute_start(k) = max(pull_end(k), compute_end(k−1))
```

- A stage hand-off costs no simulator tick (`orchestrator/impls/pp.rs:183-240`).
- Each follower costs its own whole-stage iteration over the same batch.

**[inference]** A build that receives on the same stream as the forward loses part of this pull/compute overlap. At
about 1 GB per hop on NVLink against stage computes of hundreds of ms, the cost is small.

### 7c. Layer partition

**What it does.**
- `stage_ranges` uses `layer_partition` if it is given. It must have `pp_size` positive counts summing to 45.
- Otherwise `pp_indices` gives `n // pp` layers to each stage, with the remainder added to the stages ending at
  `pp − 2`, then `pp − 3`, and so on.
- 45 / 5 = 9 exactly, so the split is **[9, 9, 9, 9, 9]**.

**Code.**
- `stage_ranges` at `arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs:92-122`.
- `pp_indices` at `arch/glm52_vllm_nvfp4_pp_dsa_moe.rs:72-92`.
- Config `layer_partition` at `arch/config.rs:900` (NVFP4; `:737` for FP8); the preset leaves it empty.

**Layer types** (from `model/config/glm53_flash_nvfp4.json`, `linear_attn_config`):
- `full_attn_layers` (DSA) = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43], every 4th starting at 3: 11 layers.
- The other 34 are KDA.
- `first_k_dense_replace` = 3, so layers 0-2 have a dense FFN and 3-44 have MoE.
- The code classifies a layer as KDA if it is in `kda_layers`, else DSA (`StageLayerCounts::of`, `:357-385`).

| stage | layers | DSA layers | KDA layers | dense-FFN layers | KV B/token | extras |
|---|---|---|---|---|---|---|
| 0 | 0-8 | 3, 7 | 7 | 3 (0-2) | 1090 | embedding, hc_expand |
| 1 | 9-17 | 11, 15 | 7 | 0 | 1090 | |
| 2 | 18-26 | 19, 23 | 7 | 0 | 1090 | |
| 3 | 27-35 | 27, 31, 35 | 6 | 0 | **1635** | |
| 4 | 36-44 | 39, 43 | 7 | 0 | 1090 | terminal mHC post, hc_contract mean, final norm, lm_head |

**Balance: claimed vs measured.**
- The study's preset states 97.1% layer balance for PP5 against 91.0% for PP8; the script behind 97.1 is not
  kept. The study's run reported "stage balance 96.7% vs 90.4%".
- **[inference]** With the study's per-layer fit at 8192 tokens (MoE+KDA 6.9 ms, MoE+DSA 8.05 ms, dense+KDA 5.95 ms), the
  stage times come out to [61.6, 64.4, 64.4, 65.6, 64.4] ms. That is mean/max ≈ 97.7%, ignoring embedding and lm_head.
- In the winning run, stage 3 is the bottleneck: busy 99.5% against 90.5% on stage 0 (`$R/reports/utilization_report.json`).

**To match.** Use 9 consecutive layers per stage. Every stage needs at least one DSA layer.

---

## 8. Relevant docs

Paths are from the ServingStudioSim root.

- `simulator/src/worker/README.md` §"5. Pipeline stages": L5 composition of the PP head and follower, even split,
  tiers, stage cadence.
- `doc/detailed_design/L5.md` §6.5: design record of the head lifecycle, all PP knobs and tier semantics.
- `doc/detailed_design/L6.md` §"Pipeline parallelism": PP stage pool, placement and session stickiness, zero-tick
  hand-off, `p2p_intra`.
- The study's scheduler trials (29 on PP4 and PP8; raw logs not included) led to the recommended policy. Rejected:
  - fresh-first;
  - contended, alone, matched and backlog caps;
  - the tail ladder;
  - deadline guards;
  - the busy-fraction budget, replaced by the backlog budget.
  - The ablation showed SRPT is the head-of-line fix; a later trial raised the budget ceiling to 16k (now 32k).
- The study's PP5 sweeps, summarized in README §3 and §9: budget (m16k/m32k × budget), the concurrency collapse
  without a read bound, and the read bound blocking vs pass-over. To rerun one, copy `pp5_8h.yaml` and change the
  stated knob.
- `trace/session_decode_wait.py` and `trace/session_closed_loop.py`: how decode time and the closed-loop warm start
  enter the workload; `make_traces.sh` runs them.
- `skills/top-compose-real-framework-from-sim/SKILL.md`: the repo's skill for building a real framework from a
  simulated design; follow it.

## 9. Summary of mechanisms

| mechanism | knob (default) |
|---|---|
| ≤ pp_size in flight; progress committed at formation | (fixed) |
| form at stage-0 idle, not ahead into a queue | n/a |
| plain chunk min(remaining, budget) | `prefill_chunk_alignment` (checkpoint) |
| exact-end-state prefix reuse | via `plain` |
| even split, floor 512 | `microbatch_split` (greedy), `min_microbatch_tokens` (0) |
| long_prefill_token_threshold | 0 |
| SPF queue | `pending_order` (fifo) |
| SRPT started + queued | `srpt` (false) |
| backlog load budget | `load_budget_*` (0 = off) |
| force schedule after 60 s | `force_schedule_after_ms` (0) |
| one pool sized by worst stage; G + 1 state blocks; 8576 block | n/a |
| full-footprint reservation, no preemption | n/a |
| tier lookup at admission, read holds blocks, landed first | `dram_tier_gb`/`ssd_tier_gb` (0) |
| write-through, free writes, FIFO per-tier channel | `*_gb_per_s` (50 / 10) |
| read-wait bound with pass-over | `prefix_tier_max_read_wait_ms` (0) |
| balanced tier loads (what-if) | `prefix_tier_balanced_load` (false) |
| warm start (benchmark artifact) | `prefix_tier_warm_start` (false) |
| external decode with KV hand-back | `external_decode` (false) |
