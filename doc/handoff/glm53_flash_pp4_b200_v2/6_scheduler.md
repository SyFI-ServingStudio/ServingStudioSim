# The scheduler

The fifth milestone, after `5_prefix_cache.md` passes. The pipeline, the kernels and the prefix cache (HBM only) stay
as they are; only the scheduler changes. It is built in four steps on top of the plain scheduler, and each step has
its own targets, so each can be checked before the next. DRAM and SSD come in `7_offloading.md`.

## 1. The workloads

| Run | Trace | Requests | Arrivals |
|---|---|---|---|
| Scheduling check | `data/bringup/sched_check.csv` | 28 prompts, one output token each, no prefix | 26 at t = 0, longest first: 1 × 128k, 1 × 32k, 8 × 4k, 16 × 1k. Then each alone on an idle pipeline: 8k at 5 s, 1k at 7 s |
| Force check | `data/bringup/force_check.csv` | 1 × 32k, then 16,000 × 1k | the 32k at t = 0; the 1k prompts at 200 per second from t = 0 to 80 s, more than the pipeline computes |
| Sessions | `data/sessions_2ps.csv` (`1_contract.md` §2.2) | 2,457 coding-agent sessions | 2 per second for 20 minutes, at most N active; N = 400 and 500 |

The two checks use req-frontend's `text-generation-independent` format; `data/bringup/make_bringup.py` writes them.

## 2. What the scheduler does

**Unchanged from `5_prefix_cache.md` §2:**
- at most 4 microbatches in flight, and the next one forms only when stage 0 is free;
- a prompt's progress is committed when its chunk is scheduled;
- chunks end at any token;
- each request reserves its whole footprint once, with no preemption;
- the prefix cache.

**Each time stage 0 is free**, the scheduler forms one microbatch: it measures the backlog, sets the budget, and
fills the budget in order.

**The backlog.** B is the prefill work the pipeline holds, in tokens:
- the tokens started prompts have not scheduled yet;
- the new tokens of the waiting prompts;
- the tokens of the microbatches in flight: formed on stage 0 and not yet finished on stage 3.

**The budget** (M is 1,000,000 tokens) is the smaller of two numbers:
- **Load budget:** 4,096 tokens while B ≤ 1M, 32,768 when B ≥ 3M, and linear in between:
  4,096 + round(28,672 × (B − 1M) / 2M), to the nearest token.
- **Even split:** B / 4, rounded up and kept between 512 and 32,768. The 4 microbatches the pipeline holds then
  share the backlog evenly. A lone 8k prompt runs as 4 chunks of 2,048 that follow each other through the stages,
  not as one or two large ones.

**The order** fills the budget:
- **(a) Overdue first.** A request whose wait since it was submitted has reached 60 s goes before everything else:
  waiting ones first, then started ones, each oldest first, whatever their size.
- **(b) Started prompts, fewest remaining tokens first.** Before each one, waiting prompts that need no more tokens
  than it has left go first, shortest first.
- **(c) Then waiting prompts, shortest first,** until the budget is spent.

The rules inside the order:
- **Size.** A waiting prompt's size is the tokens it must compute: its new tokens plus any part of its prefix that
  was not in the cache when it arrived. Ties go by arrival.
- **Chunks.** Each prompt takes as much of the budget as is left, up to its remaining tokens.
- **Admission.** A waiting prompt is admitted only if its footprint fits the pool. The first one that does not fit
  stops admission for this microbatch.
- **No waiting.** The microbatch starts at once, even below its budget.

**Built in four steps.** Each adds one rule to the plain scheduler of `5_prefix_cache.md` (step 0):

| Step | Adds | Budget | Order |
|---|---|---|---|
| 0 | the plain scheduler (`5_prefix_cache.md`) | 8,192 | started first, then arrival order |
| 1 | shortest first | 8,192 | (b), (c) |
| 2 | load budget | load budget | (b), (c) |
| 3 | even split | min(load budget, even split) | (b), (c) |
| 4 | force-schedule: the design | min(load budget, even split) | (a), (b), (c) |

## 3. Targets: scheduling check

TTFT in ms:

| Step | Microbatches | 128k | 32k | 4k (first – last) | 1k (first – last) | 8k alone | 1k alone |
|---|---|---|---|---|---|---|---|
| 0 | 28 | 1,118.2 | 1,343.5 | 1,397.8 – 1,561.2 | 1,614.7 – 1,669.1 | 223.6 | 51.9 |
| 1 | 28 | 1,678.9 | 728.5 | 327.3 – 493.9 | 215.2 – 270.0 | 223.6 | 51.9 |
| 2 | 55 | 1,741.7 | 711.0 | 243.1 – 459.8 | 119.6 – 209.6 | 154.7 | 51.9 |
| 3 | 58 | 1,741.7 | 711.0 | 243.1 – 459.8 | 119.6 – 209.6 | 131.7 | 50.8 |
| 4 | 58 | 1,741.7 | 711.0 | 243.1 – 459.8 | 119.6 – 209.6 | 131.7 | 50.8 |

The microbatches at step 4:
- **0–3:** the 16 1k prompts, 4 per microbatch. B is 213k, so the budget is 4,096.
- **4–11:** one 4k prompt each.
- **12–19:** the 32k prompt in chunks of 4,096.
- **20–51:** the 128k prompt in chunks of 4,096.
- **52–55:** the lone 8k prompt in 4 chunks of 2,048.
- **56–57:** the lone 1k prompt in 2 chunks of 512 (B / 4 = 256, kept at 512).

At step 0 the 128k prompt goes first in 16 chunks of 8,192, and every short prompt waits behind it.

These decisions follow from the rules alone, not from timing, so the real scheduler must form the same microbatches.
`data/sched_check_targets.json` lists every microbatch of every step (keys `plain`, `srpt8k`, `budget`, `even`, `good`
for steps 0–4): its chunks as (request, first token, tokens),
its start, and its tokens. To regenerate: `data/sched_check_targets.py` (usage in its docstring).

## 4. Targets: force check

| Step | 32k TTFT (s) | 1k TTFT p50 / p90 / p99 / max (s) | Last token (s) | Microbatches |
|---|---|---|---|---|
| 0 | 0.40 | 15.26 / 27.13 / 29.80 / 30.11 | 110.1 | 2,004 |
| 1 | 110.07 | 15.08 / 26.95 / 29.62 / 29.93 | 110.1 | 2,004 |
| 2 | 107.51 | 14.77 / 23.90 / 26.89 / 27.30 | 107.5 | 1,429 |
| 3 | 107.51 | 14.77 / 23.90 / 26.89 / 27.30 | 107.5 | 1,429 |
| 4 | 61.03 | 14.77 / 24.09 / 27.12 / 27.52 | 107.5 | 1,429 |

- The 1k prompts arrive faster than the pipeline computes them, so their queue grows for 80 s and drains by about
  108 s.
- **Shortest first (steps 1–3).** The 32k prompt waits until the queue is empty: it comes out last.
- **Force-schedule (step 4).** The 32k prompt goes first once it has waited 60 s, and comes out at 61 s.
- **Load budget (steps 2–4).** The backlog passes 3M tokens, so the microbatches grow up to 32,768 tokens and the
  queue drains 2.6 s sooner.

`data/force_check_targets.json` holds these, with the same keys as §3. To regenerate: `data/force_check_targets.py`.

## 5. Targets: sessions

Over the 20 minutes, by `1_contract.md` §3, with the prefix cache in HBM only. Over 60 s is the number of counted
rounds whose TTFT exceeds 60 s.

**N = 400:**

| Step | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 0 | 24.9 | 87.5k | 98.28% | 0.55 / 3.66 / 5.98 / 9.3 | 0 | 84 / 79 / 82 / 82% |
| 1 | 26.8 | 93.0k | 98.58% | 0.26 / 0.34 / 1.80 / 133.0 | 12 | 86 / 81 / 83 / 84% |
| 2 | 26.7 | 92.3k | 98.77% | 0.15 / 0.34 / 1.80 / 232.0 | 19 | 88 / 85 / 87 / 88% |
| 3 | 26.7 | 92.3k | 98.77% | 0.15 / 0.34 / 1.83 / 179.8 | 18 | 89 / 86 / 89 / 90% |
| **4** | **25.6** | **89.4k** | 98.55% | **0.15 / 0.81 / 9.85 / 63.7** | 165 | 90 / 87 / 89 / 90% |

**N = 500:**

| Step | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 0 | 22.8 | 86.6k | 97.01% | 3.01 / 13.03 / 23.93 / 28.0 | 0 | 91 / 86 / 88 / 89% |
| 1 | 29.2 | 104.2k | 98.76% | 0.26 / 0.36 / 3.39 / 564.7 | 118 | 91 / 86 / 88 / 89% |
| 2 | 28.6 | 102.9k | 98.83% | 0.16 / 0.59 / 4.61 / 577.4 | 162 | 92 / 88 / 91 / 91% |
| 3 | 28.6 | 102.9k | 98.83% | 0.16 / 0.60 / 4.38 / 579.6 | 159 | 93 / 89 / 91 / 92% |
| **4** | **25.8** | **92.6k** | 97.82% | **0.29 / 4.22 / 61.55 / 68.4** | 979 | 93 / 88 / 91 / 92% |

**At N = 400, step by step:**
- **Shortest first:** TTFT p50 / p90 / p99 from 0.55 / 3.66 / 5.98 s to 0.26 / 0.34 / 1.80 s. A short round no
  longer waits behind a long prompt's chunks.
- **The load budget:** 0.15 / 0.34 / 1.80 s; p50 falls most. At this load the backlog stays small, so microbatches of
  4,096 tokens run instead of 8,192 and come out sooner. They cost more GPU time per token: stage busy 86% → 88%.
- **The even split:** 0.15 / 0.34 / 1.83 s, little change here; its effect is on a lone prompt (8k: 154.7 →
  131.7 ms in §3).
- **Force-schedule:** 0.15 / 0.81 / 9.85 s. It bounds the worst wait, 180 s → 64 s, and in return p99 rises above the
  SLO. About 160 rounds reach 60 s, and most of them are computed again whole, a median of 99k tokens: 102 whose
  prompt ends inside the previous one and 46 whose session was evicted from the pool. Shortest first keeps passing
  them over; force-schedule serves them at 60 s, and the rounds behind them wait.
- **With DRAM and SSD,** evicted sessions are read back instead, and the same scheduler holds N = 400 within the SLO
  (`7_offloading.md`; README: 92.8k new tokens/s, TTFT 0.15 / 0.48 / 4.03 s).

**At N = 500, with HBM only, no step keeps up:**
- **Without force-schedule (steps 1–3).** TTFT p50 / p90 / p99 look fine: 0.16–0.26 / 0.36–0.60 / 3.39–4.61 s. But
  120–160 rounds wait more than a minute, up to 10 minutes, and about 150 more are still waiting after a minute when
  the 20 minutes end. Most of the rounds over a minute are computed again whole, a median of 71–79k tokens: their
  prompt ends inside the previous one (about 70 at steps 2–3), or their session was evicted from the pool (about 60).
  Shortest first keeps passing them over.
- **With force-schedule (step 4).** These rounds go in after 60 s. Each reserves its whole footprint at once and
  evicts more sessions: the rounds whose session was evicted rise from about 140 to 298. The queue behind them grows:
  TTFT 0.29 / 4.22 / 61.55 s.
- **DRAM and SSD do not lift N = 500.** `7_offloading.md` keeps evicted sessions in DRAM and SSD, which removes the
  evictions but not the rounds that end inside the previous prompt. Those still add about 21k computed tokens/s, and
  the stages, already 93% busy, cannot take them: the design does not keep up at N = 500 either.

`data/session_targets.json` holds every step (steps 0–4 under `plain_hbm`, `srpt8k_hbm`, `budget_hbm`, `even_hbm`,
`good_hbm`) at N = 100 to 700. The microbatch check of `5_prefix_cache.md` §4 applies unchanged.

## 6. How to run

- The setup of `5_prefix_cache.md` §5: no DRAM or SSD tier, the 61.3M-token pool, cached tokens reported per request.
- For each step, the scheduling check, the force check, then the sessions at N = 400 and 500, each on a freshly
  started server.
- **Scheduling check.** The 26 requests at t = 0 must all be queued before the first microbatch forms: hold the
  scheduler until the client has submitted them. Log every microbatch's chunks: request, first token, tokens.
- **Force check.** Replay with no `max_concurrency`, so the queue really grows. Check from the client log that the
  1k prompts went out at 200 per second.
- **Sessions.** As in `5_prefix_cache.md` §5, with the microbatch log.

## 7. Pass rule

**Scheduling check**, per step:
- **Microbatches:** the same sequence as `data/sched_check_targets.json`, chunk for chunk. Any difference is a
  difference in the rules; find which rule.
- **TTFT** of every request within ±10%.

**Force check**, per step:
- **32k TTFT** within ±10%. At step 4 it must come out between 60 and 65 s (target 61.03 s).
- **1k TTFT** p50, p90 and p99 within ±10%; last token within ±10%.

**Sessions**, per step and N, as in `5_prefix_cache.md` §6:
- rounds/s and new tokens/s within ±10%;
- TTFT p50 and p90 within ±10%, p99 within ±25%;
- prefix hit share and microbatches as there.
- Record the worst TTFT and the rounds over 60 s. At step 4 the worst TTFT stays within 10 s above 60 s (targets 63.7
  s at N = 400, 68.4 s at N = 500).

If the scheduling check passes and the sessions do not, the gap is in the timing the rules run on (when stage 0 is
free, host time) or in the cache, not in the rules.

## 8. Where these targets may be off

- **Host time.** The scheduler sorts and fills the queue for every microbatch, about 20 a second. Time the host
  spends on it is not in the target and comes straight out of stage 0's time.
- **Prompt sizes known at arrival.** A waiting prompt's size is fixed when it arrives. If its session is evicted
  while it waits, it is still ranked by its old size. A scheduler that re-ranks it is a difference to record.
- **N = 500 with HBM only is overloaded at every step.** Its numbers depend on which sessions get evicted, and they
  move more between implementations than the N = 400 numbers do.
- **Force-schedule at N = 400 with HBM only.** Its p99 comes from the about 160 rounds served at 60 s and the rounds
  queued behind them. Which rounds reach 60 s depends on the order and the evictions, so p99 moves more than p50
  and p90 between implementations.
- The caveats of `5_prefix_cache.md` §7 still hold.

## 9. What to record

**Scheduling check** (`measured/sched_check_<step>.jsonl`), one line per microbatch:

```json
{"index": 12, "start_ms": 368.1, "tokens": 4096, "chunks": [["sched_1", 0, 4096]]}
```

and each request's TTFT.

**Force check** (`measured/force_check.jsonl`), one line per step: the 32k TTFT, the 1k TTFT p50 / p90 / p99 / max, the
last token, the microbatch count.

**Each session run** (`measured/runs/sched_step<k>_n400/`, `sched_step<k>_n500/`, `1_contract.md` §6): as
`5_prefix_cache.md` §8.

**Summary** (`measured/scheduler.md`):
- The tables of §3, §4 and §5 with the measured numbers and their ratios beside them.
- The first microbatch where each scheduling check differs, if any, and the rule it points at.

**Gaps** (`issues.md`): every check or run above its pass rule, every scheduling-check difference, and every
difference from §2.
