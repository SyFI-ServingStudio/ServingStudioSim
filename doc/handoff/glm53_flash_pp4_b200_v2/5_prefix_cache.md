# Prefix cache and chunked prefill

The fourth milestone, after `4_iteration.md` passes. The pipeline now serves coding-agent sessions. Each round's
prompt starts with the previous round's prompt, so the server keeps every session's context and computes only what
is new. The scheduler stays the plain one of `4_iteration.md`, and the prefix cache lives in HBM only. The design's
scheduler comes in `6_scheduler.md`; DRAM and SSD come in `7_offloading.md`.

## 1. The workloads

| Run | Trace | Sessions | Arrivals |
|---|---|---|---|
| Prefix bring-up | `data/bringup/prefix_sessions.csv` | 40 two-round sessions: 5 for each (P, A) below | one session every 30 s |
| Sessions | `data/sessions_2ps.csv` (`1_contract.md` §2.2) | 2,457 coding-agent sessions | 2 per second for 20 minutes, at most N active; N = 400 and 500 |

**Prefix bring-up.** Each session has two rounds:
- Round 0 is a prompt of P tokens.
- Round 1 comes 10 s after round 0's token. It sends round 0's prompt, its output token, and A − 1 new tokens.
- P tokens should hit the cache and A should be computed: the output token was never run through the model, so it is
  computed with the new tokens.
- P is 8k, 32k, 128k or 512k, and A is 1k or 8k.
- Only one session is active at a time, so every round runs on idle GPUs.

`data/bringup/make_bringup.py` writes the file. Both traces use req-frontend's `text-generation-session-execution-v2`
format.

## 2. What runs

**What a session keeps.** When a round ends, the server keeps its prompt:
- the KV of every DSA layer, each stage for its own layers;
- the state of every KDA layer after the prompt's last token.

**A hit.** A hit is the whole previous prompt; a prompt that ends inside it recomputes.
- **The prompt extends the previous one.** This is almost every round: the next round finds the whole previous prompt
  (`prefix_len` − 1 tokens) and computes only the rest.
- **The prompt ends inside the previous one.** The round shares only its first `prefix_len` tokens with the previous
  prompt (`1_contract.md` §2.2). The session's KDA state is the one at the previous prompt's last token, and there is
  none at `prefix_len`, so the round computes its whole prompt. At N = 400 these are 1.2% of the rounds, and their
  prefix tokens are 1.2% of all prefix tokens.

The hit is exact:
- It is not rounded down to a block. A cache that can only resume on block boundaries computes the tokens from the
  last boundary again.
- The KDA state must be the one at the prompt's last token. A KDA state saved only every N tokens cannot resume
  there.
- req-frontend starts every session at its own offset in its token corpus, so sessions share no prefix. A block
  cache that matches by content finds the same hits as one that matches by session.

**Memory.**
- A hit needs the prefix on all 4 stages, so every stage keeps the same tokens: one block table for the pipeline.
  The stage with the most KV per token sets the size. Stages 0, 2 and 3 hold 3 DSA layers each (1,635 B/token) and
  stage 1 holds 2 (1,090 B/token), so the pool is 100.18 GB ÷ 1,635 B = 61.3M tokens (`1_contract.md` §1). Stage 1
  fills only about 67 GB of its 100.18 GB with them; about 33 GB of its KV memory stays unused.
- Kept sessions use the same KV pool as running requests.
- A kept session costs its prompt's tokens plus its KDA states: 6 blocks of 8,576 tokens, 51,456 tokens' worth
  (`1_contract.md` §1). A running request costs its whole prompt plus the same 6 blocks, reserved once when it is
  admitted.
- Running requests come first. When a new request needs room, kept sessions are evicted, least recently used first,
  a whole session at a time.
- The next round of an evicted session computes its whole prompt again.

**The scheduler** runs on stage 0 and is the plain one:
- **In flight.** At most 4 microbatches. The next one forms only when stage 0 is free.
- **Order.** Started prompts first, in the order they started, then waiting prompts in arrival order.
- **Budget.** 8,192 computed tokens per microbatch. Cached tokens do not count.
- **Chunking.** A prompt that does not fit the budget left is cut at whatever token the budget ends on, not on a
  block boundary. Its progress is committed when the chunk is scheduled, so its next chunk can ride the next
  microbatch.
- **Admission.** A request is admitted when its whole footprint fits the pool, after evicting kept sessions if
  needed. If it does not fit, it waits, and the requests behind it wait too. There is no preemption.

**What the targets assume.** The numbers in §3 and §4 come from exactly the system above:
- chunks end at any token;
- a hit is the whole previous prompt, resumed from the KDA state saved at its last token, and a prompt that ends
  inside the previous one is computed again from its first token;
- eviction takes whole sessions, least recently used first;
- a running request's KV is reserved whole at admission, and no request is preempted;
- no host time: a microbatch starts on stage 0 as soon as stage 0 is free.

A real system that differs in one of these misses or beats the target for that reason: record which (§8).

**What a microbatch costs.** A request computing A tokens on top of a cached prefix of P tokens:
- The DSA indexer scores all P + A keys of its context for each new token. Sparse MLA then attends to the top 2,048
  tokens, so its cost hardly depends on P.
- KDA starts from the saved state, so its cost does not depend on P at all.
- Every other kernel depends only on the microbatch's tokens.

## 3. Targets: prefix bring-up

TTFT in ms. Round 0 runs its P tokens as 8k chunks through the pipeline. Round 1 is one microbatch of A tokens, after
a hit of P; its compute per stage in ms, and each send in ms:

| P | A | Round 0 TTFT | Round 1 TTFT | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Each send |
|---|---|---|---|---|---|---|---|---|
| 8k | 1k | 223.6 | 53.3 | 12.703 | 13.273 | 13.457 | 13.684 | 0.068 |
| 8k | 8k | 223.6 | 223.8 | 57.130 | 54.159 | 55.325 | 55.698 | 0.482 |
| 32k | 1k | 396.8 | 53.8 | 12.825 | 13.355 | 13.580 | 13.807 | 0.068 |
| 32k | 8k | 396.8 | 226.4 | 57.841 | 54.633 | 56.036 | 56.409 | 0.482 |
| 128k | 1k | 1,118.2 | 55.4 | 13.272 | 13.652 | 14.027 | 14.253 | 0.068 |
| 128k | 8k | 1,118.2 | 239.0 | 61.277 | 56.924 | 59.472 | 59.845 | 0.482 |
| 512k | 1k | 4,408.1 | 64.3 | 15.710 | 15.278 | 16.465 | 16.692 | 0.068 |
| 512k | 8k | 4,408.1 | 287.7 | 74.574 | 65.789 | 72.770 | 73.143 | 0.482 |

- Without a prefix, the same microbatches take 51.9 ms (1k) and 223.6 ms (8k) (`4_iteration.md` §3). A cached
  context of 8k to 32k adds 0–4%, 128k adds 7%, and 512k adds 24–29%, almost all of it in the indexer.
- Round 0 is the chunked pipeline at work. Its 8k chunks follow each other through the stages, so 128k takes 1.12 s
  instead of the 3.33 s of one 128k microbatch.

`data/prefix_targets.json` holds every case, with each microbatch's tokens, context, stage compute and sends. To
regenerate: `data/prefix_targets.py` on the prefix bring-up run (usage in its docstring).

## 4. Targets: sessions

Over the 20 minutes, by `1_contract.md` §3. Computed tokens/s counts every prompt token the server did not take
from the cache, including prompts computed again after an eviction or because they end inside the previous prompt.
The prefix hit share is cached tokens over the
trace's prefix tokens.

| N | Rounds/s | New tokens/s | Computed tokens/s | Prefix hit share | TTFT p50 / p90 / p99 (s) | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| **400** | **24.9** | **87.5k** | 115.4k | 98.28% | **0.55 / 3.66 / 5.98** | 84 / 79 / 82 / 82% |
| 500 | 22.8 | 86.6k | 126.8k | 97.01% | 3.01 / 13.03 / 23.93 | 91 / 86 / 88 / 89% |

- **N = 400 keeps up,** just within the SLO (p99 5.98 s). 1.72% of prefix tokens miss, and computing them again adds
  32% to the computed tokens. Most of them, 1.24% of the prefix tokens, belong to the rounds whose prompt ends inside
  the previous one; the other 0.48% belong to sessions evicted from the pool.
- **N = 500 does not.** The pool no longer holds every session between its rounds. 2.99% of prefix tokens miss, and
  computing them again adds 46% to the computed tokens. The stages run close to their limit, rounds/s falls below
  N = 400, and the queue grows: TTFT p50 / p90 / p99 3.01 / 13.03 / 23.93 s.
- The full design serves N = 400 at 92.8k new tokens/s and TTFT 0.15 / 0.48 / 4.03 s (README). The design's
  scheduler (`6_scheduler.md`) does not reach it alone: with HBM only, its force-schedule step misses the SLO at
  N = 400. Keeping evicted sessions in DRAM and SSD (`7_offloading.md`) does reach it.

`data/session_targets.json` holds these and the runs at N = 100 to 700 (setting `plain_hbm`). The SLO holds up to
N = 400, just: at N = 300, TTFT is 0.22 / 2.13 / 5.83 s, against 0.55 / 3.66 / 5.98 s at N = 400. To regenerate:
`data/session_targets.py`, which runs `data/metrics.py` on each run (usage in its docstring).

**Microbatch check.** Inside the session run, each stage's compute time per microbatch follows a fit:

```
stage time (ms) = a + b · tokens / 1k + c · pairs / 1e9,    pairs = Σ over requests of A · (P + A / 2)
```

Here A is a request's computed tokens in this microbatch and P the tokens it already has: its cached prefix plus what
its earlier chunks computed.

| Stage | a (ms) | b (ms per 1k tokens) | c (ms per 1e9 pairs) |
|---|---|---|---|
| 0 | 6.29 | 6.106 | 3.879 |
| 1 | 7.31 | 5.639 | 2.485 |
| 2 | 7.38 | 5.759 | 3.914 |
| 3 | 7.58 | 5.779 | 3.915 |

For microbatches of at least 256 tokens, the fit is within 1.5% of the targets' own microbatch times at the median
and within 4.3% for 99% of them, at N = 400 and N = 500 and on the prefix bring-up. Smaller microbatches take 2–8 ms
and are left out. `data/microbatch_fit.json` holds the fit; `data/microbatch_fit.py` makes it and checks measured
microbatches against it.

## 5. How to run

- The setup of `1_contract.md` §1, with no DRAM or SSD tier: the prefix cache is HBM only, in the 61.3M-token pool.
- The serving process with the scheduler of §2 and the prefix cache on. The server must report its cached tokens per
  request (vLLM: `--enable-prompt-tokens-details`); req-frontend logs them as `cached_prompt_tokens`.
- **Prefix bring-up.** Replay `data/bringup/prefix_sessions.csv` with req-frontend on a freshly started server, as in
  `1_contract.md` §2.2 but with no `max_concurrency`. Record a trace of all 4 GPUs.
- **Sessions.** The replay of `1_contract.md` §2.2, N = 400 and N = 500, each on a freshly started server with an
  empty cache. Log every microbatch: its stage, its compute time, and each request's cached prefix and computed
  tokens.
- **Check the output.** For round 1 of every bring-up case, the token, or the logits, against the reference
  computing the whole prompt with no cache (`1_contract.md` §4). A wrong token after a hit points at the saved KDA
  state or at the KV of the cached tokens.

## 6. Pass rule

**Prefix bring-up**, per case:
- **Hit:** round 1's cached tokens equal P exactly.
- **Round 1:** each stage's compute within ±10% of §3; TTFT within ±10%.
- **Round 0:** TTFT within ±10%.

**Sessions**, per N:
- **Rounds/s and new tokens/s** within ±10%.
- **TTFT** p50 and p90 within ±10%. p99 within ±25%, because one long prompt more or less in the tail moves it.
- **Prefix hit share:** record it. A lower share than the target is a gap in the cache: what it keeps, what it
  evicts, or how large the pool is. A higher share beats the target and is welcome.
- **Microbatches:** for each stage, the median of measured ÷ fit within ±10%.

If the microbatches pass and TTFT does not, the gap is in scheduling or in the cache, not in the kernels: compare the
microbatch sizes, the order requests are served in, and the hit share.

## 7. Where these targets may be off

- **The pool decides N = 500.** The miss share, and with it the overload, follows from the pool's size and from what
  a kept session costs. A real pool of another size, a KDA footprint of another size, or eviction that keeps parts of a
  session moves the N = 500 numbers far more than any kernel does. The states themselves need less than the 6
  blocks: 39.1 MB on stage 0, about 24k tokens' worth. A build that stores them at that size keeps more sessions
  and beats the target; record the footprint it uses.
- **Stage 1's unused KV memory.** The target leaves it empty. It cannot hold more tokens on its own, since a hit
  needs every stage. A layer split with the DSA layers spread more evenly, or a pool that uses the spare memory some
  other way, raises the hit share; that beats the target and is welcome. The split [12,11,11,11] balances compute,
  and the targets keep it.
- **Partial sessions.** The target evicts whole sessions. A cache that evicts blocks from the end of a session's KV
  leaves a prefix with no KDA state at its end, which nothing can resume from: memory held for no hit.
- **Prompts that end inside the previous one.** The target computes them whole. A cache that also keeps KDA states
  inside a context, for example one per block, can resume them near their end and beats the target; record which
  states it keeps and what they cost in the pool.
- **The tokens the client sends.** The client builds round k+1 from round k's prompt and its real output token. Sent
  as token IDs, the prefix is exact. A server that tokenizes text again may not reproduce round k's tokens, and the
  hit is then shorter than P. req-frontend logs the difference as `prefix_shortfall_tokens`; it should be 0.
- **Host time.** At N = 400 the host forms 17.5 microbatches a second, one every 57 ms. Scheduling, cache lookups and
  launch time are not in the target; at N = 500, every millisecond the host adds to a stage-0 microbatch is lost
  throughput.
- **Microbatch fit.** It is a fit to the targets' own microbatches, not a kernel measurement. Below 256 tokens it does
  not apply.
- The caveats of `4_iteration.md` §7 and `3_layers.md` §6 still hold.

## 8. What to record

**Prefix bring-up** (`measured/prefix_bringup.jsonl`), one line per session:

```json
{"prefix_tokens": 131072, "computed_tokens": 8192, "session": 2, "round0_ttft_ms": 1131.0, "round1_ttft_ms": 242.6,
 "round1_cached_tokens": 131072, "round1_stages": [{"stage": 0, "compute_ms": 61.9}, "..."],
 "round1_sends_ms": [0.49, 0.49, 0.49], "token_matches_reference": true}
```

**Each session run** (`measured/runs/prefix_n400/`, `prefix_n500/`, `1_contract.md` §6):
- What `1_contract.md` §6 lists for each run, including the output of `data/metrics.py`.
- `microbatches.jsonl`, one line per (microbatch, stage) in the format of `data/microbatch_fit.py check`, and that
  check's output.
- The pool in tokens, the evictions, any preemptions, and the prefix hit share.

**Summary** (`measured/prefix_cache.md`):
- The bring-up table of §3 with the measured numbers and their ratios beside it.
- The session table of §4 the same way, with the microbatch check per stage.
- For each N that misses: whether the microbatches, the hit share or the scheduling explains it.

**Gaps** (`issues.md`): every case above its pass rule, every hit shorter than P, every output mismatch, and every
difference from what the targets assume (§2).
