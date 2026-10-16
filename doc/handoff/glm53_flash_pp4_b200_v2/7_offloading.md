# Prefix KV in DRAM and SSD

The sixth and last milestone, after `6_scheduler.md` passes. The pipeline, the kernels, the HBM prefix cache of
`5_prefix_cache.md` and the scheduler of `6_scheduler.md` stay as they are. Each GPU now also keeps session contexts
in host DRAM and on local SSD. A session evicted from HBM is then read back instead of computed again. The targets
of this file at full memory and at the two memory limits of `1_contract.md` §2.3 are the final targets of the README.

## 1. The workloads

| Run | Trace | Sessions | Arrivals |
|---|---|---|---|
| Tier bring-up | `data/bringup/tier_sessions.csv` | 15 two-round sessions, 5 for each P below, and 75 one-round fillers | one session every 30 s |
| Sessions | `data/sessions_2ps.csv` (`1_contract.md` §2.2) | 2,457 coding-agent sessions | 2 per second for 20 minutes, at most N active; N = 400 and 500 |

**Tier bring-up.** It runs with a KV pool of 1,234,944 tokens, so that HBM cannot keep a session through its
round 1. Each session has two rounds:
- Round 0 is a prompt of P tokens: 32k, 128k or 512k.
- Round 1 comes 20 s after round 0's token, with P cached tokens and 1k computed, as in `5_prefix_cache.md` §1.
- 6 s after the session starts, five fillers of 256k tokens arrive, each a one-round session. Kept in the pool, they
  push the older session out of HBM, so round 1 must find its P tokens in DRAM or SSD.

`data/bringup/make_bringup.py` writes the file, in req-frontend's `text-generation-session-execution-v2` format.

**Sessions** run at full memory (`1_contract.md` §2.2) and with the two memory limits of `1_contract.md` §2.3:

| Memory | KV pool in HBM | DRAM per GPU | SSD per GPU |
|---|---|---|---|
| Full | 61,258,368 tokens (100.18 GB) | 150 GB | 256 GB |
| Tiers | 15,273,856 tokens (25 GB) | 37.5 GB | 256 GB |
| SSD-heavy | 7,632,640 tokens (12.5 GB) | 18.75 GB | 256 GB |

## 2. What runs

**Unchanged:** the pipeline, the scheduler of `6_scheduler.md` §2, and the HBM prefix cache of
`5_prefix_cache.md` §2 (what a session keeps, exact hits, LRU eviction of whole sessions).

**The tiers.** Each GPU has host DRAM (150 GB, read at 50 GB/s) and a local SSD (256 GB, read at 10 GB/s) for prefix
KV (`1_contract.md` §1).

**What a tier keeps.** One entry per session: its latest context, the same as HBM keeps (`5_prefix_cache.md` §2):
- the KV of every DSA layer for the prompt's tokens;
- the KDA state at the prompt's last token, only that one.

A tier hit follows the rule of `5_prefix_cache.md` §2: the whole kept context, or nothing for a round whose prompt
ends inside it.

**Writes.**
- **Write-through.** When a round ends, its context is written to DRAM and to SSD. A later round's write replaces
  the session's entry.
- **Only what is new.** A write moves only what the entry did not already hold: the KV of the round's new tokens,
  plus the new KDA state.
- **In the background.** Writes never stall compute, and a session's write is done before HBM can evict it.
- **Eviction.** Each tier evicts on its own: least recently used first, a whole session at a time. A read refreshes
  its session's place in the order.

**Reads.**
- **When.** A read starts when the scheduler's order (`6_scheduler.md` §2) reaches a waiting round that DRAM or SSD
  holds more of than HBM does. It reads from the fastest tier that holds the longest prefix; since writes go to both
  tiers, that is DRAM whenever DRAM still holds the session.
- **What it moves.** Only the tokens HBM lacks, plus the KDA state.
- **HBM first.** A read starts only if the round's whole footprint fits the pool. It reserves that footprint when it
  starts and holds it until the round is admitted. If the footprint does not fit, admission stops for this
  microbatch, as for any round that does not fit (`6_scheduler.md` §2).
- **Out of the queue.** The round leaves the queue while its read runs, and the order goes on with the rounds behind
  it. Until it is admitted, it is not part of the backlog B (`6_scheduler.md` §2).
- **When it lands.** The context goes into the reserved blocks, and the round is admitted before every queued request,
  overdue ones included (`6_scheduler.md` §2, order (a)), in the order the reads landed. It computes only its new
  tokens.
- **One at a time per tier.** Each GPU's DRAM reads queue one after another at 50 GB/s, and its SSD reads at 10 GB/s.
  The two tiers read at the same time.
- **Copy up.** A read from SSD also puts the session back in DRAM.
- **Read-wait bound.** If a round's read would wait more than 1 s for its tier, the round is passed over: it stays in
  the queue, holds nothing, and the rounds behind it go ahead. The order reaches it again at the next microbatch.
- **Size for the order.** A waiting round's size (`6_scheduler.md` §2) treats what HBM or a tier holds as cached.

**Balanced storage.** The stages hold different layers, so a context's bytes differ by stage:

| Stage | DSA layers | KDA layers | KV per token | KDA state per context |
|---|---|---|---|---|
| 0 | 3 | 9 | 1,635 B | 39.1 MB |
| 1 | 2 | 9 | 1,090 B | 39.1 MB |
| 2 | 3 | 8 | 1,635 B | 34.7 MB |
| 3 | 3 | 8 | 1,635 B | 34.7 MB |
| Mean | | | 1,499 B | 36.9 MB |

- **Own stage.** If each GPU stores and reads only its own stage's bytes, stage 0 carries the most. It fills its
  DRAM and SSD first and takes the longest to read. A context needs all 4 stages, so stage 0 sets the tier's size
  and read time for the whole pipeline: 1,635 B per token plus 39.1 MB per context.
- **Balanced.** The design spreads each context's bytes evenly over the 4 GPUs: the node's DRAM and SSDs form one
  store, a context is striped over the 4 SSDs, and every GPU's link moves a quarter of each read. Each GPU then stores
  and reads the mean: 1,499 B per token plus 36.9 MB per context. A GPU that reads bytes of another stage's layers
  passes them to that stage over NVLink.
- **What it changes.** The tiers hold 9% more tokens: DRAM's 150 GB holds 100.1M tokens' worth instead of 91.7M.
  A read takes 7–8% less time.

**Read time.** For a read of P tokens, on each GPU (balanced):

```
read time = (P × 1,499 B + 36.9 MB) ÷ (50 GB/s for DRAM, 10 GB/s for SSD)
```

With own-stage storage, use 1,635 B and 39.1 MB. Round 1 then takes the read time plus what it takes after a hit in
HBM.

**Built in two steps** on the system of `6_scheduler.md` (step 0):

| Step | Adds | DRAM | SSD |
|---|---|---|---|
| 0 | the design scheduler, HBM only (`6_scheduler.md` step 4) | — | — |
| 1 | DRAM with balanced storage: writes, reads, the read-wait bound | ✓ | — |
| 2 | SSD, striped over the 4 drives: the design | ✓ | ✓ |

Balanced storage comes with the first tier: it decides where every byte is stored and how each read is split, so
adding it later means rebuilding the store. **Own stage**, each GPU keeping only its own stage's bytes, is the
alternative: a build without balanced storage passes against those targets and records the difference (§6).

**What the targets assume.** The numbers in §3 and §4 come from exactly the system above, and also:
- writes cost nothing: they take no link time from reads and no GPU time from compute;
- a read runs at the tier's full bandwidth from its first byte, whatever its size;
- passing bytes over NVLink between stages costs nothing;
- the host takes no time to look up, start or land a read.

A real system that differs in one of these misses or beats the target for that reason: record which (§8).

## 3. Targets: tier bring-up

TTFT in ms, with the read time in parentheses. Round 0 runs its P tokens in microbatches of 4,096 tokens. Round 1
computes its 1k tokens as two microbatches of 512.

| P | Round 0 | Round 1, HBM hit | Round 1, no tier | Step 1, DRAM | Step 2, SSD | Own stage, DRAM | Own stage, SSD |
|---|---|---|---|---|---|---|---|
| 32k | 343.5 | 52.3 | 363.8 | 54.0 (1.72) | 60.9 (8.60) | 54.2 (1.85) | 61.6 (9.26) |
| 128k | 1,123.8 | 53.5 | 1,145.4 | 58.1 (4.67) | 76.8 (23.34) | 58.5 (5.07) | 78.8 (25.34) |
| 512k | 4,653.4 | 57.0 | 4,679.5 | 73.5 (16.46) | 139.3 (82.28) | 74.9 (17.93) | 146.6 (89.63) |

- **HBM hit** is the same session with the full 61.3M-token pool: round 1 finds its context in HBM.
- **No tier** is step 0: round 1 finds nothing and computes P + 1k tokens again.
- **DRAM** runs with DRAM only, **SSD** with SSD only (the DRAM tier off), so that each read comes from that tier.
  The SSD run starts at step 2.
- Round 1 takes the HBM-hit time plus the read time. A 512k context read back from SSD costs 82 ms; computing it
  again costs 4.6 s.

`data/tier_targets.json` holds every case. To regenerate: `data/tier_targets.py` (usage in its docstring).

## 4. Targets: sessions

Over the 20 minutes, by `1_contract.md` §3. Over 60 s is the number of counted rounds whose TTFT exceeds 60 s.
Prefix reads are the prefix read share of `1_contract.md` §3, over the rounds that arrive in [10, 20 min). A run
with more than 500 rounds over 60 s does not keep up: its queue grows to the end, and its read share is left out (—).

**Full memory:**

| Step | N | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Prefix reads HBM / DRAM / SSD | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 400 | 25.6 | 89.4k | 98.55% | 0.15 / 0.81 / 9.85 / 63.7 | 165 | 98 / 0 / 0% | 90 / 87 / 89 / 90% |
| 1 | 400 | 26.8 | 92.7k | 98.92% | 0.15 / 0.50 / 3.93 / 62.7 | 71 | 98 / 1 / 0% | 89 / 86 / 88 / 89% |
| **2** | **400** | **26.8** | **92.8k** | 98.92% | **0.15 / 0.48 / 4.03** / 62.7 | 67 | 98 / 1 / 0% | 89 / 86 / 88 / 89% |
| own stage | 400 | 26.8 | 92.7k | 98.92% | 0.15 / 0.49 / 4.00 / 63.0 | 68 | 98 / 1 / 0% | 89 / 86 / 88 / 89% |
| 0 | 500 | 25.8 | 92.6k | 97.82% | 0.29 / 4.22 / 61.55 / 68.4 | 979 | — | 93 / 88 / 91 / 92% |
| 1 | 500 | 29.0 | 103.3k | 98.89% | 0.23 / 1.96 / 60.78 / 67.1 | 681 | — | 93 / 88 / 91 / 92% |
| 2 | 500 | 29.1 | 103.6k | 98.91% | 0.23 / 1.82 / 60.77 / 66.0 | 683 | — | 93 / 88 / 91 / 92% |
| own stage | 500 | 29.1 | 103.6k | 98.90% | 0.23 / 1.74 / 60.74 / 64.5 | 665 | — | 93 / 88 / 91 / 92% |

**Tiers (15.3M-token pool, 37.5 GB DRAM):**

| Step | N | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Prefix reads HBM / DRAM / SSD | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 400 | 9.0 | 42.6k | 75.31% | 0.94 / 110.96 / 125.81 / 132.5 | 2,477 | — | 97 / 90 / 93 / 93% |
| 1 | 400 | 15.7 | 59.3k | 91.49% | 0.19 / 3.09 / 124.30 / 372.4 | 916 | — | 96 / 91 / 94 / 94% |
| **2** | **400** | **26.7** | **92.7k** | 98.92% | **0.17 / 0.50 / 3.95** / 62.6 | 67 | 48 / 36 / 15% | 89 / 86 / 88 / 89% |
| own stage | 400 | 26.7 | 92.7k | 98.93% | 0.17 / 0.50 / 3.85 / 62.8 | 65 | 48 / 33 / 18% | 89 / 86 / 88 / 89% |
| 0 | 500 | 9.1 | 44.0k | 73.99% | 1.05 / 134.61 / 151.23 / 155.9 | 2,879 | — | 97 / 90 / 93 / 93% |
| 1 | 500 | 13.4 | 56.0k | 87.09% | 0.24 / 10.13 / 145.21 / 158.5 | 1,372 | — | 96 / 90 / 93 / 94% |
| 2 | 500 | 28.9 | 103.3k | 98.92% | 0.26 / 2.20 / 60.75 / 66.9 | 630 | — | 93 / 88 / 91 / 92% |
| own stage | 500 | 28.9 | 102.9k | 98.90% | 0.26 / 2.35 / 60.79 / 66.4 | 655 | — | 93 / 88 / 91 / 92% |

**SSD-heavy (7.6M-token pool, 18.75 GB DRAM):**

| Step | N | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Prefix reads HBM / DRAM / SSD | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 400 | 5.3 | 33.2k | 47.83% | 6.55 / 113.98 / 119.25 / 120.7 | 3,061 | — | 98 / 90 / 93 / 94% |
| 1 | 400 | 10.1 | 45.0k | 78.67% | 0.88 / 86.89 / 121.80 / 125.9 | 2,241 | — | 97 / 90 / 93 / 93% |
| **2** | **400** | **26.7** | **92.6k** | 98.91% | **0.18 / 0.53 / 4.15** / 62.6 | 66 | 20 / 29 / 49% | 89 / 86 / 88 / 89% |
| own stage | 400 | 26.7 | 92.5k | 98.92% | 0.18 / 0.54 / 3.89 / 62.6 | 64 | 20 / 26 / 53% | 89 / 86 / 88 / 89% |
| 0 | 500 | 5.5 | 35.3k | 48.31% | 61.23 / 134.92 / 139.30 / 141.8 | 3,349 | — | 98 / 90 / 93 / 94% |
| 1 | 500 | 9.8 | 46.0k | 76.40% | 1.03 / 120.72 / 149.27 / 153.9 | 2,620 | — | 97 / 90 / 93 / 93% |
| 2 | 500 | 28.7 | 102.4k | 98.91% | 0.29 / 2.55 / 60.77 / 67.7 | 619 | — | 92 / 88 / 91 / 91% |
| own stage | 500 | 28.6 | 102.2k | 98.91% | 0.29 / 2.72 / 60.77 / 68.2 | 636 | — | 92 / 88 / 90 / 91% |

**Step by step:**
- **DRAM (step 1).** At full memory and N = 400 it brings the HBM-only system of `6_scheduler.md` back within the
  SLO: 89.4k → 92.7k new tokens/s, TTFT p50 / p90 / p99 0.15 / 0.81 / 9.85 → 0.15 / 0.50 / 3.93 s. Evicted sessions
  are read back instead of computed again. With the memory limits, DRAM is too small on its own, and the pipeline
  stays overloaded.
- **SSD (step 2).** Every kept context is found in every memory setting: the 1.08% of prefix tokens that miss belong
  to the rounds whose prompt ends inside the previous one, computed again whole (`5_prefix_cache.md` §2). N = 400
  meets the SLO: TTFT p50 / p90 / p99 0.15 / 0.48 / 4.03 s at full memory, 0.17 / 0.50 / 3.95 s and
  0.18 / 0.53 / 4.15 s with the limits. With the limits, throughput stays within 0.2% of full memory: SSD serves up to
  49% of the prefix tokens, and the reads cost no compute.
- **N = 500 does not keep up at any step:** 619–683 rounds wait more than 60 s at step 2. The rounds whose prompt
  ends inside the previous one still add about 21k computed tokens/s, and the stages are 93% busy. The SLO holds up
  to N = 410 at full memory (p99 5.23 s); at N = 420, p99 is 7.79 s.
- **Own stage instead of balanced.** With the memory limits, the DRAM share falls by 3.0–3.6 points, because DRAM
  holds 8% fewer tokens. TTFT changes by −6% to −1% at p99 and by at most 0.02 s at p50 and p90. On these runs the
  tier links stay far from full (below), so faster reads gain little.
- **The read-wait bound** never acts in these runs: no read waits more than 0.6 s for its tier. It is there for a tier
  that falls behind.
- **Over 60 s.** In every run that keeps up at N = 400, about 65 rounds (0.2%) wait about 60 s. About 11 are the
  first rounds of the sessions with the largest prompts, with no prefix; about 55 are rounds whose prompt ends inside
  the previous one, computed again whole, a median of 129k tokens. Shortest first serves them last, and
  force-schedule serves them at 60 s (`6_scheduler.md` §5).

**Tier traffic** of the design (step 2) at N = 400, per GPU, averaged over the 20 minutes:

| Memory | DRAM writes | DRAM reads | SSD writes | SSD reads | DRAM used | SSD used |
|---|---|---|---|---|---|---|
| Full | 1.13 GB/s | 0.02 GB/s | 1.13 GB/s | 0.00 GB/s | 150 GB (full) | 207 GB |
| Tiers | 1.52 GB/s | 1.04 GB/s | 1.12 GB/s | 0.38 GB/s | 37.5 GB (full) | 207 GB |
| SSD-heavy | 2.50 GB/s | 1.06 GB/s | 1.12 GB/s | 1.35 GB/s | 18.75 GB (full) | 206 GB |

- **Writes are mostly KDA states.** Each round writes a new 36.9 MB state, 0.99 GB/s at 26.8 rounds/s. The KV of the
  new tokens adds only 0.14 GB/s.
- **DRAM writes** also include the copies of SSD reads.
- **Reads** use at most 2% of DRAM's bandwidth and 14% of SSD's on average. Over 10-s windows SSD reads peak at
  3.3 GB/s and writes at 1.7 GB/s (SSD-heavy). A read waits at most 0.6 s for the SSD, under the 1 s bound.
- **SSD space.** After 20 minutes it holds 206–207 GB per GPU at N = 400 (231 GB at full memory and N = 500): every
  context the run wrote, with no eviction. With less, SSD evicts and those sessions are computed again. TTFT p50 / p90
  / p99 (s) at N = 400:

| SSD per GPU | Full | Tiers | SSD-heavy |
|---|---|---|---|
| 256 GB | 0.15 / 0.48 / 4.03 | 0.17 / 0.50 / 3.95 | 0.18 / 0.53 / 4.15 |
| 192 GB | 0.15 / 0.48 / 4.03 | 0.17 / 0.50 / 3.95 | 0.18 / 0.53 / 4.15 |
| 128 GB | 0.15 / 0.50 / 3.93 | 0.17 / 0.52 / 3.88 | 0.18 / 0.55 / 4.50 |
| 64 GB | 0.15 / 0.50 / 3.93 | 0.18 / 1.50 / 60.35 | 0.19 / 1.83 / 57.12 |

  The SLO holds down to 128 GB; at 64 GB the two memory limits miss it by far (p99 57–60 s). At full memory, an SSD
  no larger than DRAM's 150 GB holds nothing DRAM does not, so 128 GB and 64 GB give the DRAM-only numbers.

`data/session_targets.json` holds every run (settings `<memory>_<variant>` with memory `full`, `h25d37`, `h12d19`
(Tiers is `h25d37`, SSD-heavy `h12d19`) and variant `hbm` = step 0, `dram` = step 1, `design` = step 2, `unbal` = own
stage; also `dram_unbal`, DRAM only with own-stage storage, `nobound`, the design without the read-wait bound, and
`ssd192`, `ssd128`, `ssd64`, `ssd32`, the design with that SSD per GPU). The `design` settings also hold N = 410 to
490 in steps of 10. To regenerate: `data/session_targets.py`; the tier traffic: `data/tier_traffic.py` (usage in each
docstring).
The microbatch check of `5_prefix_cache.md` §4 applies unchanged.

## 5. How to run

- The setup of `1_contract.md` §1 with the scheduler of `6_scheduler.md` and the tiers of §2, at the step's tiers
  and storage.
- **Start state.** Every run starts with empty HBM, DRAM and SSD. Clear the SSD store between runs; a context left
  from an earlier run makes hits that the targets do not have.
- **Memory.** Limit the KV pool by its token count (`1_contract.md` §2.3), and DRAM and SSD by their bytes per GPU.
- **Tier bring-up.** For each step, replay `data/bringup/tier_sessions.csv` on a freshly started server with the
  1,234,944-token pool and no `max_concurrency`. Run it once with DRAM only (150 GB), and from step 2 on once with SSD
  only (256 GB, the DRAM tier off). For the two reference columns of §3, run it once at step 0 (no tier: round 1
  computes again) and once at step 0 with the full 61.3M-token pool (round 1 hits HBM). Log every read: its round, its
  tier, its tokens, when it starts and when it lands.
- **Sessions.** For each step, N = 400 and 500 at each memory setting, each on a freshly started server, with the
  microbatch log of `5_prefix_cache.md` §5 and the read log above. Log each round's prefix source (HBM, DRAM, SSD or
  none) and tokens, every eviction from each tier, and every round passed over by the read-wait bound.
- **Check the output.** For round 1 of every bring-up case, the token, or the logits, against the reference computing
  the whole prompt with no cache (`1_contract.md` §4). The SSD-heavy session runs also give many rounds whose prefix
  came from DRAM or SSD; check a sample of them the same way.

## 6. Pass rule

**Tier bring-up**, per case and step:
- **Hit:** round 1's cached tokens equal P exactly, read from the tier the run allows.
- **Read time** within ±10% of §3.
- **Round 1 TTFT** within ±10%; round 0 TTFT within ±10%.

**Sessions**, per step, memory setting and N, as in `5_prefix_cache.md` §6:
- rounds/s and new tokens/s within ±10%;
- TTFT p50 and p90 within ±10%, p99 within ±25%;
- prefix hit share and microbatches as there;
- prefix read share: each tier within ±5 points;
- record the worst TTFT and the rounds over 60 s.
- Own stage and step 2 differ by less than these bounds. A system that keeps each stage's bytes on its own GPU
  passes against the own-stage targets; record that it does.

**The final targets** are step 2: at full memory those of the README "Sessions" table, at Tiers and SSD-heavy those
of its "Cache tiers" table.

If the bring-up passes and the sessions do not, compare the read log with the targets' rounds: which rounds missed,
which read, how long the reads took, and whether writes delayed them.

## 7. Where these targets may be off

- **Writes are free in the target.** A real write of 1.1 GB/s per GPU (§4) shares the host links and the SSD with
  the reads. Writes that slow the reads, or a session evicted from HBM before its write is done, miss the target.
- **The KDA state's size.** It is most of the write traffic and a fixed part of every read. A state stored larger
  than 36.9 MB per GPU per context (padded to blocks, or one per block) costs bandwidth and tier space in proportion.
- **Read bandwidth from the first byte.** The target reads a 32k context (86 MB) at full speed. A real SSD read that
  size may not reach 10 GB/s, so short reads take longer than the formula; record the measured read time per size.
- **Reads are short and rare.** The tier links run at most a seventh full on average, so a slower link shows up as
  later reads, not as lost throughput. Check the read times before the link bandwidth.
- **Balanced storage passes bytes over NVLink.** The target counts no time for it. It is a few tens of MB per read,
  next to an NVLink of hundreds of GB/s.
- **SSD space.** The run fills up to 207 of 256 GB per GPU at N = 400, and 231 GB at full memory and N = 500. A
  context stored larger than the target's (the KDA state above, or per-block overhead) fills the SSD sooner; once it
  evicts, the evicted sessions are computed again (§4: at 64 GB the memory limits miss the SLO).
- **Eviction.** The target evicts whole sessions from each tier, least recently used first. Keeping parts of a
  session, or another order, changes the read shares and moves the Tiers and SSD-heavy numbers.
- **Host time.** Lookups and starting and landing reads take host time on every microbatch; the target has none.
- The caveats of `6_scheduler.md` §8 and `5_prefix_cache.md` §7 still hold.

## 8. What to record

**Tier bring-up** (`measured/tier_bringup.jsonl`), one line per session and run:

```json
{"step": 2, "tier": "ssd", "prefix_tokens": 131072, "session": 30, "round0_ttft_ms": 1131.0, "round1_ttft_ms": 78.1,
 "round1_cached_tokens": 131072, "read_tier": "ssd", "read_ms": 24.0, "token_matches_reference": true}
```

**Each run** (`measured/runs/tier_bringup_step<k>_<tier>/` and `tiers_step<k>_<memory>_n<N>/`, `1_contract.md` §6):
- what `5_prefix_cache.md` §8 lists, including the output of `data/metrics.py`;
- `reads.jsonl` in the format of `1_contract.md` §3;
- per tier: the bytes written and read, the evictions, and the space used at the end;
- the rounds passed over by the read-wait bound.

**Summary** (`measured/offloading.md`):
- the tables of §3 and §4 with the measured numbers and their ratios beside them;
- the tier traffic table of §4, measured;
- for each run that misses: whether the reads, the writes, the cache or the scheduling explains it.

**Gaps** (`issues.md`): every case above its pass rule, every hit shorter than P, every output mismatch after a tier
read, and every difference from §2.
