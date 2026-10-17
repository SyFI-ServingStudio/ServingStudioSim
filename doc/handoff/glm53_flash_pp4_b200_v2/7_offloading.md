# Prefix KV in DRAM and SSD

The sixth and last milestone, after `6_scheduler.md` passes. The pipeline, the kernels, the HBM prefix cache of
`5_prefix_cache.md` and the scheduler of `6_scheduler.md` stay as they are. Each GPU now also keeps session contexts
in host DRAM and on local SSD, each tier holding what the tier above it evicts. A session evicted from HBM is then
read back instead of computed again. The targets of this file at full memory and at the two memory limits of
`1_contract.md` §2.3 are the final targets of the README.

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
| Full | 61,258,368 tokens (100.18 GB) | 50 GB | 128 GB |
| Tiers | 12,220,800 tokens (20 GB) | 20 GB | 128 GB |
| SSD-heavy | 7,632,640 tokens (12.5 GB) | 12.5 GB | 128 GB |

## 2. What runs

**Unchanged:** the pipeline, the scheduler of `6_scheduler.md` §2, and the HBM prefix cache of
`5_prefix_cache.md` §2 (what a session keeps, exact hits, LRU eviction of whole sessions).

**The tiers.** Each GPU has host DRAM (50 GB, read at 50 GB/s) and a local SSD (128 GB, read at 10 GB/s) for prefix
KV (`1_contract.md` §1).

**What a tier keeps.** One entry per session: a context of the session as it was when it left the tier above, in
the form HBM keeps it (`5_prefix_cache.md` §2):
- the KV of every DSA layer for the prompt's tokens;
- the KDA state at the prompt's last token, only that one.

An entry is always a prefix of the session's latest context, but it may be older than the context HBM or a faster
tier holds. A tier hit follows the rule of `5_prefix_cache.md` §2: the whole kept context, or nothing for a round
whose prompt ends inside it. A round whose prompt ends inside its latest context can therefore still resume from an
older entry that it extends.

**Writes: each tier is the victim tier of the one above (HBM -> DRAM -> SSD).**
- **DRAM takes what HBM evicts.** A finished round's context stays in HBM. When HBM evicts a session, or cannot keep
  its context, the context is written to DRAM.
- **SSD takes what DRAM evicts (write-back).** When DRAM evicts a session, or a context is larger than DRAM, the
  context is written to SSD. Nothing is written to SSD when a round ends.
- **Only what is new.** A write moves only what the tier's entry for the session lacks: the KV of the tokens past
  the entry, plus the new KDA state. A context the tier already holds whole is not written again; it only refreshes
  the session's place in the order.
- **Cut at the next round.** When a session's next round is admitted, each tier cuts its entry to what that round
  can resume from: it keeps an entry the round extends and drops one the round's prompt ends inside.
- **In the background.** Writes never stall compute. An evicted context is written before its blocks are reused.
- **Eviction.** Each tier evicts on its own: least recently used first, a whole session at a time. A read refreshes
  its session's place in the order.

**Reads.**
- **When.** A read starts when the scheduler's order (`6_scheduler.md` §2) reaches a waiting round that DRAM or SSD
  holds more of than HBM does. It reads from the tier whose entry gives the round the longest prefix, DRAM on a tie.
  A session that left HBM is in DRAM until DRAM evicts it, then on SSD; after a read, both may hold it (below).
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
- **Clean copies.** A read leaves the entry it read in its tier and refreshes it; it writes nothing to a faster tier.
  The context is in HBM now. When HBM evicts it again, DRAM writes what it lacks; when DRAM evicts it, SSD, which
  may still hold it, writes only what it lacks, or nothing.
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
- **What it changes.** The tiers hold 9% more tokens: DRAM's 50 GB holds 33.4M tokens' worth instead of 30.6M.
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
| 1 | 400 | 26.7 | 92.5k | 98.90% | 0.15 / 0.53 / 4.60 / 62.9 | 81 | 98 / 1 / 0% | 89 / 86 / 88 / 89% |
| **2** | **400** | **26.8** | **92.9k** | 98.93% | **0.15 / 0.50 / 3.80** / 62.7 | 67 | 98 / 1 / 0% | 89 / 86 / 88 / 89% |
| own stage | 400 | 26.8 | 92.9k | 98.94% | 0.15 / 0.49 / 3.82 / 62.7 | 65 | 98 / 1 / 0% | 89 / 86 / 88 / 89% |
| 0 | 500 | 25.8 | 92.6k | 97.82% | 0.29 / 4.22 / 61.55 / 68.4 | 979 | — | 93 / 88 / 91 / 92% |
| 1 | 500 | 29.0 | 102.9k | 98.86% | 0.24 / 2.05 / 60.83 / 67.2 | 707 | — | 93 / 88 / 91 / 92% |
| 2 | 500 | 29.6 | 104.8k | 99.01% | 0.23 / 1.55 / 60.69 / 65.6 | 628 | — | 93 / 88 / 91 / 92% |
| own stage | 500 | 29.5 | 104.9k | 99.01% | 0.23 / 1.72 / 60.67 / 65.7 | 625 | — | 93 / 88 / 91 / 92% |

**Tiers (12.2M-token pool, 20 GB DRAM):**

| Step | N | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Prefix reads HBM / DRAM / SSD | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 400 | 6.3 | 36.2k | 60.02% | 1.45 / 116.00 / 120.52 / 123.8 | 3,035 | — | 97 / 90 / 93 / 93% |
| 1 | 400 | 12.3 | 50.3k | 85.24% | 0.35 / 60.69 / 131.75 / 143.2 | 1,553 | — | 97 / 91 / 94 / 94% |
| **2** | **400** | **27.3** | **94.4k** | 99.18% | **0.17 / 0.39 / 2.72** / 62.6 | 29 | 39 / 31 / 28% | 88 / 85 / 87 / 88% |
| own stage | 400 | 27.2 | 94.2k | 99.15% | 0.17 / 0.40 / 2.74 / 62.6 | 33 | 39 / 30 / 30% | 88 / 85 / 87 / 88% |
| 0 | 500 | 6.5 | 38.2k | 60.50% | 1.81 / 135.35 / 141.87 / 145.0 | 3,334 | — | 97 / 90 / 93 / 93% |
| 1 | 500 | 11.4 | 50.6k | 82.47% | 0.39 / 100.94 / 161.76 / 166.3 | 1,992 | — | 97 / 91 / 93 / 94% |
| 2 | 500 | 29.4 | 104.5k | 99.03% | 0.26 / 1.99 / 60.69 / 65.6 | 590 | — | 92 / 88 / 91 / 92% |
| own stage | 500 | 29.4 | 104.6k | 99.02% | 0.27 / 2.22 / 60.68 / 64.5 | 589 | — | 92 / 88 / 91 / 92% |

**SSD-heavy (7.6M-token pool, 12.5 GB DRAM):**

| Step | N | Rounds/s | New tokens/s | Prefix hit share | TTFT p50 / p90 / p99 / max (s) | Over 60 s | Prefix reads HBM / DRAM / SSD | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
| 0 | 400 | 5.3 | 33.2k | 47.83% | 6.55 / 113.98 / 119.25 / 120.7 | 3,061 | — | 98 / 90 / 93 / 94% |
| 1 | 400 | 7.5 | 39.2k | 68.79% | 1.25 / 105.88 / 118.79 / 121.9 | 2,754 | — | 97 / 90 / 93 / 93% |
| **2** | **400** | **26.9** | **93.2k** | 99.03% | **0.18 / 0.47 / 3.39** / 63.1 | 48 | 20 / 28 / 50% | 88 / 86 / 88 / 89% |
| own stage | 400 | 26.8 | 92.9k | 99.00% | 0.18 / 0.51 / 4.10 / 62.7 | 57 | 20 / 27 / 52% | 88 / 86 / 88 / 89% |
| 0 | 500 | 5.5 | 35.3k | 48.31% | 61.23 / 134.92 / 139.30 / 141.8 | 3,349 | — | 98 / 90 / 93 / 94% |
| 1 | 500 | 7.6 | 40.6k | 67.90% | 1.40 / 124.89 / 135.31 / 138.5 | 3,156 | — | 97 / 90 / 93 / 93% |
| 2 | 500 | 28.9 | 102.9k | 98.95% | 0.29 / 2.56 / 60.79 / 66.5 | 635 | — | 92 / 88 / 90 / 91% |
| own stage | 500 | 28.8 | 102.4k | 98.92% | 0.29 / 2.60 / 60.78 / 65.6 | 624 | — | 92 / 88 / 90 / 91% |

The two memory limits split the prefix reads at N = 400 about evenly over HBM, DRAM and SSD (Tiers) and put half
of them on SSD (SSD-heavy). At N = 500 more sessions share the same HBM, and the reads move down a tier: HBM / DRAM /
SSD 26 / 34 / 38% (Tiers) and 8 / 31 / 60% (SSD-heavy).

**Step by step:**
- **DRAM (step 1).** At full memory and N = 400 it brings the HBM-only system of `6_scheduler.md` back within the
  SLO: 89.4k → 92.5k new tokens/s, TTFT p50 / p90 / p99 0.15 / 0.81 / 9.85 → 0.15 / 0.53 / 4.60 s. Evicted sessions
  are read back instead of computed again. With the memory limits, DRAM is too small on its own, and the pipeline
  stays overloaded.
- **SSD (step 2).** A context that left HBM is found again in DRAM or SSD; the sessions SSD evicts have almost all
  ended. N = 400 meets the SLO in every memory
  setting: TTFT p50 / p90 / p99 0.15 / 0.50 / 3.80 s at full memory, 0.17 / 0.39 / 2.72 s (Tiers) and
  0.18 / 0.47 / 3.39 s (SSD-heavy). SSD serves up to half of the prefix tokens, and the reads cost no compute.
- **Older entries serve shrunk prompts.** With the limits, a session's context moves down through DRAM and SSD, and
  a tier keeps it as it was when it left the tier above. A round whose prompt ends inside its latest context finds
  nothing in HBM (`5_prefix_cache.md` §2) but can resume from such an older entry when its prompt extends it. The
  prefix hit share therefore rises above full memory's (99.18% and 99.03% against 98.93%), and Tiers beats full
  memory at N = 400: 94.4k against 92.9k new tokens/s, p99 2.72 against 3.80 s, 29 against 67 rounds over 60 s.
- **N = 500 does not keep up at any step:** 590–635 rounds wait more than 60 s at step 2, and the stages are 92–93%
  busy. The SLO holds up to N = 410 at full memory (p99 4.95 s) and with SSD-heavy (4.90 s), and up to N = 420 with
  Tiers (5.89 s); one step of 10 further, p99 is 7.72, 7.22 and 7.97 s.
- **Own stage instead of balanced.** The DRAM share falls by 1–2 points, because DRAM holds 8% fewer tokens. TTFT
  changes by at most 0.04 s at p50 and p90; p99 rises by under 1% at full memory and with Tiers and by 21% with SSD-heavy
  (3.39 → 4.10 s).
- **The read-wait bound** never acts at N = 400: no read waits more than 0.6 s for its tier. At N = 500 with Tiers,
  SSD reads wait up to 0.99 s and the bound passes a few rounds over (p90 1.99 s against 1.95 s without it). It is
  there for a tier that falls behind.
- **Over 60 s.** At N = 400, 29–67 rounds (0.1–0.2%) wait about 60 s: the first rounds of the sessions with the
  largest prompts, with no prefix, and rounds whose prompt ends inside the previous one with no older entry to
  resume from, computed again whole. Shortest first serves them last, and force-schedule serves them at 60 s
  (`6_scheduler.md` §5).

**Tier traffic** of the design (step 2) at N = 400, per GPU, averaged over the 20 minutes:

| Memory | DRAM writes | DRAM reads | SSD writes | SSD reads | DRAM used | SSD used |
|---|---|---|---|---|---|---|
| Full | 0.13 GB/s | 0.02 GB/s | 0.08 GB/s | 0.00 GB/s | 50 GB (full) | 100 GB |
| Tiers | 1.35 GB/s | 1.02 GB/s | 0.41 GB/s | 0.78 GB/s | 20 GB (full) | 128 GB (full) |
| SSD-heavy | 1.99 GB/s | 1.02 GB/s | 0.58 GB/s | 1.42 GB/s | 12.5 GB (full) | 128 GB (full) |

- **DRAM writes** are what HBM evicts: mostly the KV of contexts DRAM did not hold (65–72% of the bytes), the rest
  KDA states. **SSD writes** are what DRAM evicts; a context SSD still holds from an earlier eviction adds only its
  new tokens, so with the limits 62–72% of SSD's bytes are KDA states. At full memory HBM evicts little, and SSD receives 0.08 GB/s.
- **Reads** use at most 2% of DRAM's bandwidth and 14% of SSD's on average. Over 10-s windows SSD reads peak at
  3.4 GB/s, DRAM writes at 4.1 GB/s and SSD writes at 1.0 GB/s (SSD-heavy). A read waits at most 0.6 s for its tier.
- **SSD space.** With the limits, SSD fills its 128 GB per GPU and evicts (about 650 sessions at N = 400). With
  256 GB it would hold 204–206 GB at N = 400 and 230–231 GB at N = 500; at full memory it holds 100 GB at N = 400.
  TTFT p50 / p90 / p99 (s) at N = 400:

| SSD per GPU | Full | Tiers | SSD-heavy |
|---|---|---|---|
| 256 GB | 0.15 / 0.50 / 3.80 | 0.17 / 0.39 / 2.72 | 0.18 / 0.46 / 3.03 |
| 128 GB | 0.15 / 0.50 / 3.80 | 0.17 / 0.39 / 2.72 | 0.18 / 0.47 / 3.39 |
| 64 GB | 0.15 / 0.50 / 3.80 | 0.18 / 1.08 / 14.69 | 0.19 / 1.30 / 29.19 |

  The SLO holds at 128 GB; at 64 GB the two memory limits miss it (p99 15–29 s), since the evicted sessions are
  computed again. At full memory SSD is hardly read, and its size does not change N = 400.

`data/session_targets.json` holds every run (settings `<memory>_<variant>` with memory `full`, `h20d20`, `h12d12`
(Tiers is `h20d20`, SSD-heavy `h12d12`) and variant `hbm` = step 0, `dram` = step 1, `design` = step 2, `unbal` = own
stage; also `dram_unbal`, DRAM only with own-stage storage, `nobound`, the design without the read-wait bound, and
`ssd256`, `ssd64`, the design with that SSD per GPU). The `design` settings also hold N = 410 to 490 in steps of 10.
To regenerate: `data/session_targets.py`; the tier traffic: `data/tier_traffic.py` (usage in each docstring).
The microbatch check of `5_prefix_cache.md` §4 applies unchanged.

## 5. How to run

- The setup of `1_contract.md` §1 with the scheduler of `6_scheduler.md` and the tiers of §2, at the step's tiers
  and storage.
- **Start state.** Every run starts with empty HBM, DRAM and SSD. Clear the SSD store between runs; a context left
  from an earlier run makes hits that the targets do not have.
- **Memory.** Limit the KV pool by its token count (`1_contract.md` §2.3), and DRAM and SSD by their bytes per GPU.
- **Tier bring-up.** For each step, replay `data/bringup/tier_sessions.csv` on a freshly started server with the
  1,234,944-token pool and no `max_concurrency`. Run it once with DRAM only (50 GB), and from step 2 on once with SSD
  only (128 GB, the DRAM tier off). For the two reference columns of §3, run it once at step 0 (no tier: round 1
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

- **Writes are free in the target.** With the memory limits at N = 400, DRAM takes 1.35–1.99 GB/s per GPU of what
  HBM evicts and SSD 0.41–0.58 GB/s of what DRAM evicts, in 10-s bursts of up to 4.1 and 1.0 GB/s (§4). They share
  the host links and the SSD with the reads. Writes that slow the reads, or HBM blocks reused before the evicted
  context is written to DRAM, miss the target.
- **Victim writes sit on the eviction path.** An eviction from HBM must copy the context out before its blocks take
  new KV, and an eviction from DRAM before its pages are reused. A system that makes room only after the copy
  lands admits later than the target.
- **The KDA state's size.** It is most of SSD's write traffic and a fixed part of every read. A state stored larger
  than 36.9 MB per GPU per context (padded to blocks, or one per block) costs bandwidth and tier space in proportion.
- **Read bandwidth from the first byte.** The target reads a 32k context (86 MB) at full speed. A real SSD read that
  size may not reach 10 GB/s, so short reads take longer than the formula; record the measured read time per size.
- **Reads are short and rare.** The tier links run at most a seventh full on average, so a slower link shows up as
  later reads, not as lost throughput. Check the read times before the link bandwidth.
- **Balanced storage passes bytes over NVLink.** The target counts no time for it. It is a few tens of MB per read,
  next to an NVLink of hundreds of GB/s.
- **SSD space.** With the memory limits SSD fills its 128 GB per GPU and evicts, almost only sessions that have
  ended. A context stored larger than the target's (the KDA state above, or per-block overhead) fills the SSD sooner
  and evicts sessions still in use, which are then computed again (§4: at 64 GB the memory limits miss the SLO).
- **Older entries.** A tier keeps a session's context as it was when it left the tier above, and a round whose prompt
  ends inside the latest context can resume from it (§4). A system that overwrites or drops a tier's entry whenever
  the session's context changes loses these hits and moves toward the full-memory hit share.
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
