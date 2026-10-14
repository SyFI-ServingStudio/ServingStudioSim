# Contract: what the real setup must match

A real measurement is comparable with a target only when everything in this file matches. If something here cannot
match, record it in `issues.md` before comparing, and say which target it affects.

## 1. Fixed setup

| | Must be |
|---|---|
| Model | `nvidia/GLM-5.3-Flash-NVFP4`, snapshot `da920bb0b9f4a06727223a349e55468e38352348`. NVFP4 dense FFN and routed experts; BF16 attention, routers, shared expert and `lm_head`; FP8 KV cache. MTP and speculative decoding off. |
| Context length | `max_model_len` 1,048,576 tokens. |
| GPUs | 4 × NVIDIA B200 (180 GB) in one NVLink domain. |
| HBM per GPU | 149.84 GB for weights and KV: 85% of the B200's 183,359 MiB (163.4 GB), less about 13.6 GB of activations and CUDA-graph workspace, as measured on a B200 serving process. The largest stage's weights take 49.66 GB; 100.18 GB is left for KV. |
| Host DRAM for prefix KV | 150 GB per GPU at 50 GB/s (600 GB for the node). |
| Local SSD for prefix KV | 256 GB per GPU at 10 GB/s (1 TB and 40 GB/s for the node). |
| Decode | Not on these GPUs. Every request asks for one output token (§2.2). |

**KV pool.** The target pool is 61.3M tokens (61,258,368). It is allocated in blocks of 8,576 tokens. A block holds
either 8,576 tokens of DSA KV or one KDA layer's state: 8,576 is the smallest multiple of 128 tokens whose DSA page
(545 B per token per layer) holds one KDA layer's state. On the stages with 3 DSA layers a block takes 8,576 × 1,635
B, so 100.18 GB holds 7,144 blocks; one is kept back as a null block, and 7,143 × 8,576 = 61,258,368. Each request,
running or kept, also holds 6 blocks (51,456 tokens' worth) for its KDA states, whatever its length: one per KDA cache
group (5 groups: stage 1 has 9 KDA layers to place in 2 DSA layers' tensors, ⌈9 / 2⌉ = 5) and one scratch block.
Reproduce this footprint; a smaller one is a difference to record (§6). A smaller pool is a gap to record and shrink;
activation and workspace peaks have to fit beside it. A larger pool is allowed: it raises the HBM hit rate, and the
run may then beat the target. The cache-tier benchmark (§2.3) sets a smaller pool on purpose.

## 2. Workloads and how to replay them

Three benchmarks, in the order the implementer meets them. None needs a warm-up; the session benchmarks run 20
minutes from an empty cache.

| | Workload | Memory | Exercises |
|---|---|---|---|
| §2.1 Bring-up | one prompt length at a time | full | per-stage time, pipeline throughput |
| §2.2 Sessions | coding-agent sessions, 20 min | full | scheduler, prefix cache in HBM |
| §2.3 Cache tiers | the same sessions, 20 min | KV pool and DRAM limited | prefix reads from DRAM and SSD |

### 2.1 Bring-up: one prompt length

For each length L in 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k (L = 1,024 · 2^k tokens):
- **One request.** A single prompt of L fresh tokens, 1 output token, no prefix. Run it alone, as one microbatch:
  the microbatch budget must hold the whole prompt.
- **Saturated.** max(128, 2^24 / L) such requests, all arriving at t = 0, no prefix reuse. Use the plain scheduler of
  the README: microbatches filled to 8,192 tokens in arrival order, longer prompts cut into 8k chunks.

### 2.2 Sessions: 20 minutes

**The trace.** `data/sessions_2ps.csv`: coding-agent sessions that arrive at 2 sessions/s for 20 minutes.

| File | Sessions | Rounds | Longest prompt | sha256 |
|---|---|---|---|---|
| `data/sessions_2ps.csv` | 2,457 | 207,441 | 999,888 tokens | `a1a1e78b231b7e5bbe02b7e6db7a83cd85d2e9de98262b89a70263c274eb3e72` |

To rebuild it from a ServingStudioSim checkout at the frozen commit, with this package's `data/make_trace.py`:

```bash
# 1. The session corpus (4,281 sessions, 357,161 rounds): see trace/README.md "Regenerating".
#    trace/tracelab_preserving.csv
# 2. Add the decode time of each round's outputs (80 tokens/s) to the wait that follows it.
uv run python trace/session_decode_wait.py trace/tracelab_preserving.csv mono_sessions_d80.csv --decode-tok-s 80
#    sha256 2a5fb71550179d9f7972d02e185dd77237bb76f1d27129b092efa3caa166ce49
# 3. Keep the sessions that arrive in the first 20 minutes at 2 sessions/s; one output token per round.
uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/make_trace.py mono_sessions_d80.csv sessions_2ps.csv --sessions-per-s 2
```

Step 3 writes `sessions_2ps.manifest.json`; compare it with the one in `data/`.

**CSV columns.** `request_id` (`session_X_round_Y`), `session_id`, `round_idx`, `arrival_time_ms`, `prefix_len`,
`input_len`, `output_len`, `tool_wait_after_ms`.
- `arrival_time_ms` is the session's arrival, repeated on each of its rows.
- `prefix_len` is the previous round's prompt plus its one output token. `input_len` is the new prompt tokens.
- `output_len` is 1 on every row. Decode runs elsewhere; its time is already in `tool_wait_after_ms`.

**Concurrency.** N sessions at most are active at once: N = 400 and N = 500, one run each. A session arriving while N
are active waits for one to finish, first come first served. N counts sessions, including those in a tool wait,
not requests in flight.

**Replay with req-frontend** (`alignment/load_generator/req-frontend`), on the setup of §1:

```yaml
input:
  trace: doc/handoff/glm53_flash_pp4_b200_v1/data/sessions_2ps.csv
  format: text-generation-session-execution-v2
corpus:
  text_file: <path>/enwik9      # 1,000,000,000 bytes, md5 e206c3450ac99950df65bf70ef61a12d
  tokenizer: <checkpoint snapshot>/tokenizer.json
server:
  backend: openai               # or vllm-tokens / sglang-tokens; see below
  base_url: http://<host>:<port>/v1
  model: <served model name>
  temperature: 0
replay:
  arrival_mode: trace-timed
  max_concurrency: 400          # or 500
  context:
    max_model_len: 1048576
output:
  directory: measured/runs/<run>/
```

- **Text.** enwik9 (the first 10^9 bytes of an English Wikipedia dump, as published for the Hutter Prize). The
  measured expert routing behind the MoE targets comes from enwik9 prompts, so other text changes the routing and the
  MoE time.
- **Token IDs.** Every backend sends prompts as token IDs. The next round must carry the real output token IDs: with
  `openai`, the server must honour `return_token_ids`; otherwise use a token-ID backend. req-frontend reports any
  shortfall as `prefix_shortfall_tokens`, which must be 0.

- Run the client under `timeout 20m`. Rounds not answered by then are not counted (§3).
- A session starts at its `arrival_time_ms`, or when a slot frees if N are active. Round k+1 is sent
  `tool_wait_after_ms` of round k after round k's token arrives.
- The prompt of round k+1 is the prompt of round k, its real output token, then `input_len` new tokens. The client
  builds the new tokens from enwik9, deterministically per session. The server never sees the other output
  tokens of the original round; they arrive as new tokens of the next round.

**Start state.** Every run starts with empty caches: nothing in HBM, DRAM or SSD from an earlier run.

### 2.3 Cache tiers: 20 minutes, smaller memory

The same trace, N and replay as §2.2, with less memory for prefix KV. At full size, the context held by the active
sessions after 20 minutes (about 40–50M tokens) still fits in HBM, so DRAM and SSD would carry almost nothing.

| Variant | KV pool in HBM | DRAM per GPU | SSD per GPU |
|---|---|---|---|
| Tiers | 15,273,856 tokens (25 GB per GPU) | 37.5 GB | 256 GB, unchanged |
| SSD-heavy | 7,632,640 tokens (12.5 GB per GPU) | 18.75 GB | 256 GB, unchanged |

- Limit the pool by its token count, not by a memory fraction: the count is what the targets depend on.
- Bandwidths stay as in §1.
- Leave the freed HBM unused. Do not hand it to larger batches or other buffers.

## 3. Metrics

| Metric | Definition |
|---|---|
| TTFT | First-token time − submit time, per request (per round in §2.2–2.3). The submit time is when the client sends the request, so time spent waiting for a session slot is not included. |
| Stage compute time (§2.1, one request) | GPU time of each stage for the microbatch, without the stage-to-stage sends. |
| Tokens/s (§2.1, saturated) | Prompt tokens in microbatches that start on stage 0 inside the middle 80% of the run, ÷ that window. The run spans from the first stage-0 start to the last stage's last finish. |
| Counted rounds (§2.2–2.3) | t = 0 at the first request's submit. A round counts when its first token arrives in [0, 20 min). |
| Rounds/s | Counted rounds ÷ 1,200 s. |
| New tokens/s | Sum of `input_len` over counted rounds ÷ 1,200 s: the prompt tokens the workload adds, however much of each prompt was cached. The main throughput number. |
| TTFT percentiles | p50, p90 and p99 over counted rounds. |
| SLO | TTFT p99 ≤ 6 s. |
| Prefix read share (§2.3) | Prefix tokens served from HBM, DRAM and SSD, ÷ prefix tokens, over the rounds that arrive in [10, 20 min). |
| Stage busy | Fraction of time each stage's GPU computes: over the middle 80% in §2.1, over [0, 20 min) in §2.2–2.3. |

**Server logs.** The client sees only TTFT. The other metrics come from two server logs, with times in ms from t = 0
(the first request's submit):
- `microbatches.jsonl`, one line per (microbatch, stage):
  `{"stage": 0, "index": 12, "start_ms": 368.1, "compute_ms": 57.4, "requests": [{"id": "session_3_round_000002", "prefix": 55373, "append": 1196}]}`.
  `start_ms` is when the stage starts computing the microbatch, and `compute_ms` its GPU time without the send.
  Each request adds `append` computed tokens on top of `prefix` tokens it already has: cached, or computed by
  its earlier chunks.
- `reads.jsonl` (from `7_offloading.md` on), one line per round with a prefix:
  `{"id": "session_3_round_000002", "arrival_ms": 61234.0, "prefix_tokens": 55373, "hbm_tokens": 0, "dram_tokens": 55373, "ssd_tokens": 0, "read_start_ms": 61240.2, "read_land_ms": 61244.9}`.
  The three token counts say where the prefix came from; what none of them covers was computed again.

From these:
- **Stage busy** = Σ `compute_ms` of the stage's lines that start in the window ÷ the window.
- **Saturated tokens/s** = Σ `append` of stage 0's lines whose `start_ms` falls in the middle 80% of the run ÷ that
  window. The run spans from stage 0's first `start_ms` to stage 3's last `start_ms` + `compute_ms`.
- **Prefix read share** = Σ `hbm_tokens`, Σ `dram_tokens`, Σ `ssd_tokens` ÷ Σ `prefix_tokens`, over the lines with
  `arrival_ms` in [600,000, 1,200,000).
- `data/microbatch_fit.py check measured/.../microbatches.jsonl` compares the microbatches with the fit of
  `5_prefix_cache.md` §4.

`data/metrics.py` computes the counted rounds, rounds/s, new tokens/s and TTFT percentiles from req-frontend's
`requests.jsonl`, the same way as for the targets:

```bash
uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/metrics.py <run dir>/requests.jsonl
```

## 4. Correctness

- **Reference:** vLLM serving the same checkpoint without PP, for example TP4 with expert parallelism on 4 B200
  (`--enable-expert-parallel --kv-cache-dtype fp8_e4m3`).
- **Check:** greedy outputs (or logits) of the real system against the reference, on the §2.1 prompts and on session
  rounds both with and without a cached prefix. The cached cases must cover a prefix read back from DRAM and from SSD;
  the §2.3 SSD-heavy setting produces both within minutes. Log every mismatch.
- A performance number counts only from a build that passes this check.

## 5. Running the benchmark

- Run it alone on the node. These runs are sensitive to CPU contention: a contended run read 16% low. Record
  `uptime` before each run.
- Clear every cache tier before each session run (§2.2, start state).
- When comparing iterations, align them by microbatch, not by elapsed time.

## 6. What to record

Measurements go under `measured/`; differences go into `issues.md`.

**Setup** (`measured/setup.md`), once per node and again after any change:
- Model snapshot revision; the commits or versions of the serving code and of every kernel library.
- GPUs: `nvidia-smi -q` (name, driver, CUDA, memory) and `nvidia-smi topo -m`.
- Usable HBM per GPU, the largest stage's weights, and the KV pool in tokens against 61,258,368 (§1).
- DRAM and SSD given to prefix KV per GPU, each with its measured read bandwidth against 50 GB/s and 10 GB/s.

**Trace:**
- Whether `data/sessions_2ps.csv` was rebuilt (§2.2) or used as shipped.
- If rebuilt: the sha256 of `mono_sessions_d80.csv` and of `sessions_2ps.csv` against the ones in §2.2, and the
  difference between the new `sessions_2ps.manifest.json` and the one in `data/` (2,457 sessions, 207,441 rounds).
- The req-frontend commit and the YAML of each run.

**Correctness** (§4): the reference build, the prompts checked (§2.1 lengths; session rounds without a prefix and
with a prefix read from HBM, DRAM and SSD), the number of mismatches, and pass or fail.

**Each run** (`measured/runs/<name>/`). The names: `one_<L>` and `saturated_<L>` (`4_iteration.md`), `prefix_bringup`
and `prefix_n<N>` (`5_prefix_cache.md`), `sched_check_step<k>`, `force_check_step<k>` and `sched_step<k>_n<N>`
(`6_scheduler.md`), `tier_bringup_step<k>_<tier>` and `tiers_step<k>_<memory>_n<N>` with memory `full`, `tiers` or
`ssd_heavy` (`7_offloading.md`). Each holds:
- `uptime` before the run, and how the cache tiers were cleared (§5).
- The KV pool in tokens and the DRAM per GPU the run actually had.
- The client config, `requests.jsonl`, the server logs of §3, and the output of `data/metrics.py`.
- Stage busy per stage; for §2.3, the prefix read share from HBM, DRAM and SSD.

**Differences** (`issues.md`), in one of two kinds:
- **Setup difference:** something in this file does not match, for example a smaller KV pool or a different
  checkpoint revision. Name the targets it affects.
- **Implementation gap:** the setup matches but a number misses its target. Give the measured value, the target, and
  the suspected cause.

Never change the workload, the metric definitions or the design to close a gap.
