# Single iteration: the whole model on 4 GPUs

The third milestone, after the layers of `3_layers.md` pass. The whole model is wired across the 4 GPUs as the
design splits it: embedding, 45 layers in 4 stages, the sends between stages, the final norm and `lm_head`, and one
output token per request.
- **One request** runs through it alone. This checks the wiring, the stage-to-stage transfer, and that a stage runs
  its layers with no gaps.
- **Saturated:** many requests of the same length at once. This checks that the 4 stages overlap, with up to 4
  microbatches in flight, and gives the pipeline's throughput.

Neither needs the session scheduler, the prefix cache or the cache tiers.

## 1. The workloads

The bring-up of `1_contract.md` §2.1, for each L in 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k (L = 1,024 · 2^k tokens).
Every request has L fresh tokens, no prefix, and one output token.

| Run | Trace (req-frontend, `text-generation-independent`) | Requests | Arrivals |
|---|---|---|---|
| One request | `data/bringup/one_<L>.csv` | 5 | 10 s apart |
| Saturated | `data/bringup/uni_<L>.csv` | max(128, 2^24 / L): 16,384 at 1k down to 128 at 128k | all at t = 0 |

`data/bringup/make_bringup.py` writes all 16 files.

## 2. What runs

| Stage | GPU | Layers | Then |
|---|---|---|---|
| 0 | 0 | embedding, expand to the 4 mHC streams, layers 0–11 | send to stage 1 |
| 1 | 1 | layers 12–22 | send to stage 2 |
| 2 | 2 | layers 23–33 | send to stage 3 |
| 3 | 3 | layers 34–44, final mHC post, average of the 4 streams, final RMSNorm, `lm_head` on the last token | sample the token |

- **The send.** Each stage hands the next the 4-wide residual stream of all L tokens: 4 × 4,096 × BF16 =
  32 KiB per token (256 MiB at 8k, 4 GiB at 128k). One NCCL point-to-point send over NVLink per hop.
- **Order.** With one request, nothing overlaps: a stage starts when its input has arrived, and the send starts when
  the stage's last kernel ends. TTFT is the sum of the four stage times and the three sends.
- **Inside a stage.** The layers back to back with no host gaps, as in `3_layers.md` §4. Each DSA layer writes the
  request's KV into its stage's cache; each KDA layer writes its state.
- **One request** is one microbatch: the microbatch budget must hold the whole prompt.
- **Saturated** uses the plain scheduler on stage 0. It fills each microbatch to 8,192 tokens in arrival order and
  cuts a longer prompt into 8k chunks. At most 4 microbatches are in flight, and the next one forms only when stage 0
  is free, so each stage computes one microbatch while the others work on theirs.

## 3. Targets: one request

Per stage compute, each send, and TTFT, in ms:

| Prompt | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Each send | TTFT |
|---|---|---|---|---|---|---|
| 1k | 12.326 | 13.022 | 13.081 | 13.308 | 0.068 | 51.941 |
| 2k | 18.029 | 18.260 | 18.401 | 18.649 | 0.121 | 73.702 |
| 4k | 30.945 | 30.117 | 30.594 | 30.883 | 0.247 | 123.278 |
| 8k | 57.086 | 54.130 | 55.282 | 55.655 | 0.482 | 223.599 |
| 16k | 111.727 | 103.939 | 106.838 | 107.376 | 0.952 | 432.737 |
| 32k | 220.375 | 203.103 | 209.043 | 209.906 | 1.892 | 848.104 |
| 64k | 431.220 | 395.487 | 406.900 | 408.380 | 3.774 | 1,653.308 |
| 128k | 871.480 | 792.604 | 821.195 | 823.911 | 7.536 | 3,331.797 |

The timeline at 8k, ms from the request's arrival:

| | Starts | Computes | Ends | Send to the next stage |
|---|---|---|---|---|
| Stage 0 | 0.000 | 57.086 | 57.086 | 0.482 |
| Stage 1 | 57.568 | 54.130 | 111.698 | 0.482 |
| Stage 2 | 112.180 | 55.282 | 167.462 | 0.482 |
| Stage 3 | 167.944 | 55.655 | 223.599 = TTFT | |

Each stage split by layer type: the layer count × the one-layer time of `3_layers.md` §3, in ms.

8k:

| Stage | Input head | Layer 0 | Layers 1–2 | DSA + MoE | KDA + MoE | Output head | Compute |
|---|---|---|---|---|---|---|---|
| 0 | 0.090 | 3.671 | 2 × 3.733 | 3 × 5.863 | 6 × 4.712 | | 57.086 |
| 1 | | | | 2 × 5.863 | 9 × 4.712 | | 54.130 |
| 2 | | | | 3 × 5.863 | 8 × 4.712 | | 55.282 |
| 3 | | | | 3 × 5.863 | 8 × 4.712 | 0.373 | 55.655 |

128k:

| Stage | Input head | Layer 0 | Layers 1–2 | DSA + MoE | KDA + MoE | Output head | Compute |
|---|---|---|---|---|---|---|---|
| 0 | 1.530 | 60.165 | 2 × 61.151 | 3 × 95.448 | 6 × 66.856 | | 871.480 |
| 1 | | | | 2 × 95.448 | 9 × 66.856 | | 792.604 |
| 2 | | | | 3 × 95.448 | 8 × 66.856 | | 821.195 |
| 3 | | | | 3 × 95.448 | 8 × 66.856 | 2.716 | 823.911 |

`data/iteration_targets.json` holds all eight lengths: per stage the start, the compute time and the split by layer
type; per send the bytes and the time; and TTFT. Each layer's own breakdown, down to every kernel, is in
`data/layer_targets.jsonl` and `data/layers/` (`3_layers.md` §3). To regenerate: `data/iteration_targets.py`, after
`data/layer_targets.py` (usage in its docstring).

## 4. Targets: saturated

Per length: the run from the first microbatch on stage 0 to the last one leaving stage 3; prompt tokens/s over the
middle 80% of the run (`1_contract.md` §3); TTFT over all requests, in s; and the fraction of the middle 80% each stage
computes.

| Prompt | Requests | Run (s) | Tokens/s | Requests/s | TTFT first / p50 / p90 / p99 / max | Stage busy 0 / 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 1k | 16,384 | 112.47 | 149.3k | 145.82 | 0.215 / 56.34 / 101.28 / 111.37 / 112.47 | 100.0 / 95.2 / 97.0 / 97.7% |
| 2k | 8,192 | 112.00 | 149.9k | 73.21 | 0.214 / 56.11 / 100.86 / 110.91 / 112.00 | 100.0 / 95.1 / 97.0 / 97.7% |
| 4k | 4,096 | 113.85 | 147.5k | 36.01 | 0.218 / 57.03 / 102.50 / 112.74 / 113.85 | 100.0 / 94.7 / 97.1 / 97.7% |
| 8k | 2,048 | 117.08 | 143.4k | 17.51 | 0.224 / 58.65 / 105.39 / 115.91 / 117.08 | 100.0 / 94.8 / 96.8 / 97.5% |
| 16k | 1,024 | 117.12 | 143.4k | 8.75 | 0.281 / 58.70 / 105.44 / 115.96 / 117.12 | 100.0 / 94.8 / 96.8 / 97.5% |
| 32k | 512 | 117.44 | 143.0k | 4.36 | 0.397 / 58.92 / 105.73 / 116.27 / 117.44 | 100.0 / 94.7 / 96.9 / 97.5% |
| 64k | 256 | 118.47 | 141.8k | 2.16 | 0.633 / 59.55 / 106.68 / 117.29 / 118.47 | 100.0 / 94.5 / 96.9 / 97.5% |
| 128k | 128 | 121.35 | 138.4k | 1.06 | 1.118 / 61.23 / 109.33 / 120.15 / 121.35 | 99.5 / 93.5 / 96.4 / 97.0% |

- Every run is 2^24 tokens in 2,048 microbatches of 8,192 tokens. Stage 0 is the busiest stage, so it sets the pace.
- All requests arrive at once, so TTFT is set by a request's place in the queue: it rises steadily from the first
  microbatch's to the run's end, and p50 is about half the run.
- The first TTFT is the first request's. Up to 8k it is one 8k microbatch through the 4 stages (8 requests at 1k).
  From 16k on, it is the first prompt's L / 8k chunks, which enter stage 0 one after another.

`data/saturated_targets.json` holds these numbers and every request's TTFT in trace order
(`ttft_ms_by_request`). To regenerate: `data/saturated_targets.py` on the saturated run (usage in its docstring).

## 5. How to run

- The setup of `1_contract.md` §1: 4 B200 in one NVLink domain, the model checkpoint, the stage split above.
- Run the serving process itself, not a test harness, with the scheduler of §2 for each run.
- Replay each trace with req-frontend (`alignment/load_generator/req-frontend`), one length per run, on a freshly
  started server:

  ```yaml
  input:
    trace: doc/handoff/glm53_flash_pp4_b200_v1/data/bringup/one_8k.csv   # or uni_8k.csv
    format: text-generation-independent
  replay:
    arrival_mode: trace-timed    # no max_concurrency: the saturated run must not be capped
    context:
      max_model_len: 1048576
  ```
- **One request every 10 s.** The 5 requests of `one_<L>.csv` arrive 10 s apart. The longest (128k) takes 3.3 s, so
  no two overlap and every one runs on idle GPUs. Each request has its own prompt, so nothing is served from the
  prefix cache. Record all 5; none is dropped as a warm-up.
- **Saturated.** All requests of `uni_<L>.csv` are submitted at t = 0. Check from the client log that the
  submissions really went out together; record how long submitting them took.
- Record a trace of all 4 GPUs (Nsight Systems, or CUPTI per process) and the client's `requests.jsonl`. For the
  saturated runs a trace of the middle 80% is enough.
- **Check the output.** The token, or the logits, against the reference of `1_contract.md` §4 for every length. A
  wrong token here is a wiring error: a layer on the wrong stage, a missing send, the wrong rows to `lm_head`.

## 6. Pass rule

**One request**, from the GPU trace, per request:
- **Stage compute:** stage s's first kernel start to its last kernel end. Within ±10% of the target.
- **Send:** stage s's last kernel end to stage s+1's first kernel start. Within ±10% of the target, or 0.05 ms,
  whichever is larger.
- **GPU span:** stage 0's first kernel start to stage 3's last kernel end. Within ±10% of TTFT.

Slower is an implementation gap; faster is not a failure but is recorded. A stage that misses while its layers passed
in `3_layers.md` points at the wiring: gaps between layers, an extra copy, a synchronization.

**Host time.** The client's TTFT minus the GPU span is time the target does not have: tokenization, scheduling, the
launch of stage 0, sampling, detokenization, the response. It has no pass rule; record it.

**Saturated**, per length:
- **Tokens/s** within ±10% of the target.
- **TTFT** p50, p90 and p99 within ±10% of the target.
- Stage busy has no pass rule. A stage much below its target busy while stage 0 is at 100% points at the hand-off
  between stages: a send that blocks, or a microbatch that forms late.

## 7. Where these targets may be off

- **Sends past 256 MiB.** The largest profiled point-to-point row is 256 MiB (8k tokens; 0.482 ms, 557 GB/s). Larger
  sends are priced at that rate.
- **The send waits for the whole stage.** A real system may start sending the first layers' output early, or overlap
  the send with the next stage's first kernels. That beats the target, and is welcome.
- **Stage compute** inherits every caveat of `3_layers.md` §6.
- **No host time.** The target starts the first kernel at the request's arrival and ends at the last kernel.
- **Saturated TTFT depends on the queue order.** The target serves requests in trace order. A client that submits
  thousands of requests over a noticeable time, or a server that admits them in another order, changes every
  request's TTFT without changing throughput; compare `ttft_ms_by_request` to see which.

## 8. What to record

**Every one-request run** (`measured/iteration.jsonl`), one line per (prompt length, request):

```json
{"prompt_tokens": 8192, "run": 1, "client_ttft_ms": 231.4, "gpu_span_ms": 224.8, "host_ms": 6.6,
 "stages": [{"stage": 0, "start_ms": 0.0, "compute_ms": 57.3, "gaps_ms": 0.1}, "..."],
 "sends": [{"after_stage": 0, "bytes": 268435456, "ms": 0.49}, "..."], "token_matches_reference": true}
```

- Times are from stage 0's first kernel. `gaps_ms` is the idle time between kernels inside the stage.

**Summary** (`measured/iteration.md`):
- Per length, the mean of the 5 requests, and the first one alone, beside the targets of §3: stage compute, sends,
  GPU span, client TTFT, and their ratios.
- Host time per length, and what it is made of, where the trace shows it.
- For every stage above 1.10: the measured layer times of `3_layers.md` summed beside it, and where the difference
  sits (gaps, copies, waits).
- The output check of §5: lengths checked and mismatches.

**Every saturated run** (`measured/saturated.jsonl`), one line per prompt length:

```json
{"prompt_tokens": 8192, "requests": 2048, "submit_spread_s": 0.4, "run_s": 118.9, "tokens_per_s": 141200,
 "requests_per_s": 17.24, "ttft_s": {"first": 0.23, "p50": 59.6, "p90": 107.0, "p99": 117.7, "max": 118.9},
 "stage_busy": [1.0, 0.95, 0.97, 0.97], "ttft_ms_by_request": ["..."]}
```

- `submit_spread_s` is the time from the first submission to the last.
- Keep each run's `requests.jsonl` under `measured/runs/saturated_<L>/`.

The summary adds the saturated table of §4 with the measured numbers and their ratios beside it.

**Gaps** (`issues.md`): every stage, send or GPU span above its pass rule, every output mismatch, and every saturated
length whose tokens/s or TTFT misses.
