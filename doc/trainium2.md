# Trainium2: stock vLLM Neuron Llama 3.1 8B

ServingStudio Sim models one serving target on AWS Trainium2: the original
Llama 3.1 8B BF16 checkpoint served by stock vLLM Neuron 0.24.0.1.1.0 (pinned
image `sha256:44d2eef7…`, `profiling/container/images.py`), TP4 over the four
LNC2 logical cores of one chip (eight physical cores). The paged KV pool is
6782 pages of 32 tokens, 217024 token slots, 32768 bytes per token per rank.
These constants live once per language: `profiling/runners/neuron/stock_contract.py`
and `simulator/src/timing/kernels/neuron_llama_forward.rs`
(`tests/test_neuron_full_forward.py` checks that they agree). The GPU key is
`AWS Trainium2 LNC2`, an alias of the catalog entry `Trainium2-LNC2`.

## Pieces

| Layer | What | Where |
| --- | --- | --- |
| L1 execution | Neuron backends (`device_family: neuron`) run in the vendor image on a reserved whole chip | `profiling/exec/neuron.py`, `profiling/README.md` |
| L1 kinds | `neuron_llama_forward/vllm_neuron` (whole forward), `neuron_llama_region/vllm_neuron_fx_regions`, `neuron_llama_segment/vllm_neuron_collective_segments`, `neuron_llama_segmented_forward/vllm_neuron_segmented` | `profiling/kernels/neuron_llama_*.py`, `profiling/runners/neuron/`, `simulator/src/timing/kernels/neuron_llama_*.rs` |
| L4 arch | `llama3_vllm_neuron`: `composition` and `prefill_segment_size` | `simulator/src/arch/llama3_vllm_neuron.rs` |
| L5/L6 | generic `chunked_prefill` configured as the stock scheduler | `deployment/unified.rs::ensure_stock_neuron_worker` |
| Work maps | `llama3-vllm-neuron-{unified,regions,segments}-v1` | `model/work/location_maps/` |
| Alignment | engine `vllm_neuron`, campaign pack `llama31_8b_bf16_trainium2_tp4` | `alignment/neuron/README.md` |

Host configuration for profiling (not checked in): `SERVINGSTUDIO_NEURON_DOCKER_HOST`
(the local Docker socket, required), `SERVINGSTUDIO_NEURON_DOCKER_COMMAND` (CLI
argv, default `docker`), `SERVINGSTUDIO_VLLM_NEURON_MODEL_DIR` (checkpoint,
required) and `SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR` (compile cache). Measured
rows live in private databases; the shared `profiling/profile.db` holds none.

## Whole-forward kernel

`neuron_llama_forward/vllm_neuron` measures the complete compiled forward (32
layers, TP collectives, KV updates and greedy sampling) as one leaf. Its args
are `phase, token_bucket, max_model_len, kv_blocks, block_size, tp_size, dtype`;
the Rust config adds the compiled `decode_buckets`. The cache is categorical
(`Cache1DDirect` over the compiled-variant index: prefill, then each decode
bucket), with no interpolation.

A profiling run per context:

1. compiles the context's prefill graph and the requested decode graphs (bucket
   1 only when the inventory asks for it);
2. checks full native logits of eight museum subjects against independent CPU
   FP32/BF16 references with the unmodified vendor criterion (eager attention up
   to 8192 tokens, eager over 1024-row query blocks above, the choice being part
   of the reference cache key);
3. seals the NEFF bytes it validated (`binary-provenance.json`);
4. times the same graphs from the native NRT trace: each execution is the union
   of all four ranks' physical-core intervals, and every traced call must run
   its requested decode graph at least seven times, allowing one early decode
   at a smaller bucket that stock async scheduling can produce.

## Per-operation diagnostics

`launcher kernel-profile ops neuron_llama_forward --backend vllm_neuron` captures
native instruction/DMA activity in the stock executable and emits operation
JSON/CSV, detailed Parquet evidence, and Analyzer-compatible plots. See the
[profiling guide](../profiling/README.md#per-operation-neuron-diagnostics) for
capture commands, offline analysis, executable checks, and metric definitions.

Hardware validation covers prefill512/decode16 at context512 and
prefill1024/decode16 at context1024, on rank0 and both of its physical cores,
for Llama 3.1 8B BF16 TP4. Accepted captures pass numerical/executable checks,
native execution selection, trace-loss checks, and elapsed-time accounting.

Compiler source metadata supplies operation labels. An isolated debug build
recovers missing labels through checked instruction-byte matches; timestamps
always come from the stock executable. Reports distinguish exact, inferred,
compiler-primitive, and unresolved attribution. Source locations identify
operation types; specific transformer layer-instance ownership is not verified.

Per-op activity overlaps, so operation durations are not additive latency costs.
Prefill capture adds output-completion waits around a selected prefill and a
trailing decode guard, which is excluded from prefill aggregates. A controlled
unprofiled/controller-only/profiled comparison has not been performed:
zero overhead and fidelity to production timing remain unverified. Unseen-shape,
critical-path and marginal-speedup prediction are unsupported. No timing database
rows are read or written by this diagnostic path.

## Context ladder (C512-C8192)

`presets/unified_llama3_8b_vllm_neuron_ladder.yaml` costs contexts 512..8192.
At most `217024 // C` requests are resident, and a decode bucket is a power of
two up to `min(next_pow2(217024 // C), C)`. Each context configures only the
buckets whose rows passed both gates; a decode batch pads to the smallest one
that holds it, and inventories need not start at 1.

| Context | Decode buckets | Resident | Prefill ms | Decode ms by bucket |
| ---: | --- | ---: | ---: | --- |
| 512 | 1,2,4,8,16,32,64,128,256,512 | 423 | 38.2 | 9.9, 11.3, 12.3, 15.5, 55.3, 58.1, 114.7, 84.3, 153.3, 303.1 |
| 1024 | 1,2,4,8,16,32,64,128,256 | 211 | 65.8 | 9.7, 10.9, 12.3, 14.9, 38.8, 196.4, 310.6, 309.8, 615.5 |
| 2048 | 1,2,4,8,16,32,64,128 | 105 | 121.1 | 10.1, 22.3, 35.6, 102.3, 117.1, 106.9, 1025.0, 571.2 |
| 4096 | 16,32 | 32 | 236.3 | 230.4, 112.2 |
| 8192 | 1,16,32 | 26 | 489.2 | 21.6, 120.7, 1071.0 |

Bucket 2 (11.30, 10.93 and 22.25 ms at C512, C1024 and C2048) passes only under
the one-early-decode coverage rule, because stock async scheduling can run one
batch-2 decode at bucket 1 first. C4096 B1-B8/B64 and C8192 B2-B8 fail the
vendor criterion. Decode time
is not monotonic in the bucket (C2048 B64, C8192 B32); this reproduces across
runs and is a property of the stock compiled graphs.

## Serving pairing (L4/L5/L6)

`llama3_vllm_neuron` accepts only the original 32-layer BF16 checkpoint with
Llama 3.1 RoPE, RMSNorm and unbiased SiLU. A prefill must be one prefix-free
prompt alone on the context graph; a decode-only batch pads to a configured
bucket within the pool. It pairs only with `chunked_prefill` set to the stock
scheduler: one replica, `max_batch_tokens` = C (S when segmented),
`max_prefill_requests_per_iteration: 1`,
`max_resident_requests_per_partition` = min(largest bucket, 217024 // C),
`separate-prefill-priority`, FIFO, full-footprint admission,
`prefix_cache_mode: disabled`, and `attn_gpu_memory_gb` equal to the pool per
rank (7.111442432). See `presets/unified_llama3_8b_vllm_neuron.yaml` and
`presets/predict_llama3_8b_vllm_neuron.json`.

## Compositions (experimental)

`composition: whole_forward` is the default. Two finer costings of the same
executables exist; rows of either are validated only where recorded.

- `model_head_regions` (`neuron_llama_region`): the stock FX graph split at the
  `LlamaModel.forward` return and compiled as two executables, model (embedding,
  32 layers, final norm, KV updates) then head (row selection, `lm_head`, logit
  gather, sampling). A run is accepted only if the partition proof holds, the
  vendor check passes on split and stock, split versus stock generates the same
  tokens with RMS(split−FP32) ≤ 1.10 × RMS(stock−FP32), and the region sum is
  within 5% of the whole forward. Validated: every shape at C512 and C2048,
  C1024 B128/B256, C8192 B32.
- `layer_segments` (`neuron_llama_segment`): the unchanged executable's native
  instruction timeline cut at the sublayer-closing reductions into embedding,
  per-layer attention and MLP blocks (folded as 32 × (attention + MLP)) and
  head. Gates: NEFFs byte-identical to the stock forward, complete two-core
  traces, interior layers 1-30 within 10% of the layer mean, and the composed
  forward within 5% of the whole forward. Validated at C512 `[1,16]`.

Below these blocks, QKV, attention and O-projection share one synchronization
interval; standalone probes of them were not additive, so no finer rows exist.

## Segmented prefill (C16384/C32768)

Above C8192 the engine runs `max_num_batched_tokens = S` with one compiled
S-token prefill graph; a prompt executes as `ceil(prompt / S)` calls, the k-th
attending `k × S` cached tokens. `neuron_llama_segmented_forward` (args add
`prior_tokens` and `segment_size`) accepts C in {8192 (cross-check), 16384,
32768} and S in {512, 1024, 2048, 4096} below C. Prefill rows are timed per
prior `k × S` from dedicated batch-1 calls and interpolated linearly between
measured priors; a prior beyond the largest one or off the 32-token grid is
rejected. Decode rows are exact buckets. On the arch, `prefill_segment_size: S`
(with `whole_forward`) costs both phases in the one `forward` slot, and the
worker budget becomes S.

C16384 and C32768 at S2048 are recorded. The vendor gate passes on every
validation shape through the per-position BC route (RMS ratio 1.037-1.042 at
C16384 for b1/b4/b8/b13; 1.029-1.037 at C32768 for b1/b4/b6), against
row-chunked eager references.

| S2048 rows (ms) | C16384 | C32768 |
| --- | --- | --- |
| prefill segment at prior 0 | 261.0 | 260.7 |
| prefill segment at prior 14336 | 400.2 | 400.6 |
| prefill segment at prior 30720 | - | 561.6 |
| decode b1 / b4 / b8 / b16 | 13.51 / 80.98 / 151.70 / 236.90 | 18.28 / 151.85 / 290.70 / - |

A segment costs about 20 ms more per 2048 prior tokens, and the same at both
contexts for equal priors (within 0.2%). `presets/*_vllm_neuron_segmented.*`
use decode buckets `[1,4,8,16]` at C16384 and `[1,4,8]` at C32768.

## Alignment

`engine: vllm_neuron` captures stock HTTP serving natively (`neuron` pass) and
cleanly (`workload_metrics` pass): N simultaneous (C−8)+8 museum requests at a
context C in 512..8192. The C512 `[1,16]` pack judges 13 campaign metrics. On
the ladder captures, kernel error is within ±0.7% at every context with 100%
map coverage; the end-to-end mean is −1.4% to −5.4%. From C2048 upward the
captures ran one extra solo decode, which fails four workload-geometry checks at
unchanged thresholds (see the next section).

## Async scheduling and frontend hand-off (opt-in)

Stock vLLM Neuron plans step N+1 shortly after step N starts. If the second of
a burst of simultaneous requests reaches EngineCore later than step 1's plan,
step 1 is a solo decode of the first request. The default simulator forms each
batch when the previous step ends, with every arrival visible, so it never runs
that decode. `presets/unified_llama3_8b_vllm_neuron_ladder_async.yaml` enables
two models of this race (at a 10 us tick):

- `worker.async_lookahead_us` (chunked prefill on `llama3_vllm_neuron` only,
  1..=5000): step N+1 is planned `w` after step N starts from the requests
  visible then; later arrivals wait for the next plan (preset 1040 us). With the
  hand-off model also on, the step after one that started from an idle engine
  uses a per-seed draw of the measured first-step plan offset for the context
  (`WorkerConfig::async_lookahead_from_idle`).
- `workload.frontend_handoff: vllm-neuron-0.24` with
  `workload.frontend_handoff_seed`: in a burst into an idle server, request k
  is sent at `arrival + ingest + gap_1 + ... + gap_k` and becomes visible at
  `max(visible_{k-1}, send_k + send_to_visible_k)`. Each delay is resampled
  from measured values per (seed, request id); `arrival_time` stays the trace
  arrival, so TTFT includes the hand-off.

Inputs come from 11 hand-off-stamped captures (`server.handoff_stamp: true`;
C512 x4, C2048 x4, C8192 x3), measured and never fitted to the outcome; the
frontend uses the largest measured context at or below the serving context
(C1024 borrows C512, C4096 borrows C2048). `alignment/neuron/vllm_handoff_distribution.py`
regenerates `alignment/neuron/data/vllm_neuron_handoff.json` and records which
inputs are keyed by context (disjoint ranges) and which are pooled.

| Input (ms) | C512 | C2048 | C8192 | Keyed |
| --- | --- | --- | --- | --- |
| ingest (first arrival to first send) | 17.7-20.6 | 19.6-20.1 | 27.2-27.3 | context |
| first gap (send 1 - send 0) | 1.06-1.15 | 1.24-1.33 | 0.45-0.54 | context |
| later gaps | median 0.50 | 0.56 | 0.69 | pooled |
| send to visible (idle engine) | 0.51-0.95 | 0.52-0.73 | 0.71-0.88 | pooled |
| first-step plan offset | 0.71-0.90 | 0.75-0.83 | 1.00-1.67 | context |

Probability of the extra decode (24 steps) for the 16-request burst, seeds
0-99 per context, with exact binomial 95% intervals:

| Context | Simulated | Stamped captures | Earlier unstamped captures |
| ---: | --- | --- | --- |
| 512 | 0.91 (0.84-0.96) | 3/4 | 3/3 |
| 1024 (borrows 512) | 0.91 (0.84-0.96) | — | 3/3 |
| 2048 | 1.00 (0.96-1.00) | 4/4 | 2/4 |
| 4096 (borrows 2048) | 1.00 (0.96-1.00) | — | — |
| 8192 | 0.00 (0.00-0.04) | 0/3 | 2/3 |

The model reproduces every stamped context, including the context dependence
(a short first gap and a long plan offset at C8192). It is calibrated to one
session's client spacing: the earlier unstamped session falls outside the
simulated intervals at C2048 and C8192, and its outcomes track how far apart the
load generator sent the first two requests. Inputs and comparison come from the
same captures (3-4 per context), and C1024 and C4096 have none. With both
models off, or only the lookahead on, ladder runs are bit-identical to the
default cadence.

## Work estimates

The kernel rows' FLOPs and bytes (`profiling/runners/neuron/vllm_forward_work.py`)
are source-derived per-rank estimates, not counters: graph contractions from the
stock model shapes and the NKI causal attention tile loops, and persistent
operand volume (weights, embedding rows, KV reads/writes). They omit nonmatmul
arithmetic, collectives, temporaries, spills and rereads. Optimality's R5 takes
one roofline over the whole compiled forward while R6 sums per-semantic
rooflines, so R6 can exceed R5 although executed FLOPs and bytes exceed the
independent minimum at every iteration; this is a partition mismatch, not
missing work.

## Known limits

- One chip, TP4, BF16, this checkpoint and image. Other models, dtypes, TP
  layouts or SDK versions are different executables and need their own rows.
- Region and segment rows are validated only where recorded (above); layer
  segment capture beyond small C512 buckets exceeds the 512 MB instruction
  buffer.
- Segmented prefill rows exist only for S2048 at C16384 and C32768.
- HTTP passes do not repeat the full-logit validation; they bind the observed
  graphs to the validated forward run.
- Host-side overhead is calibrated only by the separate duty multiplier
  (`presets/alignment/llama31_8b_bf16_trainium2_tp4/calibration.json`).
