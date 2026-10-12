# Stock vLLM Neuron alignment on Trainium2

`engine: vllm_neuron` uses the stock HTTP server in a declared immutable image,
with public scheduler/worker observation subclasses: Llama 3.1 8B BF16, TP4/DP1,
one whole Trainium2 chip of four LNC2 units, 6782 KV blocks and page32 (217024
slots; `profiling/runners/neuron/stock_contract.py`). A capture declares one context C in {512, 1024, 2048,
4096, 8192} (`max_model_len`) and `token_buckets`: ascending unique powers of two
(1 need not be among them), each at most the profiler's `largest_decode_bucket(C)` (512/256/128/64/32).
The server compiles one prefill bucket `[C]` and those decode buckets, with
maxseq = the largest bucket. The replay is N simultaneous requests of C-8 prompt
tokens and eight outputs, with 1 <= N <= min(maxseq, 217024 // C). The validated
example is C512, `[1,16]`, N=16 (504-token prompts).

Materialize the eight L1 museum prompts (request i replays subject i mod 8, as
the profiler's batch does) before configuring the shared req-frontend client:

```bash
uv run python -m alignment.neuron.vllm_corpus --model /absolute/model --output /absolute/corpus --context 512 --requests 16
```

The corpus uses req-frontend's actual circular pool offsets (`ordinal*9973
modulo pool.len()`). Offline validation checks all sixteen windows; scheduler
observations must then match each prompt's token-ID hash. Neutral filler is not
part of the consumed corpus. Set the workload's `text_file` to `museum-pool.txt`,
frontend path to `requests.csv`, tokenizer to the local model, token_pool_limit
to `79784`, max_items = max_concurrency = N and max_model_len to C. Warmup is
disabled; stock server initialization and client preflight precede the gate.

The server config declares `model_path`, persistent `cache_path`, immutable
`image`, private Unix `docker_host`, the Docker CLI argv `docker_command`
(default `[docker]`; e.g. `[sudo, -n, docker]`), image-owned
`python_executable`, and an explicit existing `req_frontend_binary`. No fork build or container pull occurs.
Declare `accepted_forward_path` for the separately validated stock forward run.
Fresh captures require its producer-written `binary-provenance.json`: numerical
native arrays and the unchanged vendor precision result are bound to NEFF hashes
recorded before/after accuracy and public warm-load hits on all four ranks.
The pinned lite loader passes those canonical files to `Executor`. Cold numerical
runs continue profiling but record binary binding as unavailable; warm the cache
and repeat accuracy/reference before using them as HTTP authority. No runtime
model-name suffix is decoded as a binary hash.

The registered `neuron_llama_forward:vllm_neuron` profiler produces this receipt
automatically; its graphs must be exactly prefill C plus each configured decode
bucket. Revalidate the original C512 prefill512/decode1/decode16 specs into
a fresh private DB/output root, reusing the stock compile and reference caches:

```bash
uv run --no-sync python -m profiling run neuron_llama_forward --backend vllm_neuron --specs /absolute/three-stock-specs.json --db /absolute/new-private.db --gpu-name 'AWS Trainium2 LNC2' --output-dir /absolute/new-results --json
```

Set `accepted_forward_path` to the new numerical run named in its row provenance.
A fresh accuracy/reference-only acquisition is also sufficient when it produces
the same complete three-graph receipt; profile records are not copied from an old
run. Before HTTP startup the adapter streams the eight pinned checkpoint,
tokenizer and config files, then compares NEFF bytes before/after server lifetime
and requires four-rank warm-loader evidence. Checkpoint mounts remain read-only.
Legacy captures without these acquisition receipts remain resumable and explicitly
report graph-key-only provenance; today's cache cannot upgrade historical proof.

Its numerical criterion, subject continuations and graph keys are preserved;
a different observed graph is rejected. HTTP replay JSON and timeline
persist generated-token counts and times, not the IDs or full logits, so this
pass cannot repeat the numerical validation or compare those continuations.
The adapter holds the shared whole-chip lease until server termination. HTTP
binds only loopback; host networking supports that endpoint. Source/checkpoint
mounts are read-only and telemetry/downloads are disabled. The CPU-only Explorer
export container has no networking or device mapping.

Use separate `profile_kind: neuron` and `workload_metrics` configs with fresh
log directories. The native pass uses the stock public profiler endpoints on
all four ranks. The clean pass omits profiling. Req-frontend's token-ID SSE
arrivals remain the client TTFT/TPOT authority. Separately, the observer reads
stock `QUEUED`/`SCHEDULED` events and `EngineCoreOutputs.timestamp` for the first
and last nonempty token outputs. It emits one schema-v2 EngineCore timing record
at successful completion, without a new clock or device synchronization.
Native `log_stats` must be enabled. Aborted/empty requests, missing events,
duplicate completions and mismatched emitted counts invalidate timing evidence.
The shared extractor filters preflight requests against the replay population;
new captures require every successful request. Old captures without this
instrumentation remain resumable with `request_timings_result: null`.
The record also carries an additive arrival sub-record
(`arrival_timing_schema_version` 1; `schema_version` stays 2). The observed
scheduler's `add_request` samples one (epoch, monotonic) pair when EngineCore
drains the request from its input queue between steps, before the stock QUEUED
event, and places vLLM's wall-clock `Request.arrival_time` on that monotonic
clock: `frontend_arrival_*_ns`, `engine_receive_monotonic_ns`,
`frontend_arrival_to_engine_receive_ms`, `engine_arrival_to_queued_ms` and
`server_ttft_from_arrival_ms` (= arrival-to-QUEUED + `engine_core_ttft_ms`).
`VllmNeuronPrompt` repeats both stamps beside the iteration's
`observed_start_monotonic_ns`. Captures whose provenance declares
`request_arrival_timing_schema: 1` or later require them; older rows get nulls.

Arrival sub-version 2 can split arrival-to-receive at the API server's
hand-off, using only public vLLM mechanisms. The split is opt-in through
`server.handoff_stamp: true` (default false). When false, the server argv and
env are byte-identical to unstamped captures. When true, the server command
adds `--enable-log-requests --max-log-len 0` and
`VLLM_LOGGING_CONFIG_PATH=/source/alignment/neuron/vllm_logging.json`; nothing
else in the C512 `[1,16]` argv changes. The capture's source provenance records
`handoff_stamp`. That dictConfig reproduces vLLM's
default `vllm` handler and `INFO MM-DD HH:MM:SS [file.py:N] msg` format
(`vllm_handoff_log.VllmLineFormatter`, stdlib-only) and leaves non-vLLM loggers
(e.g. the Neuron loader receipts) alone. It sets the per-request prompt loggers
(`vllm.entrypoints.logger`, and `vllm.entrypoints.serve.utils.request_logger` in
later layouts) to WARNING. It also attaches `HandoffStampFilter` to
`vllm.v1.engine.async_llm`. Stock `AsyncLLM._add_request` logs
`Added request <id>.` right after `await engine_core.add_request_async`, with
the internal id from `assign_request_id` that EngineCore sees. For each such
record the filter samples monotonic and epoch clocks and writes one
`VllmNeuronHandoff {"request_id","send_monotonic_ns","send_epoch_ns"}` line,
then drops the plain log line. Postprocessing joins stamps to rows by
`engine_request_id`; the stamp must be the scheduler's engine id for its client.
It adds `api_send_monotonic_ns`, `frontend_arrival_to_api_send_ms` and
`api_send_to_engine_receive_ms`, which sum to
`frontend_arrival_to_engine_receive_ms`. The observer emits them null because
EngineCore cannot see the API process. Validation requires each group to be
all-null or complete, with arrival <= send <= receive. Sub-version 1 rows stay
readable, with null hand-off fields. Provenance `handoff_stamp: true`, which
needs `request_arrival_timing_schema: 2`, requires a stamp for every request.
Without it, the hand-off fields are null and any stamp is an error.
These EngineCore durations exclude HTTP ingress and SSE delivery; the clean
workload pass supplies server latency comparisons. Scheduler observations
retain the stock resolved async policy and pre-increment progress, actual
padding and request IDs.

Native normalization validates one submit per observed worker call, pairs each
worker's ordered submit and execution-pre model sequences, then binds genuine
execution IDs to exactly two cores per rank. Device execution can finish after
the host call. It requires all four ranks/eight disjoint physical cores and
compiled graph metadata before taking their busy union once on chip lane0.
Disjoint busy segments remain separate. Unknown/missing/ambiguous joins fail;
runtime/export warnings about dropped events and server/observer exceptions
also reject the capture. Capture source hashes before/after shutdown are kept
separate from resumable postprocessing hashes.

`alignment profile <config> --resume` reruns only this postprocessing from the
saved server log, replay result, `system-trace.json` and graph metadata, without
a device or server. It requires the capture-time configuration (image, model
config, checkpoint path) to be unchanged and records the current source hashes
separately in `postprocess-source-provenance.json`.

Shared prediction uses `input_adapter: vllm_neuron_text`,
`arch.type: llama3_vllm_neuron` and the single whole-forward location; the
input builder refuses any other architecture for a native Neuron capture.
Native manifests name the parse `parsed_trace` (the analyzer also accepts
`parsed_nsys`), and the native producer is `neuron_system_trace`, not NSYS. Existing Phase0 graph measurements
provide kernel evidence; the HTTP passes add scheduler and client evidence.
Neither whole NEFFs nor observer host spans provide internal layer timings.

Shutdown uses the public `--shutdown-timeout 10` drain mode within Docker's
30-second stop deadline. For the pinned lite-Neuron allocator, a source and
binary checked EngineCore cleanup adapter skips only Torch's generic caching
allocator flush: this allocator frees its NRT tensors directly and has no
PyTorch allocation cache. Group destruction, garbage collection and host-cache
cleanup retain their original order; cleanup errors still invalidate capture.
The adapter changes no model operations, compiled binaries or global Torch
functions and refuses other allocator/source versions. Failed historical
shutdown captures remain rejected and are never upgraded by this fix.
