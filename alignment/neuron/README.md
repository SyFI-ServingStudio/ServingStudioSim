# NxDI alignment on Trainium2

This adapter captures the public NxDI full-model generation path and runs an
independent pass without a profiler through the existing req-frontend client.
The initial contract is Llama 3.1 8B BF16, 32 layers, TP1, batch1, one LNC2
unit, CTE128 and TKG512. Prompts contain 1..128 tokens, and prompt plus output
must fit the 512-token cache. There is no prefix caching or continuous batching.

The default compiled model contains one NEFF per forward. Normalization joins
each real `nrt_execute` model identity to its host forward bracket, then takes
the synchronized **union** of `nc_exec_running` intervals across both physical
cores. It preserves disjoint busy segments and all host/runtime gaps. The
result names the native producer `neuron_system_trace`; it is not an NSYS file.

Both phases use a loopback completions endpoint around the public
`HuggingFaceGenerationAdapter`. The pinned NxDI sampler does not invoke
Transformers streamers. A public `StoppingCriteria` emits each appended token
ID as it becomes available; req-frontend's client timing parser is unchanged.
Generation output is checked against those streamed IDs. Public model-load
warmup and optional frontend warmup complete before the measurement gate. Trace
start/stop and generation share one lock.

## Profile configuration

Keep the two configs beside the same `trace.csv` and corpus. Resolve the model,
compiled artifact and Neuron interpreter paths for the current host. Select the
port using the workspace's per-user convention. The controller rejects an
occupied port and reserves the requested whole chip with the shared Neuron
pool; the model uses one logical LNC2 unit on it.

```yaml
schema_version: 1
name: llama31_8b_nxdi_native
log_dir: ./profile_neuron
engine: nxdi
profile_kind: neuron
gpu: AWS Trainium2 LNC2
fork_python: /absolute/path/to/.venv-neuron-trace/bin/python
server:
  model_path: /absolute/path/to/llama3.1-8b
  compiled_path: /absolute/path/to/neuron-full-llama3.1-8b
  port: 63030
  neuron_device: 0
workload:
  frontend: {type: independent, path: ./trace.csv}
  backend: {type: openai}
  text_file: ./prompts.txt
  tokenizer: /absolute/path/to/llama3.1-8b
  max_concurrency: 1
  max_model_len: 512
  arrival_mode: saturated
  warmup: true
```

Make the clean pass identical except for `name`, `log_dir: ./profile_workload`
and `profile_kind: workload_metrics`. It never creates a `SystemTraceSession`.
The controller explicitly selects `PJRT_DEVICE=NEURON` and `NXD_CPU_MODE=0`.
Unset inherited `NEURON_RT_VISIBLE_CORES` before launch; allocation belongs to
the pool. Use the repository controller environment for the launcher and the
separate Neuron environment for `fork_python`.

```bash
uv run python -m launcher alignment profile profile_neuron.yaml --dry-run
uv run python -m launcher alignment profile profile_neuron.yaml
uv run python -m launcher alignment profile profile_workload.yaml
```

Run long physical jobs in named tmux sessions and record their command/log in
`progress.md`. `profile --resume` on a native pass regenerates parsed evidence
from its saved `system-trace.json` and `forward-records.json` without loading a
model or using a device. A saved clean pass resumes only its completed result.
Raw captures alone cannot supply HTTP/client E2E evidence.

## Shared prediction and analysis

The native pass writes `parsed.json` plus `parsed.kernels.parquet`, the usual
folded `kernel_sequences.json`, schema2 `metrics.jsonl`, real engine request
timings, untouched frontend replay results, token evidence and hashed runtime
provenance. The clean pass writes request/iteration/client evidence without
native trace artifacts. Shared analyzer interval reductions remain unchanged.
Reports retain the native provider, physical core IDs and whole-forward
measurement boundary.

Use `input_builder.type: engine_text` and `arch.type: llama3_nxdi` with
`context_bucket: 128` and `kv_capacity: 512`. The iteration adapter is
`nxdi_text`. Logical query lengths and decode K (resident prefix **plus the
current input**) reach the predictor and independent necessary-work model;
compiled 128/512 buckets are separate provenance.

Label the stable phase-specific measured names against the actual cost manifest:

- `NxDI Llama3.1-8B whole forward prefill CTE128 TKG512`
- `NxDI Llama3.1-8B whole forward decode CTE128 TKG512`

Each maps to the single `unified.forward` CostTree location. Use the existing
inventory initialize/apply/check workflow. Whole NEFF evidence cannot be
assigned to individual layers. The native input builder refuses the separate
`llama3_neuron` layer model. Native manifests use `parsed_trace`; the analyzer
also accepts historical `parsed_nsys` manifests.

Campaign variants select `engine: nxdi`, `backend: openai`, a
`server.compiled_checkpoint` host checkpoint key and profile passes of kinds
`neuron` and `workload_metrics`. Host profiles do not need an NSYS executable.
The case's `max_model_len: 512` renders as `arch.kv_capacity` and the workload
limit. Each device role names one whole chip, and each case must explicitly use
`max_concurrency: 1`.

## Stock vLLM Neuron TP4

`engine: vllm_neuron` uses the stock HTTP server in a declared immutable image,
with public scheduler/worker observation subclasses. It is a separate producer
from `nxdi`: Llama 3.1 8B BF16, TP4/DP1, one whole LNC2 chip, context512,
decode buckets `[1,16]`, maxseq16, 6782 KV blocks and page32. The initial replay
is sixteen simultaneous requests with 504 prompt tokens and eight outputs.

Materialize the exact eight L1 museum prompts twice before configuring the
shared req-frontend client:

```bash
uv run python -m alignment.neuron.vllm_corpus --model /absolute/model --output /absolute/corpus
```

The corpus uses req-frontend's actual circular pool offsets (`ordinal*9973
modulo pool.len()`). Offline validation checks all sixteen windows; scheduler
observations must then match each prompt's token-ID hash. Neutral filler is not
part of the consumed corpus. Set the workload's `text_file` to `museum-pool.txt`,
frontend path to `requests.csv`, tokenizer to the local model, token_pool_limit
to `79784`, max_items/max_concurrency to16 and max_model_len to512. Warmup is
disabled; stock server initialization and client preflight precede the gate.

The server config declares `model_path`, persistent `cache_path`, immutable
`image`, private Unix `docker_host`, image-owned `python_executable`, and an
explicit existing `req_frontend_binary`. No fork build or container pull occurs.
Declare `accepted_forward_path` for the separately validated stock forward run.
Fresh captures require its producer-written `binary-provenance.json`: numerical
native arrays and the unchanged vendor precision result are bound to NEFF hashes
recorded before/after accuracy and public warm-load hits on all four ranks.
The pinned lite loader passes those canonical files to `Executor`. Cold numerical
runs continue profiling but record binary binding as unavailable; warm the cache
and repeat accuracy/reference before using them as HTTP authority. No runtime
model-name suffix is decoded as a binary hash.

The registered `neuron_llama_forward:vllm_neuron` profiler produces this receipt
automatically. Revalidate the original C512 prefill512/decode1/decode16 specs into
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
These EngineCore durations exclude HTTP ingress and SSE delivery; the clean
workload pass supplies server latency comparisons. Scheduler observations retain the stock resolved async
policy and pre-increment progress, actual padding and request IDs.

Native normalization validates one submit per observed worker call, pairs each
worker's ordered submit and execution-pre model sequences, then binds genuine
execution IDs to exactly two cores per rank. Device execution can finish after
the host call. It requires all four ranks/eight disjoint physical cores and
compiled graph metadata before taking their busy union once on chip lane0.
Disjoint busy segments remain separate. Unknown/missing/ambiguous joins fail;
runtime/export warnings about dropped events and server/observer exceptions
also reject the capture. Capture source hashes before/after shutdown are kept
separate from resumable postprocessing hashes.
the raw trace remains available for `--resume` without another device run.

Shared prediction uses `input_adapter: vllm_neuron_text`,
`arch.type: llama3_vllm_neuron` and the single whole-forward location. It refuses
the NxDI architecture for stock captures. Existing Phase0 graph measurements
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
