# Serving-engine profiler setup

Alignment ground truth comes from the instrumented **vLLM and SGLang
submodules in this directory**. Both forks emit indexed per-phase NVTX scopes
and the same versioned alignment records, so capture parsing and downstream
analysis stay shared. The parent repository pins the exact fork commits:

```text
alignment/profiler/vllm    # branch: moesim-profile
alignment/profiler/sglang  # branch: vibesim-alignment
```

Initialize both from the VibeSim checkout:

```bash
git submodule update --init alignment/profiler/vllm alignment/profiler/sglang
```

## vLLM environment

The vLLM fork adds `vllm_iteration(N): <phase>` NVTX scopes and a versioned
`VibeSimAlignmentIteration {json}` model-input record. It also emits one
`VibeSimAlignmentRequestTiming {json}` record when each request completes. The
MoE popularity pass additionally emits `VibeSimAlignmentExpertLoad {json}` only
when EPLB balancedness logging is explicitly enabled. Each record contains the
EP-reduced per-layer token counts mapped back to logical expert ids; the launcher
aggregates these into `expert_popularity.json`. This pass runs without NSYS and
must be separate from the timing pass because the reduction and D2H conversion
are deliberate measurement overhead. Each request-timing record contains both
first-token and decode-span timing.

Only its `.venv` is missing — build it once. `runner.py` defaults `fork_python`
to `alignment/profiler/vllm/.venv/bin/python`; override `fork_python` in
the align config to point elsewhere.

The instrumentation on `moesim-profile` is Python-only (NVTX scopes + iteration
metrics logging), so the **precompiled fast path** works — no multi-hour CUDA
compile. The submodule is detached at the parent gitlink, so explicitly select
the last upstream commit before the `feat(alignment):` patches. Without this,
vLLM's setup cannot resolve the branch and silently falls back to the latest
nightly wheel, which can be binary-incompatible with the pinned fork:

```bash
cd alignment/profiler/vllm
git status                         # confirm branch moesim-profile, tree present

# 3.12 venv living next to the fork (the path runner.py expects)
# Archive any adopted environment first: installing over it does not prune
# stale dependencies from a different nightly/CUDA generation.
if [ -e .venv ]; then
  ENVIRONMENT_ARCHIVE=$(mktemp -d "$TMPDIR/vllm-venv-archive-XXXXXXXX")
  rmdir "$ENVIRONMENT_ARCHIVE"
  mv .venv "$ENVIRONMENT_ARCHIVE"
  echo "archived old vLLM environment at $ENVIRONMENT_ARCHIVE"
fi
uv venv --python 3.12 .venv

# Resolve the binary base and verify every checkout-local change remains Python.
FIRST_ALIGNMENT_COMMIT=$(git log --reverse --format=%H --fixed-strings \
  --grep='feat(alignment):' HEAD | sed -n '1p')
PRECOMPILED_BASE_COMMIT=$(git rev-parse "$FIRST_ALIGNMENT_COMMIT^")
if git diff --name-only "$PRECOMPILED_BASE_COMMIT"..HEAD | \
  rg '\.(c|cc|cpp|cu|cuh|h|hpp|rs)$'; then
  echo "alignment fork has native changes; build vLLM from source" >&2
  exit 1
fi

# Install the fork's Python over that base's precompiled CUDA-12 wheel. The
# hosted CUDA-12 variant is cu129; its PTX still needs a compatible driver.
VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT="$PRECOMPILED_BASE_COMMIT" \
  VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  uv pip install --python .venv/bin/python -e .

# VLLM's profiling-only NVTX scopes import this optional package when
# VLLM_NVTX_SCOPES_FOR_PROFILING=1.
uv pip install --python .venv/bin/python nvtx

# Sanity: the fork imports and exposes the instrumentation env var.
uv run --python .venv/bin/python python -c "import vllm, vllm.envs as e; print(vllm.__version__, hasattr(e, 'VLLM_NVTX_SCOPES_FOR_PROFILING'))"
```

Import success is not a CUDA-runtime qualification: an older driver can load the
cu129 extensions and still fail at their first PTX kernel with
`cudaErrorUnsupportedPtxVersion`. Run the checkout-path and representative
Marlin GPU probes in `skills/dev-create-worktree/SKILL.md`. If the host driver
is too old, set the profile's top-level `driver_compat_lib_dir` to an unpacked
matching NVIDIA `cuda-compat` directory containing `libcuda.so.1`; the launcher
scopes that library to the vLLM server and records it in launch metadata. A
local source build against the host CUDA toolkit is the other clean option.
Before the probe, inspect `uv pip tree --python .venv/bin/python` for CUDA and
CUTLASS packages and run `uv pip check --python .venv/bin/python`. Recent vLLM
can intentionally resolve Torch cu12 together with CUDA-13 CUTLASS DSL/JIT
packages, so mixed CUDA generations alone do not prove a stale environment.
Recreate the environment when the consistency check fails, a distribution is
present at multiple versions, or a package is not reachable from the current
resolution. Select `driver_compat_lib_dir` for the highest CUDA/PTX generation
exercised by the actual model path, and test that path rather than stopping at a
Marlin-only probe.

If the diff from `PRECOMPILED_BASE_COMMIT` contains native source, or if
`VLLM_USE_PRECOMPILED` cannot resolve a wheel for that commit, fall back to a
from-source build (`uv pip install --python .venv/bin/python -e .` without the
flag) — this compiles CUDA kernels and takes much longer. Never use the latest
nightly, another checkout's `.so` files, or model-specific compatibility
switches as a fallback.

## SGLang environment

The SGLang fork emits `sglang_iteration(N): <phase>` and the same iteration,
request-timing, expert-load, and worker-device records consumed by the shared
pipeline. Build its environment under the submodule's `python/` directory,
which is the default path selected when `engine: sglang` and no `fork_python`
override is supplied:

```bash
cd alignment/profiler/sglang/python
uv venv --python 3.12 .venv-sglang
uv pip install --python .venv-sglang/bin/python -e .
uv run --python .venv-sglang/bin/python python -c \
  "import sglang; print(sglang.__version__, sglang.__file__)"
```

The pinned SGLang version currently uses a CUDA 13 user-space stack. On a host
whose driver needs NVIDIA forward compatibility, set `driver_compat_lib_dir`
in `profile.yaml` to an unpacked matching `cuda-compat` library directory. The
launcher validates `libcuda.so.1` there and records the path in launch metadata.
Do not modify the global `LD_LIBRARY_PATH` to make one capture work.

Model weights: the align config uses the HF repo id `meta-llama/Meta-Llama-3-8B`,
resolved from the shared cache at `HF_HOME=/m-coriander/coriander/hf`. Confirm the
launching shell has `HF_HOME` set (the alignment env inherits it) and the repo is
present under `$HF_HOME/hub/models--meta-llama--Meta-Llama-3-8B`.

Verify end to end before a real capture:

```bash
export NSYS_BIN=/absolute/path/to/nsys
"$NSYS_BIN" --version
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader   # pick an idle GPU
```

## NVTX marker contract

`alignment.runner` sets `VLLM_NVTX_SCOPES_FOR_PROFILING=1`. The instrumented
fork then emits one canonical inline, indexed range family:

```text
vllm_iteration(N): preprocess
vllm_iteration(N): forward
vllm_iteration(N): postprocess
vllm_iteration(N): sample
vllm_iteration(N): bookkeep
vllm_iteration(N): eplb
```

Indexed text is part of the profiling contract, not merely a display label.
NSYS may intern it into `StringIds` during SQLite export, so readers resolve
`text`/`textId` while still accepting only indexed labels. Do not add a paired
`gpu_model_runner: <phase>` alias around the same work: it contains no extra
information and historical dual-marker traces double-counted kernels.

The default OpenAI-server capture mode is worker-owned `cuda_profiler_api`, not
an NVTX capture trigger. The outer command uses
`--trace-fork-before-exec=true --capture-range=cudaProfilerApi
--capture-range-end=stop`; the server is launched with
`--profiler-config.profiler=cuda`; and the alignment runner calls
`/start_profile` immediately before replay and `/stop_profile` after the server
returns idle. This is the vLLM-documented boundary that arms CUPTI inside the
spawned CUDA EngineCore. On Nsight Systems 2024.6, an outer NVTX-triggered
capture can retain all child NVTX ranges and graph-creation metadata while
silently recording zero replay kernels. Keep `capture_mode: nvtx` only as an
explicit diagnostic fallback.

The runner never searches `PATH` or machine-specific CUDA/Nsight installation
directories. Set `nsys.executable` in the profile config or export `NSYS_BIN`
with an absolute executable path. That exact resolved path and its reported
version are reused for capture and SQLite export and persisted in launch/result
provenance.

Unpaired markers such as `gpu_model_runner: ModelRunnerOutput` remain because
they describe separate output/bookkeeping work. `execute_context_C(CT)_generation_G(GT)`
remains a diagnostic annotation. The human `Iteration(N): ...` line is not an
analyzer API: the separate `VibeSimAlignmentIteration {json}` line is the
authoritative request/token shape, stage, and full-run EngineCore observation
source. Schema v2 carries `observed_start_monotonic_ns`,
`observed_end_monotonic_ns`, and `observed_elapsed_ms`. Adjacent starts measure
the complete cadence including the intervening scheduler/bookkeeping gap;
start-to-end preserves the narrower historical result-wait/sampling elapsed
time. These host-clock fields continue after bounded CUPTI capture stops and do
not replace NSYS GPU timestamps for kernel attribution. The record carries exact
`prefill_chunk_pairs` as `[prefix_len, append_len]` and exact
`decode_kv_lens`; the extractor does not parse historical prose formats. The parser
accepts only indexed iteration ranges and attributes kernels by
`kernel.correlationId → runtime.correlationId → indexed phase`. CUDA-graph
profiles must also use `--cuda-graph-trace=node` and verify non-null
`graphNodeId` rows. The launcher defaults Nsight's optional device-side CUDA
Event completion tracing to `--cuda-event-trace=false`: it is not needed for
kernel/NVTX attribution and can trigger cross-stream false dependencies and
Xid 32 faults with multi-GPU NVLink all-to-all backends. The explicit
`nsys.cuda_event_trace` field can override that default for a controlled
diagnostic capture.

Request TTFT/TPOT instrumentation reuses vLLM's native EngineCore events and clock:
`QUEUED` is recorded when `Scheduler.add_request` appends the request to its
waiting queue, and the end is the `EngineCoreOutputs.timestamp` created after
the iteration producing the first token has returned and its model output has
been processed. `engine_core_ttft_ms` spans those boundaries and is decomposed
into `engine_queue_wait_ms` and
`engine_first_schedule_to_first_token_ms`. For chunked prefill the first-token
boundary remains the iteration that actually emits the first token, not the
first prompt chunk.

The same schema-v2 record is emitted once, when the request completes. Its
`engine_core_decode_ms` spans the first-token EngineCore output timestamp to the
last-token EngineCore output timestamp, and `engine_core_tpot_ms` is that span
divided by `num_output_tokens - 1`. A one-token request records a null TPOT
because it has no inter-token interval. The v1 TTFT-only record remains readable
for older captures.
The extractor removes vLLM completions' `cmpl-<X-Request-Id>-0` envelope into
the canonical req-frontend `request_id` and retains the raw `engine_request_id` for
audit.
The NSYS profile runner extracts these lines to
`profile/vllm/<name>_request_timings.jsonl` and records that path in
`profile_result.json`. The full scheduler metrics JSONL is likewise retained in
`profile_result.metrics_jsonl`; e2e workload analysis consumes it directly and
uses the NSYS iteration IDs only to identify the measured replay's contiguous
segment. These engine-core metrics exclude HTTP/frontend ingress,
tokenization, detokenization, SSE return, `[DONE]`, and client post-processing;
req-frontend's fields remain the client-observed latency measurements.

When API timing instrumentation is enabled, the same request artifact also
contains API-process durations from endpoint entry through the first token SSE
yield. API timing schema v3 decomposes lazy generator activation,
`AsyncLLM.add_request`, waiting for the first EngineCore output, output
processing/fan-out to the per-request collector, collector wakeup, generator
resume, and first-token serialization. The collector interval closes as API
EngineCore-output wait plus output fan-out. The extractor compares durations
within their originating clock domains and never subtracts EngineCore and API
absolute timestamps across processes.

The separate `expert_popularity` pass sets
`VLLM_NVTX_SCOPES_FOR_PROFILING=0`, so it neither emits nor requires this
request-timing artifact. Its only model-side ground truth is the expert-load
record stream described above; timing evidence always comes from the NSYS pass.
The resulting summary follows
[`alignment/schema/expert_popularity_v2.schema.json`](../schema/expert_popularity_v2.schema.json):
`counts_by_layer[layer][logical_expert]` is authoritative, while EP degree,
top-k, aggregation scope, and the simulator's rank-major logical partition are
explicit provenance rather than implicit loader assumptions.

For long multi-GPU workloads, set `nsys.capture_duration_seconds` with
`capture_mode: cuda_profiler_api`. The launcher stops CUPTI at that deadline but
does not stop the replay: the bounded window supplies representative kernel
segments while all requests still finish and emit EngineCore TTFT/TPOT. This
also avoids letting graph-node tracing perturb multi-rank collectives for the
entire benchmark.

NSYS exports the CUPTI runtime and kernel tables without lookup indexes. Before
attribution, the parser adds persistent `vibesim_` indexes on
`runtime(globalTid, start)` and `kernel(globalPid, correlationId, start)` to the
exported SQLite derivative. The `.nsys-rep` remains the immutable capture. The
indexes do not change event rows or attribution results; they prevent one full
runtime/kernel table scan per indexed phase range and are reused by later parse
runs.

## Parser output contract

`python -m alignment parse` writes two views of the same capture. The
`iteration_details` list is the ground-truth timeline; `kernel_names` maps its
integer `name_id` values to complete demangled kernel names. (`json` represents
those integer dictionary keys as strings.) `by_stage` and `all` are derived
summaries for later comparison and statistics. Do not treat a summary category
as a replacement for the underlying kernel launches.

`kernel_sequences` is the folded human-labeling inventory derived from that
timeline. For each captured phase it exact-deduplicates the complete ordered
kernel sequence, then replaces only exact contiguous repetition with a
`repeat.count + body.kernels` node. Stored kernel occurrences carry their full
demangled `name` and parser-owned `suggested_category`; there is no numeric
kernel catalog. Each unique sequence owns its source `iterations`, so no
materialized iteration-assignment index is stored. Expanding the program and
deriving `sequence_id:ordinal` must reproduce the original sequence exactly.
The profile runner writes this schema-v2 catalog separately as
`profile/kernel_sequences.json` for copying and labeling.

Each `iteration_details` entry contains the indexed iteration, its
`iteration_type` (`prefill` or `decode`), the raw vLLM metrics row, and one or
more selected NVTX ranges. Each range contains every owned kernel in
chronological launch order:

```text
iteration_details[]
  iteration, iteration_type, stage, metrics
  ranges[]
    device_id, worker, phase, start_ns, end_ns
    kernels[]
      ordinal, name_id, category, start_ns, end_ns
      stream_id

kernel_sequences.<phase>
  unique_sequences[]
    sequence_id, iterations, expanded_kernel_count
    program[]
      kernels[] {name, suggested_category}
      or repeat {count, body.kernels[]}
```

Complete demangled names occur once in `kernel_names`; timeline records carry
only `name_id`. `ordinal` is one-based within its range. All timestamps are the
raw NSYS nanosecond timeline. Relative offsets and durations are intentionally
not serialized because they are derived directly from the range and kernel
start/end timestamps. Runtime correlation, launch-shape, and resource fields
are used only while parsing and are not repeated in every kernel record. The
parser's `--range-mode` controls whether those ranges cover only `forward`, all
indexed `phases`, or an `envelope`; `phases` is the default so the ground-truth
timeline does not silently omit preprocessing, sampling, or bookkeeping.
`by_phase` keeps the derived summaries separated by phase, and the later
timing-predict config explicitly selects `input_builder.measured_phase`. The
range mode does not change the per-kernel record schema.

## Launch contract

Do not launch this profiler directly or keep run configs in this implementation
directory. Create one dated experiment directory and preserve every phase config
there; each may be YAML or JSON. The profile is the first capture-producing stage;
the simulation runs later (after kernel-align derives the multiplier it injects):

```bash
uv run python -m launcher alignment profile logs/<experiment>/profile.yaml
# ... timing-predict, kernel-align ...
uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml \
  --gpu-time-multiplier-from logs/<experiment>/analysis_kernel
```

`profile.yaml` is only the vLLM/req-frontend/NSYS run input. It owns one artifact
root and contains no timing-predict or analyzer policy:

```yaml
schema_version: 1
name: llama3_8b
log_dir: ./profile
gpu: NVIDIA H200
server: ...
nsys: ...
workload: ...
```

The launcher removes an inherited `VLLM_API_KEY` from the local vLLM server
subprocess. The paired req-frontend replay client intentionally targets an
unauthenticated loopback endpoint; ambient shell credentials must not turn that
private measurement boundary into an authenticated API and fail preflight with
HTTP 401.

The same launch boundary enables vLLM prompt-token details by default. req-frontend
performs a mandatory two-request prefix-cache preflight and reads
`usage.prompt_tokens_details.cached_tokens`; experiment presets do not need to
repeat `--enable-prompt-tokens-details` in `server.extra_args`.

The simulation preset independently uses `io.log_dir`. Paths in `profile.yaml`
are resolved relative to that file.

The profiling config's `workload.frontend.path` should equal the simulation
config's sole `workload.trace_files` entry. `workload.frontend.type` selects a
typed req-frontend trace format: `independent` for
`id,input_len,output_len,arrival_time`, or `session` for the canonical
round/prefix/tool-wait schema. Request construction is
currently synthetic text and reads `workload.text_file` plus
`workload.tokenizer` directly. Add a tagged request-builder config only when
another implementation has real runtime dispatch. No missing fields are filled
with zeros.

After inspecting `profile/parsed.json`, write the typed `vllm_text/single`
conversion in `timing_predict.yaml`. After timing-predict exposes simulator
slots, copy `profile/kernel_sequences.json`, add an explicit embedded label to
every stored kernel occurrence, and reference that labeled JSON from
`analyze.yaml`. Capture and timing-predict never consume those labels. See
`../README.md`; the labeled copy belongs beside the experiment configs.
