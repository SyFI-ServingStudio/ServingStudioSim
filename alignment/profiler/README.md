# vLLM profiler setup

The alignment ground truth comes from the **vLLM submodule in this directory**,
which adds `vllm_iteration(N): <phase>` NVTX scopes and a versioned
`VibeSimAlignmentIteration {json}` model-input record. The fork source is already
vendored (as a git submodule) at:

    alignment/profiler/vllm      # branch: moesim-profile

Only its `.venv` is missing — build it once. `runner.py` defaults `fork_python`
to `alignment/profiler/vllm/.venv/bin/python`; override `fork_python` in
the align config to point elsewhere.

The instrumentation on `moesim-profile` is Python-only (NVTX scopes + iteration
metrics logging), so the **precompiled fast path** works — no multi-hour CUDA
compile:

```bash
cd alignment/profiler/vllm
git status                         # confirm branch moesim-profile, tree present

# 3.12 venv living next to the fork (the path runner.py expects)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the fork's Python over vLLM's precompiled binaries for the pinned
# base version (skips the from-source CUDA build). Match the box's CUDA (12.8).
export VLLM_USE_PRECOMPILED=1
pip install -e .

# VLLM's profiling-only NVTX scopes import this optional package when
# VLLM_NVTX_SCOPES_FOR_PROFILING=1.
uv pip install nvtx

# Sanity: the fork imports and exposes the instrumentation env var.
python -c "import vllm, vllm.envs as e; print(vllm.__version__, hasattr(e, 'VLLM_NVTX_SCOPES_FOR_PROFILING'))"
```

If `VLLM_USE_PRECOMPILED` cannot resolve a wheel for the pinned version, fall
back to a from-source build (`pip install -e .` without the flag) — this compiles
CUDA kernels and takes much longer.

Model weights: the align config uses the HF repo id `meta-llama/Meta-Llama-3-8B`,
resolved from the shared cache at `HF_HOME=/m-coriander/coriander/hf`. Confirm the
launching shell has `HF_HOME` set (the alignment env inherits it) and the repo is
present under `$HF_HOME/hub/models--meta-llama--Meta-Llama-3-8B`.

Verify end to end before a real capture:

```bash
nsys --version                     # /usr/local/cuda-12.8/bin/nsys
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

Unpaired markers such as `gpu_model_runner: ModelRunnerOutput` remain because
they describe separate output/bookkeeping work. `execute_context_C(CT)_generation_G(GT)`
remains a diagnostic annotation. The human `Iteration(N): ...` line is not an
analyzer API: the separate `VibeSimAlignmentIteration {json}` line is the
authoritative request/token shape and stage source. It carries exact
`prefill_chunk_pairs` as `[prefix_len, append_len]` and exact
`decode_kv_lens`; the extractor does not parse historical prose formats. The parser
accepts only indexed iteration ranges and attributes kernels by
`kernel.correlationId → runtime.correlationId → indexed phase`. CUDA-graph
profiles must also use `--cuda-graph-trace=node` and verify non-null
`graphNodeId` rows.

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
directory. Create one dated experiment directory and preserve all four phase
configs there; each may be YAML or JSON. Run the two capture-producing stages
independently:

```bash
uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml
uv run python -m launcher alignment profile logs/<experiment>/profile.yaml
```

`profile.yaml` is only the vLLM/TraceLab/NSYS run input. It owns one artifact
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

The simulation preset independently uses `io.log_dir`. Paths in `profile.yaml`
are resolved relative to that file.

The profiling config's `workload.frontend.path` should equal the simulation
config's sole `workload.trace_files` entry. `workload.frontend.type` selects a
typed TraceLab frontend: `vibesim` for `id,input_len,output_len,arrival_time`, or
`session` for TraceLab's round/prefix/tool-wait schema. Request construction is
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
