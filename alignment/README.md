# VibeSim Alignment

Alignment validates VibeSim against one measured vLLM run. It is an explicit
four-phase workflow; each phase has one config and one disjoint artifact root.

```text
simulation.yaml ── alignment sim ────────────────→ simulation/
profile.yaml ───── alignment profile ────────────→ profile/
timing_predict.yaml ─ alignment timing-predict ─→ timing_predict/
analyze.yaml ────── alignment analyze ───────────→ analysis/
```

No phase implicitly launches the next one. YAML and JSON are both accepted.
`simulation.yaml` remains an ordinary VibeSim preset. Paths in the other three
configs are resolved relative to the declaring config file.

## Ownership

```text
alignment/
  runner.py                 measured profile controller
  load_generator/           thin TraceLab invocation adapter
  profiler/                 vLLM lifecycle and NSYS capture
  nsys/                     NSYS SQLite normalization and sequence catalog
  timing_predict_input/     measured iteration → generic predictor inputs

launcher/
  alignment.py              four commands and cross-stage orchestration
  alignment_config.py       strict phase YAML/JSON schemas and path resolution

analyzer/
  rust + python             comparison, statistics, payloads, and plots
```

`alignment/timing_predict_input` does not parse phase configs or launch the
predictor. It receives resolved artifacts from the launcher and writes only
timing-predict inputs. The analyze launcher alone creates
`analysis/alignment_manifest.json`.

## Phase configs

Keep the four files beside each other in one dated experiment directory.

### `simulation.yaml`

This is the normal simulation preset. Its output remains `io.log_dir`:

```yaml
deployment: unified
workload:
  trace_files: [trace/requests.csv]
io:
  log_dir: logs/<experiment>/simulation
pools: ...
```

### `profile.yaml`

This file owns only the real vLLM/TraceLab/NSYS run:

```yaml
schema_version: 1
name: llama3_8b
log_dir: ./profile
gpu: NVIDIA H200
cuda_visible_devices: "0"
fork_python: ../../alignment/profiler/vllm/.venv/bin/python

server:
  model_path: meta-llama/Meta-Llama-3-8B
  tp_size: 1
nsys:
  capture_mode: cuda_profiler_api
workload:
  frontend:
    type: vibesim
    path: ../../trace/requests.csv
  text_file: ../../trace/prompts.txt
  tokenizer: meta-llama/Meta-Llama-3-8B
  max_concurrency: 64
```

It writes the replay log, server log, NSYS report/SQLite, `parsed.json`, the
folded label-ready `kernel_sequences.json`, engine-core per-request TTFT/TPOT in
`vllm/<name>_request_timings.jsonl`, and `profile_result.json` beneath
`profile/`. `profile_result.replay_result` identifies the TraceLab per-request
JSONL used later for client-observed E2E comparison, while
`profile_result.request_timings_jsonl` identifies the vLLM engine-core timing
records. It has no timing-predict or analysis fields.

### `timing_predict.yaml`

After inspecting `profile/parsed.json`, select an explicit tagged input builder:

```yaml
schema_version: 1
simulation_log_dir: ./simulation
profile_log_dir: ./profile
log_dir: ./timing_predict
input_builder:
  type: vllm_text
  measured_phase: forward
  group_assignment: single
```

The builder writes `timing_predict_cases.json`,
`timing_predict_case_map.json`, `timing_predict_config.json`, and
`timing_predict_input_manifest.json`, then the launcher invokes the generic
timing-predict command. The generated predictor config copies the normalized
simulation's complete per-role backend policy along with its arch and GPU, so
iteration prediction rebuilds the same CostTree instead of reverting to an
arch's default best-of-N candidates. Future multimodal/sharded builders are
sibling tagged variants with their own required fields; no sparse all-purpose
record is used.

This phase compares the simulation and profile source traces. A mismatch emits
a warning and continues. It never reads embedded kernel labels or writes
`analysis/`.

### `analyze.yaml`

After timing prediction, copy the folded inventory and label every stored kernel
occurrence. Repeat-body labels apply to every exact expansion:

```bash
cp profile/kernel_sequences.json kernel_sequences_labeled.json
```

Every occurrence must carry either `{"label": {"status": "unmapped"}}` or a
complete mapped label with `operation`, `type`, `role`, and a non-empty
`simulated_slots` list. One measured operation may own multiple simulated leaf
slots; the analyzer counts its measured kernels once and sums the selected
folded leaf workloads. This is an operation-workload comparison, not an
overlap-aware wall-clock sum when the slots sit under `Max` branches.
The labeled JSON is the only mapping source; there is no separate mapping YAML.

```json
{
  "label": {
    "status": "mapped",
    "operation": "layer.attention",
    "type": "attention",
    "role": "attention main and combine",
    "simulated_slots": [
      "unified.attn.main",
      "unified.attn.combine"
    ]
  }
}
```

```yaml
schema_version: 1
simulation_log_dir: ./simulation
profile_log_dir: ./profile
timing_predict_log_dir: ./timing_predict
log_dir: ./analysis
iteration:
  enabled: true
  labeled_kernel_sequences_file: ./kernel_sequences_labeled.json
workload:
  enabled: true
e2e:
  enabled: true
  throughput_bins: 20
```

Iteration, workload, and E2E subjects can be enabled independently. Workload
analysis plots each side against its recorded iteration ids and emits fine-grained
prefill-token, decode-batch-size, scheduled-KV-workload, and actual iteration-cycle
series. It also plots decode batch size against each side's independently
normalized elapsed time. vLLM cycle time is first-kernel to next-first-kernel;
simulation cycle time is one actual `wall_start_ms` to the next, so it already
includes the worker's `gpu_time_multiplier`, tick quantization, and scheduler
gaps. Scheduled KV
workload is `sum(decode_kv_lens) + sum(prefill_prefix_len +
prefill_append_len)`. It is deliberately not resident KV-pool occupancy. A
labeled folded inventory is required only when iteration analysis is enabled.
The launcher validates and snapshots it into `analysis/`; the analyzer
losslessly expands all captured phases, validates names/categories against
`parsed.json`, and validates mapped slots against the timing-predict cost
manifest.

E2E analysis writes two distinct TTFT overlays: TraceLab client-observed TTFT vs
simulator TTFT, and vLLM engine-core queued-to-first-output TTFT vs the same
simulator TTFT. It likewise keeps client-accounted TPOT and vLLM engine-core
first-output-to-last-output TPOT as separate overlays against simulator TPOT,
then overlays client E2E. Each measured/simulated pair is compared as independent
distributions; no per-request latency ratio is computed. Request ids only audit
whether either side lost requests, which matters when the two schedulers execute
simultaneous arrivals in different orders.
An older profile captured before engine-core request timing instrumentation can
still provide client TTFT, TPOT, E2E, and throughput analysis: its manifest
records `request_timings_result: null`, the report marks server TTFT/TPOT
unavailable, and no server plots are rendered. A schema-v1 request-timing file
continues to provide server TTFT while server TPOT is unavailable. Missing timing
data is never reconstructed or substituted from client measurements.

## Commands

```bash
uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml
uv run python -m launcher alignment profile logs/<experiment>/profile.yaml
uv run python -m launcher alignment timing-predict logs/<experiment>/timing_predict.yaml
uv run python -m launcher alignment analyze logs/<experiment>/analyze.yaml
```

Standalone NSYS normalization remains available as:

```bash
uv run python -m alignment parse --sqlite capture.sqlite --metrics metrics.jsonl \
  --iteration-start 24 --iteration-end 48 --output parsed.json
```

After a profile completes, derive that capture's experiment-specific GPU-cycle
correction with:

```bash
uv run python -m alignment gpu-kernel-ratio \
  --profile-dir logs/<experiment>/profile \
  --output logs/<experiment>/gpu_kernel_ratio.json
```

The command defines a GPU cycle as one iteration's first attributed kernel start
to the next kernel-bearing iteration's first kernel start on the same device.
It reports the attributed kernel union for ownership auditing and independently
unions every CUPTI kernel on that device inside each cycle.  The pooled
`gpu_time_multiplier` is `sum(GPU cycle) / sum(global kernel busy)`; copy that
value into the selected simulation worker and rerun the simulation.  The final
iteration per device is excluded because it has no next boundary.  Never reuse
the factor as a GPU-wide constant across unrelated captures.

## Current v1 boundary

- one direct `deployment: unified` simulation target;
- one main group and one replica;
- one vLLM model rank (`server.tp_size: 1`);
- `vllm_text/single` is the first input-builder variant;
- exact folded sequence positions and unmatched measured/simulated kernels remain explicit.
