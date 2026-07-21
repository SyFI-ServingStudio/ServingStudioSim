# VibeSim Alignment

Alignment validates VibeSim against one measured vLLM run. It is an explicit
phased workflow; each phase has one config and one disjoint artifact root. The
GPU-cycle duty-cycle correction (`gpu_time_multiplier`) is a pure measured
quantity: the analyzer's **kernel-align** pass derives it before the simulation,
and the simulation phase injects it automatically. So `analyze` splits into two
semantic passes that bracket the simulation:

```text
profile.yaml ─────── alignment profile ──────────────→ profile/
timing_predict.yaml ─ alignment timing-predict ──────→ timing_predict/   (reads simulation.yaml preset)
analyze_kernel.yaml ─ alignment analyze (kernel-align)→ analysis_kernel/  (emits recommended_gpu_time_multiplier)
simulation.yaml ──── alignment sim ──────────────────→ simulation/        (auto-injects the multiplier)
analyze_e2e.yaml ─── alignment analyze (e2e-align) ──→ analysis_e2e/      (consumes the completed sim)
```

No phase implicitly launches the next one. YAML and JSON are both accepted.
`simulation.yaml` remains an ordinary VibeSim preset. Paths in the other configs
are resolved relative to the declaring config file.

## Ownership

```text
alignment/
  runner.py                 measured profile controller
  load_generator/           thin TraceLab invocation adapter
  profiler/                 vLLM lifecycle and NSYS capture
  nsys/                     NSYS SQLite normalization and sequence catalog
  timing_predict_input/     measured iteration → generic predictor inputs

launcher/
  alignment.py              phase commands and cross-stage orchestration
  alignment_config.py       strict phase YAML/JSON schemas and path resolution

analyzer/
  rust + python             comparison, statistics, payloads, and plots
```

`alignment/timing_predict_input` does not parse phase configs or launch the
predictor. It receives resolved artifacts from the launcher and writes only
timing-predict inputs. The analyze launcher alone creates
`analysis/alignment_manifest.json`.

## Phase configs

Keep the config files beside each other in one dated experiment directory.

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
simulation_preset: ./simulation.yaml
profile_log_dir: ./profile
log_dir: ./timing_predict
input_builder:
  type: vllm_text
  measured_phase: forward
  group_assignment: single
```

Timing prediction is kernel-only, so it reads the simulation **preset**
(`simulation_preset`), not a completed run — it takes the gpu, arch, and backend
policy straight from `simulation.yaml`. This lets it run before the simulation,
so kernel-align can derive the multiplier the simulation later bakes in. The
builder writes `timing_predict_cases.json`, `timing_predict_case_map.json`,
`timing_predict_config.json`, and `timing_predict_input_manifest.json`, then the
launcher invokes the generic timing-predict command. The generated predictor
config re-nests the preset's flat `"pool/role": [...]` backend policy into
`{pool: {role: [...]}}` along with its arch and GPU, so iteration prediction
rebuilds the same CostTree the simulation will use instead of reverting to an
arch's default best-of-N candidates. Future multimodal/sharded builders are
sibling tagged variants with their own required fields; no sparse all-purpose
record is used.

This phase compares the preset and profile source traces. A mismatch emits a
warning and continues. It never reads embedded kernel labels or writes
`analysis/`.

### `analyze_kernel.yaml` and `analyze_e2e.yaml`

`analyze` splits into two semantic passes selected purely by which subjects are
enabled — there is no new flag:

- **kernel-align** (`iteration.enabled`) — measured↔predicted per-kernel/iteration
  cost accuracy. It needs no completed simulation, so `simulation_log_dir` is
  omitted. It emits `recommended_gpu_time_multiplier` into
  `reports/alignment_iteration_report.json`.
- **e2e-align** (`workload.enabled` / `e2e.enabled`) — both read the completed DES
  simulation (cost_log / request_slo), so `simulation_log_dir` is required and
  points at the sim that already baked in the multiplier.

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
overlap-aware wall-clock sum when the slots sit under `Max` branches. Conversely
one simulated slot may be owned by several operations (a fused aggregate boundary
and an unfused split boundary sharing one `tp_allreduce` slot); the analyzer
resolves the per-iteration owner from the operations present. One *operation*
still keeps a single consistent `type`/`role`/`simulated_slots`.
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
# analyze_kernel.yaml — kernel-align; no simulation needed
schema_version: 1
profile_log_dir: ./profile
timing_predict_log_dir: ./timing_predict
log_dir: ./analysis_kernel
iteration:
  enabled: true
  labeled_kernel_sequences_file: ./kernel_sequences_labeled.json
```

```yaml
# analyze_e2e.yaml — e2e-align; consumes the completed sim
schema_version: 1
simulation_log_dir: ./simulation
profile_log_dir: ./profile
timing_predict_log_dir: ./timing_predict
log_dir: ./analysis_e2e
workload:
  enabled: true
e2e:
  enabled: true
  throughput_bins: 20
```

One analyze config is exactly one phase: enable `iteration` (kernel-align) or
`workload`/`e2e` (e2e-align), never both — the two write distinct typed manifests
and mixing them is rejected. Every subject defaults to disabled, so a phase is
opted into by naming only its block; `simulation_log_dir` is required only for
the e2e-align phase. Workload
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
Throughput reports both client completion time and server GPU time. The client
rate uses TraceLab's earliest post/submit through latest completion. The server
GPU rate uses parsed NSYS's first observed kernel start through last observed
kernel end, so it includes inter-iteration gaps and the terminal iteration that
first-kernel-to-next-first-kernel cycle sums omit. The per-bin plot remains a
client-completion versus simulation comparison because NSYS does not attribute
completed output tokens to individual server timestamps; its summary box shows
all three aggregate rates without inventing a server token-production series.
An older profile captured before engine-core request timing instrumentation can
still provide client TTFT, TPOT, E2E, and throughput analysis: its manifest
records `request_timings_result: null`, the report marks server TTFT/TPOT
unavailable, and no server plots are rendered. A schema-v1 request-timing file
continues to provide server TTFT while server TPOT is unavailable. Missing timing
data is never reconstructed or substituted from client measurements.

## Commands

```bash
uv run python -m launcher alignment profile logs/<experiment>/profile.yaml
uv run python -m launcher alignment timing-predict logs/<experiment>/timing_predict.yaml
uv run python -m launcher alignment analyze logs/<experiment>/analyze_kernel.yaml
uv run python -m launcher alignment sim logs/<experiment>/simulation.yaml \
  --gpu-time-multiplier-from logs/<experiment>/analysis_kernel
uv run python -m launcher alignment analyze logs/<experiment>/analyze_e2e.yaml
```

`--gpu-time-multiplier-from <kernel-align-dir>` makes the simulation read that
pass's `recommended_gpu_time_multiplier` and inject it as
`--override pools.main.groups.0.worker.gpu_time_multiplier=<v>` — no manual copy.
Omit the flag to run the simulation with whatever `gpu_time_multiplier` the
preset's worker already carries (defaults to 1.0).

Standalone NSYS normalization remains available as:

```bash
uv run python -m alignment parse --sqlite capture.sqlite --metrics metrics.jsonl \
  --iteration-start 24 --iteration-end 48 --output parsed.json
```

The duty-cycle correction is derived by the analyzer's kernel-align pass, not a
standalone command. It pools `Σ measured_gpu_cycle_ms / Σ measured_ms` over
iterations that have a next-iteration GPU cycle (a GPU cycle is one iteration's
first attributed kernel start to the next kernel-bearing iteration's first kernel
start on the same device; the terminal iteration per device is excluded). The
numerator and denominator share the analyzer's per-occurrence cross-rank
reduction, so the kernel-layer and GPU-cycle-layer gaps stay consistent. Treat
the factor as experiment-specific, never a GPU-wide constant.

## Current boundary

- one direct `deployment: unified` simulation target;
- one main group and one replica;
- one symmetric tensor-parallel vLLM replica; the visible CUDA-device count must
  equal `server.tp_size`, and the simulation arch must carry the same `tp_size`;
- `vllm_text/single` is the first input-builder variant;
- exact folded sequence positions and unmatched measured/simulated kernels remain explicit.

For tensor parallelism, every rank keeps its own normalized ranges. The folded
labeling inventory stores one representative sequence only after proving that
the ordered `(name, suggested_category)` sequence is identical on every device
for every phase/iteration. Analysis applies those labels independently to each
rank and reduces per occurrence across ranks (an independent op takes the
max-rank duration; a synchronizing collective takes `max(end) − max(start)`,
dropping arrival wait), never by summing GPU durations. The kernel-align
multiplier is derived from this same per-occurrence measured population, so its
`measured_ms` numerator matches the breakdown the analyzer reports.
