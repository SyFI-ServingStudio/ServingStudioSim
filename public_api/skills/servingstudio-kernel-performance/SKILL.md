---
name: servingstudio-kernel-performance
description: >-
  Look up measured GPU kernel performance for LLM serving (GEMM, attention, MoE,
  normalization, communication) from the ServingStudio Kernel Library API, and
  find which shapes a model deployment runs. Not for end-to-end serving latency.
---

# ServingStudio Kernel Performance

The ServingStudio Kernel Library publishes GPU kernel measurements that the
ServingStudio simulator uses to predict LLM serving performance. Each kernel
kind, such as `single_gemm` or `nvfp4_fused_moe`, has a table of rows, one per
combination of argument values, GPU and backend. Use this skill to answer
questions such as "how long does a bf16 GEMM with m=64, n=6144, k=4096 take on
H200?" or "which attention kernels does GLM-5.2 run on B200, and at what
shapes?"

The API is read-only JSON over HTTPS. Set the base URL once:

```bash
API=${SERVINGSTUDIO_API:-https://servingstudio.cs.washington.edu/api/public/v1}
curl -s "$API/health"
```

If `/health` does not return `{"status": "ok", ...}`, ask the user for the base
URL; do not guess another host.

## 1. Find the kernel kind

The catalog lists every kind with its title, category, backends, precisions,
row count and the models that use it.

```bash
curl -s "$API/kernels" | jq -r '.kernels[] | [.kind, .category, .title] | @tsv'
curl -s "$API/kernels" | jq '.kernels[] | select(.used_by | index("glm52_nvfp4")) | .kind'
```

`used_by` holds model-config stems; `.models[]` in the same response maps each
stem to its display name and Hugging Face checkpoint. `.gpus[]` gives each GPU's
spec-sheet peaks (dense `peaks.tflops.by_dtype`, `peaks.memory_bandwidth_gbps`),
which put a measured number in context.

## 2. Read what the kind measures

```bash
curl -s "$API/kernels/single_gemm" | jq '{title, description, formula, method, caveats, args}'
```

Read `args` before filtering rows. Each entry gives the argument's `unit` and
meaning (`doc`). Its `role` says whether a model fixes the value (`config`) or
the value grows with the batch (`sweep`); `role` is null when no supported
deployment runs the kind.

The other fields explain the numbers. `formula` defines the FLOPs and bytes
behind the throughput metrics, `method` states how time was measured, and
`caveats` lists what the measurement leaves out. `backends` describes each
implementation (for example `torch`, `deepgemm`, `flashinfer_trtllm_sm100`)
with a link to its source.

To see the shapes a model actually runs, read `used_by`. Each deployment names
its `model_config` and `gpu` and lists `shapes[]`, where `layer` names the model
layer and `db` holds the fixed argument values to filter rows by.

```bash
curl -s "$API/kernels/single_gemm" \
  | jq '.used_by[] | select(.model_config == "llama3_8b" and .gpu == "NVIDIA H200")
      | .shapes[] | {layer, db}'
```

## 3. Get the rows

```bash
curl -s "$API/kernels/single_gemm/rows?gpu=NVIDIA%20H200&dtype=bf16&n=6144&k=4096"
curl -s "$API/kernels/single_gemm/rows?gpu=NVIDIA%20H200&n=6144&format=csv" > rows.csv
```

Every query parameter except `format` filters one column by exact equality:
`gpu` (full name, such as `NVIDIA H200`), `backend`, or any argument; an
unknown column returns 400. There are no range filters, so download a superset
and filter ranges yourself. The JSON is column-oriented:

```bash
curl -s "$API/kernels/single_gemm/rows?gpu=NVIDIA%20H200&n=6144&k=4096" | jq -r '
  .columns as $c | .rows[] | [$c, .] | transpose | map({(.[0]): .[1]}) | add
  | [.backend, .m, .time_ms, .tflops] | @tsv'
```

In JSON, each row's last column, `provenance`, indexes the response's
`provenance` list, which records when the row was measured (`profiler_run_at`)
and with which CUDA, driver and backend versions. CSV repeats those fields on
every row instead.

## 4. Interpret and report

- Read the kind's `method` before quoting `time_ms`; the timing differs by kind
  and sometimes by backend. Most kernels report CUPTI kernel time with the L2
  cache flushed before each launch, so the numbers are cold-cache times. Some
  `torch` backends use CUDA events around back-to-back calls, which includes
  the gaps between launches. Collectives time CUDA graph replays or CUDA events
  across ranks. When two backends use different methods, say so before
  comparing them.
- A `method` can say that nothing is timed and the values come from a stored
  table or a formula. Report those rows as modeled, not measured.
- `tflops` and `memory_bandwidth_gbps` are logical rates: the work or bytes in
  the kind's `formula`, divided by time. A value of 0 means the kind does not
  compute that metric.
- Quote only rows the API returns. If the exact shape is missing, say so and
  give the nearest rows; label any interpolation as your own estimate.
- When you answer, name the kind, GPU, backend, argument values, the metric with
  its unit, and `profiler_run_at`.

## Kernel configs (optional)

`/kernels/{kind}/configs` lists the grids the simulator reads for each
deployment, and `/kernels/{kind}/configs/{config_hash}` returns one grid with
every cell's measured metrics per backend. Use these only when the user asks
how the simulator uses the data; rows answer most performance questions.
