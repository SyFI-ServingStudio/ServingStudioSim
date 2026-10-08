---
name: servingstudio-kernel-performance
description: >-
  Look up measured GPU kernel performance for LLM serving (GEMM, attention, MoE,
  normalization, communication) from the ServingStudio Kernel Library API. Not
  for end-to-end serving latency.
---

# ServingStudio Kernel Performance

The ServingStudio Kernel Library publishes GPU kernel measurements that the
ServingStudio simulator uses to predict LLM serving performance. Each kernel
kind, such as `single_gemm` or `nvfp4_fused_moe`, has a table of rows, one per
combination of argument values, GPU and backend. Use this skill to answer
questions such as "how long does a bf16 GEMM with m=64, n=6144, k=4096 take on
H200?"

The API is read-only JSON over HTTPS. Set the base URL once:

```bash
API=${SERVINGSTUDIO_API:-https://servingstudio.cs.washington.edu/api/public/v1}
curl -s "$API/health"
```

If `/health` does not return `{"status": "ok", ...}`, ask the user for the base
URL; do not guess another host. `$API/openapi.json` lists every route.

## 1. Find the kernel kind

The catalog lists every kind with its `title`, `summary`, category, backends,
precisions, row count, `coverage` (rows per GPU, backend and precision) and
`used_by` (the public deployments that call it, by preset id). Match the
user's wording against `title` and `summary`; when the question names a model,
match it against `used_by` instead:

```bash
curl -s "$API/kernels" | jq -r '.kernels[] | [.kind, .category, .title] | @tsv'
curl -s "$API/kernels" | jq '.kernels[] | select(.kind == "single_gemm") | .coverage'
curl -s "$API/kernels" | jq -r '.kernels[] | select(any(.used_by[]; startswith("GLM-5.3-Flash/"))) | [.kind, .category] | @tsv'
```

A kind's name does not fix its precision: `nvfp4_fused_moe` also has fp8
backends. Read the precision from `coverage`.

Check `coverage` before asking for rows: a GPU, backend or precision missing
from it has no rows. `.gpus[]` in the same response names each GPU as rows
spell it (`name`, such as `NVIDIA H200`) with its spec-sheet peaks
(`peaks.tflops.by_dtype.<dtype>`, dense; `peaks.memory_bandwidth_gbps.value`;
`peaks.busbw_gbps.value`, NVLink one direction), which put a measured number in
context.

## 2. Read what the kind measures

```bash
curl -s "$API/kernels/single_gemm" | jq '{title, description, formula, method, caveats, args, backends}'
```

Read `args` before filtering rows. Each entry gives the argument's `unit` and
meaning (`doc`).

The other fields explain the numbers. `formula` defines the FLOPs and bytes
behind the throughput metrics, `method` states how time was measured, and
`caveats` lists what the measurement leaves out. `backends` describes each
implementation (for example `torch`, `deepgemm`, `flashinfer_trtllm_sm100`)
with a link to its source. Its `supports.gpus` names spec-sheet SKUs (such as
`B200-SXM-180GB`), not the GPU names rows carry; filter rows by
`.gpus[].name`.

## 3. Get the rows

```bash
curl -s "$API/kernels/single_gemm/rows?gpu=NVIDIA%20H200&dtype=bf16&n=6144&k=4096"
curl -s "$API/kernels/single_gemm/rows?gpu=NVIDIA%20H200&n=6144&format=csv" > rows.csv
```

Every query parameter except `format` filters one column by exact equality:
`gpu` (a `.gpus[].name`, such as `NVIDIA H200`), `backend`, or any argument; an
unknown column returns 400. An unknown value is not an error: `gpu=H200` returns
200 with no rows (CSV: the header only), as does a GPU that was never measured,
so confirm the pair in `coverage` before you report "no data". There are no
range filters, so download a superset and filter ranges yourself. A response
holds every matching row; there are no pages. The JSON is column-oriented:

```bash
curl -s "$API/kernels/single_gemm/rows?gpu=NVIDIA%20H200&n=6144&k=4096" | jq -r '
  .columns as $c | .rows[] | [$c, .] | transpose | map({(.[0]): .[1]}) | add
  | [.backend, .m, .time_ms, .tflops] | @tsv'
```

Rows also carry `energy_j`, the energy of one call in joules, where measured.
In JSON, each row's last column, `provenance`, indexes the response's
`provenance` list, which records when the row was measured (`profiler_run_at`)
and with which CUDA, driver and backend versions. CSV repeats those fields on
every row instead.

## 4. Kernels a deployment calls

A kind's `used_by` lists the deployments that call it. For the other direction,
which kernels one deployment calls and whether each has rows, read the
deployment's tree (the `servingstudio-timing-predict` skill explains it). The
tree route takes the preset `id` from `/models`, `{checkpoint directory}/{arch}`
(`GLM-5.3-Flash/...`, not the Hugging Face name), and one value of each of the
preset's `axes` as a query parameter:

```bash
curl -s "$API/models" | jq -r '.checkpoints[] | .presets[] | [.id, .gpu, (.axes | map(.name) | join(","))] | @tsv'
curl -s "$API/models/GLM-5.3-Flash/glm53_flash_vllm_fp8_kda_dsa_moe/tree?tp_size=4&enable_expert_parallel=true&max_model_len=8192&workload=diverse_100" > tree.json
jq -r '[.sections[].slots[].kernel] | unique[]' tree.json
jq -r '.sections[] | .slots[] | [.kernel, (.backends | join(",")), .config, .name] | @tsv' tree.json
```

A deployment runs on its preset's one `gpu`. Slots repeat per layer kind and
rank, so the slot list is long; the first jq gives the kinds once. The tree's
`missing` maps a kind to the rows the deployment needs and lacks; empty means
every kernel it calls is measured (`/models` gives the same `missing` per
member). `configs` maps each slot's `config` id to its kind, identity and
missing rows. `/kernels/{kind}/configs/{config}` serves one config's cells and
how many each backend measured, so a slot leads to its measurements.

## 5. Interpret and report

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
