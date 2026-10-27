# Llama 3.1 8B on Trainium2 (vLLM Neuron 0.29): modeling notes and evidence

The branch models Llama 3.1 8B BF16 served by vLLM Neuron 0.29 (Neuron SDK
2.33) at TP4 over the four LNC2 logical cores of one Trainium2 chip, and
checks it against the engine on a trn2.3xlarge. This record describes the
pieces and holds the evidence.

## Deployment modeled

- One chip per replica, TP4 over its four LNC2 logical cores (two physical
  cores each), BF16 weights and KV, the original 32-layer checkpoint. The GPU
  key is `AWS Trainium2 LNC2`, an alias of the catalog entry `Trainium2-LNC2`.
- The engine is started with a fixed paged KV pool,
  `--num-gpu-blocks-override 6782`: 6782 pages of 32 tokens, 217,024 token
  slots, 32768 bytes per token per rank. The 0.29 engine accepts the override
  and reports "GPU KV cache size: 217,024 tokens, Maximum concurrency for 512
  tokens per request: 423.88x". These constants live once per language:
  `profiling/runners/neuron/stock_contract.py` and
  `simulator/src/timing/kernels/neuron_nki.rs`.
- `--max-model-len 512`, `--max-num-batched-tokens 512`, prefix caching off,
  one compiled prefill graph `[512]` and one compiled decode graph per
  `num_seqs_buckets` entry.
- Two decode attention paths, selected by `decode_attention`:
  - `torch_fallback`: the stock Llama3 `forward_decode`. It passes a 2-D block
    table to `attention_decode`, which routes a GQA request at TP4 to the
    functional's torch fallback (a GpSimd gather of the K/V blocks, dense
    attention).
  - `nki_kernel` (the default): a `forward_decode` patched to the per-head
    block table (the qwen3_vl production decode path), which routes to the
    `attention_block_tkg` NKI kernel. The patch is a 50-line change applied
    to the engine for the measurement; it is not part of this repository.

## Engine image

One vendor image, pinned in `profiling/container/images.py`:

| Image (profile env) | Release | Route |
| --- | --- | --- |
| `vllm-neuron-native` (`vllm_neuron_native_env`), `sha256:d6602265…`, tag `0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | vLLM Neuron 0.29.0.1.2.0, Neuron SDK 2.33, vLLM 0.29.0, torch 2.13 | native: `libtorch_neuronx_lite` 2.13 (Dynamo compile backend `neuron_native_lite`, the `neuron` PrivateUse1 device), neuronx-cc 2.28, nki 0.7; no `torch_neuronx` |

SDK 2.33 makes the native route the default of vLLM Neuron; SDK 2.34 removes
the PyTorch/XLA route. Facts of the image that the runners depend on:

- The compile backend is `vllm_neuron.envs.get_compile_backend_name()`
  (`neuron_native_lite`). `neuron` and `neuron_libtorch` are not registered.
- NRT profiling is `torch.classes.neuron.Runtime().start_profiling(dir,
  activities, None, None, None)`, the binding of the engine's own
  `/start_profile`. `neuron-explorer` 2.33 still exports JSON.
- A single-rank process must set `NEURON_RT_ROOT_COMM_ID`, or the NEFF fails
  to load with `NRT_INVALID ... cannot load NEFF with collective operations`.
  The lite backend emits collectives even for one rank.
- `torch.distributed` has a `neuron` backend after
  `libtorch_neuronx_lite._compiler.distributed.backend._register_neuron_backend()`.
  `dist.barrier()` is unusable (no accelerator hooks); a one-element
  all-reduce is the fence.

## How the engine's behavior is modeled

| Engine behavior (0.29, observed) | Where it is modeled |
| --- | --- |
| Static graphs: a prefill pads to the smallest prefill bucket, a decode batch to the smallest decode bucket | `llama3_neuron_ops::lower_variant` / `lower_decode`; leaves read `Cache1DDirect` rows per bucket |
| One prefill per step, never mixed with decode; FIFO; no prefix caching | `ensure_stock_neuron_worker` pins `chunked_prefill` to it, with `max_prefill_requests_per_iteration: 1` |
| Residents capped by the largest decode bucket and the pool | `max_resident_requests_per_partition` = min(largest bucket, 217024 // C) |
| The decode step is one graph on four ranks with the block all-reduces, the vocab-sharded embedding, the lm_head and the on-device sampler (`all_greedy=False`, `max_top_k=256`) inside | the decode leaves `neuron_decoder_layer` and `neuron_decode_fixed`, measured in a reduced-depth TP4 forward |
| Host time: 2.9 ms per prefill step, none per decode request, 8 ms front end per request, next step planned about 4 ms after a step starts | `host_us_per_prefill_request`, `host_us_per_decode_request`, `frontend_latency_us`, `async_lookahead_us` in the preset |

**Composition.** The arch `llama3_neuron_ops` (`max_model_len`,
`prefill_buckets`, `decode_buckets`, `tp_size`, `decode_attention`,
`decode_granularity`) composes the leaves as `vllm_neuron/model/llama3/model.py`
launches them:

```text
Sum(Scale{32}(Sum(attn_block, mlp_block, glue[, decoder_layer])), head, head_all_gather[, decode_fixed])
attn_block prefill: all_gather, qkv, attention_cte, o_proj, reduce_scatter
attn_block decode:  [attention_decode, all_reduce]     per_op only
mlp_block  prefill: all_gather, mlp, reduce_scatter
mlp_block  decode:  [mlp, all_reduce]                  per_op only
glue, head, head_all_gather: prefill; decode only under per_op
decoder_layer, decode_fixed: decode, under layer (the default)
```

- A prefill must be one prefix-free prompt alone. It pads to the smallest
  configured token bucket that holds it (`num_batched_tokens_buckets`). The
  default inventory `[max_model_len]` is the stock engine's one prefill graph.
- A decode-only batch pads to the smallest decode bucket (`num_seqs_buckets`).
- The branch a phase does not run is charged zero. Its slots stay in the tree,
  so the cost log keeps one manifest: 15 slots under `layer`, 14 under
  `per_op`.
- The block collectives move the padded `bucket x 4096 x 2` bytes. The head's
  all-gather moves `sampled x 128256 x 2` bytes (one sampled row in prefill,
  the batch bucket in decode).
- Every leaf config is validated at build, so a bucket the engine cannot
  compile is a build error.
- Under `decode_granularity: layer` (the default) a decode step is
  `neuron_decode_fixed + 32 x neuron_decoder_layer`. Every decode collective,
  the head and the sampler are inside those two terms, so the blocks' decode
  all-reduce, the head and its all-gather are zero in decode. `per_op` charges
  the standalone decode units instead: the attribution view.

**Serving pairing.** `llama3_neuron_ops` accepts only the original 32-layer
BF16 checkpoint with Llama 3.1 RoPE, RMSNorm and unbiased SiLU. It pairs only
with `chunked_prefill` set to the stock scheduler
(`ensure_stock_neuron_worker`): `max_batch_tokens` = C,
`max_prefill_requests_per_iteration: 1`, `max_resident_requests_per_partition`
= min(largest bucket, 217024 // C), `separate-prefill-priority`, FIFO,
full-footprint admission, `prefix_cache_mode: disabled`, and
`attn_gpu_memory_gb` equal to the pool per rank (7.111442432). Each replica is
one chip running its own engine. The engine confirms this scheduler at
start-up: "Neuron only supports chunking prefills with batch size of 1. Mixing
prefill and decode in the same batch is not supported".

## Kernels

| Kind | Backends | Unit | Phase |
| --- | --- | --- | --- |
| `neuron_nki_qkv` | `vllm_neuron_nki` | `nkilib` `qkv` + RoPE, `[1,T,4096] @ [4096,1536]` | prefill |
| `neuron_nki_attention_cte` | `vllm_neuron_nki` | `attention_cte`, 8 heads x 128, causal over T | prefill |
| `neuron_nki_output_projection` | `vllm_neuron_nki` | `output_projection_cte`, `[1,8,128,T] @ [1024,4096]` | prefill |
| `neuron_nki_mlp` | `vllm_neuron_nki` | `mlp`, gate/up `[4096,3584]`, down `[3584,4096]` | both |
| `neuron_glue` | `vllm_neuron_nki` | the layer's torch ops (norms, residuals, layouts, prefill KV write) | both |
| `neuron_head` | `vllm_neuron_nki` | embedding gather, final norm, the vocab-sharded `lm_head`, argmax | both |
| `neuron_attention_decode` | `vllm_neuron_nki`, `vllm_neuron_torch` | `attention_decode` on the kernel path or the stock torch fallback | decode (per-op view) |
| `neuron_decoder_layer` | `vllm_neuron_nki`, `vllm_neuron_torch` | one decode layer with both block all-reduces, `(t(32) - t(8)) / 24` of a TP4 decode forward | decode |
| `neuron_decode_fixed` | `vllm_neuron_nki`, `vllm_neuron_torch` | the rest of that forward per step: embedding and its all-reduce, RoPE tables, final norm, `lm_head` shard, sampler, the graph's fixed cost; `t(8) - 8 x layer` | decode |
| `neuron_collective` | `vllm_neuron_dist` | `all_reduce` / `all_gather` / `reduce_scatter` over the four ranks, host-timed over 100 back-to-back calls; `Cache1DLinear` over 1 KB..256 MB | prefill, `per_op` |

**Standalone units** (`profiling/runners/neuron/nki_bench.py`). Each unit is
one `torch.nn.Module` at the per-rank shapes of TP4 (`nki_units.py`). It is
compiled as one Dynamo graph on the engine's compile backend with the serving
compiler flags (`-O2`), warmed, then executed 50 times on one reserved LNC2
core under the NRT system profile. `time_ms` is the median device execution
(`nc_exec_running`, union of the core's two physical cores), raw. The
empty-graph floor (0.026 ms) is measured once per batch and kept in the row's
provenance, not subtracted. Rows exist for compiled variants only: prefill
buckets are contexts of the stock ladder (128 to 8192), decode buckets are
powers of two.

**Decode leaves** (`profiling/runners/neuron/forward_bench.py`). Four
processes, one per LNC2 core, compile `nki_units.DecodeForward` at 8 and 32
layers and run each 50 times under the NRT system profile. Per execution the
shortest of the four ranks is kept, because the last rank to launch does not
wait for the others; `layer = (t32 - t8) / 24`, `fixed = t8 - 8 x layer`.
Measurements are cached on the host by a key of the measuring code, the
depths and the image's engine package versions.

C512 unit times (ms, per rank, median of 50):

| Unit | T512 | B1 / B16 |
| --- | ---: | ---: |
| `neuron_nki_qkv` | 0.087 | |
| `neuron_nki_attention_cte` | 0.066 | |
| `neuron_nki_output_projection` | 0.051 | |
| `neuron_nki_mlp` | 0.365 | 0.164 / 0.170 |
| `neuron_attention_decode`, kernel path | | 0.108 / 0.143 |
| `neuron_attention_decode`, torch fallback | | 0.151 / 0.273 |
| `neuron_decoder_layer`, kernel path | | 0.267 / 0.340 |
| `neuron_decode_fixed`, kernel path | | 0.734 / 0.842 |
| `neuron_glue` | 0.134 | 0.021 / 0.038 |
| `neuron_head` | 0.772 | 0.768 / 0.689 |

The head (a 262 MB BF16 `lm_head` shard per rank) is weight-bandwidth bound
at every variant. The collective rows read 20-44 us for 8 KB to 4 MB (the
all-reduce 20-29 us): host-timed queued calls read the issue rate, not the
link. Under `decode_granularity: layer` only prefill reads them.

## Check 1: step alignment against the engine's own profile

The engine is started in the image with the serving arguments
(`vllm serve /model --tensor-parallel-size 4 --max-model-len 512
--max-num-batched-tokens 512 --max-num-seqs 16 --num-gpu-blocks-override 6782
--no-enable-prefix-caching`, `neuron_config` with `num_batched_tokens_buckets`
`[512]` and `num_seqs_buckets` `[1, 16]`, or `[1..128]` for the bucket
ladder). One profile window per shape is opened with `/start_profile` and
`/stop_profile`, and `neuron-explorer` exports the system trace. A shape's
device time is the union of a rank's two physical cores' `nc_exec_running`
spans per execution. The four ranks agree within 0.01 ms. Consecutive
executions overlap by 0.02-0.16 ms, so the step period equals the device
time: the engine is device-bound.

| Shape | Engine | Simulator | Ratio |
| --- | ---: | ---: | ---: |
| prefill T512 (stock / patched engine) | 26.09 / 26.0-27.2 | 26.43 | 1.00-1.01 |
| decode B1 (patched) | 9.77 | 9.27 | 0.95 |
| decode B16 | 11.90 | 11.71 | 0.98 |
| decode B32 | 14.62 | 14.25 | 0.97 |
| decode B64 | 19.24 | 18.72 | 0.97 |
| decode B128 | 32.13 | 30.94 | 0.96 |

The decode rows are `fixed + 32 x layer` from the table above (B32: 1.206 +
32 x 0.408; B64: 1.504 + 32 x 0.538; B128: 2.220 + 32 x 0.898). The 2-5% left
is the same sign at every bucket: device work of the engine's step beyond the
reduced forward (per-step input staging and metadata ops).

Per-op sums against each engine, decode, C512:

| Shape | Stock engine | `torch_fallback`, `per_op` | Patched engine | `nki_kernel`, `per_op` |
| --- | ---: | ---: | ---: | ---: |
| decode B1 | 10.37 | 12.82 (+24%) | 9.77 | 11.45 (+17%) |
| decode B16 | 23.01 | 17.62 (-23%) | 11.90 | 13.46 (+12%) |

Why the decode leaves are a forward and not a sum of units. An in-graph
ablation and the device profile of the units show the cause. Every unit except
the MLP is latency-bound (DMAs outstanding for more than 90% of its span, no
engine busy for half of it), so neighbouring units overlap in one graph by an
amount that depends on the neighbours. The MLP alone is bandwidth-bound (88 MB
of per-rank weights at about 510 GB/s). On the stock engine the fallback's K/V
gathers stretch over the MLP's weight stream, which is why its B16 sum reads
low. Single-core stacks of the layer reproduced B1-B32 only by cancellation:
they missed the in-graph collectives (an all-reduce is 13 us at 8 KB and 27
us at 1 MB in the graph, against 20-29 us host-timed) and the sampler (0.2-1.4
ms a step), and read B64 and B128 2 ms short. Stacks of 1 and 4 layers read
the layer 2-6% low; the compiler schedules a short graph differently from the
32-layer one.

## Simulation against the engine

The engine was replayed with streaming clients that record every output
token's arrival. The first four replays fitted the host terms; the last three
are held out: the engine restarted with `num_seqs_buckets` `[1, 2, 4, 8, 16,
32, 64, 128]` and `max-num-seqs 128`, at loads not used for fitting, with the
simulator's predictions recorded before the engine ran. Ratios simulator /
engine:

| Replay | TTFT p50 | TTFT p90 | TPOT p50 | latency p50 | latency p90 | duration | out tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 isolated requests | 1.00 | 0.99 | 0.96 | 0.98 | 0.97 | 1.00 | 1.00 |
| 32 at once (smoke trace) | 0.92 | 0.95 | 1.00 | 0.96 | 0.96 | 0.96 | 1.05 |
| chat, 2 req/s (300 requests) | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 |
| chat, 8 req/s (600 requests) | 0.98 | 0.91 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 |
| held out, bucket ladder, 2 req/s (200) | 0.98 | 0.99 | 0.95 | 0.95 | 0.95 | 1.00 | 1.00 |
| held out, bucket ladder, 16 req/s (600) | 0.98 | 0.98 | 0.97 | 0.98 | 0.98 | 1.00 | 1.00 |
| held out, bucket ladder, 24 req/s (saturated) | 0.91 | 0.91 | 0.98 | 0.97 | 0.96 | 0.98 | 1.02 |

Host terms, measured from the same replays of the patched engine
(`decode_attention: nki_kernel`). Every streamed token of one step reaches the
client within about 1 ms, so splitting the merged token times at 2 ms gaps
gives one cluster per step. A cluster's interval minus the simulator's device
time for its padded variant is the host time that the device does not hide.

| Term | Knob | Value | Evidence |
| --- | --- | ---: | --- |
| front end: arrival to the scheduler (HTTP, tokenization, hand-off to the engine process) | `workload.frontend_latency_us` | 8000 | isolated requests: TTFT 37.4 ms = 26.4 device + 2.9 prefill host + 8.1 |
| host time per prefill step | `worker.host_us_per_prefill_request` | 2900 | 915 prefill steps: 29.3-29.6 ms against 26.43 device |
| host time per decode request | `worker.host_us_per_decode_request` | 0 | batch 1 to 128: real minus device has no per-request slope; the capture shows no gap between executions |
| plan offset of the async scheduler | `worker.async_lookahead_us` | 4000 | a request that reaches the scheduler within about 4 ms of a decode step's start is prefilled after that step; a later one waits one more step |

The host terms replace `gpu_time_multiplier`, which must stay 1.0: a
multiplier scales with device time, and on this engine the host time does not.
Decode host time is zero because the async scheduler overlaps it with the
device step. Without the host terms the fitted replays read TTFT 0.62-0.84.

## Changes that reach other models

- **`workload.frontend_latency_us`** (default 0): a trace-timed request
  reaches the scheduler this long after it arrives; its TTFT still counts from
  arrival. A non-zero value is an error in saturated mode.
- **Host time per step**: `chunked_prefill` gains `host_us_per_prefill_request`
  and `host_us_per_decode_request`, added to the device time in
  `CostBuffers::run_iter`. They are accepted only with `llama3_neuron_ops` and
  `gpu_time_multiplier: 1.0`; the cost log stays device-only.
- **Request-count limits** on `chunked_prefill`:
  `max_prefill_requests_per_iteration` and
  `max_resident_requests_per_partition` (default unset, bit-identical).
- **Async one-step lookahead**: `chunked_prefill.async_lookahead_us`, accepted
  only with `llama3_neuron_ops`.
- **`Cache1DDirect`** accepts a one-point grid (no kernel on main uses it).
- **Profiling plumbing**: `BackendSupport.device_family` and `architectures`,
  `KernelProfilerSpec.neuron_logical_cores`, the Neuron Docker pool
  (`profiling/exec/neuron.py`), vendor images in `profiling/container/images.py`,
  Neuron rows keep NULL CUDA provenance, and `migrate-db` drops four retired
  0.24 tables (`RETIRED_TABLES`).
- **`gpu/spec.json`**: a `Trainium2-LNC2` entry (alias `AWS Trainium2 LNC2`).

## Modeling choices

- **Prefill per unit, decode per forward.** Prefill units are large GEMMs and
  add up within 1.3% (26.43 against 26.09 ms at T512); decode units overlap
  inside one graph, so decode is measured where it runs.
- **Two decode terms from one measurement.** `fixed + num_layers x layer`
  keeps the layer count a composition input and shows the per-step part
  (embedding, head, sampler) apart from the layers.
- **Depths 8 and 32.** Short stacks schedule differently (see Check 1).
- **Host time is per request, not a multiplier.** It does not scale with
  device time on this engine.
- **`per_op` decode granularity** stays as the attribution view and as the
  only decode path for `torch_fallback` until its forward rows are measured.

## Known gaps and follow-up

- **Small decode buckets read 4-5% fast:** the engine step is about 0.4 ms
  longer than the reduced forward; the source is not located.
- **At saturation (24 req/s) TTFT is 9% low:** a 2% throughput error becomes a
  queueing error.
- **32 simultaneous arrivals read TTFT 8% low:** the front end ingests them one
  after another (about 1.4 ms each), which a per-request latency does not
  express.
- **The stock engine (`torch_fallback`) has no forward rows yet.** It runs
  through `per_op`, checked at the step level only: decode B1 +24%, B16 -23%.
- **The host terms belong to this vLLM release, host and front end.** A new
  target needs the same replay.
- **Coverage:** C512 only (prefill `[512]`; decode leaves `[1..128]` on the
  kernel path; per-op decode units `[1, 16]`); one chip, TP4, BF16, this
  checkpoint and image. Other contexts fill through the launcher's cold cache
  on a Trainium host.
- **Collective rows are host-timed**; a device-side timing is not
  implemented. They are read only by prefill and by `per_op`.
- **No alignment engine runs against vLLM Neuron 0.29 yet**; the Check 1
  capture (`/start_profile`) is the seed of one.

## Reproduce

Timing predict and simulation read the committed profile.db and need no
Trainium:

```bash
uv run python -m launcher presets/unified_llama3_8b_neuron_ops.yaml --dry-run
uv run python -m launcher presets/unified_llama3_8b_neuron_ops.yaml
```

Filling rows runs on a Trainium2 host with Docker and the pinned image. The
launcher's cold cache measures exactly the rows the compiled cost tree looks
up, one vendor-image worker per chunk:

```bash
export VIBESIM_PROFILE_RUNTIME=container
export SERVINGSTUDIO_NEURON_DOCKER_HOST=unix:///var/run/docker.sock
export SERVINGSTUDIO_NEURON_DOCKER_COMMAND="sudo -n docker"
export SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR=/scratch/neuron-cache
uv run python -m launcher presets/unified_llama3_8b_neuron_ops.yaml --refresh
uv run python -m profiling merge-db <checkout>/profiling/profile.db <host>/profiling/profile.db --output merged.db
```

Notes from the measurements:

- Under `--network=none` the worker container needs a resolvable host name
  (`--add-host`).
- Under a whole-chip reservation (`NEURON_VISIBLE_DEVICES=0-3`) a process's
  device index must equal its rank; the forward runner gives each rank its own
  core only.
- `neuron-explorer` Parquet export needs a writable `TMPDIR` and the NEFF whose
  UUID matches the trace in `<profile>/neffs/`.
