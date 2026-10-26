# Trainium2 stock Llama 3.1 8B: prefill bucket inventories

2026-10-22, branch `trn2-prefill-buckets` (base `f61ecb0f`), trn2.3xlarge,
vendor image `pytorch-inference-vllm-neuronx:0.24.0.1.1.0`.

## Why

The profiler and the simulator pinned vllm-neuron's
`num_batched_tokens_buckets` to `[C]`, so every prompt ran the whole-context
prefill graph. vllm-neuron itself defaults to powers of two from 128 up to C
(`vllm_neuron/utils/bucket_utils.py`). A hardware replay on 2026-10-20 showed
that three prefill buckets alone cut TPOT p90 by 15% at 8 req/s. The simulator
could not model that inventory. Decode buckets stay powers of two, the
vllm-neuron default.

## What changed

| Layer | File | Change |
| --- | --- | --- |
| L1 Rust | `simulator/src/timing/kernels/neuron_llama_forward.rs` | `prefill_buckets` (empty = `[C]`); inventory = prefill buckets, then decode buckets |
| L1 Rust | `neuron_llama_region.rs`, `neuron_llama_segment.rs`, `neuron_llama_segmented_forward.rs` | keep the `[C]` prefill inventory their rows were measured with |
| L4 | `simulator/src/arch/llama3_vllm_neuron.rs`, `arch/config.rs`, `arch/build.rs`, `deployment/unified.rs` | selector `prefill_buckets`; a prompt pads to the smallest prefill bucket that holds it; an inventory other than `[C]` requires `composition: whole_forward`; `[C]` builds the same config as empty |
| L1 Python | `profiling/runners/neuron/stock_contract.py` | `PREFILL_BUCKET_STEP`, `prompt_length` |
| L1 Python | `vllm_forward.py` | prefill rows at multiples of 128 up to C; `validate_whole_context_shape` keeps the whole-context rule for regions, segments, per-op captures |
| L1 Python | `vllm_forward_engine.py`, `vllm_forward_trace.py` | engine compiles every requested prefill bucket plus C; each bucket below C is validated and timed on its own `(T-8)`-token prompts; those requests (`prefill_only`) add no decode samples |
| L1 Python | `vllm_identity.py` | the binary seal covers every compiled prefill graph |
| Tests | `tests/test_neuron_full_forward.py`, `tests/test_stock_neuron_identity.py`, Rust unit tests | prefill inventory rules, prompt cases, engine buckets, seal |
| Doc | `doc/trainium2.md` | inventory rules |

Every existing preset and row is unchanged: an empty `prefill_buckets` is `[C]`.

Not changed: the alignment pipeline (`alignment/neuron/`) still serves `[C]`.
Its stock graph check and iteration metric assume it, so the hardware replay
below started the server directly with the same engine flags.

## Measured rows (whole forward, ms)

Measured through the repository runner (`vllm_forward.profile_forward_batch`):
full-logit check against CPU FP32/BF16 references, binary seal, NRT timing,
median. The rows are in `profiling/profile.db`, each with its provenance. The
instance checkpoint has the pinned weights (all shards and index
byte-identical) with the unsloth mirror's `config.json`/tokenizer; the
measurement gated on the weights and records the metadata hashes in each
row's provenance.

| C512 prefill | 128 | 256 | 384 | 512 |
| --- | ---: | ---: | ---: | ---: |
| ms | 22.5 | 21.6 | 36.5 | 38.4 (existing 38.2) |

| C1024 prefill | 128 | 256 | 384 | 640 | 896 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ms | 22.1 | 21.6 | 35.9 | 50.9 | 64.5 | 66.4 (existing 65.8) |

Eight new rows. Refused by the vendor numerical criterion: C1024 prefill 512
and 768. Repeated existing rows agree within 1% (C512 prefill 512 38.39 vs
38.23; decode 16 55.31 vs 55.26; decode 128 84.34 vs 84.26; C1024 prefill 1024
66.43 vs 65.80; decode 16 38.97 vs 38.85). Prefill 128 costs slightly more
than 256.

## Simulator sweep (C512, chat trace, async plan offset and duty multiplier on)

96 runs: 4 prefill inventories x 6 power-of-two decode inventories x 4 rates,
on a 600-request chat trace with in+out <= 512 tokens. The worker is the stock
`chunked_prefill` pairing with `async_lookahead_us: 1040` and
`gpu_time_multiplier: 1.062` (fitted on the 2026-10-20 hardware replay). SLO:
TTFT p90 < 1 s, TPOT p90 < 150 ms.

| prefill | decode | SLO rate | sat tok/s | 8 req/s TTFT/TPOT p90 |
| --- | --- | ---: | ---: | ---: |
| [256, 384, 512] | [1, 2, 4, 8, 16, 32, 128] | **10** | 2,917 | 195 / 119 |
| [128, 256, 384, 512] | [1, 2, 4, 8, 16, 32, 128] | 10 | 2,912 | 195 / 119 |
| [256, 512] | [1, 2, 4, 8, 16, 32, 128] | 10 | 2,903 | 196 / 120 |
| [512] (2026-10-19 best) | [1, 2, 4, 8, 16, 32, 128] | 8 | 2,640 | 214 / 140 |

Decode 64 costs more than 128 at C512 (114.7 vs 84.3 ms). Inventories that
include 64 rank lower. With more than 128 residents, TPOT p90 is more than
150 ms at 10 req/s.

The new best as an arch and worker configuration:

```yaml
arch:
  type: llama3_vllm_neuron
  model_config: model/config/llama3_8b.json
  max_model_len: 512
  prefill_buckets: [256, 384, 512]
  decode_buckets: [1, 2, 4, 8, 16, 32, 128]
  tp_size: 4
worker:
  type: chunked_prefill
  attn_gpu_memory_gb: 7.111442432
  max_batch_tokens: 512
  max_prefill_requests_per_iteration: 1
  max_resident_requests_per_partition: 128
  batch_policy: separate-prefill-priority
  kv_admission_policy: full-footprint
  prefix_cache_mode: disabled
```

## Hardware validation

The simulator's request schedules (same arrival times and lengths) replayed
against the clean stock server through its OpenAI-compatible endpoint, 600
requests per run, 0 errors. TTFT and TPOT use the simulator's definitions,
measured on the client clock.

| prefill (decode [1, 2, 4, 8, 16, 32, 128]) | rate | TTFT p90 | TPOT p90 | tok/s |
| --- | ---: | ---: | ---: | ---: |
| [512] (previous best) | 8 | 218 | 141.2 | 2,296 |
| [128, 256, 512] (2026-10-20) | 8 | 202 | 120.4 | 2,306 |
| [256, 384, 512] (new best) | 8 | **198** | **119.9** | 2,304 |
| [256, 384, 512] (new best) | 10 | **566** | **130.4** | 2,783 |
| [512] (previous best) | 12 | 12,033 | 149.4 | 2,628 |
| [128, 256, 512] (2026-10-20) | 12 | 7,051 | 132.8 | 2,851 |

Simulated vs real for the new best (real/sim): 8 req/s TTFT p90 1.02, TPOT
p90 1.01; 10 req/s TPOT p90 1.02, TTFT p50 1.15, TTFT p90 2.46. Near
saturation the simulator under-reads the TTFT tail, as in the 2026-10-20
validation.

Outcome on hardware: the new best meets the SLO at 10 req/s; at 8 req/s TPOT
p90 -15% and TTFT p90 -9% against the previous best. The previous best was not
replayed at 10 req/s (the simulator puts its TTFT p90 at 4 s there).

## With the GQA decode patch (decode attention on the NKI kernel)

The numbers above are the stock engine: at TP4, Llama3 decode attention runs
the Torch fallback, because each rank holds 2 KV heads and the model passes a
2-D block table. A local patch to `vllm_neuron/model/llama3/model.py` passes a
per-head block table, as `qwen3_vl/model_bf16.forward_decode` does. With it,
decode attention runs the NKI `attention_block_tkg` kernel. The patch is not
part of this PR, and its rows are not in this PR's `profile.db`. It was
bind-mounted read-only into the same image; every patched row passed the same
vendor full-logit check.

| C512 decode bucket | 16 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock ms | 55.3 | 114.7 | 84.3 | 153.3 | 303.1 |
| patched ms | 12.9 | 20.3 | 33.7 | 51.4 | 99.4 |

Prefill rows do not change. Hardware replay with the patch, prefill
`[256, 384, 512]` (this PR's inventory), decode `[1, 2, 4, 8, 16, 32, 64, 128,
256]`, 256 resident, same schedules, 0 errors:

| rate | TTFT p90 ms | TPOT p90 ms | tok/s |
| ---: | ---: | ---: | ---: |
| 10 | 97 | 32.6 | 3,111 |
| 16 | 133 | 81.1 | 4,734 |
| 18 | 156 | 128.5 | 5,055 |
| 20 | 1,200 | 145.6 | 5,275 |

With the patch, the SLO rate is 18 req/s (stock best: 10), and 5,055 tok/s at
the SLO (1.8x). At 10 req/s, TTFT p90 is 83% lower and TPOT p90 is 75% lower
than the stock best.

## Measurement notes

- Measure in small groups (one context, a few buckets). A timing run with
  more than a few hundred requests fills the NRT trace buffer, and the trace
  drops events.
- Set `VLLM_CACHE_ROOT` to `<profile cache>/cache`. The runner reads compiled
  geometry there.
- Keep `NKI_COMPILE_CACHE_URL` across containers. If you do not, cached graphs
  point at NKI kernel files that were deleted.
