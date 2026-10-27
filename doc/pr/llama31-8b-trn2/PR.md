# feat(trn2): Llama 3.1 8B on Trainium2 with vLLM Neuron 0.29, aligned with the engine

## Purpose

Model Llama 3.1 8B BF16 as vLLM Neuron 0.29 (Neuron SDK 2.33) serves it at
TP4 on one Trainium2 chip, and check it against the engine on a trn2.3xlarge.
This replaces #96, which targeted vLLM Neuron 0.24. SDK 2.34 removes the
PyTorch/XLA route that 0.24 used.

- **Neuron profiling.** `BackendSupport.device_family: neuron`, a Docker chip
  pool (`profiling/exec/neuron.py`) that reserves a whole chip and runs the
  worker in the vendor image pinned by digest, Neuron row provenance, and a
  `Trainium2-LNC2` catalog entry.
- **L1 kinds.**
  - Prefill launch units timed alone on one core: `neuron_nki_qkv`,
    `neuron_nki_attention_cte`, `neuron_nki_output_projection`,
    `neuron_nki_mlp`, `neuron_glue`, `neuron_head`.
  - Decode leaves `neuron_decoder_layer` and `neuron_decode_fixed`: the
    per-layer and per-step terms of a reduced-depth TP4 decode forward (8 and
    32 layers) on the chip's four cores. The block all-reduces, the embedding,
    the head and the engine's sampler are in the measured graph.
  - `neuron_attention_decode` (per-op view) and `neuron_collective`.
  - Two decode attention paths as backends: `vllm_neuron_nki` (a patched
    `forward_decode`, the default) and `vllm_neuron_torch` (the stock call).
- **Worklets and the arch.** `attn_block_neuron_tp`, `mlp_block_neuron_tp`;
  arch `llama3_neuron_ops` with `decode_attention` and `decode_granularity`;
  `ensure_stock_neuron_worker` pins `chunked_prefill` to the engine's
  scheduler.
- **Host time.** `host_us_per_prefill_request` / `_decode_request`,
  `workload.frontend_latency_us` and `async_lookahead_us`, fitted on the
  engine's replays.
- **Preset** `presets/unified_llama3_8b_neuron_ops.yaml` with
  `trace/trainium_stock_smoke.csv`.
- **Changes that reach other models** (all default off, so other presets are
  unchanged): `frontend_latency_us` (default 0), the host-time fields, the
  `chunked_prefill` request-count limits and async lookahead (default unset),
  a one-point grid in `Cache1DDirect`, the profiling device-family plumbing,
  and `RETIRED_TABLES` in `migrate-db`.

`doc/pr/llama31-8b-trn2/README.md` has the modeling notes, the evidence and
the open items.

**Dependencies.**

- Engine image `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`,
  pinned by image ID `sha256:d6602265…` in `profiling/container/images.py`.
- Overlap with #107 (`trn2-flashnext`, open). Both add the `Trainium2-LNC2`
  catalog entry, the Neuron device family and Docker pool, and a NeuronLink
  fabric variant. The profiling DB plumbing is byte-identical.
  `profiling/exec/neuron.py`, `gpu/spec.json` and `common/fabric.rs` differ and
  need reconciling in whichever PR lands second.

## Test Plan

```bash
uv run cargo test -p simulator --lib
uv run pytest tests/test_neuron_nki_ops.py tests/test_neuron_execution.py \
  tests/test_profile_db_device_provenance.py tests/test_profile_env_registry.py \
  tests/test_profile_env_isolation.py tests/test_l1_single_gemm.py -q
uv run pytest tests/test_public_presets.py tests/test_public_sim_presets.py -m presets
uv run python -m launcher presets/unified_llama3_8b_neuron_ops.yaml
```

On a trn2.3xlarge:

- every row filled through the launcher's cold cache;
- the engine's own profile of each prefill and decode shape;
- streaming replays of four traces for the host terms, and three held-out
  replays on a restarted engine with eight decode buckets.

Not run: `just test-cpu` as a whole (on this macOS host without CUDA,
`just sync` cannot build deep-gemm; the Rust and pytest halves above ran
separately), the GPU tier (`just test-gpu`, including the throughput golden),
`just test-bench`, `just test-agent`.

## Test Result

- Simulator lib: 1618 passed, 8 ignored. Neuron and env pytest: 316 passed,
  2 skipped. Preset tier: 9 passed. Preset run: 32/32 requests finished, no
  missing row.
- **Step alignment** (simulator / engine, per rank, C512):

  | Shape | Engine (ms) | Simulator (ms) | Ratio |
  |---|---:|---:|---:|
  | prefill T512 | 26.09 | 26.43 | 1.01 |
  | decode B1 | 9.77 | 9.27 | 0.95 |
  | decode B16 | 11.90 | 11.71 | 0.98 |
  | decode B32 | 14.62 | 14.25 | 0.97 |
  | decode B64 | 19.24 | 18.72 | 0.97 |
  | decode B128 | 32.13 | 30.94 | 0.96 |

- **Request level** (simulator / engine):

  | Replay | TTFT p50 | TPOT p50 | latency p50 | duration | out tok/s |
  |---|---:|---:|---:|---:|---:|
  | 10 isolated requests | 1.00 | 0.96 | 0.98 | 1.00 | 1.00 |
  | 32 at once | 0.92 | 1.00 | 0.96 | 0.96 | 1.05 |
  | chat, 2 req/s | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 |
  | chat, 8 req/s | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 |
  | held out, 8 buckets, 2 req/s | 0.98 | 0.95 | 0.95 | 1.00 | 1.00 |
  | held out, 8 buckets, 16 req/s | 0.98 | 0.97 | 0.98 | 1.00 | 1.00 |
  | held out, 8 buckets, 24 req/s (saturated) | 0.91 | 0.98 | 0.97 | 0.98 | 1.02 |

  The held-out predictions were recorded before the engine ran.

This is evidence for the patched-engine deployment (`decode_attention:
nki_kernel`). The stock engine (`torch_fallback`) is checked at the step level
only: decode B1 +24%, B16 -23% through `per_op`. No golden is recorded.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here (project, URL, version, license — or "none"): none.

---
<details>
<summary>Checklist</summary>

- [x] **Doc-backed — name the file.** `doc/pr/llama31-8b-trn2/README.md` for the
      modeling and evidence; `simulator/src/arch/README.md` (L4 contract) and
      `simulator/src/worklet/README.md` (TP worklet suffix) for the arch and
      worklet shapes; `profiling/README.md` for the Neuron execution backend.
- [x] **Formatting radius matches the change.** Rust was formatted per file
      (`rustfmt --edition 2021 --config skip_children=true <file.rs>`); in shared
      files only this branch's hunks were touched. Python got an explicit file list.
- [x] **`profiling/profile.db`:** merged row-wise with `merge-db` onto `main`'s DB,
      0 conflicts. 10 new `neuron_*` tables, 95 rows, all measured on Trainium2
      (trn2.3xlarge, vLLM Neuron 0.29 image above): `neuron_collective` 57,
      `neuron_head` 9, `neuron_decoder_layer` 8, `neuron_decode_fixed` 8,
      `neuron_attention_decode` 4, `neuron_nki_mlp` 3, `neuron_glue` 3,
      `neuron_nki_qkv` / `neuron_nki_attention_cte` / `neuron_nki_output_projection`
      1 each; 31 profiling-run records. No row of `main` is changed or removed.
- [x] **Unmeasured shapes still fail.** Every leaf config is validated at build, and
      `Cache1DDirect` rows exist per compiled bucket only: an unlisted bucket or a
      missing row is a build error or a missing cache identity, not a neighbour's time.
- [ ] ~~**Goldens.**~~ n/a: no per-GPU golden covers Trainium2, and the new knobs default
      off for every other preset. The GPU-tier throughput golden was not run.

</details>
