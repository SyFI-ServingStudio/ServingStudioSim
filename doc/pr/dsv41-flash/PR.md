# feat(dsv41): DeepSeek-V4.1-Flash on vLLM, B200 TP4/EP4, with kernel alignment

## Purpose

Model DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, 40-layer MoE
with mHC, Engram, three compression ratios, cross-layer KV and index sharing,
candidate-block selection) as vLLM serves it on 4x B200 with TP4 attention and
EP4 experts, and check it against a vLLM capture.

- **L1 kinds and backends.**
  - Three new kinds: `compressed_sparse_mla_rope_cast` (the fused FlashMLA
    attention), `q_pad_kv_rope_mxfp8_insert` and `engram_lookup`.
  - New backends on existing kinds: `single_gemm` / `flashinfer_mxfp8`,
    `nvfp4_fused_moe` / `flashinfer_trtllm_sm100_mxfp4` (MXFP4 experts),
    `mhc_fused_post_pre_rms_norm` / `deepgemm_mega` and `batched_gemm` /
    `deepgemm_mxfp8_einsum_grouped_o_proj`.
  - `gemm_fp32_output` / `torch_cublas` and `all_reduce_fusion` /
    `flashinfer_mnnvl` gain B200 rows for the router gate, the compressor and
    the TP4 all-reduce.
  - Dtypes `mxfp8_e4m3` and `mxfp4_e2m1`.
  - Each new kind or backend has a Python runner, a Rust cache and a fidelity
    check against fresh measurements.
- **Ops, worklets and the arch.**
  - L2 ops `deepseek_v41_mega_attn` and `deepseek_v41_indexer`.
  - Worklets for attention, Engram, MoE, prologue and head.
  - Arch selectors `deepseek_v41_vllm` and `deepseek_v41_vllm_serial_streams`.
  - Side streams under `CostNode::Parallel`, gated at vLLM's thresholds.
  - `max_model_len` defaults to the checkpoint's 1048576 and is a timing
    input.
  - `decoder_swa_bounded_replay` is an opt-in counterfactual (vLLM PR #58132,
    SGLang `--enable-decoder-swa-bounded-replay`).
  - The KV accounting is reconciled with vLLM's block pool on both captures.
- **model.work** accountant and location map for the TP4/EP4 unified
  deployment.
- **Presets.**
  - The capture-2 simulation preset reads the tracked
    `trace/quadrant_c48_96.csv`.
  - Public arch presets: `max_model_len` 131072 or 1048576, bounded replay off
    or on.
  - Public sim presets: `server` ctx128k / ctx1m, replicas 1 / 2, bounded
    replay off or on. The ctx1m KV pool is an estimate.
  - Routing reads the pinned `diverse_100` capture in
    `UW-SyFI/servingstudio-workload`.
- **Alignment.**
  - Label rules for the capture (`presets/alignment/dsv41_flash_b200_tp4_ep4/`).
  - A `name_exact` rule key, because two vLLM launches are literally named
    `kernel`.
- **Changes that reach other models.**
  - The CUPTI cold-L2 flush read a misspelled torch property, so it always
    flushed the 64 MiB floor; on B200 it now flushes 2 x L2.
  - 308 B200 `elementwise` / `triton` rows that other models read take the
    re-measured values.
  - `vllm_upstream_fork_env` profiles on the fork's `.venv`, because the
    `vllm_env` container has no DeepSeek-V4.1.

[`README.md`](README.md) has the modeling notes, the evidence and the open
items.

## Test Plan

CPU tier on the branch head:

```bash
just test-cpu
```

Also run:

- **Preset tests.** `uv run pytest tests/ -k "public_preset or sim_preset or
  public_api"`, which builds every public member and checks that it finds every
  profile.db row it reads.
- **Capture-2 preset.** `uv run python -m launcher
  presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml`.
- **Product path.** Through a local public API and the Intro site:
  - Live predict on every DeepSeek-V4.1 member, including contexts up to
    1,048,000 at `max_model_len` 1048576;
  - simulations on the 131072 and 1M members, with the `diverse_100` capture
    and with 400K and 900K prompts.
- **Kernel alignment against capture 2** with `launcher alignment
  timing-predict` and `alignment analyze` (configs in the experiment tree).
- **Cache fidelity** of every new kind and backend against fresh B200
  measurements, on Slurm.

Not run: the GPU tier (`just test-gpu`), `just test-bench` and the analyzer
suite. The analyzer is unchanged.

## Test Result

- `just test-cpu`: simulator lib 1221 passed, 8 ignored; pytest 3976 passed,
  1 skipped.
- Preset and public API tests: 110 passed.
- **Kernel alignment against capture 2** (2236 iterations): the simulated
  critical path is −4.55% overall, −4.65% in decode and −4.00% in mixed
  iterations. The recommended `gpu_time_multiplier` is 1.0361.
- **Simulation of capture 2's workload** (96 requests at 48 in flight):

  | | Simulated | Measured |
  |---|---|---|
  | Makespan | 51.85 s | 54.0 s, nsys stall removed |
  | Throughput | 6872.5 tok/s | 6598.9 tok/s |
  | TTFT mean | 539.2 ms | 567.7 ms |

  TPOT is not compared: the profiler stall falls inside 48 requests.
- **Merges of `main`.** The timing predictions of four capture-2 shapes are
  bit-identical after each merge of `main`, the `max_model_len` change and the
  move to `CostNode::Parallel`: 10.6022 / 43.5739 / 15.4371 / 42.0314 ms.
- **Long context.** At `max_model_len` 1048576, a 2048-token chunk at prefix
  1,046,528 is 288.09 ms, and a decode of 8 x 1,048,575 is 8.57 ms. Attention
  and indexer-logits leaves stay on the measured grid; three indexer
  placeholders read past the `elementwise` grid and hold its edge bandwidth.

[`README.md`](README.md) explains the remaining deviations: routed MoE +6-7%
in mixed iterations from a corpus without prompt routes, under-priced
placeholders, and isolated-vs-in-server timing of the large prefill kernels.
This is review evidence, not an accepted baseline. No golden is recorded.

## Dependencies

- **vLLM fork** `servingstudio-alignment` moves to `892da082`, rebased onto
  upstream main `04730e8`. The previous line, `3f667d7`, stays on the fork as
  `backup/servingstudio-alignment-pre-04730e8-20261006`.
- **Token corpus**: the `diverse_100` capture of the Hugging Face dataset
  `UW-SyFI/servingstudio-workload`, pinned at `c3f5ecaa`.
- **`profiling/profile.db`** gains 5,450 B200 rows, all measured by this
  branch, merged row-wise by semantic key onto `main`'s DB with 0 conflicts:

  | Kind | Rows |
  |---|---:|
  | `compressed_sparse_mla_rope_cast` | 2,540 |
  | `elementwise` | 1,449 |
  | `single_gemm` | 474 |
  | `batched_gemm` | 272 |
  | `gemm_fp32_output` | 204 |
  | `q_pad_kv_rope_mxfp8_insert` | 182 |
  | `engram_lookup` | 122 |
  | `mhc_fused_post_pre_rms_norm` | 98 |
  | `nvfp4_fused_moe` | 73 |
  | `all_reduce_fusion` | 36 |

  It also gains 20 profiling-run records. 308 existing B200 `elementwise`
  rows take the post-fix values. No row of `main` is removed. The file is
  49 MB.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
