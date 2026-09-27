# feat(glm53): GLM-5.3-Flash FP8 on vLLM, hybrid chunked-prefill worker and B200 TP4/EP4 alignment

## Purpose

Model GLM-5.3-Flash (FP8, hybrid KDA + DSA with kpool indexer, MoE with mHC
boundaries) as served by the vLLM fork on B200 under TP4/DP1/EP4, and validate
it against a fifteen-case campaign.

- **L1 kinds and profiling backends** for the serving stack's kernels:
  - KDA chunked prefill and plain decode;
  - FlashInfer MNNVL all-reduce and FP8 block-scale fused MoE;
  - vLLM's DSA kernels (sparse MLA, paged MQA logits, persistent and
    prefill top-k at the kpool width 512, index remap);
  - mHC on B200, fused q/kv RMSNorm, FP8 group quant, DeepGEMM and cuBLAS
    GEMMs.
- **One profiling stack, no fork backends.** The existing backends admit the
  GLM-5.3-Flash shapes: kpool top-k 512, index-table width 2176, rope-free
  sparse MLA, UE8M0 scale layouts and FP32-input GEMM. `vllm_env` now builds
  from the fork's vLLM (`3f667d7`, 0.28.1rc0, FlashInfer 0.6.18) as
  `vibesim-profiler-vllm:cu130-3f667d7e`, replacing the v0.23.0 image. The
  build bakes every FlashInfer cubin into the image, because workers run with
  `--network none` and 0.6.18 ships no cubin wheel.
- **Worklets, ops and the `glm53_flash_vllm_fp8_kda_dsa_moe` arch**:
  - kernels outside the attention graph break run on CUDA-graph-padded rows;
  - the aux-stream shared expert contends with its routed slice;
  - the prefill indexer is timed on each row's pooled context.
- **Necessary-work accountant** and semantic location map for the unified
  deployment.
- **Hybrid chunked-prefill worker.** `chunked_prefill` now runs on the
  `HybridGdnKv` store for Qwen3.6 local and GLM-5.3-Flash. It carries vLLM's
  Mamba `align` chunk-end quantum.
- **`prefill_gpu_time_multiplier`** for iterations that schedule prefill
  tokens, on every co-located worker selector. It is a calibration knob.
- **Alignment infrastructure:**
  - vLLM launches through `vllm serve` with 4 API server processes by
    default, verified from the startup log;
  - the `glm53_flash_fp8_b200_tp4_ep4` campaign pack with 138 label rules;
  - an analyzer fix for PDL waits parked under a same-stream collective;
  - the request-population audit refuses a `trace_timed` simulation whose
    arrival-time scale differs from the measured replay's by more than 10%.
    `alignment analyze` exits on it and names the `request_rate` to use.
- **Kernel Library (merged from `main`, #53).** GLM-5.3-Flash is a catalog model
  (`glm53_flash`) with a `#[supported]` row on B200: FP8, TP4, `max_model_len`
  8192 to 524288. Every kind it runs has a `DOC`, argument docs and a
  `BackendDoc`, and the registry holds its supported and alignment configs.
  The alignment sources name the routing corpus on the hub
  (`hf://UW-SyFI/servingstudio-corpora@869a5d39/glm53_flash_fp8_tp4_ep4`).
  The merge renamed three kinds and backends to name their mechanism:
  `deepseek_v4_fused_q_kv_rmsnorm` became `q_kv_rms_norm`, and
  `torch_mla_{q_absorb,v_up}_glm53` became `torch_mla_q_absorb_no_rope` and
  `torch_mla_v_up_unpadded`. The FP8 block-scale MoE backend's
  `weight_format` is now `fp8_e4m3`, which is also its precision.

## Test Plan

CPU tier, on the branch head, as a 16-CPU Slurm job:

```bash
uv run cargo test -p simulator --lib
uv run pytest -m "not gpu and not agent and not bench" -n 16
uv run cargo test --manifest-path analyzer/rust/Cargo.toml --bin analyze
```

Each simulator commit on this branch also passes
`cargo check -p simulator --lib --tests` on its own.

The analyzer test `a_log_dir_outside_any_checkout_falls_back_to_the_process_cwd`
needs `TMPDIR` outside every Git checkout. The workspace `.env` points `TMPDIR`
inside the root checkout, so run the analyzer suite with another `TMPDIR`.

The GPU tier (`just test-gpu`) was not run. The model is validated by the
fifteen-case B200 alignment campaign instead.

[`README.md`](README.md#reproduce) has the per-phase commands, from `check`
through `compare --markdown`.

## Test Result

After merging `main` (head `e708af5`):

- `just test-cpu`: simulator lib 1119 passed, 8 ignored; pytest 3535 passed,
  6 skipped.
- All 15 campaign cases rerun, on this branch before and after the merge:
  throughput, SLO, batch, KV, time-share and input-distribution reports are
  identical, after renaming. Only kernel throughput and optimality differ:
  main's mHC kinds report bytes, so nine mHC locations gain GB/s and a
  bandwidth floor. The merged analyzer run on the pre-merge output
  reproduces the pre-merge optimality exactly.

- Profiling image: `profiling run --fresh` on B200 with the new image
  reproduces the stored GLM-5.3-Flash rows for all 11 kinds it runs. Large
  shapes are within about 3%; a few microsecond-scale shapes differ by 4-13%.
  Rows from GLM-5.2 on v0.23 differ more: rope-64 sparse MLA reads
  0.82-1.12x on 8 us shapes (see the mixed-version gap below).

Before the merge:

- Simulator lib: 1143 passed, 8 ignored.
- Pytest CPU tier: 3904 passed, 5 skipped.
- Analyzer: 274 passed.
- Campaign: all 15 cases complete, and all 15 request-population audits pass.
  - Re-audited with the arrival-scale check: case 09, the one `trace_timed`
    case, replays at 0.370 on both sides (`--rate 2.7`). The other 14 are
    saturated, so the check does not apply.
  - Kernel absolute error is 1.15-5.41%. Only case 11 exceeds the 5% tolerance.
  - E2E is within 7% for 14 of 15 cases. Case 11 is -9.99%.
  - TTFT is within 12% for 14 of 15 cases. Case 08 is +15.8%.
  - TPOT is within tolerance for all 15 cases.
  - Workload structure exceeds the 1% tolerance for cases 01, 02 and 09.

See [the generated matrix](alignment_matrix.md) and
[the evidence notes and known gaps](README.md). This is review evidence, not an
accepted baseline, and no golden is recorded.

### Known gap: paged MQA logits decode reads the v0.23 rows

`dsa_paged_mqa_logits_decode` on B200 has 1583 rows measured with the vLLM
fork's DeepGEMM 2.6.1. They are stored as `deepgemm_fp8_vllm_fork`, a backend
that is no longer registered. Their keys collide with older `deepgemm_fp8`
rows measured with vLLM v0.23's DeepGEMM for GLM-5.2. The rename kept the
older rows, so GLM-5.3-Flash reads v0.23 timings for this kernel. The fork's
rows are slower: 1.055x at the median, 1.12x at p95 and 1.58x at the maximum.
180 keys differ by more than 10%. This kernel is likely predicted about 5% too
fast. Both sides of the before/after comparison read the same rows, so the
comparison cannot show it.

The follow-up profiling cleanup separates rows by vLLM version with a version
guard. That lets GLM-5.2 keep the v0.23 rows and GLM-5.3-Flash read the fork's
rows.

### Known gap: `vllm_env` backends mix vLLM v0.23 and v0.28 rows

The GLM-5.2 and DeepSeek rows in `vllm_env` backends were measured on the
v0.23.0 image and are not re-profiled here. They sit next to the v0.28
GLM-5.3-Flash rows in the same backends, told apart only by
`backend_version`, which lookups ignore. Re-profiling them on the new image is
follow-up work, along with the version guard above.

## Dependencies

- vLLM fork `3f667d7` (`servingstudio-alignment`) and req-frontend `e3a400f`;
  unchanged from `main`.
- Profiling image `vibesim-profiler-vllm:cu130-3f667d7e`, built by
  `profiling/container/build.sh` from the same vLLM commit.
- `profiling/profile.db` gains 12,261 B200 rows for these kernels, all
  additions (`7a20220`). `592bfd6` moves them from the fork backends to the
  canonical backend names. It is now 91 MiB, close to GitHub's 100 MiB
  per-file limit.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
