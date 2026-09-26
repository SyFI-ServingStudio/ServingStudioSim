# feat(glm53): GLM-5.3-Flash FP8 on vLLM, hybrid chunked-prefill worker and B200 TP4/EP4 alignment

## Purpose

Model GLM-5.3-Flash (FP8, hybrid KDA + DSA with kpool indexer, MoE with mHC
boundaries) as served by the vLLM fork on B200 under TP4/DP1/EP4, and validate
it against a fifteen-case campaign.

- **L1 kinds and profiling backends** for the serving stack's kernels:
  - KDA chunked prefill and plain decode;
  - FlashInfer MNNVL all-reduce and FP8 block-scale fused MoE;
  - the fork's DSA kernels (sparse MLA, paged MQA logits, persistent and
    prefill top-k at the kpool width 512, index remap);
  - mHC on B200, fused q/kv RMSNorm, FP8 group quant, DeepGEMM and cuBLAS
    GEMMs.
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
  tokens. It is a calibration knob.
- **Alignment infrastructure:**
  - vLLM launches through `vllm serve` with 4 API server processes by
    default, verified from the startup log;
  - the `glm53_flash_fp8_b200_tp4_ep4` campaign pack with 138 label rules;
  - an analyzer fix for PDL waits parked under a same-stream collective.

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

- Simulator lib: 1143 passed, 8 ignored.
- Pytest CPU tier: 3896 passed, 6 skipped.
- Analyzer: 274 passed.
- Campaign: all 15 cases complete, and all 15 request-population audits pass.
  - Kernel absolute error is 1.15-5.41%. Only case 11 exceeds the 5% tolerance.
  - E2E is within 7% for 14 of 15 cases. Case 11 is -9.99%.
  - TTFT is within 12% for 14 of 15 cases. Case 08 is +15.8%.
  - TPOT is within tolerance for all 15 cases.
  - Workload structure exceeds the 1% tolerance for cases 01, 02 and 09.

See [the generated matrix](alignment_matrix.md) and
[the evidence notes and known gaps](README.md). This is review evidence, not an
accepted baseline, and no golden is recorded.

## Dependencies

- vLLM fork `3f667d7` (`servingstudio-alignment`) and req-frontend `e3a400f`;
  unchanged from `main`.
- `profiling/profile.db` gains 12,261 B200 rows for these kernels, all
  additions (commit `data(profiling): B200 rows for GLM-5.3-Flash FP8 on the
  vLLM fork`).

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
