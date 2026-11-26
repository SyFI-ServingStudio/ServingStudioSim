# feat(kimi_k31): Kimi-K3.1-Pro NVFP4 on vLLM with DSpark k=7, GB200 TP16/EP16 and 2x B200 TP8 x DP2/EP16

## Purpose

Model Kimi-K3.1-Pro NVFP4 (`nvidia/Kimi-K3.1-Pro-NVFP4`: KDA linear attention
plus sparse MLA, 92 routed-MoE layers with top-16 of 896 experts) as vLLM serves
it with the DSpark k=7 proposer (`Inferact/Kimi-K3.1-Pro-DSpark`). The PR covers
two deployments and checks each one against a vLLM capture of
`trace/diverse_100.csv`:

- **GB200 NVL72**: four trays of four GPUs running one TP16 replica. Attention is
  TP16 and routed experts are EP16 over MNNVL.
- **Two B200 NVL8 nodes**: attention TP8 per node and DP2 across the nodes.
  Routed experts are EP16, using NCCL all-gather / reduce-scatter over
  InfiniBand.

What it adds:

- **Arch.** `kimi_k31_vllm_nvfp4_tp_attn_ep_moe_dspark` with its worklets
  (`simulator/src/worklet/kimi_k31_local.rs`). It covers:
  - KDA RecoverSSM verify and conv decode
  - ShardQ sparse-MLA prefill under attention TP
  - the GateLinear router
  - EP all-gather / reduce-scatter at the DP-padded totals
  - the GB200 decode capture grid
  - the hybrid block size derived from the topology
- **L1 kernels.**
  - A new kind, `sparse_mla_shard_q`, for vLLM's ShardQ token/head exchange.
  - New backends: NCCL `row_all_gather`, vLLM-env Triton `elementwise`, and
    `dsa_persistent_topk_decode` / `vllm_cuda_graph`. The last one is the
    capture-time `max_seq_len = max_model_len` bound that a full CUDA graph
    replays.
  - The MLA Q-absorb / V-up shapes for qk_nope 128 + RoPE 64 and v_head 128.
  - FlashInfer MNNVL fused all-reduce at TP16 across trays, allowed only where
    one NVLink domain holds the TP group (GB200 NVL72; B200 NVL8 still refuses
    TP16).
  - Multi-node runners for the NCCL / MNNVL all-reduce and the EP collectives.
- **Routed MoE activation and tactic rule.**
  - `nvfp4_fused_moe` gains an optional arg `gated_activation`. Kimi-K3.1 runs
    SiTU-GLU (`situ:beta=4,linear_beta=25`, read from the checkpoint's
    `hidden_act` / `activation_situ_beta` / `activation_situ_linear_beta`), and
    serving's GEMM1 is the `siTuGlu` kernel. Absent means SwiGLU: it is stored as
    NULL and left out of `args_hash`, so every existing row keeps its hash.
  - The FlashInfer runner has a tactic policy. The default is
    `runner_autotune` (unchanged behavior). With
    `VIBESIM_FLASHINFER_TACTIC_POLICY=vllm`, `MAX_NUM_BATCHED_TOKENS` and
    `DP_SIZE`, the runner reproduces serving by following vLLM's tactic selection
    rule, including the fallback to the default tactic where autotune does not
    cover a size. The Kimi rows are measured this way (budget 4096; DP 2 on B200,
    DP 1 on GB200). The DB key does not include the budget or DP.
- **Grid extensions, pricing unchanged.**
  - `nvfp4_fused_moe.extra_token_points` and
    `moe_ep_{quantized_all_gather,reduce_scatter}.extra_total_tokens` let an
    arch add the row counts it actually runs (CUDA-graph capture sizes,
    DP-padded EP totals) to the shared grid.
  - Both are empty by default and left out of the serialized config, so
    existing identities do not change. Only the Kimi arch sets them.
- **Analyzer.**
  - Per-node clock alignment of multi-node captures
    (`analyzer/rust/src/alignment_iteration/clock.rs`). Each node is aligned on
    the iteration's node-spanning collectives. When an iteration has too few of
    them, the node takes the run-wide median offset.
  - The `incomplete_device_coverage` exclusion.
- **Profiling and capture infrastructure.**
  - One TP replica across nodes, with each peer's nsys `PROCESSES` rows in the
    merged export.
  - GB200 captures under pyxis.
- **vLLM reference engine.**
  - The base moves from upstream `5629b26` to `c5db9bd`.
  - The 22 patches on `main` (0001-0022) are folded into one patch carried onto
    the new base:
    `reference_engines/vllm/patches/0001-alignment-carry-the-reference-patch-line-onto-upstre.patch`.
- **Presets.**
  - The public arch preset and the public sim preset, on GB200 TP16/EP16 only.
    `public_api` allows one public preset per checkpoint and arch type, with a
    static `gpu`, so the B200x2 TP8/DP2 deployment is an alignment pack only.
  - Catalog entries for the checkpoint and the arch.
  - Two alignment packs: `presets/alignment/kimi_k31_pro_nvfp4_gb200_tp16_ep16/`
    and `presets/alignment/kimi_k31_pro_nvfp4_b200x2_tp8dp2_ep16/`.
  - Each deployment's routing reads its own capture, pinned to
    `UW-SyFI/servingstudio-workload@6ad8e60b` on the dataset's `main`.

[`README.md`](README.md) has the modeling notes, the evidence and the known gaps.

## Test Plan

- **Alignment Phase 1** for both packs, per `operate-run-alignment` at `AP=1`:
  Check 1 (kernel), Check 2 (duty cycle) and Check 3 (E2E). The tables come
  from `alignment-campaign compare --markdown`.
- **Unit tests** for the Kimi arch, `moe_ep_all_gather`, `nvfp4_fused_moe`,
  `gated_activation`, the tactic policy, `all_reduce_fusion`, `fixed_exchange`
  and the multi-node profiler.
- **Full test suite** (`cargo test` lib and `pytest`) against `origin/main`.
- **`export-missing`** on both deployments against the shipped
  `profiling/profile_db/`.
- **Public presets**: `pytest -m presets` (`just test-presets`) and the
  `test_public_*` files, with the captures fetched from the pinned `hf://`
  paths.

Not run: `just test-gpu` and `just test-bench`.

## Test Result

- **Unit tests.** All pass: Kimi arch 5, `moe_ep` 5, `fixed_exchange` 4, plus
  the new `nvfp4_fused_moe` grid, `gated_activation` and tactic-policy tests.
- **Full test suite.** Rust lib: 1,935 passed, 0 failed (`origin/main`
  f3720afa: 1,926 passed). `pytest`: 6,103 passed, 10 failed, 28 skipped, 0
  errors (`origin/main` f3720afa on the same host: 5,903 passed, 10 failed, 40
  skipped). The 10 failures are the same tests on both: GPU and environment
  tests (CUPTI, single GEMM, logits top-k, the `l1_measure` end-to-end test); no
  failure is new.
- **Public presets.** `pytest -m presets`: 9 passed, 1 skipped. The
  `test_public_*` files: 209 passed, 10 skipped. The routes behind both pinned
  manifests match their sha256.
- **`export-missing`.** 0 missing specs on both deployments: B200 has 10,044
  specs and GB200 has 10,184.
- **Alignment Phase 1** (`diverse_100`). The tables are pasted unchanged from
  [`alignment_phase1.md`](alignment_phase1.md) (`alignment-campaign compare
  --markdown`). The metrics are in `metrics_phase1_gb200_tp16.json` and
  `metrics_phase1_b200x2_tp8dp2.json`.

GB200 TP16/EP16 (`kimi_k31_pro_nvfp4_gb200_tp16_ep16`):

| Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| tp16_ep16_c4096_s32/01_tp16_diverse_100 | +1.98% | -1.84% | 98.14% | -3.26% | +10.18% | -6.50% | -6.01% |

2x B200 TP8 x DP2/EP16 (`kimi_k31_pro_nvfp4_b200x2_tp8dp2_ep16`):

| Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| tp8dp2_ep16_c4096_s32/01_tp8dp2_diverse_100 | +4.07% | -3.20% | 97.06% | -14.78% | +13.52% | -25.50% | -24.36% |

The `gpu_time_multiplier` for B200 is 1.0239
(`metrics_phase1_b200x2_tp8dp2.json`) and for GB200 1.0276. Both files list
their out-of-tolerance items (workload structure, GB200 output TPS and
iteration cycle, B200 E2E; the kernel metrics of both are within tolerance).

Check 1 by bucket is the same-batch-shape step time from the same Analyzer
reports. Signed error is (sim − meas) / meas.

GB200 (iteration 0 excluded; the multiplier also excludes iterations 3, 61 and
374-383):

| Bucket | n | Measured (ms) | Simulated (ms) | Signed | Abs |
|---|---:|---:|---:|---:|---:|
| decode, 1 request | 2 | 15.58 | 15.50 | −0.49% | 2.70% |
| decode, 2 | 102 | 18.28 | 18.14 | −0.77% | 2.08% |
| decode, 3-4 | 212 | 19.86 | 19.87 | +0.07% | 1.07% |
| decode, 5-8 | 20 | 25.71 | 24.83 | −3.43% | 3.82% |
| decode, 9-16 | 261 | 28.24 | 27.04 | −4.24% | 4.24% |
| decode, 17-32 | 20 | 36.05 | 33.05 | −8.31% | 8.31% |
| decode, all | 617 | 23.84 | 23.19 | −2.73% | 3.24% |
| mixed, cut chunk | 235 | 171.06 | 168.42 | −1.54% | 1.54% |
| mixed, last chunk | 12 | 127.54 | 125.95 | −1.25% | 1.27% |
| all | 864 | 65.33 | 64.12 | −1.84% | 1.98% |

B200 (iterations 0-3 and 16-18 excluded by `incomplete_device_coverage`):

| Bucket | n | Measured (ms) | Simulated (ms) | Signed | Abs |
|---|---:|---:|---:|---:|---:|
| decode, 1 request | 29 | 33.53 | 32.40 | −3.36% | 3.36% |
| decode, 2 | 107 | 38.19 | 36.69 | −3.92% | 3.92% |
| decode, 3-4 | 17 | 40.69 | 39.26 | −3.50% | 3.50% |
| decode, 5-8 | 242 | 43.30 | 43.88 | +1.35% | 1.85% |
| decode, 9-16 | 54 | 48.03 | 49.36 | +2.78% | 3.38% |
| decode, all | 449 | 41.92 | 41.91 | −0.02% | 2.65% |
| mixed, cut chunk | 188 | 196.41 | 187.49 | −4.54% | 4.54% |
| mixed, last chunk | 10 | 135.67 | 120.73 | −11.01% | 11.01% |
| mixed, last chunk, without iterations 52 and 293 | 8 | 140.59 | 132.19 | −5.97% | 5.97% |
| all | 647 | 88.26 | 85.43 | −3.20% | 4.07% |

Alignment Phase 2 has not been run.

### Known gaps

Each gap is explained in [`README.md`](README.md#known-gaps).

- **B200 prefill (mixed steps) is about 4.5% fast** (cut chunk −4.54%). Known
  parts:
  - EP all-gather / reduce-scatter on large messages in prefill/prefill steps
    runs about 19% slower in serving than in the microbenchmark: −19.4%
    all-gather and −17.9% reduce-scatter (sim vs meas, 110 steps). The cause is
    unknown, and no correction is applied.
  - A host gap of about 4.4 ms per step in eager mode. No general model is
    added for it.
- **B200 routed MoE rows.** The routed rows (SiTU-GLU, vLLM's tactic rule)
  match serving's kernel names exactly. At M=8192 the median-rank SiTU row is
  423.5 µs, +14 to +21 µs slower than serving's 402.5-409.6 µs; the cause is
  unknown. The B200 simulation prices the heaviest EP rank's row (480.0 µs at
  M=8192, 480.8 µs with SwiGLU), so the switch to SiTU moved the B200 tables
  by less than 0.2 points.
- **GB200 routed MoE tactics.** On some token buckets the tactic that vLLM's
  autotune picks is not deterministic, so the measured row may use a different
  tactic from serving there. The effect is within a few percent.
- **B200 E2E TPOT is −25.50%.** It has three parts, explained in the README (the per-part split has not been measured, so the parts are qualitative):
  - short requests weigh as much as long ones in the per-request mean
  - the per-request acceptance distribution: the trace carries per-position
    probabilities, not per-round outcomes
  - the exposed fraction of mixed steps
- **B200 mixed last-chunk outliers, iterations 52 and 293**, at −42.2% and
  −29.1%. Only part of each TP group did host-side lazy loading and
  `SetAttribute`, and the group's collective waited for those GPUs. Both steps
  are kept in the data. The bucket table gives the figures with and without
  them.
- **Decode by batch size** (Check 1 buckets above). GB200 is within ±0.8% at
  1-4 requests and grows to −8.31% at 17-32 requests (20 steps). B200 is
  −3.4% to −3.9% at 1-4 requests and +1.35% / +2.78% at 5-8 / 9-16.
- **GB200 DP16 is not in this PR.**

### Changes that reach other models

- `gpu/spec.json`: `NVIDIA GB200` / `GB200` now resolve to a per-GPU row
  (`GB200-GPU-186GB`). They used to resolve to the two-GPU superchip row. This
  affects any GB200 member that reads spec peaks.
- `all_reduce_fusion` / `flashinfer_mnnvl` accepts TP16 on GPUs whose NVLink
  domain holds 16 GPUs (GB200 NVL72, from `gpu/spec.json`). B200 TP16 is still
  refused with the same message as on `main`.
- `nvfp4_fused_moe` gains the optional `gated_activation` arg. It is unset for
  every other arch, stored as NULL and not hashed, so existing rows and
  identities are unchanged.
- The FlashInfer runner's default tactic policy stays `runner_autotune`.
- Analyzer: multi-node clock alignment and the `incomplete_device_coverage`
  exclusion. Single-node captures are not shifted.
- `profiling/profile_db/` changes 29 kind files (27 existing, 2 new) and only
  adds rows, plus the nullable `gated_activation` column in
  `nvfp4_fused_moe.db`. The other 118 kind files are untouched. `merge-db` onto
  `main`'s directory had 0 conflicts, and no existing row changes. The Kimi
  routed MoE rows are SiTU-GLU, measured with vLLM's tactic rule: B200 run_key −6570559150466200488 (2,336 rows), GB200
  run_key −7687545119914990384 (1,232 rows).
- The new config fields are empty by default and leave other archs' identities
  unchanged.

## Dependencies

- **Token corpus and acceptance**: two captures for
  `UW-SyFI/servingstudio-workload`, at commit `6ad8e60bf522a04db0fde4efffd8de2b33c87a1a` on `main`:
  - `kimi_k31_pro_nvfp4/vllm_dspark_k7/diverse_100/capture/20261120/` (GB200)
  - `kimi_k31_pro_nvfp4/vllm_dspark_k7/diverse_100/capture/20261125/` (B200,
    `replicate_of` the GB200 capture)
- **vLLM reference engine**: upstream `c5db9bd` plus the folded patch above.
- **`profiling/profile_db/`** gains 24,065 kernel rows in 29 kind files,
  merged with `kernel-profile split-db` / `merge-db` onto `main`'s directory: 0
  conflicts, every `main` row unchanged. Each touched file also gains the
  `_profile_run` records of the runs whose rows it holds (65 records across
  the 29 files). `sparse_mla_shard_q.db` and `kda_recoverssm_verify.db` are
  new files. The directory grows from 98.2 MiB to 116.3 MiB; the largest file,
  `nvfp4_fused_moe.db`, is 31.2 MiB.

  | Kind file | Rows added | Size on `main` (KiB) | Size with this PR (KiB) |
  |---|---:|---:|---:|
  | `all_reduce.db` | 19 | 64 | 64 |
  | `all_reduce_fusion.db` | 115 | 76 | 96 |
  | `batched_gemm.db` | 272 | 344 | 400 |
  | `dsa_index_cache_append.db` | 70 | 88 | 104 |
  | `dsa_mqa_logits_prefill.db` | 474 | 464 | 564 |
  | `dsa_paged_mqa_logits_decode.db` | 698 | 5,076 | 5,248 |
  | `dsa_persistent_topk_decode.db` | 5,526 | 4,232 | 5,300 |
  | `dsa_sparse_mla_attention.db` | 1,258 | 5,980 | 6,416 |
  | `dsa_sparse_mla_prefill.db` | 2,168 | 720 | 1,652 |
  | `dsa_topk_prefill.db` | 412 | 388 | 460 |
  | `elementwise.db` | 2,164 | 1,584 | 1,884 |
  | `flashinfer_attn_rect.db` | 768 | 792 | 928 |
  | `gdn_causal_conv_decode.db` | 89 | 56 | 72 |
  | `gdn_causal_conv_prefill.db` | 288 | 244 | 292 |
  | `gemm_fp32_output.db` | 300 | 224 | 276 |
  | `kda_chunk_prefill.db` | 1,024 | 844 | 1,000 |
  | `kda_recoverssm_verify.db` | 260 | new file | 76 |
  | `mla_cache_append.db` | 70 | 100 | 112 |
  | `moe_ep_quantized_all_gather.db` | 37 | 64 | 68 |
  | `moe_ep_reduce_scatter.db` | 61 | 76 | 92 |
  | `nvfp4_fused_moe.db` (adds column `gated_activation`) | 3,568 | 17,856 | 31,928 |
  | `nvfp4_quant.db` | 136 | 148 | 168 |
  | `rms_norm.db` | 528 | 152 | 228 |
  | `row_all_gather.db` | 42 | 56 | 64 |
  | `sigmoid_gated_rms_norm.db` | 50 | 40 | 52 |
  | `silu_and_mul.db` | 408 | 56 | 108 |
  | `single_gemm.db` | 3,028 | 2,912 | 3,428 |
  | `sparse_mla_shard_q.db` | 96 | new file | 44 |
  | `static_fp8_quant.db` | 136 | 80 | 100 |

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.
