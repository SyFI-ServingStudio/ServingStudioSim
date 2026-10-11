# feat(inkling): Inkling-NVFP4 and Inkling-Small-NVFP4 on vLLM, B200 TP8/EP8 and TP4/EP4, with kernel alignment

## Purpose

Model Inkling (`thinkingmachines/Inkling-NVFP4`, 66 layers, and
`thinkingmachines/Inkling-Small-NVFP4`, 42 layers) as the vLLM fork serves it
on B200: TP attention and EP routed experts over the same ranks, 256 routed
experts top-6 plus two sink experts, sliding-window and global
relative-position attention, and a short convolution fused into each
sublayer's collective. Check it against vLLM captures on 4 and 8 B200s.

- **L1 kinds and backends.**
  - Four new kinds:
    - `rel_bias_attention` / `tml_fa4`, the relative-position FA4 attention;
    - `sconv_qkvr_prep` / `vllm_triton`, the fork's two-stream QKVR prep;
    - `rs_sconv_ag_residual_rms_norm` / `vllm_lamport_triton`, the Lamport
      reduce-scatter, short conv, all-gather, residual add and RMSNorm call;
    - `moe_sink_gate_select` / `vllm_triton`.
  - New backends on existing kinds:
    - `gemm_fp32_output` / `vllm_ll_bf16_auto`, the router dispatch;
    - `nvfp4_fused_moe` / `flashinfer_trtllm_routed_sm100_fork`, the routed
      MoE on the fork's FlashInfer 0.7.0. The pinned image's 0.6.18 runs about
      2x slower at T >= ~1792.
  - Each new kind or backend has a Python runner and a Rust cache. The four
    kinds and the router backend also have a cache-fidelity caller.
- **Ops, worklets and the arch.**
  - The `inkling_vllm_nvfp4_rel_attn_moe` arch, its model config and the
    attention and router worklets. The dense FFN and the sinks reuse the
    GLM-5.3 gated MLP worklet.
  - The sinks run under `CostNode::Parallel` only where vLLM keeps them on the
    aux stream: uniform-decode FULL graph or eager steps at <= 256 tokens.
  - FULL-graph decode prices global attention at the graph-padded request
    count.
  - The unified deployment arm and the Analyzer's necessary-work labeler.
- **model.work** accountant for Inkling, a `short_conv` mixer and the location
  maps for the TP4/EP4 and TP8/EP8 unified deployments.
- **Presets.**
  - Public arch presets for both checkpoints, EP on. Inkling-NVFP4 sweeps
    `max_model_len` 131072 and 1048576. Inkling-Small-NVFP4 offers 131072
    only: 8 of its 639 global attention specs at 1048576 do not fit one GPU
    at TP4.
  - Public sim presets, replicas 1 / 2. Inkling-NVFP4 adds `server` ctx128k /
    ctx1m; the ctx1m KV pool reuses the measured 131072 pool.
  - Catalog entries in `model/catalog.yaml` and `model/arch_catalog.yaml`.
  - Routing reads each checkpoint's `diverse_100` token corpus in
    `UW-SyFI/servingstudio-workload`, pinned to commit `a6959f41`.
  - The top-level Inkling case presets moved to the experiment trees.
- **Alignment.** The `inkling_nvfp4_b200_tp8` pack: the 15 operating points of
  `glm52_nvfp4_b200_tp8`, its default tolerances, and the label rules.
- **Changes that reach other models.**
  - The FlashInfer autotune cache keeps a configuration's live tactics across
    its specs. Clearing them per spec re-tuned every bucket on every spec.
  - The CUPTI multi-launch check compares kernel names as a multiset under
    `interval_union`.
  - `VllmTpGroupLauncher` for profiling callables that need vLLM's TP group.
    It shares one spawn helper with `VllmLauncher`.
  - The NVFP4 and BF16 MoE worklets gain an `eval_or_zero` that GLM does not
    call.
  - No existing profile.db row changes.

[`README.md`](README.md) has the modeling notes, the evidence and the open
items. [`alignment_matrix.md`](alignment_matrix.md) is the generated campaign
matrix.

## Test Plan

Test tiers on the branch head, after the merge of `main`:

```bash
just test-cpu
just test-presets
just test-gpu    # on one B200, through Slurm
```

Also run:

- **Kernel alignment on `diverse_100`**, Small TP4/EP4 and Inkling-NVFP4
  TP8/EP8, with `launcher alignment timing-predict` and `alignment analyze`
  (configs in the experiment trees).
- **Simulation of each `diverse_100` workload pass**, compared by the E2E
  Analyzer.
- **The 15-case campaign** on Inkling-NVFP4 TP8/EP8 through
  `alignment-campaign`, from capture to `compare`.
- **Merge of `main` and the review fixes.** Simulations before and after each
  on three workloads: Inkling-NVFP4 `diverse_100`, Small `diverse_100` and
  campaign case 05.
- **Cache fidelity** of the new kinds against fresh B200 measurements, on
  Slurm.

Not run: the product path, live predict and simulation on the public members
through a local public API and the Intro site.

## Test Result

- `just test-cpu`: pytest 4227 passed, 16 skipped; Rust 1382 passed, 8
  ignored.
- `just test-presets`: 7 passed. Every public member, Inkling included, is
  fully measured in the committed profile.db.
- `just test-gpu` (job 5340, after the review fixes): 14 passed, 3 skipped. The throughput regression
  monitor raised no drift warning.
- **Merge of `main` and the review fixes.** The three simulations write an
  identical `request_slo.parquet` before the merge, after it, and after the
  review fixes.
- **Kernel alignment on `diverse_100`.** The signed median of the simulated
  critical path against measured, per iteration:

  | Deployment | Decode | Mixed | All | Coverage | Multiplier | Source |
  |---|---:|---:|---:|---:|---:|---|
  | Small TP4/EP4 | -0.59% | -0.17% | -0.56% | 98.85% | 1.0465 | `logs/20261008_10_inkling_standard/check1e_derived.md` |
  | Inkling-NVFP4 TP8/EP8 | +5.92% | -1.21% | +5.69% | 99.50% | 1.0276 | `logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/check1_derived.md` |

- **Simulation of `diverse_100`** (100 requests, all ids shared):

  | Deployment | TTFT client mean | TPOT client mean | E2E mean | Throughput | Source |
  |---|---:|---:|---:|---:|---|
  | Small TP4/EP4 | -2.6% | +14.5% | +11.6% | -3.4% | `logs/20261008_10_inkling_standard/e2e_b2b_table.md`, `b2b_m1046` |
  | Inkling-NVFP4 TP8/EP8 | -4.4% | +2.3% | +2.5% | -2.8% | `logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/e2e_table.md` |

- **Campaign.** All 15 cases completed. Kernel abs error is 0.61-5.47% and
  coverage 98.78-99.66%. E2E is -4.68% to +4.34%, TPOT -6.31% to +4.41%.
  Out of tolerance:
  - the iteration population of cases 01, 02 and 04 (2.69-2.77%, 2.51-2.58%,
    1.25-1.27% against 1%);
  - decode kernel error in cases 06 and 09 (+5.40% and +5.19% signed against
    5%);
  - case 15 server TTFT (+27.73% against 12%), from the order of the requests
    submitted at t=0.

  Source: [`alignment_matrix.md`](alignment_matrix.md).

[`README.md`](README.md) explains each miss and the remaining gaps: Small TPOT
+14.5%, the TP8 decode cost items (sink overlap, `topk_copy` L2 timing,
critical-path share of three GEMMs), which need approval, and the KV
capacity. The arch counts sliding-window layers at full length, so at
131072 tokens per request its capacity is about 3.9x below vLLM's on TP8.
No simulation here reaches it; a follow-up PR models vLLM's cache groups. This is review
evidence, not an accepted baseline. No golden is recorded.

## Dependencies

- **vLLM fork** `servingstudio-alignment` at `892da082`, the same commit
  `main` uses.
- **Token corpora**: the `diverse_100` captures of the Hugging Face dataset
  `UW-SyFI/servingstudio-workload`, pinned at `a6959f415133a183260ecba335fbdb0fb12c3db8`. Host names and
  absolute paths were stripped from the provenance files before upload:
  - `inkling_nvfp4/vllm/diverse_100/capture/20261009`;
  - `inkling_small_nvfp4/vllm/diverse_100/capture/20261008`.
- **`profiling/profile.db`** gains 11,265 B200 rows, all measured by this
  branch, merged row-wise by semantic key onto `main`'s DB with 0 conflicts:

  | Kind | Rows |
  |---|---:|
  | `rel_bias_attention` | 6,472 |
  | `nvfp4_fused_moe` | 1,588 |
  | `bf16_fused_moe` | 1,324 |
  | `single_gemm` | 748 |
  | `elementwise` | 348 |
  | `rs_sconv_ag_residual_rms_norm` | 254 |
  | `gemm_fp32_output` | 232 |
  | `moe_sink_gate_select` | 152 |
  | `sconv_qkvr_prep` | 147 |

  `rel_bias_attention` includes 1,554 rows at `max_model_len` 1048576 (job
  5317). It also gains 26 profiling-run records. No row of `main` is changed
  or removed. The file grows from 59 MB to 69 MB.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
