# Kimi-K3.1-Pro NVFP4 evidence (vLLM + DSpark k=7; GB200 TP16/EP16 and 2x B200 TP8 x DP2/EP16)

This is review evidence for the `kimi_k31_vllm_nvfp4_tp_attn_ep_moe_dspark` arch.
It covers what the simulator models, how the model was checked against vLLM
captures of `trace/diverse_100.csv`, and what is still open. It is not an
accepted baseline, and no golden is recorded.

Captures, Analyzer bundles and profiling jobs stay in the untracked experiment
trees under `logs/`. Numbers are tagged by origin:

- measured: a vLLM capture or a profiling run
- simulated: a timing prediction or a simulation
- catalog: the checkpoint config or the fork source
- derived: arithmetic on the above

## Deployment modeled

Both deployments share these server settings:

- vLLM V1 at upstream `c5db9bd` with the alignment patch line
- chunked prefill at 4096 tokens
- `--kv-cache-dtype fp8`, `max_num_seqs` 32, `max_model_len` 131072
- full and piecewise CUDA graphs
- EPLB off
- DSpark k=7 with probabilistic draft sampling and block rejection
- all-to-all backend `allgather_reducescatter`

| Deployment | Host key | Ranks | Collectives |
|---|---|---|---|
| GB200 NVL72, 4 trays x 4 GPUs | `gb200_nvl72` | one TP16 replica, EP16 | TP all-reduce: FlashInfer MNNVL at TP16. EP: all-gather / reduce-scatter over MNNVL. |
| 2x B200 NVL8 | `b200x2_nvl8_ib` | attention TP8 per node, DP2, EP16 | EP: NCCL all-gather / reduce-scatter over InfiniBand at the DP-padded total |

Each deployment has its own capture of `diverse_100` (100 requests, enwik9
prompts). The captures are published in `UW-SyFI/servingstudio-workload` under
`kimi_k31_pro_nvfp4/vllm_dspark_k7/diverse_100/capture/`:

- `20261120` (GB200)
- `20261125` (B200, a replicate of the GB200 capture)

Each capture contains:

- the token corpus of routed-expert ids: 135,144 generated tokens x 92 MoE
  layers x top-16 of 896. Prompt routes and rejected-draft routes are not
  recorded.
- a `trace.csv` with the per-request observed acceptance

The MoE profile rows are measured on each deployment's own routes, so the two
deployments do not share one capture.

## How each model feature is modeled

| Feature | vLLM | Simulator |
|---|---|---|
| KDA linear attention | RecoverSSM verify over the k+1 draft positions; causal conv decode | `kda_recoverssm_verify` and `gdn_causal_conv_decode` rows from the current Kimi-K3 fork's Triton kernels |
| Sparse MLA + indexer | DSA indexer logits, persistent top-k, sparse MLA decode / prefill | DSA kinds at qk_nope 128 + RoPE 64, v_head 128; under a full CUDA graph the top-k reads `vllm_cuda_graph` rows (capture-time `max_seq_len = max_model_len`) |
| ShardQ prefill under attention TP | token/head exchange around sparse-MLA prefill | new kind `sparse_mla_shard_q` |
| Router | GateLinear | priced as the router GEMM vLLM launches |
| Routed MoE | NVFP4 fused MoE on each EP rank, SiTU-GLU activation (`siTuGlu` GEMM1), tactics by vLLM's selection rule | `nvfp4_fused_moe` on the critical EP rank with `gated_activation = situ:beta=4,linear_beta=25` from the checkpoint, routes from the deployment's own corpus; rows measured with the runner following vLLM's tactic rule; capture sizes added to the grid (`extra_token_points`) |
| EP exchange | all-gather / reduce-scatter (`allgather_reducescatter`); under DP the total is padded to DP x the padded per-rank tokens | `moe_ep_quantized_all_gather` / `moe_ep_reduce_scatter`; totals capture_sizes x dp_size added to the grid (`extra_total_tokens`) |
| TP all-reduce | FlashInfer MNNVL at TP16 across GB200 trays; NCCL past the MNNVL limit | `all_reduce_fusion` / `flashinfer_mnnvl` at TP16, allowed where the NVLink domain holds TP16; the NCCL slot only when a step can outgrow MNNVL |
| DSpark k=7 | draft layers on the target's hidden states, block rejection | speculative worker, draft 7, per-request observed acceptance from the capture's `trace.csv` |
| KV and hybrid cache | hybrid block size from the topology | `attn_gpu_memory_gb` read from the capture log: 60.805999 (GB200), 35.293894 (B200) |

## Kernels

New kind `sparse_mla_shard_q`; new backends `row_all_gather` / NCCL, `elementwise` / vLLM-env Triton,
`dsa_persistent_topk_decode` / `vllm_cuda_graph`; multi-node runners for NCCL and FlashInfer MNNVL all-reduce
and for the vLLM pynccl EP collectives over InfiniBand. Rows for both deployments are in `profiling/profile_db/`;
`export-missing` reports 0 missing specs on both.

`nvfp4_fused_moe` gains the optional arg `gated_activation`; absent means SwiGLU, is stored as NULL and is not part
of `args_hash`. The FlashInfer runner keeps `runner_autotune` as its default tactic policy. With
`VIBESIM_FLASHINFER_TACTIC_POLICY=vllm`, `MAX_NUM_BATCHED_TOKENS` and `DP_SIZE` it reproduces serving by following
vLLM's tactic selection rule, including the fallback to the default tactic where autotune does not cover a size.
The Kimi routed rows are measured this way (budget 4096; DP 2 on B200, DP 1 on GB200). The DB key does not include
the budget or DP.

## Check 1 and simulation (alignment Phase 1)

The official tables and the Check 1 buckets are in [`PR.md`](PR.md#test-result), pasted from
[`alignment_phase1.md`](alignment_phase1.md). Both deployments are regenerated with the routed MoE rows measured
with SiTU-GLU and vLLM's tactic rule. An earlier B200 run had used the routes of an older B200 capture by
mistake; with the pinned routes of the 20261125 capture, B200 kernel abs moved from 4.18% to 4.07%.

Excluded iterations:

- B200 removes 7 steps under the `incomplete_device_coverage` rule: iterations
  0, 1, 2, 3, 16, 17, 18.
- GB200 removes iteration 0.
- GB200 multiplier-excluded iterations: 0, 3, 61, 374-383.

## Changes that reach other models

- **`gpu/spec.json`.** The `NVIDIA GB200` and `GB200` aliases move from the
  `GB200-Superchip` row (two GPUs combined) to a new per-GPU row,
  `GB200-GPU-186GB`. Any model that runs on GB200 and reads spec peaks
  (roofline fallbacks, memory size) now sees the per-GPU values. Before this
  change those values were double.
- **`all_reduce_fusion` / `flashinfer_mnnvl`** accepts TP16 on GPUs whose NVLink
  domain holds the group (`nvl_domain_size` from `gpu/spec.json`: GB200 72,
  B200 8). A GB200 TP16 member of any model can now price the fused all-reduce
  from rows. B200 TP16 is still refused with the same message as on `main`.
- **`nvfp4_fused_moe`** gains the optional `gated_activation` arg. Other archs
  leave it unset, so their rows, hashes and identities are unchanged. The
  runner's default tactic policy stays `runner_autotune`.
- **Analyzer.**
  - A per-node clock alignment of multi-node captures (`clock.rs`).
  - The `incomplete_device_coverage` exclusion.
  - Single-node captures have one clock domain and are not shifted.
- **`moe_ep_quantized_all_gather` / `moe_ep_reduce_scatter`** gain
  `extra_total_tokens`, and **`nvfp4_fused_moe`** gains `extra_token_points`.
  They are empty by default and left out of the serialized config, so existing
  identities and predictions are unchanged. Other archs (GLM-5.2, GLM-5.3-Flash,
  DeepSeek-V4/V4.1, `fixed_exchange`) only gain the empty field.
- **`dsa_persistent_topk_decode`** gains a `vllm_cuda_graph` backend. The
  existing `vllm_cuda` backend and its rows are unchanged.
- **`profiling/profile_db/`** only adds rows and the nullable `gated_activation`
  column of `nvfp4_fused_moe.db`. `merge-db` adds 24,065 kernel rows to 29 kind
  files of `main`'s directory (2 of them new files) with 0 conflicts, and no
  existing row changes.

## Modeling choices

- The EP grid is extended to capture_sizes x dp_size; pricing is unchanged.
- The B200 eager prefill host gap (~4.4 ms per step) is not modeled; no general model is added for one case.
- The B200 mixed last-chunk outliers (iterations 52, 293) are kept, not filtered by a new rule.

## Known gaps

The figures below come from the official alignment Phase 1 run in this PR,
with the SiTU-GLU routed MoE rows.

### B200 prefill and EP collectives

- Mixed (prefill) steps are about 4.5% fast (cut chunk −4.54%).
  - The routed MoE tactic mismatch is fixed. The runner used to autotune
    `t128x256`, while serving uses vLLM's selection (tuned buckets up to the
    4096-token budget, the default tactic −1 above it, e.g. `(192,-1)` at
    M=8192). The rows are now measured with that rule, which moved cut-chunk
    steps from −8.54% to −4.61% with SwiGLU rows.
  - Another part is a host gap of about 4.4 ms per step in eager mode, which the
    simulator does not model. No general model is added for it.
- NCCL EP all-gather / reduce-scatter time is high, mostly waiting for the peer
  node.
- In prefill/prefill steps, EP all-gather / reduce-scatter on large messages
  runs about 18-19% slower in serving than in the microbenchmark. The cause is
  unknown. The simulated minus measured gap is −19.4% for
  all-gather and −17.9% for reduce-scatter, over 110 steps.

### B200 routed MoE rows

- The SiTU-GLU rows match serving's GEMM1 and GEMM2 kernel names exactly.
- At M=8192 the median-rank SiTU row is 423.5 µs (401.9 µs with SwiGLU), which
  is +14 to +21 µs slower than serving's 402.5-409.6 µs. The cause is unknown.
- The B200 simulation prices the heaviest EP rank's row: 480.0 µs at M=8192
  (480.8 µs with SwiGLU). That is why switching to SiTU moved the B200 tables by
  less than 0.2 points.

### GB200 routed MoE tactics

- With SiTU-GLU and vLLM's tactic rule, decode-size rows change by at most
  ±2.5%, 1.3k-4k-token rows are 3-7% slower, and rows above 4096 tokens (the
  default tactic −1) are 10-29% slower than the SwiGLU autotuned rows. GB200
  cut chunk moved from −2.54% to −1.54%, and E2E from −4.92% to −3.26%.
- On some token buckets the tactic that vLLM's autotune picks is not
  deterministic, so a measured row may use a different tactic from serving.
  The effect is within a few percent.

### B200 E2E TPOT

TPOT is −25.50% (−30.13% before the routed MoE rows followed vLLM's tactic rule). It breaks into three parts:

- **Short requests in the per-request average.** Mean TPOT weights every request
  equally, so a short request's few steps count as much as a long request's
  thousands.
- **Per-request acceptance distribution.** The trace gives each request its
  observed per-position acceptance probability, not its per-round outcomes.
- **Exposed fraction of mixed steps.** The share of decode steps that carry a
  prefill chunk differs between simulation and serving.

The per-part split has not been measured; the parts are qualitative.

### B200 mixed last-chunk outliers (iterations 52 and 293)

Only part of each TP group did host-side lazy loading and `SetAttribute` in
these steps. The rest of the group waited for those GPUs in the group's
collective. Iteration 52 measured −42.2% and iteration 293 at
−29.1%.

These two steps are kept in the data and listed as known outliers. The
mixed last-chunk figures with and without them:

| Bucket | n | Signed |
|---|---:|---:|
| With iterations 52 and 293 | 10 | −11.01% |
| Without them | 8 | −5.97% |

### Decode residuals

From the official Check 1 bucket tables in [`PR.md`](PR.md#test-result): the
same-batch-shape step time from the Analyzer reports, decode steps grouped by
the number of decode requests, signed error (sim − meas) / meas over the
bucket's summed step time.

- GB200: decode overall is −2.73% (abs 3.24%). 1-4 requests are within ±0.8%;
  5-8, 9-16 and 17-32 requests are −3.43%, −4.24% and −8.31% (the last bucket
  has 20 steps).
- B200: decode overall is −0.02% (abs 2.65%). 1-4 requests are −3.4% to
  −3.9%; 5-8 and 9-16 requests are +1.35% and +2.78%.

### Not in this PR

GB200 DP16 (`gb200_dp16_ep16`) is kept on a separate branch.

## Reproduce

Run the command block in `skills/operate-run-alignment` ("Alignment Phase 1 and alignment Phase 2") with
`AP=1` for the packs `kimi_k31_pro_nvfp4_gb200_tp16_ep16` and `kimi_k31_pro_nvfp4_b200x2_tp8dp2_ep16`.
