# GLM-5.3-Flash FP8 alignment evidence (B200, TP4/EP4)

Review evidence for the `glm53_flash_vllm_fp8_kda_dsa_moe` model and the
`glm53_flash_fp8_b200_tp4_ep4` campaign pack. The measurement queue is finished.
This is not an accepted baseline: tolerance failures and the known gaps below
stay explicit, and no golden has been recorded.

## Maintained artifacts

| Artifact | Responsibility |
|---|---|
| `presets/alignment/glm53_flash_fp8_b200_tp4_ep4/campaign.yaml` | Fifteen-case matrix (the GLM-5.2 shapes, repeats and concurrency), framework flags, per-case calibrated inputs with status and evidence |
| Pack `acceptance.yaml` | Tolerances; no per-case exceptions |
| Pack `label_rules/` | 138 order-free kernel label rules and the generator that emits them |
| `alignment_matrix.md` | Generated case matrix and tolerance failures |
| `campaign_metrics.json` | `extract` output: every case, formula and value behind the matrix |

Raw captures, server logs and Analyzer reports stay in the experiment tree
`logs/20260925_4_glm53_flash_campaign/` (untracked).

## Calibrated inputs

- **KV budget.** `attn_gpu_memory_gb` is each case's measured per-GPU
  `Available KV cache memory` (GiB x 1.0737). Cases 14 and 15 pin the pool with
  `--kv-cache-memory-bytes` to keep their terminal-demand ratios, and cap
  `--max-num-seqs` at 256: vLLM needs one Mamba cache block per decode sequence
  and the pinned pools hold only 526 / 407 of them.
- **Prefill iterations.** `worker.prefill_gpu_time_multiplier: 1.18` is the
  workload-pass mixed-iteration cycle ratio at multiplier 1.0, pooled over cases
  01-07 (08 held out). Decode iterations stay at 1.0 (0.99-1.07x in the same
  passes). It is a calibration, not a prediction; the gap it covers is host-side
  (see known gaps). It was measured before the prefill-indexer fix, which moves
  01-07 mixed kernel sums by at most 0.2%, so it was not re-derived.
- **Expert routing.** Measured token corpus (`routing: corpus`), never uniform.
- **API servers.** Every vLLM case runs 4 API server processes
  (`vllm serve --api-server-count 4`, verified from the startup log). With one,
  request ingress serialized at ~0.75 ms per request and inflated server TTFT.

## Current evidence

Source: `logs/20260925_4_glm53_flash_campaign/`, extracted into
`campaign_metrics.json`. The committed copy drops the per-request identity
lists and makes paths relative to the repository root.

- **Coverage.** All 15 cases completed every phase: workload pass, NSYS
  pass, timing-predict, label, kernel analysis, simulation and E2E analysis.
  All 15 request-population audits pass. Case 09 replays its 1 req/s baseline
  trace at 2.7 req/s, so its arrival scale is 1/2.7.
- **Kernel timing.** Absolute error 1.15-5.41%, signed -3.79% to +3.55%.
  Mapping coverage is 97.93-99.27%. Only case 11 exceeds the 5% kernel
  tolerance, at 5.41% (signed +1.91%).
- **End to end.** 14/15 cases are within the 7% E2E tolerance, all 15 within
  the 11% TPOT tolerance, and 14/15 within the 12% TTFT tolerance. Case 11
  E2E is -9.99%; case 08 TTFT is +15.8%.
- **Workload structure.** Cases 01, 02 and 09 exceed the 1% tolerance: 01/02
  run 2.7-2.9% fewer iterations and 09 runs 5.1% more.

`alignment_matrix.md` lists every value and every tolerance failure.

## Known gaps

- **Fixed host time per prefill iteration.** Kernel timing matches in the long
  cases (11 signed +1.91%, 12 +2.13%). In both, each prefill-bearing
  iteration spends a roughly constant non-kernel interval that does not grow
  with context (K = simulated kernel sum at multiplier 1.0):

  | Case 11 iterations | Measured - K | Measured / K |
  |---|---:|---:|
  | 2048-token chunk | +16.3 to +18.8 ms | 1.25-1.33 |
  | 128-token chunk | +23.0 to +24.4 ms | 2.08-2.31 |
  | decode | -0.3 to -0.6 ms | 0.90-0.95 |

  - A multiplier cannot express an additive interval. The 1.18 calibrated on
    01-07 therefore under-bills these c1 iterations, and case 11/12 E2E run
    fast (sim faster than measured).
  - The same interval does not fit the 01-07 mixed-iteration regression
    either, which is `max(K + 6.0 ms, 49.0 ms)`.
  - Suspected sources, from the capture: a blocking device-to-host copy in the
    KDA chunk-prefill setup and ~1150 eager launches per mixed forward.
  - Modeling this interval is follow-up work.
- **Prefill top-k timing template.** The `vllm_fork_cuda` `dsa_topk_prefill`
  rows use a linspace logits template. At 2048 rows x 65K pools they read
  0.33 ms, against ~0.21 ms per layer measured.
- **Decode top-k.** The measured `topk_decode` runs 9-19% slower than its
  stride-ramp rows. That is the other direction from the fork's overflow
  exemption, so the rows stay.
- **Scheduler population, cases 01/02.** 2.7-2.9% fewer iterations with
  correspondingly larger batches. The pack's 1% workload tolerance is not
  relaxed.
- **Case 08 server TTFT (+15.8%).** Requests queue behind chunked prefill in
  the simulator (mean queue 535 ms simulated vs 409 ms measured).
- **Measurement hygiene.** vLLM is host-bound in prefill here, so workload
  passes are sensitive to other CPU work on the node. Case 04's first pass with
  4 API servers overlapped a compile job. Its prefill iterations read 78.4 ms;
  re-measured alone they read 63.8 ms (64.9 ms with 1 API server). Cases 03
  and 10 were re-measured alone as well, and the committed evidence uses the
  isolated passes. The contended passes are kept as `*.api4_contended` in the
  experiment tree.
- **Four API servers and decode.** Measured start to next start, decode
  iterations are 4.1% slower with 4 API processes than with 1 in case 03,
  consistent across batch sizes, and 1.9% slower in case 10. Cases 01, 02 and
  08 differ by at most 1%. The logged per-iteration elapsed time
  over-states this. With 4 API processes some host work moves from the
  inter-iteration gap into the logged window, so elapsed alone reads +8% in
  case 03 and +25% in case 10. Each API-server setting was run once, so
  run-to-run variation is not separated from the 03 effect.

## Reproduce

From the repository root, with `PACK=presets/alignment/glm53_flash_fp8_b200_tp4_ep4`
and `OUT` a fresh experiment directory:

```bash
uv run python -m launcher alignment-campaign check --pack "$PACK"
uv run python -m launcher alignment-campaign render --pack "$PACK" \
  --host <host> --out-root "$OUT"
for phase in profile_workload profile_nsys timing_predict; do
  uv run python -m launcher alignment-campaign run --pack "$PACK" --out-root "$OUT" --phase "$phase"
done
uv run python -m launcher alignment-campaign label --pack "$PACK" --out-root "$OUT"
for phase in analysis_kernel simulation analysis_e2e; do
  uv run python -m launcher alignment-campaign run --pack "$PACK" --out-root "$OUT" --phase "$phase"
done
uv run python -m launcher alignment-campaign extract --pack "$PACK" --runs "$OUT" --out metrics.json
uv run python -m launcher alignment-campaign compare --pack "$PACK" --measured metrics.json --markdown
```

- `<host>` names a host file under `presets/alignment/hosts/`, modeled on
  `example.yaml`. This evidence used `b200_raid`.
- The two profile phases need four B200s per case. Run each workload pass
  alone on its node (see Measurement hygiene).
- `run` also takes `--case` and `--parallelism`. Each phase writes a
  `.complete` marker, so a rerun skips finished cases.
