# Spec5 Alignment Evidence

This directory follows the campaign workflow used by
[PR 25](https://github.com/SyFI-ServingStudio/ServingStudioSim/pull/25),
[PR 27](https://github.com/SyFI-ServingStudio/ServingStudioSim/pull/27), and
[PR 28](https://github.com/SyFI-ServingStudio/ServingStudioSim/pull/28).
The workload queue is finished. This is retained review evidence, not an accepted
baseline: missing pairs and tolerance failures remain explicit.

## Maintained Artifacts

| Artifact | Responsibility |
|---|---|
| `presets/alignment/glm52_nvfp4_b200_spec5/campaign.yaml` | Workload matrix, framework settings, calibration values and their evidence/status |
| Pack `traces/invariants.json` | Reproducible workload digests |
| Pack `acceptance.yaml` | Acceptance tolerances and explained exceptions, independent of golden drift |
| `campaign_metrics.json` | Compact public extraction: all 17 cases, formulas, values, report hashes and audit status |
| `alignment_matrix.md` | Generated report values and tolerance failures |
| `server_latency.md` / `.json` | Analyzer server-side P50/P90/P99, raw values and report hashes; 15 completed cases |
| `kernel_evidence.md` / `.json` | All 17 cases, 15 available kernel reports, explicit warm/historical sources and chunk sizes |
| `tests/golden/alignment_glm52_nvfp4_b200_spec5/NVIDIA_B200.json` | Future recorded baseline; not yet created |
| Golden `.provenance.json` | Formula/report versions, evidence paths and provisional inputs, written by the recorder |
| `profiling/profile.db` and `profiling/spec5_db_manifest.json` | Committed kernel timing data and import provenance |

Raw captures, frontend results and detailed request audits remain in the experiment
tree. Machine-specific paths belong in the local host profile. Do not copy
experimental calibration into defaults without its source and status, or update
a golden merely to remove a regression warning.

## Current Evidence

Source: `logs/20260906_1_spec5_warm_matrix/extracted_final.json`.
Extraction SHA-256: `aa1392be0fdd640318e495f4a878ea85b36913f0476f004a489172f227f029a6`.

All 15 valid cases completed warm, no-NSYS workload measurement, simulation and
E2E analysis. All 15 request-population audits pass. The queue's final seven cases
returned `ok`, and its process exited. Case11/12 retain their original framework
capacity failures: max length524288 needed27.31GiB KV with17.88GiB available
(the historical server logs are under `logs/20260905_2_spec5_matrix/<case>/profile_nsys/vllm/`).
There is no remaining measurement queue or requested GPU profiling.

The following table is derived from the two Analyzer JSON sidecars. Kernel source
includes its actual chunk size; historical kernel rows are not paired warm
captures. All displayed TTFT/TPOT values below are server-side P50 relative errors.
Full P90/P99 and raw milliseconds remain in `server_latency.json`.

| Case | Kernel source | Kernel signed error | Critical coverage | Sim coverage | Server TTFT P50 | Server TPOT P50 |
|---|---|---:|---:|---:|---:|---:|
| 01 | warm / 2052 | -2.44% | 94.05% | 99.27% | +3.94% | -2.77% |
| 02 | warm / 2052 | -5.25% | 94.60% | 99.15% | -3.50% | -4.71% |
| 03 | warm / 4098 | -3.44% | 96.95% | 99.42% | -0.72% | -1.39% |
| 04 | warm / 8196 | -3.43% | 97.60% | 99.42% | -6.37% | +7.28% |
| 05 | warm / 4098 | -5.64% | 96.48% | 99.25% | +241.60% | -17.37% |
| 06 | warm / 4098 | +7.82% | 95.97% | 98.58% | -3.07% | +10.60% |
| 07 | warm / 8196 | -3.74% | 97.84% | 99.64% | +3.34% | -4.22% |
| 08 | warm / 2052 | -5.28% | 93.38% | 99.68% | +4.53% | -3.40% |
| 09 | historical / 2048 | +7.22% | 93.86% | 97.62% | -5.39% | +4.27% |
| 10 | historical / 2048 | -4.28% | 94.10% | 97.44% | +2.94% | -6.92% |
| 11 | KV-capacity failure | N/A | N/A | N/A | N/A | N/A |
| 12 | KV-capacity failure | N/A | N/A | N/A | N/A | N/A |
| 13 | historical / 4096 | +2.91% | 94.87% | 98.11% | -36.46% | +9.99% |
| 14 | historical / 4096 | -1.13% | 95.28% | 98.54% | +5.18% | +0.33% |
| 15 | historical / 8192 | -2.31% | 94.99% | 98.20% | +23.30% | -5.37% |
| 16 | historical / 8192 | -3.53% | 94.67% | 97.85% | +2.24% | +2.32% |
| 17 | historical / 4096 | +2.99% | 95.38% | 98.17% | +3.40% | +16.03% |

Only cases01-08 have full matching warm kernel/workload/E2E sets. Cases09/10/13-17
have warm E2E and explicitly separate historical kernel evidence. The public
comparison exits1 because missing pairs and real tolerance failures remain.
No acceptance limit has been widened and no golden has been recorded.
This is the TP4/EP4 matrix; TP8 reproduction is separate evidence.

## Metric Boundaries

The standard matrix preserves the existing campaign formula contract:
`map_coverage_pct` reads `mapping.coverage.measured_duration_fraction`.
It is not the critical-path coverage shown in earlier diagnostic summaries.
Kernel signed/absolute errors, raw mapping coverage, critical-path coverage and
simulator coverage must remain separately named when comparing reports.

`server_ttft_mean_pct` and `server_tpot_mean_pct` are server-side means.
They are not P50. The standard `e2e_mean_pct` is the request E2E metric, not a
server-only TTFT metric. This legacy comparison table is retained for regression
compatibility; the user-facing latency table uses Analyzer server-side
P50/P90/P99, with explicit statistic names. Never substitute client TTFT for server
TTFT or change an existing metric key's meaning.

Per-request acceptance conditioned on observed outcomes is calibration, not an
independent prediction. Keep the prepared trace manifest and request-population
audit beside the simulation, and record explicit GPU-time multiplier sources and
approximations. A successful audit does not certify timing accuracy.

The refreshed campaign uses measured capacity with the FullIndex KV footprint.
Previously completed simulations retain their snapshotted `raw/params.json`;
cases01-03/05-08 used the earlier 55224 bytes/token conversion instead of 55932.
Case04/09 use the corrected conversion. Do not claim those historical simulations
were generated with the refreshed campaign. Case09 borrows warm case02's explicit
time multiplier; its experiment `CALIBRATION.md` documents the approximation.

Case10 also uses the corrected conversion and the same explicitly borrowed warm
case02 multiplier. Its 512-request workload and E2E analysis completed after the
standard matrix snapshot; `server_latency.md` already includes that result.
Case13 also completed, with 48 requests and no missing acceptance samples. It uses
the measured 474560-token KV capacity and explicitly borrows warm case06's time
multiplier (same concurrency8/chunk4098, shorter context); see its `CALIBRATION.md`.
Case14 completed 60 requests with no missing acceptance samples. It uses the
measured 499776-token KV capacity and explicitly borrows warm case03's multiplier
(same chunk4098, different concurrency/context), recorded in its `CALIBRATION.md`.
Case15 completed 72 requests with no missing acceptance samples. It uses the
measured 438784-token KV capacity and explicitly borrows warm case07's multiplier
(same chunk8196, different concurrency/context), recorded in its `CALIBRATION.md`.
Case16 completed 96 requests and its request-population audit passed. Its measured
KV capacity is 472448 tokens, and it explicitly borrows warm case07's multiplier.
The prepared manifest lists 24 request-position fallbacks to this run's aggregate
acceptance; these are retained approximations, not additional measured samples.
Case17 completed 32 requests in 193.25 seconds of formal measurement, with measured
KV capacity440896 tokens and an explicit warm case03 multiplier. Its manifest
retains 14 request-position fallbacks to the run aggregate. All original requests
remain present; no context was shortened to avoid cache work.
Historical case09/10 NSYS uses 2048 chunks and remains separate from warm 2052 E2E.

`kernel_evidence.md` likewise retains historical 4096/8192 captures for cases13-17,
alongside the eight available warm captures. It does not imply matching warm
kernel evidence for all E2E cases. Both coverage denominators and each report's
original prediction/label provenance remain explicit.

The server latency sidecar selects only the `server_ttft` and `server_tpot` fields
from each completed warm bundle's `/api/analyzer/v1/alignments/{id}/subjects/e2e/report`.
It preserves each metric's sample count, measured/simulated percentiles in ms,
resource ID and SHA-256 of the HTTP response. Refresh these exact resources when reports change; derive displayed errors as `(simulated/measured - 1)*100`.
Keep unavailable cases visible and never substitute client fields or means.

## Regeneration

Run from the repository root; choose the experiment root explicitly:

```bash
uv run python -m launcher alignment-campaign check --pack glm52_nvfp4_b200_spec5
uv run python -m launcher alignment-campaign extract \
  --pack glm52_nvfp4_b200_spec5 \
  --runs logs/20260906_1_spec5_warm_matrix \
  --out logs/20260906_1_spec5_warm_matrix/extracted_current.json
uv run python -m launcher alignment-campaign compare \
  --pack glm52_nvfp4_b200_spec5 \
  --measured logs/20260906_1_spec5_warm_matrix/extracted_current.json --markdown
```

Replace the generated matrix from command output and preserve the comparison exit
status. A nonzero status currently represents real missing/out-of-tolerance evidence.
After completing the intended report sets and documenting calibrated inputs:

```bash
uv run python -m launcher alignment-campaign compare \
  --pack glm52_nvfp4_b200_spec5 \
  --measured logs/20260906_1_spec5_warm_matrix/extracted_current.json --record
```

Commit the generated golden and its provenance together with any corresponding
campaign and evidence-document changes. Do not silently filter missing cases or
use `--accept-provisional` to conceal unmeasured inputs. Recording a baseline and
meeting acceptance tolerances are separate judgments. PR 27 likewise deferred
golden recording while its analysis was incomplete.
