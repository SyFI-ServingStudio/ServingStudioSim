# Spec5 Alignment Evidence

This directory follows the campaign workflow used by
[PR 25](https://github.com/SyFI-VibeSim/VibeSim/pull/25),
[PR 27](https://github.com/SyFI-VibeSim/VibeSim/pull/27), and
[PR 28](https://github.com/SyFI-VibeSim/VibeSim/pull/28).
The generated matrix is a partial evidence snapshot, not an accepted baseline.

## Maintained Artifacts

| Artifact | Responsibility |
|---|---|
| `presets/alignment/glm52_nvfp4_b200_spec5/campaign.yaml` | Workload matrix, framework settings, calibration values and their evidence/status |
| Pack `traces/invariants.json` | Reproducible workload digests |
| Pack `acceptance.yaml` | Acceptance tolerances and explained exceptions, independent of golden drift |
| `alignment_matrix.md` | Generated report values and tolerance failures |
| `server_latency.md` / `.json` | Analyzer server-side P50/P90/P99, raw values and report hashes; partial snapshot |
| `kernel_evidence.md` / `.json` | All 17 cases, 15 available kernel reports, explicit warm/historical sources and chunk sizes |
| `tests/golden/alignment_glm52_nvfp4_b200_spec5/NVIDIA_B200.json` | Future recorded baseline; not yet created |
| Golden `.provenance.json` | Formula/report versions, evidence paths and provisional inputs, written by the recorder |
| `profiling/profile.db` and `profiling/spec5_db_manifest.json` | Committed kernel timing data and import provenance |

Raw captures, frontend results and detailed request audits remain in the experiment
tree. Machine-specific paths belong in the local host profile. Do not copy
experimental calibration into defaults without its source and status, or update
a golden merely to remove a regression warning.

## Current Evidence

Source: `logs/20260906_1_spec5_warm_matrix/extracted_progress.json`, extracted with
clean implementation `0eef224`; calibrated campaign inputs are committed in `f9ef858`.
The snapshot includes completed case04/09 E2E and case07/08 kernel analysis.
It is limited to the TP4/EP4 matrix; TP8 reproduction is separate evidence.

Extraction SHA-256:
`5f7dc7eabc6884b7c3d713a69c2bf9fc84082f3d0d06fb02eababbc69c0c587f`.

| Cases | Evidence at snapshot |
|---|---|
| 01-08 | Full report sets available; failures remain in the generated comparison |
| 09, 10, 13, 14 | Warm workload, simulation and E2E available; matching warm kernel reports absent |
| 15-17 | Warm workload pending |
| 11, 12 | Previously observed framework KV-capacity failures; no successful warm result |

The current renderer lists cases with report values only. Absence from that table
does not mean a case passed or has no raw measurement. All 17 declarations remain
in extraction, including unavailable cases. Completed workload measurements are
reused; remaining measurements use warmup, corrected chunk sizes and no NSYS.

## Metric Boundaries

The standard matrix preserves the existing campaign formula contract:
`map_coverage_pct` reads `mapping.coverage.measured_duration_fraction`.
It is not the critical-path coverage shown in earlier diagnostic summaries.
Kernel signed/absolute errors, raw mapping coverage, critical-path coverage and
simulator coverage must remain separately named when comparing reports.

`server_ttft_mean_pct` and `server_tpot_mean_pct` are server-side means.
They are not P50. The standard `e2e_mean_pct` is the request E2E metric, not a
server-only TTFT metric. This legacy comparison table is retained for regression
compatibility; the final user-facing latency table also needs Analyzer server-side
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
Historical case09/10 NSYS uses 2048 chunks and remains separate from warm 2052 E2E.

`kernel_evidence.md` likewise retains historical 4096/8192 captures for cases13-17,
alongside the eight available warm captures. It does not imply matching warm
kernel evidence for all E2E cases. Both coverage denominators and each report's
original prediction/label provenance remain explicit.

The server latency sidecar selects only the `server_ttft` and `server_tpot` fields
from each completed warm bundle's `/api/v1/alignments/{id}/subjects/e2e/report`.
It preserves each metric's sample count, measured/simulated percentiles in ms,
resource ID and SHA-256 of the HTTP response. Refresh these exact resources as
new reports complete; derive displayed errors as `(simulated/measured - 1)*100`.
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
