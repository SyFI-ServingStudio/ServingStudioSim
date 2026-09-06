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
| `tests/golden/alignment_glm52_nvfp4_b200_spec5/NVIDIA_B200.json` | Future recorded baseline; not yet created |
| Golden `.provenance.json` | Formula/report versions, evidence paths and provisional inputs, written by the recorder |
| `profiling/profile.db` and `profiling/spec5_db_manifest.json` | Committed kernel timing data and import provenance |

Raw captures, frontend results and detailed request audits remain in the experiment
tree. Machine-specific paths belong in the local host profile. Do not copy
experimental calibration into defaults without its source and status, or update
a golden merely to remove a regression warning.

## Current Evidence

Source: `logs/20260906_1_spec5_warm_matrix/extracted_current.json`, extracted with
clean implementation `d47bcd4`. This snapshot predates the remaining workload retry.
It is limited to the TP4/EP4 matrix; TP8 reproduction is separate evidence.

Extraction SHA-256:
`5ef457379450940bb31b7b099e556cae1e569beb926480aa666356c42e4f9c99`.

| Cases | Evidence at snapshot |
|---|---|
| 01, 02, 03, 05, 06 | Full report sets available; failures remain in the generated comparison |
| 04 | Kernel report available; warm workload retry pending |
| 07 | Warm workload and E2E report available; kernel report pending |
| 08 | Warm workload and simulation complete; E2E analysis pending |
| 09 | Warm workload complete; simulation and analysis pending |
| 10, 13-17 | Warm workload pending |
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
