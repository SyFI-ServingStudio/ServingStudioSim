# feat(glm52): add Spec5 simulation and reproducible framework alignment

Draft preparation only. Complete the evidence matrix and replace pending entries
before publishing; this document is not a claim that the campaign passed.

## Purpose

Support GLM-5.2 NVFP4 speculative decoding with explicit draft depth, target/draft
timing and routing inputs, and scheduler accounting for executed verification
work, including rejected candidates. Wire the same execution contract into
offline prediction, necessary-work lower bounds and request conservation.

The serving evidence path preserves speculative query widths and committed output
progress. Req-frontend warms representative shapes, drains requests and resets
prefix caching before an independent measurement. Warm E2E uses workload metrics
without NSYS; kernel alignment reuses existing captures.

Model-specific inference and automatic report-driven calibration have been removed
from shared launchers. Observed acceptance preparation and request identity checks
are explicit reusable alignment commands. Unwired vocabulary gather/copy/argmax
work remains on `spec5-logits-wip`; experimental scripts remain on
`spec5-alignment-experiments`.

## Test Plan

- `uv run pytest -q tests/test_alignment_campaign.py`: 240 passed after the measured calibration update.
- `uv run python -m launcher alignment-campaign check --pack glm52_nvfp4_b200_spec5`: 0 errors; 13 provisional warnings at this checkpoint.
- `uv run --no-sync cargo test -p simulator --lib` (via the CPU recipe with libpython configured): 1062 passed, 6 ignored.
- `uv run --no-sync cargo test --manifest-path analyzer/rust/Cargo.toml --bin analyze`: 255 passed.
- `uv run --no-sync pytest -m 'not gpu and not agent and not bench' -n 8`: 3414 passed, 4 skipped.
- No new GPU kernel profiles or NSYS captures were run during PR preparation.

## Test Result

See [the generated matrix](alignment_matrix.md) and [evidence definitions](README.md).
The [17-case kernel evidence table](kernel_evidence.md) contains 15 available
reports: eight warm captures and seven historical captures with their original
2048/4096/8192 chunk settings. Historical rows are not matched warm E2E evidence.
Eight cases currently have complete kernel/workload/E2E report sets. Cases09/10/13 E2E
are complete; four warm workload measurements and remaining report work are pending.
Cases11/12 retain framework capacity failures. Missing cases and tolerance failures
remain visible; no alignment golden has been recorded yet.

The [server latency table](server_latency.md) includes Analyzer P50/P90/P99 for
eleven completed cases, backed by raw values and report hashes in its JSON sidecar.
Raw mapping coverage remains distinct from critical-path and simulator coverage.
Observed per-request acceptance and borrowed time multipliers are calibration,
not independent prediction; use each result's recorded parameters for reproduction.

`profiling/profile.db` retains 66174 previous rows and adds 21680 rows, with zero
conflicting rows across the 76-table comparison. The committed manifest preserves
source identities and historical provenance limitations. Its SHA-256 remains
`b0fc741fec9a1faf31438b8a4b4167fe74c755ca6b5598fd8001848294f04779`.

## Dependencies

- vLLM: `0b8edfb` to `2c2f73f`; speculative observation and target/draft routing records.
- Req-frontend: `dbe0f9e` to `ec54cd0`; acceptance trace support and bounded warmup.
- GitHub commit API returned HTTP422 for both new gitlinks on 2026-09-06.
  Publish and verify dependency branches before publishing the parent PR.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before merge.

Author acknowledgements remain for the contributor. Serving wrappers depend on
the pinned vLLM and FlashInfer implementations; finalize the third-party version
and license disclosure against those sources before publication.
