# feat(glm52): add Spec5 simulation and reproducible framework alignment

Prepared for review after the measurement queue finished. This is not a claim
that all alignment tolerances passed; retained gaps are listed below.

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
- `uv run python -m launcher alignment-campaign check --pack glm52_nvfp4_b200_spec5`: 0 errors; 5 provisional warnings at this checkpoint.
- `uv run --no-sync cargo test -p simulator --lib` (via the CPU recipe with libpython configured): 1062 passed, 6 ignored.
- `uv run --no-sync cargo test --manifest-path analyzer/rust/Cargo.toml --bin analyze`: 255 passed.
- `uv run --no-sync pytest -m 'not gpu and not agent and not bench' -n 8`: 3414 passed, 4 skipped.
- No new GPU kernel profiles or NSYS captures were run during PR preparation.
- Full GPU unit, bench and agent test tiers were not rerun; this gate is CPU tests
  plus the completed real B200 workload campaign.
- `git diff master..HEAD --check` passed. Changes after the full CPU gate are
  calibration data, evidence documents and generated-data diff attributes only.

## Test Result

See [the generated matrix](alignment_matrix.md) and [evidence definitions](README.md).
The [17-case kernel evidence table](kernel_evidence.md) contains 15 available
reports: eight warm captures and seven historical captures with their original
2048/4096/8192 chunk settings. Historical rows are not matched warm E2E evidence.
All 15 valid cases completed warm workload measurement, simulation and E2E,
with all request-population audits passing. Eight cases have matching warm
kernel/workload/E2E report sets; cases09/10/13-17 retain historical kernel evidence.
Cases11/12 retain framework capacity failures. Missing cases and tolerance failures
remain visible; no alignment golden has been recorded yet.

The [server latency table](server_latency.md) includes Analyzer P50/P90/P99 for
fifteen completed cases, backed by raw values and report hashes in its JSON sidecar.
Case05's TTFT P50 error remains +241.60%, case13's -36.46%, and case15's +23.30%;
these are disclosed model-alignment limitations, not passing baseline claims.
Case17 TTFT/TPOT P50 errors are +3.40%/+16.03%. The public comparison exits1;
all17 declarations, formulas and available numbers are retained in
[campaign_metrics.json](campaign_metrics.json). No new measurement is pending.
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
- Remote master still equals the review baseline `fe65028`; API permission checks
  confirm push access to all three repositories. No remote changes made yet.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before merge.

Author acknowledgements remain for the contributor. Changed vendored dependencies:
[vLLM](https://github.com/serendipity-zk/vllm), commit `2c2f73f`, and
[req-frontend](https://github.com/uw-syfi/request-factory), commit `ec54cd0`,
both retain their Apache-2.0 LICENSE files. The profiling wrappers call
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) APIs; the preparation
environment reports `flashinfer-python 0.6.11.post3`, Apache-2.0 package metadata.
That environment version does not replace per-source historical DB provenance
in `profiling/spec5_db_manifest.json`.

## Review Checklist

- Design/API basis: `alignment/README.md`, `launcher/alignment_campaign/README.md`,
  `doc/detailed_design/L5.md` and `model/work/README.md`.
- Mainline has no equivalent Spec5 workload prediction; the committed tables are
  new measured evidence, not a claimed before/after speedup over master.
- DB rows were merged row-wise; GPU is NVIDIA B200. Per-table counts and source
  limitations are retained in `profiling/spec5_db_manifest.json`.
- Missing/unsupported kernel identities remain explicit. Final cases15/16/17
  required no JIT fills (0/8019, 0/7915, 0/8019 missing respectively).
- Alignment golden recording is deferred because warm kernel pairs are absent
  for seven cases and two cases failed capacity. Throughput/sim-speed goldens
  were not changed. Acceptance standards remain unchanged.
- Clean and review are separate worktrees. Review's existing staging is preserved.
  Uncommitted local audit/goal notes are excluded from the PR.
