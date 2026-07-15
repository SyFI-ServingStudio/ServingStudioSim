# Analyzer catalog resource bounds

## Summary

- Moved the `/api/v1/runs` lifecycle projection into `ui_service/catalog.rs`
  and added a one-entry metadata-stamped response cache.
- The stamp covers the ordered `run_id` values, discovery `updated_at` values,
  `.complete`/`.failed` marker identities, and the securely opened metadata of
  each pipeline/timing input.
- Cold lifecycle builds are single-flight inside the bounded blocking artifact
  pool. Cache hits, including conditional `304` responses, inspect metadata but
  do not read or parse pipeline/timing bodies.
- Each cold attempt retains the exact pipeline/timing descriptors used for
  preflight accounting and parses those same descriptors. An atomic replacement
  between preflight and parsing cannot redirect the read to a new inode.
- A cold build sums pipeline/timing lengths across all discovered runs and
  rejects more than 16 MiB with HTTP 413 and stable code
  `catalog_state_too_large` before body reads begin.
- Aggregate accounting runs before per-file classification. An input over the
  total 16 MiB budget fails the catalog, while an input over only its smaller
  per-file cap is not read and projects that run's analysis as failed.
- Catalogs are capped at 1,024 runs and fail with HTTP 413 and stable code
  `catalog_too_many_runs` before per-run metadata opens begin.
- Catalog bytes are published only when before/after metadata stamps agree.
  Three consecutive changes fail closed with HTTP 409 and stable code
  `artifact_generation_changed`.
- Response ETags are strong SHA-256 digests of final serialized bytes. Metadata
  stamps are used only for cache invalidation.

## Validation

- `cargo test -p analyzer --offline`: 96 passed.
- Catalog regressions prove concurrent cold requests perform one lifecycle body
  build, cached `200` and conditional `304` requests perform no lifecycle body
  reads, marker/timing mutation invalidates the cached ETag and lifecycle, and
  aggregate oversize is rejected before reading the oversized input.
- A barrier-controlled concurrency regression guarantees the second cold request
  reaches the build lock while the first retains it. Atomic-replacement coverage
  proves the first attempt decodes the retained old inode and the retry decodes
  the new inode.
- Tests bind the public ETag to the exact response bytes and prove any serialized
  representation change produces a new validator. The 1,025-run case proves the
  count boundary and stable problem code.
- Regressions prove one timing input above 16 MiB returns
  `catalog_state_too_large` without a body read, while a pipeline input one byte
  over its 1 MiB cap yields catalog `200` plus a row-local failed lifecycle and
  no pipeline body read.
- A deterministic churn regression proves exactly three attempts precede the
  stable `artifact_generation_changed` response.
- Scoped `rustfmt` and `cargo check -p analyzer --offline` are run on the final
  change set; `git diff --check` reports no whitespace errors.
