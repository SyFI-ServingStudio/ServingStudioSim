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
- A cold build sums pipeline/timing lengths across all discovered runs and
  rejects more than 16 MiB with HTTP 413 and stable code
  `catalog_state_too_large` before body reads begin.
- Catalog bytes are published only when before/after metadata stamps agree.
  Three consecutive changes fail closed with HTTP 409 and stable code
  `artifact_generation_changed`.

## Validation

- `cargo test -p analyzer --offline`: 91 passed.
- Catalog regressions prove concurrent cold requests perform one lifecycle body
  build, cached `200` and conditional `304` requests perform no lifecycle body
  reads, marker/timing mutation invalidates the cached ETag and lifecycle, and
  aggregate oversize is rejected before reading the oversized input.
- A deterministic churn regression proves exactly three attempts precede the
  stable `artifact_generation_changed` response.
- Scoped `rustfmt` and `cargo check -p analyzer --offline` are run on the final
  change set; `git diff --check` reports no whitespace errors.
