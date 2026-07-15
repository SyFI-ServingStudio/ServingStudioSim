# Analyzer discovery scan bounds

## Contract checked

- `doc/analyzer.md` and `analyzer/README.md` require a 30-second catalog cache,
  one single-flight refresh, and blocking-worker filesystem scans.
- `opaque_run_id` is the source of truth for identifiers: `r_` followed by the
  64 lowercase hexadecimal digits of a SHA-256 digest.

## Root cause

`resolve_run` searched the cached vector first, then forced a refresh whenever
an id was absent. Distinct unknown ids could therefore each trigger a recursive
logs-root scan during the 30-second TTL. The force path also meant run lookup
and catalog lookup followed different cache rules.

## Change

- Removed the force-refresh argument and path.
- Run resolution now validates the exact opaque-id format before consulting
  discovery, then uses the same cached discovery function as the catalog.
- A fresh empty/missing result is the bounded negative result for every id until
  the catalog expires. Only an uninitialized or expired catalog can scan.
- Added a test-only per-service scan counter at the actual blocking-scan launch
  boundary; production state and behavior gain no counter.

## Evidence

- A malformed id returns `run_not_found` with zero scans.
- Concurrent cold-cache requests for three distinct valid unknown ids collapse
  to one scan; a fourth distinct unknown id within the TTL does not rescan.
- A run created during a fresh TTL remains absent without rescanning; after the
  cached timestamp is expired, lookup scans once and discovers it.
- `rustfmt --edition 2021 --config skip_children=true` was scoped to the three
  changed Rust files.
- `cargo test -p analyzer`: 68 passed.
- `git diff --check`: passed.
