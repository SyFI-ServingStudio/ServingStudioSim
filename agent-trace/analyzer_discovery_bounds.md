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

## Review

主代理复核了 opaque-id 负缓存与 TTL singleflight 的边界，确认未知 ID 不再
触发按请求重扫，也没有改变合法 run 的发现顺序。补丁已以 `a65c9aa` 合入；
后续完整 analyzer 测试继续通过。

## Feedback

实现和回归测试都聚焦且易审。后续若 discovery 规模继续增长，建议把扫描
计数以只读 telemetry 暴露，而不是依赖测试专用 counter；本次不需要扩大范围。
