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

- `cargo test -p analyzer --offline`: 102 passed on the integrated main tree.
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

## Review

独立审查首先发现预检后重新按路径打开文件会破坏累计预算，以及 metadata
stamp 不能作为响应 bytes 的强 ETag；修复后又发现 per-file 上限先于 aggregate
预算执行，会让一个坏 run 毒死整个 catalog。这三项均已修复并由 atomic
replacement、response-body digest、single timing 16 MiB+1 与 pipeline 1 MiB+1
回归锁定。最终复审确认 exact-FD fence、row-local failed 投影、catalog 413、
singleflight、run cap 与三次 churn 后 409 均无 blocker。

## Feedback

当前冷建最多可暂存约两倍 run-count 的 lifecycle 文件描述符，且 singleflight
锁位于 artifact permit 之后；1,024-run 上限使其在当前部署的 65,535 soft FD
limit 内可控，但部署文档应保留 FD 预算要求。若规模继续增长，应优先降低或
动态计算 run cap，并把 permit 获取移动到 singleflight leader 路径，避免等待者
占用读取许可。
