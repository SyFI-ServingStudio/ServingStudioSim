# Analyzer artifact resource bounds

## Contract checked

- `doc/analyzer.md` defines revision-scoped artifact reads and requires report
  and payload readiness to describe one valid pair from one analysis generation.
- UI handlers run in Tokio's async runtime, so filesystem access and JSON parsing
  must not block executor workers.
- Conditional requests may skip body work only when the validator is derived
  from the exact securely opened file that would otherwise be served.

## Root cause

Artifact handlers performed synchronous filesystem reads and JSON parsing on
async executor threads. A subject request reconstructed readiness by reading and
parsing both members of the report/payload pair, even though it returned only
one member. Each JSON file was allowed to reach 128 MiB, and concurrent requests
had no service-wide admission bound. Conditional requests also reached full body
read/hash work before deciding that a metadata-stable response was unchanged.

## Change

- Filesystem and JSON work is dispatched through bounded blocking workers. At
  most four expensive artifact reads may be in flight; saturation fails fast
  with HTTP 503, stable code `artifact_read_busy`, and `Retry-After: 1`.
- Descriptor construction validates each report/payload pair once and caches a
  revision-bound readiness proof behind an opened-file metadata fence. A subject
  request then reads and parses only the requested member while rechecking both
  members before and after the read.
- Descriptor cold builds are single-flight, preventing duplicate full legacy
  revision and readiness computation within one process.
- JSON artifacts are capped at 16 MiB before allocation. Trace streaming retains
  its separate 512 MiB bound.
- JSON conditional GETs use an ETag derived from the securely opened file's
  device, inode, size, and nanosecond mtime; a matching validator returns 304
  before body read or JSON parsing.
- Legacy subject and trace reads reuse the cached content-derived revision while
  preserving strong metadata and generation fences. Versioned reads retain the
  revision/seqlock correctness checks.

## Evidence

- Tests prove artifact I/O leaves the async runtime worker, saturation returns
  the stable 503 response, and the JSON cap is applied before allocation.
- Tests prove report requests do not read or parse payload bodies, while a
  changed counterpart invalidates the cached readiness proof and fails closed.
- Tests prove a matching metadata ETag returns 304 without reading invalid JSON.
- Tests prove legacy subject and trace requests reuse the cached content-derived
  revision without rehashing unrelated bodies.
- Generation cutover, stale revision, and opened-trace-fd consistency tests from
  the revision binding change continue to pass.
- `cargo test -p analyzer`: 83 passed.

## Residual risk

- The first descriptor request for a legacy run must still content-hash all
  subject artifacts once because legacy revision identity is content-derived.
  That cold work is globally bounded and single-flight, then reused in process.
- Proof and descriptor caches are in-memory only, so a process restart repeats
  cold validation.
- Trace responses are bounded and streamed from an opened descriptor, but a
  client may still consume up to the documented 512 MiB trace limit.

## Review

主代理 cherry-pick 后重跑了 analyzer 与 simulator 测试。独立审查确认核心
generation 隔离正确，同时发现 trace permit 在准备完成后过早释放、wildcard
ETag 可跳过 JSON 校验，以及两处 readiness 状态不一致；这些已在 `b1bc2bb`
修复并由流 body-drop、wildcard malformed JSON、404/409 与 pending-simulation
回归测试覆盖。Catalog 的累计预算与缓存作为后续独立补丁处理。

## Feedback

资源上限和 opened-file fence 的实现扎实，但 `mod.rs`、`lifecycle.rs` 已超过
适合继续堆叠职责的体量。下一轮维护优先提取 `readiness.rs`，并让 stream
许可生命周期成为所有大对象响应的统一 helper 契约。
