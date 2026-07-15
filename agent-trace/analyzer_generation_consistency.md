# Analyzer generation read consistency

## Contract checked

- `doc/analyzer.md` makes `analysis.revision` the stable identity of one
  published artifact generation.
- Ready resources must come only from the current pipeline generation, while
  legacy runs use a content-derived revision.
- Analyzer artifacts are atomically replaced at fixed on-disk names, and trace
  responses must stream the same securely opened descriptor they inspected.

## Root cause

Descriptor links did not carry their claimed revision. A client could retain an
old descriptor while a publisher replaced the fixed report, payload, or trace
path, then fetch new bytes through the old link. Subject readiness also read the
pipeline, timing, report, and payload only once, so a cutover between those reads
could assemble state from two generations. The trace endpoint selected and
opened its file without a final pipeline check. Finally, the descriptor cache
stamp always selected the newest trace even for a versioned pipeline that named
an exact trace artifact.

## Change

- Ready report, payload, and Perfetto hrefs now contain
  `revisions/{analysis.revision}`; stale revision requests fail with HTTP 409 and
  stable code `artifact_generation_changed`.
- A shared bounded seqlock captures parsed pipeline plus timing state, performs
  the whole descriptor/resource read, then captures state again. A cutover
  retries the whole read up to three times and repeated churn fails closed.
- Legacy snapshots use their content-derived revision as the fence; that
  revision also binds the selected newest trace's opened-file identity.
- Trace handling opens and bounds the exact selected file before the final
  snapshot comparison, then streams that same descriptor.
- Versioned descriptor stamps follow only the pipeline-recorded trace path;
  newest-trace selection remains limited to legacy runs.

## Evidence

- Descriptor cutover test discards an old assembled descriptor and returns only
  the new revision and links.
- Subject-pair cutover test discards old report bytes and retries the whole pair.
- A stale revision-linked HTTP request returns 409 rather than current bytes.
- A trace cutover after fd open drops that prepared old fd instead of serving it.
- Repeated cutovers terminate after the bounded attempts with the stable 409.
- An unclaimed newer trace does not perturb a versioned descriptor stamp, while
  changing the pipeline-selected trace does.
- `cargo test -p analyzer`: 70 passed.
- Scoped `rustfmt --edition 2021 --config skip_children=true`: passed.
- `git diff --check`: passed.

## Review

主代理与独立审查均未发现跨 generation 返回错误 report/payload/trace 的
correctness blocker；revision link、pair proof、有限 seqlock 与 opened trace
fd 共同封闭了切代窗口。补丁已以 `d11f221` 合入。

## Feedback

一致性证明覆盖充分。后续应把 descriptor/proof/readiness 从大文件中拆成独立
模块，并用命名结构体替代三元 proof tuple，降低继续扩展协议时的认知负担。
