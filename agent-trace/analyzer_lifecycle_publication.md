# Analyzer 生命周期持久化修复记录

## 文档依据

- `doc/analyzer.md` 的 “Launcher integration and publication lifecycle” 规定：launcher 在计算、渲染、trace 的有序 generation 中原子替换 pipeline sidecar；计算失败必须令 pipeline 失败；渲染或 trace 失败仍可令 orchestration 完成；launcher 崩溃时允许留下合法的 `pending` generation。
- 同一节要求 reader 只依据有界 sidecar 判断当前 generation，不能从零散 artifact 或 stdout 推断状态。因此每一次已落盘 sidecar 都必须是 reader 可接受的完整状态，而不能依赖紧随其后的第二次写入来补全状态。
- Rust 的 `analyzer/rust/src/ui_service/lifecycle.rs::validate_pipeline_state` 是 v1 sidecar 的实际读取契约。

## 根因

`AnalyzerPipelinePublisher` 原先暴露通用的 `complete_stage`、`start_stage`、`fail_stage` 和 `finish`。launcher 在 stage 边界通过多个独立持久化写入拼装状态，产生两类 Rust validator 明确拒绝的快照：

1. compute 成功后先发布 `pipeline=pending, compute=complete, render=not_started`，再单独把 render 改为 `pending`；
2. compute 失败后先发布 `pipeline=pending, compute=failed`，再单独把 pipeline 改为 `failed`。

虽然第二次写入通常很快到达，但 reader、崩溃或写失败都可能观察到第一份非法持久化状态。

## 改动

- 用语义化的阶段边界 API 取代通用 stage setter：
  - `complete_compute_and_start_render`
  - `fail_compute_and_finish`
  - `complete_render_and_start_trace` / `fail_render_and_start_trace`
  - `complete_trace_and_finish` / `fail_trace_and_finish`
- 每个方法先验证当前状态形状，在内存中完成整个边界转换，然后只调用一次 durable atomic replace。
- launcher 的成功、render 失败、trace 失败、compute 失败、缺失 binary 和 subprocess 异常路径均改用上述 API。
- 新增共享契约表 `tests/fixtures/analyzer_pipeline_lifecycle_v1.json`：
  - Rust 测试逐行交给真实 `validate_pipeline_state`，确认合法表全部接受、两个回归快照全部拒绝；
  - Python 测试在 `atomic_write_json` 成功后记录每一次真实 pipeline 发布，确认其状态形状均属于同一张 Rust 已验证的表。

## 验证

- `uv run ruff check launcher/analyzer_pipeline.py launcher/exec.py tests/test_analyzer_pipeline.py`
- `uv run ruff format --check launcher/analyzer_pipeline.py launcher/exec.py tests/test_analyzer_pipeline.py`
- `uv run pytest -q tests/test_analyzer_pipeline.py`：10 passed
- `rustfmt --edition 2021 --check analyzer/rust/src/ui_service/tests.rs`
- `cargo test -p analyzer`：65 passed
- `git diff --check`

## 残余风险

- 共享表只刻画 lifecycle status 组合；producer identity、subject token、trace artifact path 等其他字段仍由 Rust validator 单独校验。现有 launcher 继续通过既有受控生成路径提供这些字段，本修复没有改变它们。
- artifact 已产生但 sidecar 边界写入前进程崩溃时，读者仍会看到上一份合法 `pending` 状态。这是文档明确允许的 crash 语义，而不是非法中间状态。

## Review

主代理逐条对照 Rust validator 与共享 fixture，确认每次原子发布都是完整合法
状态，而不是依赖下一次写入修复中间态。补丁已以 `db08d11` 合入，并在后续
Rust/Python 联合测试中保持通过。

## Feedback

用语义化 transition API 代替通用 setter 是本次最有价值的可维护性改进。
未来扩展 stage 时应继续先扩共享状态表，再同步 producer 与 reader validator。
