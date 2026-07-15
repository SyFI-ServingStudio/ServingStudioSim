# Analyzer producer provenance / subject contract 修复记录

## 文档与问题

- `doc/analyzer.md` 要求 successful generation 携带真实 producer
  version/revision/binary digest，并且 `requested_subjects` 只能是 flat
  registry 的 Run-scope token。
- 原 launcher 通过 `analyze --version` 取 version，却用 launcher 当前
  checkout 的 `git rev-parse HEAD` 作为 revision。旧 binary 会因此被错误归属
  到新 HEAD；identity 失败后仍可携带 `unavailable` 进入 compute/complete，
  最终被 reader 永久判 invalid。
- Rust `registry::select` 只打印 unknown token，且把另一 Scope 的 token
  静默过滤，因此 typo / `analyze run ... alignment-e2e` 会以零工作成功。
- launcher 在 `requested_subjects` 中直接记录 preset 字符串，未由实际执行
  binary 的 registry 确认；Cargo target 路径也硬编码为 `./target`。

## 实现

1. `analyzer/rust/build.rs` 在编译时嵌入 full 40-hex source commit，并跟踪
   当前 worktree 的 HEAD 与 symbolic branch ref；`analyze identity` 输出
   schema-v1 machine JSON：producer name、package version、build-time source
   revision，以及同一 executable 内 flat registry 的 `{name, scope}` rows。
2. `registry::select` 现在返回 `Result`：unknown token、跨 Scope token 在
   DataFusion session/输出目录创建前 hard error；合法 subject 仍按 catalog
   顺序执行，deployment applicability 仍是独立的 best-effort gate。
3. launcher 只通过 `snapshot_analyzer_binary` 获得 producer contract。它先
   对 Cargo output 建立 per-process hardlink 来 pin inode，再从 pinned path
   执行 `identity` 并计算 SHA-256；compute 与 trace 都执行同一个 pinned
   path。`(dev, ino, size, mtime_ns)` lock/cache 是 singleflight：同一约
   940 MB debug binary 每进程只 hash/query 一次，不复制文件，进程正常退出
   时清理 hardlink。Cargo atomic rename rebuild 不影响已开始 generation；
   pinned path 异常 mutation/replacement 则 fail closed。
4. Cargo artifact 路径由 `CARGO_TARGET_DIR` 或 `cargo metadata.target_directory`
   统一解析，simulator/analyzer/schema build 与后续 identity/执行不会分叉。
   `cargo_build` 在启动 async sweep 前预热 snapshot singleflight。
5. Run intent 由 binary contract 严格验证并按 registry 顺序 canonicalize；
   identity 或 subject selection 失败会先发布合法 terminal compute failure，
   不会执行 compute，也不会把非法 raw token 写入 `requested_subjects`。
   alignment 的 explicit/default selection 同样来自 binary contract，不再在
   launcher 硬编码三项列表。

## revision / dirty / worktree 语义

- `producer.revision` 明确定义为 **build-time source commit**，不声称 build
  worktree clean。相同 commit 上 dirty build 的精确区别由 executable
  `binary_sha256` 提供。
- launcher 不再读取当前 checkout HEAD；revision 只来自实际执行的 pinned
  binary 的 machine identity。
- build script 的 rerun paths 指向该 worktree 自己的 HEAD/branch ref；Cargo
  的 path package identity 还包含 canonical manifest source path，因此共享
  `CARGO_TARGET_DIR` 的不同 worktree 不复用另一 worktree 的 build-script
  identity output。提交后应重建并核对 `analyze identity.revision == commit`；
  本 worktree 的 dirty-build smoke 显示的是构建时 HEAD，符合上述定义。
- target directory 是 trusted-writer boundary；hardlink pin 防止正常 Cargo
  rename 替换，metadata checks 防止普通异常修改。对能恶意原地改写并恢复
  inode 全部 metadata 的 target writer 不提供安全隔离。

## 验证

- rebase 当前 analyzer 主线后 `cargo test -p analyzer`：89 passed。
- `target/debug/analyze identity`：输出 concrete `0.1.0`、40-hex embedded
  revision 和 13 个 Run/Alignment registry rows。
- direct CLI smoke：unknown `through-put` 与 Run 下的 `alignment-e2e` 均
  exit 1，且未创建 reports/payloads。
- `pytest -q tests/test_analyzer_pipeline.py tests/test_alignment_launcher.py`：
  37 passed。
- 新增回归覆盖：machine identity/digest、hardlink inode pin、Cargo atomic
  replacement、相同 bytes 新 inode 不覆盖既有 snapshot、singleflight cache、
  pinned path mutation fail-closed、identity failure terminal state、unknown /
  cross-scope terminal state、canonical requested tokens、Cargo env/config target、
  alignment registry default 自动包含未来 subject。
- scoped `ruff format/check`、scoped `rustfmt --check`、`git diff --check` 通过。

## 集成注意

- 本分支修改 `doc/analyzer.md`、`analyzer/README.md`、`launcher/exec.py`、
  `launcher/analyzer_pipeline.py`、`tests/test_analyzer_pipeline.py`；这些与
  lifecycle/generation/revision-route 系列提交最可能产生文本冲突。
- 不修改 generation reader、discovery 或 artifact resource bounds；合并时应
  保留主线 revision-scoped route 的 URL/read validation，并保留本提交的
  binary-owned producer/subject preflight。
