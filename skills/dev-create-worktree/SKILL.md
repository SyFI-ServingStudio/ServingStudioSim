---
name: dev-create-worktree
description: Use when the user asks to create, adopt, or set up a git worktree for VibeSim development, including converting an existing agent checkout into a wt-topic tree. Creates a sibling worktree from current main, provisions required working-copy artifacts, and rebinds checkout-local profiler environments so editable packages and CUDA extensions never point at another checkout. NOT for the parallel-writing-subagent isolation flow (that is dev-orchestrate-parallel-subagents).
---

# Create an VibeSim worktree

Spin up an isolated git worktree for a piece of VibeSim work AND stage the test
inputs that a bare `git worktree add` leaves behind, so the new tree can run
`uv run python -m launcher …` and `just test-*` without first re-profiling
kernels or hunting for trace files.

## Why this skill exists

`git worktree add` only materializes **committed** files. Three things a run/test
needs are therefore missing or unsafe in a fresh or adopted worktree:

1. **`profiling/profile.db`** — tracked, but the primary tree's *working copy* is
   almost always richer than the committed snapshot (every GPU run JIT-fills new
   kernel rows into it without committing). The worktree would get the stale
   committed `profile.db` and re-profile on the next GPU run — slow and needs a
   GPU. Copy the working copy over.
2. **Untracked traces** — large workload CSVs are deliberately **not**
   committed, so they never reach a new worktree: `trace/aime_long.csv` (~5 MB)
   and the full-dataset TraceLab traces `trace/tracelab_reported.csv` /
   `trace/tracelab_preserving.csv` (~23 MB each). Presets that reference them
   (`presets/*_aime*.yaml`) fail without them. Copy them over. Their committed
   `*.manifest.json` sidecars say which policy produced each file, so a worktree
   missing the CSV still records what it is supposed to contain.
3. **Checkout-bound profiler environments** — an editable install records the
   absolute source checkout, while vLLM's precompiled CUDA extensions are
   materialized as untracked `.so` files beside that source. Copying or moving
   `.venv` can therefore leave the interpreter in `wt-topic` importing Python
   from `main/`, or leave `wt-topic` source without `_C`, `_moe_C`,
   `_flashmla_C`, and the FlashAttention extensions. Rebind the environment to
   the new checkout before any alignment run.

The top-level `.venv/`, `target/`, and `__pycache__` are git-ignored and rebuilt
on demand. Do **not** copy them: a copied environment may retain absolute
editable-install paths, and a copied `target/` may carry stale incremental
state. The nested vLLM profiler environment is not created by an ordinary
top-level `uv run`; provision it explicitly in step 4 when the worktree will
run alignment.

## Convention (see memory `vibesim-worktree-convention`)

Worktrees are **siblings of `main/`**, named `wt-<topic>/` — never nested inside
`main/`. The workspace root holds `main/` and every `wt-*/` next to it:

```
/m-coriander/coriander/kanzhu/MLSim_workspace/
├── main/          ← primary tree (source of the working profile.db + traces)
├── wt-<topic>/    ← what this skill creates
└── …
```

Set these once:

```bash
WORKSPACE_ROOT=/m-coriander/coriander/kanzhu/MLSim_workspace
MAIN_WORKTREE=$WORKSPACE_ROOT/main
WORKTREE_TOPIC=<topic>              # short kebab, e.g. kv-cache-logging
NEW_WORKTREE=$WORKSPACE_ROOT/wt-$WORKTREE_TOPIC
BRANCH_NAME=$WORKTREE_TOPIC         # or a name the user gave
```

## Steps

### 1. Create the worktree off the current `main/` HEAD

Branch from whatever `main/` currently has checked out (the code you explored),
NOT from `master`/`origin` — the active mainline branch here is usually an
`afd-*` / feature branch, and its committed line numbers are what any plan was
written against.

```bash
BASE_BRANCH=$(git -C "$MAIN_WORKTREE" rev-parse --abbrev-ref HEAD)
git -C "$MAIN_WORKTREE" worktree add -b "$BRANCH_NAME" "$NEW_WORKTREE" "$BASE_BRANCH"
```

If `$BRANCH_NAME` already exists, drop `-b` and pass the branch as the last arg
instead. Confirm with `git -C "$MAIN_WORKTREE" worktree list`.

### 2. Provision the working-copy `profile.db` (rich, but git-invisible)

Copy the primary tree's working `profile.db`, then mark it `skip-worktree` in
the new worktree so it stays present for runs but never shows as modified and is
never swept into a `git commit -am`:

```bash
cp "$MAIN_WORKTREE/profiling/profile.db" "$NEW_WORKTREE/profiling/profile.db"
git -C "$NEW_WORKTREE" update-index --skip-worktree profiling/profile.db
```

(`skip-worktree` is the key: `profile.db` is a tracked binary, so without this
the copy would leave the worktree permanently dirty and risk an accidental
commit of the primary tree's kernel cache.)

### 3. Provision untracked traces

`rsync` the whole `trace/` dir — tracked files already match, so this only adds
the untracked ones (`aime_long.csv`, the `tracelab_*.csv` pair, any others).
Untracked files are not swept by `git commit -am`, so no extra guarding is
needed:

```bash
rsync -a "$MAIN_WORKTREE/trace/" "$NEW_WORKTREE/trace/"
```

(For an unusually large trace set you may symlink instead — `ln -s
"$MAIN_WORKTREE/trace/<file>" "$NEW_WORKTREE/trace/<file>"` — but copy is the robust default and
keeps the worktree self-contained.)

### 4. Rebind the vLLM profiler environment when alignment is in scope

Run this step when the worktree will profile vLLM, or when adopting/rebasing a
checkout that already contains `alignment/profiler/vllm/.venv`. Do not accept a
working interpreter or `import vllm` alone as proof: both can silently resolve
an editable install from `main/`.

Always create a fresh environment, then install the current nested checkout over
the wheel for the fork's **last upstream commit before the alignment patches**.
Installing over an adopted environment is not sufficient: package installers do
not prune dependencies that belonged to a previous nightly. Reversibly move an
existing `.venv` under `$TMPDIR` before recreating it; do not delete it.
The nested checkout is intentionally detached at the parent repo's gitlink, so
vLLM's default `git branch --show-current` lookup cannot identify that commit:
it silently falls back to the latest nightly and can inject binary extensions
from a different vLLM/CUDA version. Do not use `git merge-base HEAD
origin/main` either; this long-lived fork's `origin/main` can lag behind the
upstream commits already present on `moesim-profile`.

The current fork names its checkout-local patches `feat(alignment): ...` and
keeps them Python-only. Resolve the first such commit and use its parent as the
precompiled-wheel commit. Stop if the marker is absent or any native source has
changed since that base; that checkout needs the from-source fallback documented
in `alignment/profiler/README.md`. Never copy `.so` files from `main/`, symlink
the source tree, or add another checkout to `PYTHONPATH`:

```bash
cd "$NEW_WORKTREE/alignment/profiler/vllm"
if [ -e .venv ]; then
  ENVIRONMENT_ARCHIVE=$(mktemp -d \
    "$TMPDIR/vllm-venv-$WORKTREE_TOPIC-archive-XXXXXXXX")
  rmdir "$ENVIRONMENT_ARCHIVE"
  mv .venv "$ENVIRONMENT_ARCHIVE"
  echo "archived old vLLM environment at $ENVIRONMENT_ARCHIVE"
fi
uv venv --python 3.12 .venv

FIRST_ALIGNMENT_COMMIT=$(git log --reverse --format=%H --fixed-strings \
  --grep='feat(alignment):' HEAD | sed -n '1p')
if [ -z "$FIRST_ALIGNMENT_COMMIT" ]; then
  echo "cannot resolve the first alignment commit" >&2
  exit 1
fi
PRECOMPILED_BASE_COMMIT=$(git rev-parse "$FIRST_ALIGNMENT_COMMIT^")
if git diff --name-only "$PRECOMPILED_BASE_COMMIT"..HEAD | \
  rg '\.(c|cc|cpp|cu|cuh|h|hpp|rs)$'; then
  echo "alignment fork has native changes; build vLLM from source" >&2
  exit 1
fi

VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT="$PRECOMPILED_BASE_COMMIT" \
  VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  UV_CACHE_DIR="$TMPDIR/uv-cache-vllm-$WORKTREE_TOPIC" \
  uv pip install --python .venv/bin/python -e .
UV_CACHE_DIR="$TMPDIR/uv-cache-vllm-$WORKTREE_TOPIC" \
  uv pip install --python .venv/bin/python nvtx
```

The hosted CUDA-12 wheel variant is named `cu129`. That name describes the
precompiled extension build; it does **not** describe every runtime selected by
the current Python source. Recent vLLM dependency graphs can intentionally pair
Torch cu12 with CUDA-13 CUTLASS DSL/JIT packages. A successful import only proves
that the dynamic linker found the extensions. Qualify the highest CUDA/PTX
generation exercised by the actual model path. Before accepting the worktree for
GPU alignment, either:

- use a host driver that supports the wheel's CUDA toolchain;
- point the profile's top-level `driver_compat_lib_dir` at an unpacked matching
  NVIDIA `cuda-compat` directory containing `libcuda.so.1`; or
- build the fork from source with the host's CUDA toolkit.

Keep forward compatibility scoped to the profiled server through
`driver_compat_lib_dir`; do not modify the global `LD_LIBRARY_PATH`. The
precompiled install may need network access. If installation or the GPU probe
cannot complete, report the worktree as not ready for alignment; do not fall
back to a different checkout's binaries.

First verify both Python and every required binary extension resolve inside the
new nested checkout. Also inspect the installed CUDA package families and run
the package consistency check:

```bash
PYTHONPATH="$NEW_WORKTREE/alignment/profiler/vllm" \
  UV_CACHE_DIR="$TMPDIR/uv-cache-vllm-$WORKTREE_TOPIC" \
  uv run --no-project \
    --python "$NEW_WORKTREE/alignment/profiler/vllm/.venv/bin/python" \
    python - <<'PY'
import importlib
from pathlib import Path

import vllm

expected_checkout = Path.cwd().resolve()
module_names = (
    "vllm._C",
    "vllm._moe_C",
    "vllm._flashmla_C",
    "vllm.vllm_flash_attn._vllm_fa2_C",
    "vllm.vllm_flash_attn._vllm_fa3_C",
)
loaded_modules = (vllm, *(importlib.import_module(name) for name in module_names))
for loaded_module in loaded_modules:
    module_path = Path(loaded_module.__file__).resolve()
    if not module_path.is_relative_to(expected_checkout):
        raise RuntimeError(f"module escaped worktree: {module_path}")
print(vllm.__version__, vllm.__file__)
PY

uv pip tree --python .venv/bin/python | \
  rg 'torch v|nvidia-cutlass-dsl|nvidia-cuda-runtime'
uv pip check --python .venv/bin/python
```

CUDA-12 and CUDA-13 families in one resolved tree are not by themselves proof
of stale state: confirm which package requires each family and which kernel path
uses it. Stop and recreate the environment if `uv pip check` fails, the same
distribution appears at multiple versions, or a package is not reachable from
the current resolution. Do not repair that state by uninstalling individual
namespace packages; partial removal is not a reliable clean state.

Then run a fresh-process GPU probe under the same driver environment the
profile will use. If `driver_compat_lib_dir` is set, prepend that directory only
for this probe. The Marlin repack call is intentional: a plain tensor allocation
can pass while a cu129 extension later fails with
`cudaErrorUnsupportedPtxVersion` during model loading.

```bash
cd "$NEW_WORKTREE/alignment/profiler/vllm"
DRIVER_COMPAT_LIB_DIR=<empty-or-the-profile-driver_compat_lib_dir>
LD_LIBRARY_PATH="${DRIVER_COMPAT_LIB_DIR:+$DRIVER_COMPAT_LIB_DIR:}${LD_LIBRARY_PATH:-}" \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$NEW_WORKTREE/alignment/profiler/vllm" \
UV_CACHE_DIR="$TMPDIR/uv-cache-vllm-$WORKTREE_TOPIC" \
  uv run --no-project --python .venv/bin/python python - <<'PY'
from pathlib import Path

import torch
from vllm import _custom_ops as ops

size_k = 128
size_n = 128
packed_weight = torch.randint(
    0, 256, (size_n, size_k // 2), dtype=torch.uint8, device="cuda"
)
repacked_weight = ops.gptq_marlin_repack(
    b_q_weight=packed_weight.view(torch.int32).T.contiguous(),
    perm=torch.empty(0, dtype=torch.int, device="cuda"),
    size_k=size_k,
    size_n=size_n,
    num_bits=4,
    is_a_8bit=False,
)
torch.cuda.synchronize()
loaded_cuda_driver = next(
    line.split()[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "/libcuda.so" in line
)
print(torch.cuda.get_device_name(), repacked_weight.shape, loaded_cuda_driver)
PY
```

The printed `libcuda` path must match the intended host or compatibility
driver. This Marlin probe is a lower bound, not full model qualification. If the
dependency inventory contains a newer CUTLASS DSL/JIT stack, also execute the
smallest production kernel on that path (for DeepSeek-V4, its fused indexer) or
run a bounded server warmup before spending time on a profile. Treat an
import-only check, a basic allocation-only check, a Marlin-only check when the
model also uses a newer JIT path, or a probe that loaded another `libcuda.so.1`
as incomplete.

### 5. Integrate current `main/` without importing unrelated WIP

When an existing worktree needs a coherent implementation that currently lives
in `main/`, verify the feature closure there first: producer, persisted/service
state, route/registry, consumer, and behavior tests. Make one scoped commit in
`main/` containing only that closure, then checkpoint the target worktree by its
own responsibilities and rebase it onto the commit. Do not copy selected files
or use a blanket stash/add to imitate a rebase.

Before and after rebasing, inspect `git status --short`, staged and unstaged
diffs, nested repository/submodule gitlinks, and
`git ls-files -v profiling/profile.db`. Preserve unrelated WIP and remember that
an `S` profile DB can differ while status stays clean. If the closure cannot be
committed without absorbing unrelated changes, stop and ask rather than
silently broadening the commit.

### 6. Report + first-run note

Tell the user the worktree path, its branch, and that the **first** `uv run` /
`just test-*` inside it will `uv sync` + build the release binary (a few minutes,
one-time). All later commands are fast. Every command must run **from inside
`$NEW_WORKTREE`** and under `uv` (see `CLAUDE.md` env rules). If step 4 applied,
also report the exact vLLM version and the checkout path printed by its probe.

## Verify (optional but recommended)

Cheap CPU check that the tree is wired up:

```bash
cd "$NEW_WORKTREE" && just test-cpu          # Rust --lib + mocked pytest, no GPU
```

Or a dry-run of an aime preset to confirm the trace resolved:

```bash
cd "$NEW_WORKTREE" && uv run python -m launcher presets/unified_aime.yaml --dry-run
```

If the dry-run errors on a missing `trace/aime_long.csv`, step 3 did not land.

## Relationship to other skills

- Once the worktree is ready, use `operate-run-simulation` to launch sims and
  `dev-run-tests` for the test tiers — both assume the artifacts this skill staged.
- For isolating **multiple concurrent writing subagents**, use
  `dev-orchestrate-parallel-subagents` instead; this skill is for a single
  developer/agent worktree.
