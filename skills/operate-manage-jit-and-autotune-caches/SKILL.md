---
name: operate-manage-jit-and-autotune-caches
description: >-
  Use before restarting a serving engine or submitting an L1 profiling run, or
  when startup or profiling spends minutes compiling and tuning work an earlier
  process already did. Covers which caches persist, where they live, how to reuse
  them across restarts and worker processes, and when to invalidate. Not for the
  simulator's own kernel cost cache (that is impl-validate-kernel-cache).
---

# Operate Manage JIT And Autotune Caches

Two jobs in this repo pay the same tax over and over: a serving engine that
recompiles and re-tunes on every restart, and a profiling submission that
re-derives tactics in every worker process. Both are avoidable, and the cost is
measured rather than guessed — profiling eight `nvfp4_fused_moe` shapes once
spent **166 s tuning and 5.7 s measuring**, so 97% of that submission re-derived
tactics an earlier process already had.

This skill is about getting that time back. Reuse the work; do not rebuild it.

## Know which cache you are talking about

They persist differently, so do not treat them as one thing.

| What | Built when | Persists via | Survives restart? |
|------|-----------|--------------|-------------------|
| FlashInfer JIT `.so` | first use of a module | prebuilt `flashinfer-jit-cache` wheel, else `~/.cache/flashinfer` | yes, if the wheel is pinned |
| FlashInfer autotune tactics | first call per **process** | `profiling/runners/autotune_cache.py` | only where a runner wraps it |
| Triton compiled binaries | first launch per shape key | `TRITON_CACHE_DIR` | yes |
| `triton.autotune` **selections** | first launch per **process** | nothing | **no** |
| torch.compile / inductor | first compiled graph | `TORCHINDUCTOR_CACHE_DIR`, vLLM's `VLLM_CACHE_ROOT` | yes |
| deep_gemm JIT | `just sync` on a cold uv cache | uv caches the built wheel by commit | yes |
| CUDA graph capture | every engine start | nothing | no, always paid |

The trap is the two `autotune` rows. `TRITON_CACHE_DIR` persists *compilation*,
not *which config was selected* — `triton.autotune` re-benchmarks its config list
in every new process, and there is no persistent equivalent. Setting
`TRITON_CACHE_DIR` does not buy that search back.

## Restarting a serving engine

Make compilation a setup failure, not a startup cost. The pinned SGLang profile
in `alignment/profiler/README.md` already does this: it declares the prebuilt
FlashInfer jit-cache wheel and sets `FLASHINFER_DISABLE_JIT: "1"`, so a missing
module fails **before model weights load** instead of compiling while the server
comes up. Copy that shape for any engine you restart repeatedly.

- Point `VLLM_CACHE_ROOT` / `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` at a
  path that outlives the run and is **not inside a worktree** — a worktree-local
  cache re-warms per tree and dies with it.
- Autotune is not covered by the jit-cache wheel. An NVFP4 MoE engine still runs
  its tactic search after JIT finishes; budget for it or persist it separately.
- CUDA-graph capture is paid every start. If startup is still slow once the
  caches are warm, capture is the remaining cost, not the caches.

## Submitting a profiling run

`profiling/runners/autotune_cache.py` persists FlashInfer tactics across worker
processes. Wrap the tuned region:

```python
from profiling.runners.autotune_cache import autotune_cached
from flashinfer.autotuner import autotune

with autotune_cached(autotune, f"<kernel>.<variant>"):
    ...
```

What it does, and the parts that bite:

- **Path.** `$VIBESIM_AUTOTUNE_CACHE_DIR`, else `~/.cache/vibesim-autotune`.
  (The `VIBESIM_` prefix predates the rename and is still the live variable.)
- **Keyed by kernel + device name + FlashInfer version**, one file each. Do not
  force two card types onto one file: FlashInfer rejects a mismatched stamp by
  ignoring the whole file *and declining to save*, so a shared path tunes forever
  and caches nothing.
- **Fails open, silently.** An unset-but-blank variable, an unwritable directory,
  or a FlashInfer whose `autotune()` has no `cache=` parameter all disable
  persistence and the run still succeeds. So **check the file appeared** rather
  than assuming the tax is gone.
- **Container workers already have it.** `HOME=/cache/home` is bind-mounted from
  the host cache dir; nothing extra to mount.
- Right now only `profiling/runners/moe/nvfp4_fused_moe.py` wraps it. Any other
  FlashInfer-autotuned runner is still paying per process — wrapping one is a
  small change with a large return.

## When you must invalidate by hand

The file key covers GPU model and FlashInfer version, so those are handled.
Delete the cache yourself when:

- the kernel's source, args meaning, or dtype/layout changed — the key does not
  see this, and a stale tactic is silently kept;
- you are deliberately reproducing a cold-start cost and need the tuning time
  back in the measurement.

Deleting is just removing the file under the cache dir. There is no invalidation
command; the next run re-tunes and re-saves.

Do not clear caches "to be safe" before a run. That converts the thing this
skill exists to avoid back into the default.
