# Reference engines

The serving engines alignment captures measure and whose kernels L1 profiles,
kept as an official base plus our patches. One directory per engine:

| Path | Holds |
| --- | --- |
| `<engine>/BASE` | the upstream repository and commit the patches apply to, the official image built from that commit, and that image's digest per platform |
| `<engine>/patches/` | the instrumentation, one `git format-patch` file per commit: the source of truth |
| `<engine>/Dockerfile` | the engine's one image, for both L1 kernels and alignment captures |
| `<engine>/src/` | a materialized source tree (git-ignored) |
| `capture-requirements.txt` | the capture runner's interpreter in both images |

| Engine | Upstream base | Official image |
| --- | --- | --- |
| `vllm` | main `5629b26` (2026-11-07) | `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:5629b26…` (vLLM's post-merge CI image); arm64: `…:b4dffa50…-arm64` |
| `sglang` | main `9d547e19` (2026-11-08) | `lmsysorg/sglang:nightly-dev-cu13-20261108-9d547e19` |

vLLM's CI publishes no arm64 image of `5629b26`; the arm64 base is
`b4dffa50`, 3 commits earlier, which differs from it only in `recipes/` and
`tests/`, so the same patches apply (BASE: `image.linux/arm64`,
`upstream_commit.linux/arm64`). SGLang's tag is a multi-platform index; BASE
pins both platforms' manifests. The arm64 images are build-verified only.

Both bases are main/nightly builds, not releases: vLLM is on main because
DeepSeek-V4.1 needs SwapAB and other paths v0.33.0 does not have. The next bump
should move both to a stable release and its official image.

## Commands

```bash
just ref materialize vllm                  # BASE + patches -> reference_engines/vllm/src
just ref export-patches vllm               # src (or --from CHECKOUT --ref REF) -> patches/
just ref bump-base vllm --commit SHA --image REPO:TAG
just ref build-image vllm                  # this host's platform
just ref warm-caches vllm PROFILE.yaml     # compile + tune a capture's deployment, record it
```

(`just ref VERB ENGINE` is `uv run python -m reference_engines VERB ENGINE`.)

**materialize** fetches BASE's commit once into a bare object cache
(`~/.cache/vibesim-reference-engines`, or `VIBESIM_REFERENCE_CACHE`), checks it
out, and applies the patches with `git am`. Every commit gets a fixed committer
and its author date as committer date, so a BASE and a patch series give the
same commit ids on every host, on branch `reference/<engine>`. It refuses to
replace an existing tree without `--force`, and a tree with uncommitted changes
even then.

**export-patches** rewrites `patches/` from the commits a checkout has over
BASE's commit. A file depends only on its own commit (no `n/m` numbering in the
subject, full blob ids, no source commit id or signature), so an unchanged
commit gives an unchanged file wherever it is exported from. The checkout is the materialized tree unless `--from` names a
fork. The series must be linear. A fork's own `ALIGNMENT_BASE` is not
exported: BASE replaces it.

Editing the patches: commit in `src/` (amend, rebase, add commits), then
`export-patches`, and review the diff of `patches/`.

**bump-base** points BASE at a new upstream commit and official image. It reads
each platform's image-manifest digest from the registry, so nothing in BASE is a
moving tag; `--platform-image linux/arm64=REPO:TAG` adds a platform published
under another tag, and `--platform-commit linux/arm64=SHA` the upstream commit
that image was built from when it is not `--commit`. It rewrites BASE whole
(comments included) and then materializes on the new base. A patch that no longer
applies stops `git am` in `src/`: resolve it there (`git am --continue`), then
`export-patches`.

**build-image** builds the image for this host's platform
(`profiling/container/images.py`, which names its tag). The tag is a hash of the
Dockerfile, BASE, every patch, the capture requirements, the req-frontend
gitlink and the pinned base image, so a host that has the tag has that image.

**warm-caches** starts the engine of a capture config in its image, on the
config's GPUs, with autotuning allowed; it lets the engine finish JIT, compile,
autotune and CUDA-graph capture, stops it at ready (no requests, no capture;
logs in `<log_dir>.warm-caches/`), and writes the manifest a capture needs
(next section). Run it on the GPU type the captures will use, inside the
GPU job as any capture.

## Autotuning

A measurement runs the tactics a deliberate warm-up chose; it never tunes.
`VIBESIM_AUTOTUNE=allow|forbid` carries the decision, set only by the parent:

| Run | Policy |
| --- | --- |
| capture (`launcher alignment profile --runtime image`) | forbid |
| capture with `--allow-autotune`, or `warm-caches` | allow |
| L1 jit-fill (rows missing from the DB) | allow, and each tune is logged per kernel |
| L1 `--force` / `--fresh` (rows already there) | forbid, unless `--allow-autotune` |

Two layers enforce it. The preflight runs in the parent before any container
starts: a capture needs a warm-caches manifest whose key (engine, base and GPU
architecture, model, GPU count, server config and runtime environment;
`alignment/warm_caches.py`) matches, and an L1 refresh needs each spec's
persisted tactic file (`profiling/runners/tactic_labels.py`); a miss fails at
once and says what to run. The backstop is `autotune_guard.py`, which the
images load at interpreter start (a `.pth`) and L1 workers install: it raises at
every cache-miss tactic measurement (FlashInfer's autotuner, Triton
`@autotune`, inductor max-autotune and its runtime autotuners) and stops the
process; the parent reports it, and keeps no row of that chunk. If the
preflight said warm, the error says "manifest/actual mismatch": the key misses
an input of the tuning and must be completed.

What the preflight cannot see, the guard still stops: a Triton `@autotune`
key never warmed (the engine reaches it only under load), and configs with a
`pre_hook`, which Triton never writes to disk. The KDA rows that pin Triton
selections (`AutotunePin`) re-tune in every worker by design, so they refresh
only with `--allow-autotune`.

## Persistent caches

A container's `/cache` (JIT, compile, autotune caches and the warm-caches
manifests) is `~/.cache/vibesim-profiler/<engine>-<base digest[:12]>-sm<cc>`:
the image's `ai.vibesim.cache-key` label plus the GPUs' architecture
(`profiling/exec/local.py:image_cache_key`). A rebuild on the same official
base keeps its caches; the caches that patches could affect are content-addressed
(the reasoning is beside the key function). Images without the label keep one
directory per tag.

## The image

`FROM` the official image at this platform's digest, then, from what changes
least to most:

1. system tools: Nsight Systems 2025.5.2 (the hosts' version, at
   `/opt/nvidia/nsight-systems/2025.5.2`), gdb, procps;
2. the capture runner's venv (`/opt/vibesim-capture/.venv`) with py-spy;
3. what the engine's interpreter lacks (SGLang: `nvtx`, and FlashInfer's
   `tinygemm2` in the sm100a/sm103a jit-cache providers: captures run with
   `FLASHINFER_DISABLE_JIT=1` and the 0.7.3 wheel does not ship it;
   `complete_flashinfer_aot.py` compiles it without a GPU and registers it in
   each provider's manifest);
4. the autotune guard, in the engine interpreter's site-packages with a `.pth`
   that installs it at interpreter start;
5. the req-frontend `session_runner`, built in a stage on the same base;
6. the patches, applied to the engine as the official image installed it:
   vLLM's wheel in `/usr/local/lib/python3.12/dist-packages` (only `vllm/`
   paths), SGLang's editable checkout at `/sgl-workspace/sglang`. A patch that
   does not apply means the image is not BASE's commit, and the build fails;
7. this checkout's `profiling/`, `launcher/`, `gpu/` and the guard for the build's self-check
   (workers and captures overlay the checkout's own at run time);
8. the manifest (`/etc/vibesim-profiler-image.json`) and labels.

A patch-only or Sim-only change rebuilds only the last layers. FlashInfer's
cubins and AOT JIT cache come with the official images. The vLLM image is vLLM's
CI image, which sets `VLLM_GPU_SYNC_CHECK=error` for its tests; the image drops
it at interpreter start (a `.pth`), since a Dockerfile cannot unset an inherited
variable and a measurement must not run vLLM's sync checker.

The image has no entrypoint. An L1 worker runs `python -m
profiling.exec.local_worker` on the engine's interpreter (`/usr/bin/python3`,
`/opt/sglang/bin/python`); a capture runs `python -m launcher alignment profile`
on the capture interpreter (`alignment/image_runtime.py`).

Pull the official image before a build on a new host; keep it (the vLLM CI
repository deletes old tags). The digest is what the build uses, so a re-pushed
tag cannot change the image.
