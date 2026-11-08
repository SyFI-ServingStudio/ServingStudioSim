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
| `vllm` | main `5629b26` (2026-11-07) | `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:5629b26…` (vLLM's post-merge CI image) |
| `sglang` | main `9d547e19` (2026-11-08) | `lmsysorg/sglang:nightly-dev-cu13-20261108-9d547e19` |

Both bases are main/nightly builds, not releases: vLLM is on main because
DeepSeek-V4.1 needs SwapAB and other paths v0.33.0 does not have. The next bump
should move both to a stable release and its official image.

## Commands

```bash
just ref materialize vllm                  # BASE + patches -> reference_engines/vllm/src
just ref export-patches vllm               # src (or --from CHECKOUT --ref REF) -> patches/
just ref bump-base vllm --commit SHA --image REPO:TAG
just ref build-image vllm                  # this host's platform
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
BASE's commit (`--zero-commit`, no signature, so an unchanged commit gives an
unchanged file). The checkout is the materialized tree unless `--from` names a
fork. The series must be linear. A fork's own `ALIGNMENT_BASE` is not
exported: BASE replaces it.

Editing the patches: commit in `src/` (amend, rebase, add commits), then
`export-patches`, and review the diff of `patches/`.

**bump-base** points BASE at a new upstream commit and official image. It reads
each platform's image-manifest digest from the registry, so nothing in BASE is a
moving tag; `--platform-image linux/arm64=REPO:TAG` adds a platform published
under another tag. It then materializes on the new base. A patch that no longer
applies stops `git am` in `src/`: resolve it there (`git am --continue`), then
`export-patches`.

**build-image** builds the image for this host's platform
(`profiling/container/images.py`, which names its tag). The tag is a hash of the
Dockerfile, BASE, every patch, the capture requirements, the req-frontend
gitlink and the pinned base image, so a host that has the tag has that image.

## The image

`FROM` the official image at this platform's digest, then, from what changes
least to most:

1. system tools: Nsight Systems 2025.5.2 (the hosts' version, at
   `/opt/nvidia/nsight-systems/2025.5.2`), gdb, procps;
2. the capture runner's venv (`/opt/vibesim-capture/.venv`) with py-spy;
3. what the engine's interpreter lacks (SGLang: `nvtx`);
4. the req-frontend `session_runner`, built in a stage on the same base;
5. the patches, applied to the engine as the official image installed it:
   vLLM's wheel in `/usr/local/lib/python3.12/dist-packages` (only `vllm/`
   paths), SGLang's editable checkout at `/sgl-workspace/sglang`. A patch that
   does not apply means the image is not BASE's commit, and the build fails;
6. this checkout's `profiling/`, `launcher/`, `gpu/` for the build's self-check
   (workers and captures overlay the checkout's own at run time);
7. the manifest (`/etc/vibesim-profiler-image.json`) and labels.

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
