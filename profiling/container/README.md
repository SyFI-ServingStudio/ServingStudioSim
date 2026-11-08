# Profiler images

Each profile env (`profiling/exec/env.py`) runs its worker chunks in one image,
the dependency boundary for its kernels. `images.py` assigns the images and
names their tags:

| Image | Profile envs | Holds |
| --- | --- | --- |
| `default` | `default_env`, `flashinfer_pip_env`, `causal_conv1d_env` | the project's `profiling` group from `uv.lock` (Torch 2.10 cu128, FlashInfer 0.6, Triton, NVSHMEM, DeepGEMM), plus the causal-conv1d wheel |
| `vllm` | `vllm_env` | the vLLM reference engine (`reference_engines/vllm`): vLLM's official image pinned by digest, with the patches applied to its installed vLLM. Also runs vLLM alignment captures |
| `sglang` | `sglang_env` | the SGLang reference engine (`reference_engines/sglang`): SGLang's official image pinned by digest, with the patches applied to its checkout, plus `nvtx`. Also runs SGLang alignment captures |
| `flashinfer-kda` | `flashinfer_kda_env` | FlashInfer's KDA nightly, Torch 2.11 cu130, TVM/TIRx (`requirements/flashinfer_kda.txt`) |

```bash
just profile-container-build              # every image
just profile-container-build default      # some of them
just ref build-image vllm                 # an engine image (reference_engines/README.md)
uv run python -m profiling.container.images list           # tag and envs of each
uv run python -m profiling.container.images env vllm_env   # the tag one env uses
```

A tag is a hash of the Dockerfile, the files it copies into its layers (an
engine's `BASE` and every patch), the gitlink of each submodule it builds, and
the pinned image it starts `FROM` (an engine's official image at this host's
platform digest). A checkout with the same inputs names the same image, and a
host that has the tag has the image. The build refuses a submodule that is not
at its gitlink or has uncommitted changes, so the tag never names something
else. The commit that
built the image is only a label, written last, so a new commit reuses every
dependency layer. Images are never pulled: a registry copy under the same tag
could be any build. `VIBESIM_PROFILE_IMAGES` (a JSON object, env name -> tag)
overrides the computed tags; `--remote` uses it because a shipped tree has no
git to compute them from.

Workers overlay the checkout's `profiling/`, `launcher/` and `gpu/` read-only,
so a code change never rebuilds an image; the copy baked in at build time only
serves the image's self-check. A worker names its interpreter and module
(`--entrypoint`, `Image.python`), never relying on the image's own entrypoint,
since an engine image runs captures on another interpreter. Workers run with
`--network none` (the engine images take every FlashInfer cubin and its AOT JIT
cache from the official image) and get a
temporary JSON exchange directory and a persistent JIT cache at `/cache`: one
directory per image tag under `VIBESIM_PROFILE_CACHE_DIR` (default
`~/.cache/vibesim-profiler`), since Torch extensions, Triton and FlashInfer
builds link that image's libraries. A use touches its directory, and an
earlier tag's directory of the same image unused for 30 days is removed.
`kernel-profile measure` also mounts its
output directory so the CUPTI/NVML artifacts survive the worker; the images
carry no matplotlib, so its plots are skipped there (`VIBESIM_PROFILE_RUNTIME=host`
renders them). The host keeps GPU
selection, profile DB writes and artifacts. Set `VIBESIM_PROFILE_GPUS` only to
GPUs reserved for the profiling job; otherwise idle-GPU selection applies.

```bash
uv run python -m profiling run nvfp4_quant \
  --backend vllm_cuda \
  --gpu-name "NVIDIA B200" \
  --spec '{"num_tokens":32,"hidden_size":6144,"group_size":16,"input_dtype":"bf16","scale_format":"linear_e4m3"}' \
  --json
```

The engine images (`reference_engines/README.md`) add, over the official
image: Nsight Systems 2025.5.2 (apt, pinned version), gdb and py-spy for hang
stacks, the capture runner's venv (`reference_engines/capture-requirements.txt`),
the req-frontend `session_runner` built at its gitlink on the same userland, the
patches, and this checkout's worker code for the self-check. Inside them,
`VIBESIM_CAPTURE_RUNTIME=in-image` makes the capture runner use the baked
`session_runner` instead of calling cargo. A capture runs `python -m launcher`
on the capture venv (`alignment/image_runtime.py`), with the same overlay as the
workers', applied around submodules.

Every image also writes `/etc/vibesim-profiler-image.json`: its input hash
(the tag's suffix), the envs it serves, its interpreter and import path; an
engine image adds its capture interpreter, Nsight Systems, official base image
and upstream commit, and req-frontend revision. A
process that itself runs in an image (a pod started from it) sets
`VIBESIM_PROFILE_RUNTIME=in-image`; the envs in that manifest then run on the
image's interpreter, and any other env, or an image whose inputs differ from
the tag this checkout computes, is refused (launcher/README.md, "Remote
execution").

The container does not virtualize the GPU driver. After an image or host-driver
change, validate the target GPU, the driver, and a real production kernel
against the rows already in the DB.
