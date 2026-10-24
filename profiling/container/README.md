# Profiler images

Each profile env (`profiling/exec/env.py`) runs its worker chunks in one image,
the dependency boundary for its kernels. `images.py` assigns the images and
names their tags:

| Image | Profile envs | Holds |
| --- | --- | --- |
| `default` | `default_env`, `flashinfer_pip_env`, `causal_conv1d_env` | the project's `profiling` group from `uv.lock` (Torch 2.10 cu128, FlashInfer 0.6, Triton, NVSHMEM, DeepGEMM), plus the causal-conv1d wheel |
| `vllm` | `vllm_env` | CUDA 13.0, the alignment vLLM checkout at its gitlink on the precompiled wheel of its upstream base (with the FlashKDA build it vendors as `_flashkda_C`), every FlashInfer cubin |
| `sglang` | `sglang_env` | CUDA 13.0, the alignment SGLang checkout at its gitlink with the `sglang-kernel` and FlashInfer 0.6.15.post1 builds it pins |
| `flashinfer-kda` | `flashinfer_kda_env` | FlashInfer's KDA nightly, Torch 2.11 cu130, TVM/TIRx (`requirements/flashinfer_kda.txt`) |

A Neuron env runs a vendor image instead (`VENDOR_IMAGES`): an AWS release named
by its content digest, which is never built or pulled here (`profiling/README.md`,
"Neuron execution").

```bash
just profile-container-build              # every image
just profile-container-build vllm sglang  # some of them
uv run python -m profiling.container.images list           # tag and envs of each
uv run python -m profiling.container.images env vllm_env   # the tag one env uses
```

A tag is a hash of the Dockerfile, the files it copies into its dependency
layers, and the gitlink of each alignment checkout it installs. A checkout with
the same inputs names the same image, and a host that has the tag has the
image. The build refuses an alignment checkout that is not at its gitlink or has
uncommitted changes, so the tag never names something else. The commit that
built the image is only a label, written last, so a new commit reuses every
dependency layer. Images are never pulled: a registry copy under the same tag
could be any build. `VIBESIM_PROFILE_IMAGES` (a JSON object, env name -> tag)
overrides the computed tags; `--remote` uses it because a shipped tree has no
git to compute them from.

Workers overlay the checkout's `profiling/`, `launcher/` and `gpu/` read-only,
so a code change never rebuilds an image; the copy baked in at build time only
serves the image's self-check. Workers run with `--network none` (the vLLM
image therefore downloads every FlashInfer cubin at build time) and get a
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

The vLLM image builds the fork on the precompiled wheel of the upstream commit
it is rebased on (the parent of its first `feat(alignment):` commit,
`alignment/profiler/README.md`), so the fork's Python changes need no CUDA
build; `images.py` refuses a fork that changes native code. The SGLang image
installs the checkout editable against the cu130 Torch index.

Every image also writes `/etc/vibesim-profiler-image.json`: its input hash
(the tag's suffix), the envs it serves, its interpreter and import path. A
process that itself runs in an image (a pod started from it) sets
`VIBESIM_PROFILE_RUNTIME=in-image`; the envs in that manifest then run on the
image's interpreter, and any other env, or an image whose inputs differ from
the tag this checkout computes, is refused (launcher/README.md, "Remote
execution").

The container does not virtualize the GPU driver. After an image or host-driver
change, validate the target GPU, the driver, and a real production kernel
against the rows already in the DB.
