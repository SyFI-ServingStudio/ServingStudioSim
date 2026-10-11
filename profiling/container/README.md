# vLLM profiler container

This image is the dependency boundary for ServingStudio Sim kernels registered with
`subprocess_env="vllm_env"`. It pins CUDA 13.0, the instrumented vLLM checkout,
the matching cu130 native wheel, Torch, FlashInfer, and the CUPTI build tools.
Workers run with `--network none`, so the build also downloads every FlashInfer
cubin into `/opt/flashinfer-cubins`.

By default, profiling workers overlay the current worktree's `profiling/`,
`launcher/`, and `gpu/` directories read-only. This development mode lets newly
registered profilers run immediately while keeping vLLM, its virtual
environment, and native extensions frozen inside the image.

For a release measurement whose source must also be frozen, use the source
snapshot baked into a clean image:

```bash
VIBESIM_PROFILE_SOURCE_MODE=image uv run python -m profiling run ...
```

Build from a clean checkout:

```bash
profiling/container/build.sh
```

During implementation only, a dirty image must be requested explicitly:

```bash
VIBESIM_ALLOW_DIRTY_PROFILE_IMAGE=1 profiling/container/build.sh
```

Build the image, then use the existing host CLI. Registry rows selecting
`vllm_env` automatically execute their worker chunk in the container:

```bash
uv run python -m profiling run nvfp4_quant \
  --backend vllm_cuda \
  --gpu-name "NVIDIA B200" \
  --spec '{"num_tokens":32,"hidden_size":6144,"group_size":16,"input_dtype":"bf16","scale_format":"linear_e4m3"}' \
  --json
```

The host remains responsible for GPU selection, profile DB writes, and artifact
creation. Development workers additionally receive the three read-only source
overlays described above. All workers receive a temporary JSON exchange
directory and a persistent JIT cache mounted at `/cache`; the cache-free
`kernel-profile measure` diagnostic additionally mounts its explicit output
directory so the container-owned CUPTI/NVML artifacts survive the worker. Plot
rendering remains optional and never controls measurement success. Set
`VIBESIM_PROFILE_GPUS` only to GPUs reserved for the profiling job; otherwise
the existing idle-GPU selection remains active.

Release measurements require `VIBESIM_PROFILE_SOURCE_MODE=image` and an image
built from a clean tree. Record both the immutable image digest and the labels
reported by:

```bash
docker image inspect vibesim-profiler-vllm:cu130-3f667d7e
```

The container does not virtualize the GPU driver. Validate the target GPU,
driver, and a real production kernel after every image or host-driver change.

## Images for the host envs

`default.Dockerfile` is the project's own `profiling` dependency group (Torch
2.10 cu128, FlashInfer 0.6, Triton, NVSHMEM, DeepGEMM, and the causal-conv1d
wheel), installed from `uv.lock` the way `just sync` installs it. It serves
`default_env`, `flashinfer_pip_env` and `causal_conv1d_env` when
`VIBESIM_PROFILE_RUNTIME=container`:

```bash
uv run python -m profiling.container.images build default   # prints the tag
uv run python -m profiling.container.images env default_env # the tag that env uses
```

The tag is a hash of the Dockerfile and the files it copies before the source,
so a checkout with the same dependencies names the same image and a host that
has the tag has the image. The commit is only a label, written last so a new
commit reuses every dependency layer. Workers overlay the checkout's source as
for `vllm_env`.
