# vLLM profiler container

This image is the reproducibility boundary for VibeSim kernels registered with
`subprocess_env="vllm_env"`. It pins CUDA 13.0, the instrumented vLLM checkout,
the matching cu130 native wheel, Torch, FlashInfer, and the CUPTI build tools.
The image contains the profiling source snapshot; it never imports Python or
native extensions from the mounted host worktree.

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
creation. Ordinary worker containers receive only a temporary JSON exchange
directory and a persistent JIT cache mounted at `/cache`; the cache-free
`kernel-profile measure` diagnostic additionally mounts its explicit output
directory so the container-owned CUPTI/NVML artifacts survive the worker. Plot
rendering remains optional and never controls measurement success. Set
`VIBESIM_PROFILE_GPUS` only to GPUs reserved for the profiling job; otherwise
the existing idle-GPU selection remains active.

Release measurements require an image built from a clean tree. Record both the
immutable image digest and the labels reported by:

```bash
docker image inspect vibesim-profiler-vllm:cu130
```

The container does not virtualize the GPU driver. Validate the target GPU,
driver, and a real production kernel after every image or host-driver change.
