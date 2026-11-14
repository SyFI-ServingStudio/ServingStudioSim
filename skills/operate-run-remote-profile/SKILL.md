---
name: operate-run-remote-profile
description: >-
  Run the GPU half of an alignment campaign (nsys capture, workload pass,
  missing-kernel fill) on a remote GPU host over ssh, with everything else on
  this machine. Not for simulations or timing-predict on a remote CPU box
  (launcher/README.md "Remote execution").
---

# Run a campaign's GPU work on a remote

A remote with `profile_runtime: docker` (the default) runs each GPU job as one
container of the env's profiler image, on the image's own interpreters over the
shipped tree. Nothing GPU-side is built there, so a remote used only for GPU
work needs ssh, rsync and docker. Labeling, timing-predict, simulation, analysis
and the services all run here, on the results that come back.
`launcher/README.md` "Remote execution" (its "GPU work" part) is the reference;
this skill is the operating sequence.

## 1. Check the remote once

The remote needs:
- ssh with a passphrase-free key (ssh runs with BatchMode);
- rsync;
- docker with GPU support, and your user in the docker group;
- an NVIDIA driver new enough for the image's CUDA (CUDA 13.0 needs >= 580);
  the image carries its own CUDA user space, so there is no compat-library path;
- disk for the image (~30 GB), the checkpoint, and `<workdir>/cache`.

```bash
ssh HOST 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; id -nG; docker info --format "{{.ServerVersion}}"; df -h ~'
```

If the host runs Slurm (`sinfo`), use `launch: {slurm: ...}` and follow the
GPU-job rules for partitions and time limits, unless the user asks for the
direct path. On a host without a scheduler, or when the user asks for the
direct path, set `gpus: auto`: each job takes idle GPUs when it starts.
When other users run GPU work outside the scheduler, a Slurm GPU may still
be busy. Then set `gpu_wait: {min_free_mib: ...}` to at least what the
engine claims (e.g. 0.75 x 80 GB for `gpu_memory_utilization: 0.75`).

## 2. Configure

- `remote.yaml` (untracked, at the checkout root): the remote, with `workdir`
  on a disk with space and `mounts` listing every host directory the case paths
  point into. Include the checkpoint, any symlink targets inside it, and the
  corpus.
- A host profile `presets/alignment/hosts/<host>-remote.local.yaml` with the
  remote's checkpoint and corpus paths and `device_roles: {primary: "0"}` (the
  container sees only its own GPUs, from 0). Its `fork_python` and
  `nsys_executable` do not matter: in the image the engine and nsys are the
  image's own.
- The checkpoint must already be on the remote under the mounted HF cache.

**A RunPod pod** has no docker inside, so it boots from the profiler image
itself (`profile_runtime: in-image`, `ephemeral: true`), which runs the same
captures and fills. Push the image to a private registry RunPod can pull,
register a RunPod registry credential, and start the pod with the entrypoint
`bash -c` command in launcher/README.md ("A remote that is a profiler image"),
`22/tcp` exposed, `PUBLIC_KEY` set to this machine's key, and a `/workspace`
volume. Before the first capture, put the checkpoint and corpus on that volume:
run the image's `huggingface_hub` `snapshot_download` over ssh, and rsync the
corpus from here. The host profile then names `/workspace/...` paths. Terminate
the pod and delete the credential when the results are back.

## 3. Build the image here

```bash
just ref build-image vllm        # or sglang
```

One reference-engine image serves both the capture and the kernels it is
aligned against (`reference_engines/README.md`). Its tag is a hash of its
inputs; the first job sends it to the remote, later jobs reuse it. A fill part
for another env (e.g. `default_env`) needs that env's image built here too
(`python -m profiling.container.images build default`).

## 4. Run the campaign

```bash
P=presets/alignment/<pack>; R=logs/<date>_<topic>; H=presets/alignment/hosts/<host>-remote.local.yaml
python -m launcher alignment-campaign render --pack $P --host $H --out-root $R --case C
python -m launcher alignment-campaign run --pack $P --out-root $R --phase profile_nsys --case C --remote HOST
python -m launcher alignment-campaign run --pack $P --out-root $R --phase profile_workload --case C --remote HOST
python -m launcher $R/<case>/simulation.yaml --export-missing $R/work.json
python -m launcher kernel-profile fill --work $R/work.json --output-dir $R/fill --remote HOST
python -m launcher $R/<case>/simulation.yaml --export-missing $R/work.json   # must report 0 spec(s)
python -m launcher alignment-campaign run --pack $P --out-root $R --phase timing_predict --case C
python -m launcher alignment-campaign label --pack $P --out-root $R --case C
for p in analysis_kernel simulation analysis_e2e; do
  python -m launcher alignment-campaign run --pack $P --out-root $R --phase $p --case C; done
python -m launcher alignment-campaign extract --pack $P --runs $R --out $R/metrics.json
python -m launcher alignment-campaign compare --pack $P --measured $R/metrics.json --markdown
```

- The first capture of a deployment on a remote warms its caches first, in the
  same job (logged under `<phase>.warm-caches`), and records a warm-caches
  manifest in `<workdir>/cache/<key>`; the capture itself then runs with
  autotuning forbidden. Later captures of the same deployment skip the warm-up.
  Budget the first job's time limit for both.
- Keep each remote job inside the Slurm time limit: pass one `--case` per capture call.
- Run each command in a named tmux session, and record the session and log in `progress.md`.
- Fill before `timing_predict`, and rerun `--export-missing` until it reports
  `0 spec(s)`. `timing_predict` builds the same kernel cache as the simulation, and
  it measures any missing row on this machine's GPU, outside the scheduler and on the
  wrong GPU model. `label` reads the cost manifest that `timing_predict` writes.

## 5. Check what came back

- Each remote phase directory has `remote-run.json`, which records the host,
  Slurm id, image, exit code and copy status. The phase's `.complete` marker
  means the copy finished too.
- A fill writes `<output-dir>/<env>/fill_report.json` and a merge report.
  A conflicting row merges nothing: read the report before re-running.

## When a step fails

| Symptom | Action |
| --- | --- |
| `missing docker-access` | add the user to the docker group on the remote, then log in again |
| `no GPUs for this job` | Slurm: the job got no `--gres`. Direct: set `gpus` (or `auto`) in remote.yaml |
| vLLM: `Free memory on device ... is less than desired` | someone else holds the GPU: set `gpu_wait.min_free_mib`, or use `gpus: auto` |
| exit 75 `never had ... MiB free` / `no N idle GPU(s)` | the host stayed busy for `timeout_s`. Retry later, or raise the timeout |
| the container exits at start with a CUDA driver error | the driver is too old for the image. Use a host with driver >= 580 |
| `VIBESIM_AUTOTUNE_FORBIDDEN` in the capture's server log | the capture tuned what its warm-up did not: compare the warm-up's and the capture's server args, then rerun; `--allow-autotune` only to tune inside the capture deliberately |
| model or corpus not found inside the container | add its directory, including symlink targets, to `mounts` |
| fill part refused: the env is not served by the image | that env's image must be built here and is sent as its own job |
