---
name: operate-run-full-remote-profile
description: >-
  Run the GPU half of an alignment campaign (nsys capture, missing-kernel fill)
  on a remote GPU host over ssh, with everything else on this machine. Not for
  simulations or timing-predict on a remote CPU box (launcher/README.md "Remote
  execution").
---

# Run a full remote profile

A **full remote profile** remote (`profile_runtime: full-remote-profile` in
`remote.yaml`) runs only GPU work: the nsys capture of a serving run and kernel
profiling. Each job is one container of the profiler image. The remote needs
no ServingStudio setup. Labeling, timing-predict, simulation, analysis and the
services all run here, on the results that come back.
`launcher/README.md` "Remote execution" is the reference; this skill is the
operating sequence.

## 1. Check the remote once

The remote needs:
- ssh with a passphrase-free key (ssh runs with BatchMode);
- rsync;
- docker with GPU support, and your user in the docker group;
- an NVIDIA driver new enough for the image's CUDA (CUDA 13.0 needs >= 580);
- disk for the image (~30 GB) and the checkpoint.

```bash
ssh HOST 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; id -nG; docker info --format "{{.ServerVersion}}"; df -h ~'
```

If the host runs Slurm (`sinfo`), use `launch: {slurm: ...}` and follow the
GPU-job rules for partitions and time limits. Never run the GPU job outside the
scheduler on a Slurm host.

## 2. Configure

- `remote.yaml` (untracked, at the checkout root): the remote, with
  `profile_runtime: full-remote-profile`, `workdir` on a disk with space, and
  `mounts` listing every host directory the case paths point into. Include the
  checkpoint, any symlink targets inside it, and the corpus.
- A host profile `presets/alignment/hosts/<host>-remote.local.yaml` with the
  remote's checkpoint and corpus paths and:

  ```yaml
  fork_python: /opt/vibesim/alignment/profiler/vllm/.venv/bin/python
  nsys_executable: /usr/local/bin/nsys
  device_roles: {primary: "0"}   # the container sees only its own GPUs, from 0
  ```

- The checkpoint must already be on the remote under the mounted HF cache.

## 3. Build the image here

```bash
python -m profiling.container.images build vllm
```

The image carries the fork, nsys and the prebuilt load generator. Its tag is a
hash of its inputs. The first job sends it to the remote, and later jobs reuse
it. If your network is slow over IPv4, build with a host-network buildx builder.

## 4. Run the campaign

```bash
P=presets/alignment/<pack>; R=logs/<date>_<topic>; H=presets/alignment/hosts/<host>-remote.local.yaml
python -m launcher alignment-campaign render --pack $P --host $H --out-root $R --case C
python -m launcher alignment-campaign run --pack $P --out-root $R --phase profile_nsys --case C --remote HOST
python -m launcher alignment-campaign run --pack $P --out-root $R --phase profile_workload --case C --remote HOST
python -m launcher alignment-campaign label --pack $P --out-root $R --case C
python -m launcher $R/<case>/simulation.yaml --export-missing $R/work.json
python -m launcher kernel-profile fill --work $R/work.json --output-dir $R/fill --remote HOST
for p in timing_predict analysis_kernel simulation analysis_e2e; do
  python -m launcher alignment-campaign run --pack $P --out-root $R --phase $p --case C; done
python -m launcher alignment-campaign extract --pack $P --runs $R --out $R/metrics.json
python -m launcher alignment-campaign compare --pack $P --measured $R/metrics.json --markdown
```

- Keep each remote job inside the Slurm time limit: pass one `--case` per capture call.
- Run each command in a named tmux session, and record the session and log in `progress.md`.
- `--export-missing` reports `0 spec(s)` when profile.db already covers the case. Skip the fill then.

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
| `no GPUs for this job` | Slurm: the job got no `--gres`. Direct: set `gpus` in remote.yaml |
| the container exits at start with a CUDA driver error | the driver is too old for the image. Use a host with driver >= 580, or set `driver_compat_lib_dir` in the profile config |
| `fork_python does not exist` | the host profile names a host path. Use the image's interpreter (above) |
| model or corpus not found inside the container | add its directory, including symlink targets, to `mounts` |
| fill part refused: the env is not served by the image | that env's image (`profiling.container.images env ENV`) must be built here and is sent as its own job |
