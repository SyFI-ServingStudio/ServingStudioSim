"""Starting, watching and stopping one remote job (`launch: direct | slurm`).

Either way the job is ``node.py run-job JOBDIR`` and its state lives in files
under ``JOBDIR``: ``stdout.log`` grows while it runs and ``exit_code`` appears
when it ends. Polling reads those files and, to tell a running job from a lost
one, asks the process table (direct) or the scheduler (Slurm).

direct
    A detached session (``setsid nohup``) on the access host, so the job
    outlives the ssh connection that started it.
slurm
    An ``sbatch`` script submitted from the access host. Slurm exports the
    submitting environment by default, so the script clears the managed-job
    context before anything runs (node.py clears it again).

Under ``profile_runtime: full-remote-profile`` either form runs
``JOBDIR/container.sh`` instead (`container_script`): one ``docker run`` of a
profiler image whose own interpreter runs ``node.py run-job``. Its pid file is
the container's, so the host side is tracked by ``launch.pid`` alone.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from .config import AUTO_GPUS, GpuWait, Remote, Slurm
from .transport import Transport, q

# Larger chunks are left for the next poll.
LOG_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Submitted:
    jobdir: PurePosixPath
    # The direct job's session leader pid, or the Slurm job id.
    handle: str
    slurm: bool
    # A container job: its `pid` file is in the container's pid namespace.
    container: bool = False


@dataclass(frozen=True)
class Status:
    # None while the job runs or is queued.
    exit_code: int | None
    # Neither running nor finished: the process or allocation vanished.
    lost: bool
    # The log offset after this chunk, and the log's size when polled.
    log_size: int
    log_total: int
    log_chunk: str
    detail: str = ""


def node_command(remote_python: PurePosixPath, jobdir: PurePosixPath) -> str:
    return f"{q(remote_python)} -m launcher.remote.node run-job {q(jobdir)}"


@dataclass(frozen=True)
class Container:
    """A full-remote-profile job: run ``node.py`` in ``image`` on ``python``."""

    image: str
    python: PurePosixPath
    workdir: PurePosixPath
    node_local: PurePosixPath
    mounts: tuple[PurePosixPath, ...] = ()
    # The GPUs of a direct job; a Slurm job takes those it was allocated.
    gpus: tuple[str, ...] = ()
    gpu_wait: GpuWait | None = None
    # How many GPUs the job needs; `gpus: auto` takes that many idle ones.
    gpu_count: int = 1


# Quoted twice: docker reads the value as CSV, so a GPU list needs its own quotes.
GPU_REQUEST = '"\\"device=$gpus\\""'


def container_name(job_id: str) -> str:
    return f"vibesim-{job_id}"


# Exit status of a job whose GPUs never had the memory it waits for.
GPU_WAIT_TIMEOUT = 75
# `gpus: auto`: a GPU is idle with no compute process and less memory used than this.
IDLE_USED_MIB = 2000


def _auto_gpu_lines(container: Container, jobdir: PurePosixPath) -> list[str]:
    """Shell that takes ``gpu_count`` idle GPUs once the same set stays idle for ``quiet_s``."""

    wait = container.gpu_wait or GpuWait(min_free_mib=0)
    log = q(jobdir / "stdout.log")
    return [
        f'if [ "$gpus" = {AUTO_GPUS} ]; then',
        f"  count={container.gpu_count}; quiet=0; waited=0; prev=",
        f'  echo "[remote] waiting for $count idle GPU(s)" >> {log}',
        "  while :; do",
        "    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)",
        "    pick=$(nvidia-smi --query-gpu=index,uuid,memory.used,memory.free "
        "--format=csv,noheader,nounits | while IFS=', ' read -r i u used free; do "
        f'case "$busy" in *"$u"*) continue;; esac; [ "$used" -lt {IDLE_USED_MIB} ] '
        f'&& [ "$free" -ge {wait.min_free_mib} ] && echo "$i"; done | head -n "$count" | paste -sd, -)',
        '    if [ -n "$pick" ] && [ "$(echo "$pick" | tr , "\n" | wc -l)" -eq "$count" ] '
        '&& [ "$pick" = "$prev" ]; then',
        f'      [ "$quiet" -ge {wait.quiet_s} ] && break',
        "      quiet=$((quiet + 5))",
        "    else quiet=0; fi",
        "    prev=$pick",
        f'    if [ "$waited" -ge {wait.timeout_s} ]; then echo "[remote] no {container.gpu_count} '
        f'idle GPU(s) for {wait.quiet_s} s within {wait.timeout_s} s" >> {log}; '
        f"echo {GPU_WAIT_TIMEOUT} > {q(jobdir / 'exit_code')}; exit {GPU_WAIT_TIMEOUT}; fi",
        "    sleep 5; waited=$((waited + 5))",
        "  done",
        "  gpus=$pick",
        "fi",
    ]


def _gpu_wait_lines(wait: GpuWait | None, jobdir: PurePosixPath) -> list[str]:
    """Shell that waits until every assigned GPU has ``min_free_mib`` free for ``quiet_s``.

    Other users' processes need not go through the scheduler, so an allocated
    GPU may still be busy; the job reports who holds it while it waits.
    """

    if wait is None:
        return []
    log = q(jobdir / "stdout.log")
    return [
        f"need={wait.min_free_mib}; quiet=0; waited=0; last=0",
        "while :; do",
        # A Slurm job's nvidia-smi sees only its own GPUs (renumbered from 0).
        '  if [ -n "${SLURM_JOB_ID:-}" ]; then free=$(nvidia-smi --query-gpu=memory.free '
        "--format=csv,noheader,nounits); else free=$(for g in ${gpus//,/ }; do nvidia-smi "
        '-i "$g" --query-gpu=memory.free --format=csv,noheader,nounits; done); fi',
        "  low=$(echo \"$free\" | awk -v n=$need 'NF && $1 < n' | wc -l)",
        '  if [ "$low" -eq 0 ]; then',
        f'    [ "$quiet" -ge {wait.quiet_s} ] && break',
        "    quiet=$((quiet + 5))",
        "  else quiet=0; fi",
        f'  if [ "$waited" -ge {wait.timeout_s} ]; then echo "[remote] GPU(s) $gpus never had '
        f'$need MiB free for {wait.quiet_s} s within {wait.timeout_s} s" >> {log}; '
        f"echo {GPU_WAIT_TIMEOUT} > {q(jobdir / 'exit_code')}; exit {GPU_WAIT_TIMEOUT}; fi",
        '  if [ "$low" -ne 0 ] && [ $((waited - last)) -ge 300 -o "$waited" -eq 0 ]; then last=$waited; '
        "echo \"[remote] waiting for $need MiB free on GPU(s) $gpus; held by: $(nvidia-smi "
        "--query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r p m; do "
        'printf "%s(%s,%s) " "$(ps -o user= -p "$p" 2>/dev/null)" "$p" "$m"; done)\" >> ' + f"{log}; fi",
        "  sleep 5; waited=$((waited + 5))",
        "done",
    ]


def container_script(container: Container, *, job_id: str, jobdir: PurePosixPath, cwd: PurePosixPath) -> str:
    """The host side of one container job.

    The container sees the workdir, node-local scratch and each mount at the
    same absolute path, so ``job.json`` names the same files on both sides.
    It runs in the background so a TERM (``kill`` here, ``scancel``) reaches
    the trap, which stops the container; node.py inside records the exit code.
    A container that never started records the ``docker run`` status instead.
    """

    name = container_name(job_id)
    volumes = []
    for path in dict.fromkeys((container.workdir, container.node_local, *container.mounts)):
        volumes += ["--volume", f"{path}:{path}"]
    # The profiler images point their JIT caches (Triton, torch extensions) at
    # /cache, as the container workers mount it; keep them across jobs.
    volumes += ["--volume", f"{container.workdir / 'cache'}:/cache"]
    configured = ",".join(container.gpus)
    home = container.workdir / "cache" / "home"
    run = [
        "docker", "run", "--rm", "--init", "--name", name,
        "--ipc", "host",
        "--gpus", GPU_REQUEST,
        "--user", "$(id -u):$(id -g)",
        "--env", f"HOME={home}", "--env", "USER=vibesim", "--env", "LOGNAME=vibesim",
        *volumes,
        "--workdir", str(cwd),
        "--entrypoint", str(container.python),
        container.image,
        "-m", "launcher.remote.node", "run-job", str(jobdir),
    ]
    quoted = " ".join(
        part if part in (GPU_REQUEST, "$(id -u):$(id -g)") else shlex.quote(part)
        for part in run
    )
    return "\n".join(
        [
            "#!/bin/bash",
            "set -uo pipefail",
            f"gpus={shlex.quote(configured)}",
            # Slurm names the allocated GPUs by index; docker, outside the
            # job's cgroup, addresses the same physical devices.
            'if [ -z "$gpus" ]; then gpus=${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}; fi',
            'if [ -z "$gpus" ]; then echo "[remote] no GPUs for this job: set gpus in remote.yaml '
            'or ask Slurm for them" >> ' + f"{q(jobdir / 'stdout.log')}; "
            f"echo 2 > {q(jobdir / 'exit_code')}; exit 2; fi",
            f"mkdir -p {q(home)} {q(container.workdir / 'cache')}",
            *_auto_gpu_lines(container, jobdir),
            *_gpu_wait_lines(container.gpu_wait, jobdir),
            f"echo \"[remote] container {name} on GPU(s) $gpus ({container.image})\" >> {q(jobdir / 'stdout.log')}",
            f"stop() {{ docker kill --signal TERM {name} >/dev/null 2>&1; }}",
            "trap stop TERM INT HUP",
            f"{quoted} &",
            "child=$!",
            # `wait` returns early when a trapped signal arrives; wait again for the container.
            'while :; do wait "$child"; status=$?; kill -0 "$child" 2>/dev/null || break; done',
            f"[ -f {q(jobdir / 'exit_code')} ] || echo $status > {q(jobdir / 'exit_code')}",
            "exit $status",
            "",
        ]
    )


def sbatch_script(
    slurm: Slurm, *, name: str, jobdir: PurePosixPath, cwd: PurePosixPath, command: str
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --output={jobdir}/slurm.out",
        f"#SBATCH --chdir={cwd}",
        f"#SBATCH --time={slurm.time}",
    ]
    if slurm.partition:
        lines.append(f"#SBATCH --partition={slurm.partition}")
    if slurm.cpus_per_task:
        lines.append(f"#SBATCH --cpus-per-task={slurm.cpus_per_task}")
    if slurm.mem:
        lines.append(f"#SBATCH --mem={slurm.mem}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    if slurm.gpus:
        lines.append(f"#SBATCH --gres=gpu:{slurm.gpus}")
    lines += [f"#SBATCH {extra}" for extra in slurm.extra]
    lines += [
        # sbatch exports the submitting shell; the sender owns the managed job.
        "unset VIBESIM_MANAGED_JOB_CONTEXT VIBESIM_MANAGED_RUN_CONTEXT",
        f"exec {command}",
        "",
    ]
    return "\n".join(lines)


def submit(
    transport: Transport,
    remote: Remote,
    *,
    job_id: str,
    jobdir: PurePosixPath,
    cwd: PurePosixPath,
    remote_python: PurePosixPath,
    container: Container | None = None,
) -> Submitted:
    if container is None:
        command = node_command(remote_python, jobdir)
    else:
        script = jobdir / "container.sh"
        transport.write_text(
            script, container_script(container, job_id=job_id, jobdir=jobdir, cwd=cwd)
        )
        command = f"bash {q(script)}"
    if remote.slurm is None:
        # Braced so `&` backgrounds only the job: backgrounding the whole `&&`
        # list would leave a subshell holding this command's output open until
        # the job ends.
        script = (
            f"mkdir -p {q(jobdir)} && cd {q(cwd)} && "
            f"{{ setsid nohup {command} >> {q(jobdir / 'node.log')} 2>&1 < /dev/null & "
            "echo $!; }"
        )
        pid = transport.run(script).stdout.strip()
        transport.write_text(jobdir / "launch.pid", pid + "\n")
        return Submitted(jobdir, pid, slurm=False, container=container is not None)
    batch = jobdir / "job.sbatch"
    transport.write_text(
        batch,
        sbatch_script(
            remote.slurm, name=f"vibesim-{job_id}", jobdir=jobdir, cwd=cwd, command=command
        ),
    )
    output = transport.run(f"sbatch --parsable {q(batch)}").stdout.strip()
    # `--parsable` prints "jobid" or "jobid;cluster".
    slurm_id = output.splitlines()[-1].split(";", 1)[0]
    if not slurm_id.isdigit():
        raise RuntimeError(f"sbatch printed no job id: {output!r}")
    transport.write_text(jobdir / "slurm_id", slurm_id + "\n")
    return Submitted(jobdir, slurm_id, slurm=True, container=container is not None)


def status_script(submitted: Submitted, offset: int) -> str:
    jobdir = q(submitted.jobdir)
    if submitted.slurm:
        alive = (
            f"state=$(squeue -h -j {shlex.quote(submitted.handle)} -o %T 2>/dev/null); "
            'if [ -n "$state" ]; then echo alive=1; echo "detail=$state"; else '
            f'echo alive=0; echo "detail=$(sacct -n -X -P -j {shlex.quote(submitted.handle)} '
            '-o State,ExitCode 2>/dev/null | head -1)"; fi'
        )
    else:
        pid = (
            f"cat {jobdir}/launch.pid"
            if submitted.container
            else f"cat {jobdir}/pid 2>/dev/null || cat {jobdir}/launch.pid"
        )
        alive = (
            f"pid=$({pid}); "
            'if kill -0 "$pid" 2>/dev/null; then echo alive=1; else echo alive=0; fi'
        )
    return (
        f"if [ -f {jobdir}/exit_code ]; then echo exit=$(cat {jobdir}/exit_code); "
        f"else {alive}; fi; "
        f"size=$(stat -c %s {jobdir}/stdout.log 2>/dev/null || echo 0); echo size=$size; "
        "echo ---; "
        # Exactly the bytes up to the size printed above, so the next offset is
        # known without measuring decoded text.
        f"n=$((size - {offset})); [ $n -gt {LOG_CHUNK_BYTES} ] && n={LOG_CHUNK_BYTES}; "
        f"if [ $n -gt 0 ]; then tail -c +{offset + 1} {jobdir}/stdout.log | head -c $n; fi"
    )


def parse_status(text: str, offset: int) -> Status:
    head, _, chunk = text.partition("---\n")
    fields = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
    exit_code = int(fields["exit"]) if fields.get("exit", "").strip().isdigit() else None
    size = int(fields.get("size", "0") or 0)
    return Status(
        exit_code=exit_code,
        lost=exit_code is None and fields.get("alive") == "0",
        log_size=max(offset, min(size, offset + LOG_CHUNK_BYTES)),
        log_total=size,
        log_chunk=chunk,
        detail=fields.get("detail", "").strip(),
    )


def poll(transport: Transport, submitted: Submitted, offset: int) -> Status:
    return parse_status(transport.run(status_script(submitted, offset)).stdout, offset)


def cancel(transport: Transport, submitted: Submitted) -> None:
    if submitted.slurm:
        transport.run(f"scancel {shlex.quote(submitted.handle)}", check=False)
        return
    # node.py runs as its session's leader and forwards the signal to the job;
    # a container job's leader is container.sh, which stops its container.
    jobdir = q(submitted.jobdir)
    pid = (
        f"cat {jobdir}/launch.pid"
        if submitted.container
        else f"cat {jobdir}/pid 2>/dev/null || cat {jobdir}/launch.pid"
    )
    # A job that has ended may have had its pid reused: signal only a live one.
    transport.run(
        f'[ -f {jobdir}/exit_code ] || {{ pid=$({pid}); kill -TERM "$pid"; }}',
        check=False,
    )
