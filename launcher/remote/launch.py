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
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from .config import Remote, Slurm
from .transport import Transport, q

# Larger chunks are left for the next poll.
LOG_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class Submitted:
    jobdir: PurePosixPath
    # The direct job's session leader pid, or the Slurm job id.
    handle: str
    slurm: bool


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
) -> Submitted:
    command = node_command(remote_python, jobdir)
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
        return Submitted(jobdir, pid, slurm=False)
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
    return Submitted(jobdir, slurm_id, slurm=True)


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
        alive = (
            f"pid=$(cat {jobdir}/pid 2>/dev/null || cat {jobdir}/launch.pid); "
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
    # node.py runs as its session's leader and forwards the signal to the job.
    jobdir = q(submitted.jobdir)
    # A job that has ended may have had its pid reused: signal only a live one.
    transport.run(
        f"[ -f {jobdir}/exit_code ] || {{ pid=$(cat {jobdir}/pid 2>/dev/null || "
        f'cat {jobdir}/launch.pid); kill -TERM "$pid"; }}',
        check=False,
    )
