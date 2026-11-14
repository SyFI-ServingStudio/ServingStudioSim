"""Run one `alignment profile` pass inside the engine's alignment image.

The host keeps what belongs to the machine: Slurm, which GPUs, where the
outputs go. The image owns everything that decides what the capture measures:
the engine (the official image with our patches, `reference_engines/`) and its
native extensions, Nsight Systems, the req-frontend binary, and the capture
runner's interpreter (`profiling/container/images.py`, image `vllm` or
`sglang`, chosen by the config's `engine`; the same images run L1 kernels). The
capture code itself is not baked in: this checkout's `alignment/` (minus its
submodules, which the image carries), `launcher/`, `profiling/` and `gpu/` are
overlaid read-only, exactly as L1 profiler workers overlay theirs.

Inside the container every path is fixed, so a pass is described by the same
config wherever it runs:

- `/models/` (ro): the Hugging Face hub directory holding the checkpoints the
  config names (`/models/models--org--name/snapshots/<sha>`), whole, so its
  relative blob links resolve.
- `/work/` (ro, file by file): `profile.yaml` (this pass in container paths),
  the trace, the text corpus.
- `/out/` (rw): the pass's `log_dir`.
- `/cache/` (rw): JIT, compile and tactic caches, one directory per engine,
  official base and GPU architecture (`profiling/exec/local.py:image_cache_key`,
  `HOME=/cache/home`), with the warm-caches manifests (`alignment/warm_caches.py`).

The container runs with `--network none` (every artifact is in the image or a
mount), `--ipc host` (the engines' shared-memory queues and CUDA IPC), unlimited
memlock, the job's GPUs by UUID and IMEX channel 0 (fabric-handle all-reduce
workspaces). When it exits, `profile_result.json` names container paths;
`rebind_result` rewrites them to this host's, keeping the originals beside, so
every later phase reads the pass exactly like a host capture.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from profiling.exec.container_user import container_user

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Set by the engine images (`reference_engines/*/Dockerfile`); the code that
#: behaves differently in it reads this, never the filesystem.
RUNTIME_ENV = "VIBESIM_CAPTURE_RUNTIME"
IN_IMAGE = "in-image"

CONTAINER_ROOT = PurePosixPath("/opt/vibesim")
MODELS = PurePosixPath("/models")
WORK = PurePosixPath("/work")
OUT = PurePosixPath("/out")
CACHE = PurePosixPath("/cache")
NSYS = PurePosixPath("/opt/nvidia/nsight-systems/2025.5.2/bin/nsys")

#: Written into the pass's `log_dir`: the config the container ran, and how.
CONTAINER_CONFIG_NAME = "profile.container.yaml"
CONTAINER_RUN_NAME = "container_run.json"

_SOURCE_DIRS = ("profiling", "launcher", "gpu", "reference_engines")
_IMEX_CHANNEL_CDI_DEVICE = "nvidia.com/imex-channel=0"


@dataclass(frozen=True)
class EngineImage:
    """The image that serves one engine: its server and capture-runner interpreters."""

    image: str
    fork_python: PurePosixPath
    capture_python: PurePosixPath


def _engine_image(name: str) -> EngineImage:
    from profiling.container.images import IMAGES

    image = IMAGES[name]
    assert image.capture_python is not None, f"image {name} runs no captures"
    return EngineImage(name, PurePosixPath(image.python), PurePosixPath(image.capture_python))


ENGINE_IMAGES = {engine: _engine_image(engine) for engine in ("vllm", "sglang")}


def in_image() -> bool:
    return os.environ.get(RUNTIME_ENV) == IN_IMAGE


class ImageRunError(RuntimeError):
    """A pass cannot be expressed in, or run by, the image."""


@dataclass
class Mounts:
    """Host path -> container path, read-only unless listed in `writable`."""

    entries: dict[str, str] = field(default_factory=dict)
    writable: set[str] = field(default_factory=set)

    def add(self, host: Path, container: PurePosixPath, *, writable: bool = False) -> str:
        host_text = str(host)
        existing = self.entries.get(host_text)
        if existing is not None and existing != str(container):
            raise ImageRunError(f"{host} is mounted at both {existing} and {container}")
        taken = {target: source for source, target in self.entries.items()}
        if str(container) in taken and taken[str(container)] != host_text:
            raise ImageRunError(f"{container} would hold both {taken[str(container)]} and {host}")
        self.entries[host_text] = str(container)
        if writable:
            self.writable.add(host_text)
        return str(container)

    def args(self) -> list[str]:
        arguments = []
        for host, container in self.entries.items():
            mode = "rw" if host in self.writable else "ro"
            arguments += ["--volume", f"{host}:{container}:{mode}"]
        return arguments


def _check_resolves_inside(file: Path, root: Path) -> None:
    """`file` reaches a real file under `root` through relative links only.

    An absolute link names a host path, which the container does not have even
    when it points under the mounted directory.
    """
    current = file
    for _ in range(40):
        if not current.is_symlink():
            break
        link = Path(os.readlink(current))
        if link.is_absolute():
            raise ImageRunError(
                f"{file} reaches its data through the absolute link {current} -> {link}, "
                "which does not exist in the container"
            )
        current = current.parent / link
    target = file.resolve()
    if not target.is_relative_to(root):
        raise ImageRunError(
            f"{file} resolves to {target}, outside {root}, which is all the container "
            "sees of this checkpoint"
        )
    if not target.is_file():
        raise ImageRunError(f"{file} is a broken link (to {target})")


def _model_mount(path: Path, mounts: Mounts) -> str:
    """A checkpoint path in the container, under /models.

    `<hub>/models--org--name/snapshots/<sha>` mounts the whole hub directory at
    /models, so the path keeps its shape (`/models/models--org--name/...`) and
    the snapshot's relative symlinks resolve as on the host, including a hub
    whose repo `blobs/` link on into a shared store (`<hub>/blobs/xx/<sha>`, as
    on /scratch/hf). Any other directory mounts as itself. Either way every file
    under the path must resolve inside what is mounted; the server would
    otherwise fail on it minutes into model loading.
    """
    parts = path.parts
    for index, part in enumerate(parts):
        if part.startswith("models--"):
            host_root, container_root = Path(*parts[:index]), MODELS
            break
    else:
        host_root, container_root = path, MODELS / path.name
    root = host_root.resolve()
    for directory, _, files in os.walk(path, followlinks=True):
        for name in files:
            _check_resolves_inside(Path(directory, name), root)
    mounts.add(host_root, container_root)
    return str(container_root.joinpath(*path.relative_to(host_root).parts))


def _work_file(path: Path, name: str, mounts: Mounts) -> str:
    if not path.is_file():
        raise ImageRunError(f"input file not found: {path}")
    return mounts.add(path, WORK / name)


def _map_server_arg(value: str, mounts: Mounts) -> str:
    """Absolute host paths inside server arguments (plain or in a JSON object)."""
    if value.startswith("{"):
        try:
            document = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(document, dict):
            mapped = {
                key: _map_server_arg(item, mounts) if isinstance(item, str) else item
                for key, item in document.items()
            }
            if mapped != document:
                return json.dumps(mapped, separators=(",", ":"))
        return value
    if value.startswith("/") and Path(value).exists():
        path = Path(value)
        if path.is_dir():
            return _model_mount(path, mounts)
        return _work_file(path, f"arg-{path.name}", mounts)
    return value


def container_config(config_path: Path, log_dir: Path) -> tuple[dict[str, Any], Mounts, dict]:
    """The pass in container paths, the mounts that make them real, and the path map.

    Refuses what the image cannot honour rather than dropping it: another
    engine, a driver compatibility library, or a GPU list that is not the job's.
    """
    from launcher.alignment_config import load_profile_config

    config_path = config_path.resolve()
    cfg = load_profile_config(config_path)
    engine = ENGINE_IMAGES.get(cfg.engine)
    if engine is None:
        raise ImageRunError(f"no alignment image runs engine {cfg.engine!r}")
    if cfg.driver_compat_lib_dir:
        raise ImageRunError(
            "driver_compat_lib_dir names a host library; the image carries its own CUDA "
            "user space, so the host driver must support it directly"
        )
    document = yaml.safe_load(config_path.read_text())
    mounts = Mounts()
    mounts.add(log_dir, OUT, writable=True)

    server = document["server"]
    server["model_path"] = _model_mount(Path(cfg.server.model_path), mounts)
    server["extra_args"] = [_map_server_arg(str(arg), mounts) for arg in cfg.server.extra_args]

    workload = document["workload"]
    trace = Path(cfg.workload.frontend.path)
    workload["frontend"]["path"] = _work_file(trace, f"trace{trace.suffix}", mounts)
    corpus = Path(cfg.workload.text_file)
    workload["text_file"] = _work_file(corpus, f"text_corpus{corpus.suffix}", mounts)
    tokenizer = Path(cfg.workload.tokenizer)
    if tokenizer.is_absolute():
        workload["tokenizer"] = (
            _model_mount(tokenizer, mounts)
            if tokenizer.is_dir()
            else _work_file(tokenizer, f"tokenizer-{tokenizer.name}", mounts)
        )

    devices = [token.strip() for token in cfg.cuda_visible_devices.split(",") if token.strip()]
    document["cuda_visible_devices"] = ",".join(str(index) for index in range(len(devices)))
    document["log_dir"] = str(OUT)
    document["fork_python"] = str(engine.fork_python)
    document.pop("driver_compat_lib_dir", None)
    if cfg.profile_kind == "nsys":
        document.setdefault("nsys", {})["executable"] = str(NSYS)

    path_map = {container: host for host, container in mounts.entries.items()}
    return document, mounts, {"host_devices": devices, "paths": path_map}


def overlay_mounts(root: Path = REPO_ROOT) -> Mounts:
    """This checkout's capture code over the image, read-only.

    `alignment/` is mounted entry by entry around its submodules, so the engine
    fork and the req-frontend binary stay the image's own.
    """
    mounts = Mounts()
    for directory in _SOURCE_DIRS:
        mounts.add(root / directory, CONTAINER_ROOT / directory)
    submodules = {
        PurePosixPath(line.split()[-1])
        for line in subprocess.run(
            ["git", "-C", str(root), "config", "--file", ".gitmodules", "--get-regexp", "path"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    }

    def walk(relative: PurePosixPath) -> None:
        for entry in sorted((root / relative).iterdir()):
            child = relative / entry.name
            if child in submodules or entry.name == "__pycache__":
                continue
            if any(child in sub.parents for sub in submodules):
                walk(child)
            else:
                mounts.add(root / child, CONTAINER_ROOT / child)

    walk(PurePosixPath("alignment"))
    return mounts


def image_tag(engine: str, root: Path = REPO_ROOT) -> str:
    from profiling.container.images import IMAGES

    return IMAGES[ENGINE_IMAGES[engine].image].tag(root)


def container_cache_path(path: Path, cache_dir: Path) -> PurePosixPath:
    """Where a file under the host `cache_dir` appears inside the container.

    Anything handed to the container environment uses this form: nsys records
    the environment in the `.nsys-rep`, and a host path there names the host's
    user and layout in a published capture.
    """
    return CACHE / PurePosixPath(Path(path).resolve().relative_to(Path(cache_dir).resolve()))


def docker_argv(
    image: str,
    mounts: Mounts,
    *,
    capture_python: PurePosixPath,
    gpu_uuids: list[str],
    cache_dir: Path,
    name: str,
    rootless: bool | None = None,
    autotune_env: dict[str, str] | None = None,
    warm_caches: bool = False,
) -> list[str]:
    # Container root under a rootless runtime (it is the caller there), else the
    # caller's uid, so the outputs in /out belong to the caller either way.
    user = container_user(rootless)
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--ipc",
        "host",
        "--ulimit",
        "memlock=-1:-1",
        "--gpus",
        f'"device={",".join(gpu_uuids)}"',
        "--device",
        _IMEX_CHANNEL_CDI_DEVICE,
        # Explicit: the image's L1 workers run another interpreter and module.
        "--entrypoint",
        str(capture_python),
        "--user",
        user,
        "--env",
        "HOME=/cache/home",
        "--env",
        "USER=vibesim",
        "--env",
        "TMPDIR=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/cache/home/.cache/torchinductor",
        "--env",
        "HF_HUB_OFFLINE=1",
        # The autotune policy (alignment/warm_caches.py); the capture runner
        # hands its environment to the engine, so every engine process sees it.
        *[arg for item in (autotune_env or {}).items() for arg in ("--env", "=".join(item))],
        "--volume",
        f"{cache_dir}:{CACHE}:rw",
        *mounts.args(),
        image,
        "-m",
        "launcher",
        "alignment",
        "profile",
        str(WORK / "profile.yaml"),
        *(["--warm-caches"] if warm_caches else []),
    ]


def _rebind(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        for container, host in sorted(path_map.items(), key=lambda item: -len(item[0])):
            if value == container or value.startswith(container + "/"):
                return host + value[len(container) :]
        return value
    if isinstance(value, dict):
        return {key: _rebind(item, path_map) for key, item in value.items()}
    if isinstance(value, list):
        return [_rebind(item, path_map) for item in value]
    return value


def rebind_result(log_dir: Path, path_map: dict[str, str], host_devices: list[str]) -> dict:
    """Rewrite `profile_result.json` from container paths to this host's.

    The container's own copy is kept as `profile_result.container.json`, so the
    rewrite is auditable and reversible.
    """
    result_path = log_dir / "profile_result.json"
    original = json.loads(result_path.read_text())
    shutil.copy2(result_path, log_dir / "profile_result.container.json")
    rebound = _rebind(original, path_map)
    rebound["cuda_visible_devices"] = ",".join(host_devices)
    rebound["capture_runtime"] = {"kind": IN_IMAGE, "record": CONTAINER_RUN_NAME}
    result_path.write_text(json.dumps(rebound, indent=2))
    return rebound


def run_profile_in_image(
    config_path: Path,
    *,
    image: str | None = None,
    allow_autotune: bool = False,
    warm_caches: bool = False,
    startup_timeout: float | None = None,
) -> int:
    """Run `alignment profile <config>` in the image; return its exit code.

    `warm_caches` instead starts the engine with autotuning allowed, lets it
    reach ready and stops it (no requests, no capture; its logs go to
    `<log_dir>.warm-caches`), then records the manifest a capture's preflight
    looks for. A capture without `allow_autotune` needs that manifest, and
    fails here, before the container starts, when there is none.

    `startup_timeout` replaces the config's `server.startup_timeout` in the
    container's copy: a warm-up compiles and tunes what its captures then find
    cached, so it can take much longer to reach ready (SGLang's GLM-5.3-Flash
    smoke warm-up ran ~1050 s, over the 900 s default). The manifest key leaves
    the field out, so the captures still match.
    """
    from launcher.alignment_config import load_profile_config
    from profiling.exec.local import gpu_uuids_for_indices, image_cache_dir
    from reference_engines.autotune_guard import (
        POLICY_ENV,
        PREFLIGHT_ENV,
        AutotuneForbiddenError,
        forbidden_line,
    )

    from . import warm_caches as warm

    cfg = load_profile_config(config_path)
    log_dir = Path(cfg.log_dir)
    if warm_caches:
        log_dir = log_dir.with_name(log_dir.name + ".warm-caches")
    log_dir.mkdir(parents=True, exist_ok=True)
    document, mounts, path_info = container_config(config_path, log_dir)
    if startup_timeout is not None:
        document["server"]["startup_timeout"] = float(startup_timeout)
    tag = image or image_tag(cfg.engine)
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode:
        raise ImageRunError(
            f"image {tag} is not on this host; build it with `just ref build-image {cfg.engine}`"
        )
    host_devices = path_info["host_devices"]
    uuids = gpu_uuids_for_indices([int(token) for token in host_devices])
    cache_dir = image_cache_dir(tag, uuids)
    cache_key = cache_dir.name  # image_cache_key(tag, uuids)
    policy = warm.capture_policy(allow_autotune=allow_autotune, warm_caches=warm_caches)
    autotune_env = {POLICY_ENV: policy}
    if policy == "forbid":
        # Instant, and before anything starts: the guard in the engine is only
        # the backstop for a manifest whose key turns out incomplete.
        manifest = warm.preflight(cfg, config_path, cache_dir, cache_key)
        autotune_env[PREFLIGHT_ENV] = (
            f"warm-caches manifest {container_cache_path(manifest, cache_dir)}"
        )
        print(f"[profile] warm caches: {manifest}", flush=True)

    container_config_path = log_dir / CONTAINER_CONFIG_NAME
    container_config_path.write_text(yaml.safe_dump(document, sort_keys=False))
    mounts.add(container_config_path, WORK / "profile.yaml")
    for host, container in overlay_mounts().entries.items():
        mounts.add(Path(host), PurePosixPath(container))

    name = f"vibesim-capture-{os.getpid()}"
    argv = docker_argv(
        tag,
        mounts,
        capture_python=ENGINE_IMAGES[cfg.engine].capture_python,
        gpu_uuids=uuids,
        cache_dir=cache_dir,
        name=name,
        autotune_env=autotune_env,
        warm_caches=warm_caches,
    )
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", tag],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    record = {
        "schema_version": 1,
        "image": tag,
        "image_id": image_id,
        "argv": argv,
        "host_config": str(config_path.resolve()),
        "host_devices": host_devices,
        "gpu_uuids": uuids,
        "cache_dir": str(cache_dir),
        "autotune": autotune_env,
        "paths": path_info["paths"],
    }
    (log_dir / CONTAINER_RUN_NAME).write_text(json.dumps(record, indent=2))
    print(
        f"[profile] in image {tag} ({image_id[:19]}) on GPUs {','.join(uuids)}, "
        f"cache {cache_dir.name}, autotune {policy}",
        flush=True,
    )
    before = warm.snapshot(cache_dir) if warm_caches else {}
    started = time.monotonic()
    returncode = subprocess.run(argv).returncode
    record["exit_code"] = returncode
    (log_dir / CONTAINER_RUN_NAME).write_text(json.dumps(record, indent=2))
    if returncode != 0:
        server_log = log_dir / cfg.engine / f"{cfg.name}_server.log"
        line = (
            forbidden_line(server_log.read_text(errors="replace")) if server_log.is_file() else None
        )
        if line is not None:
            raise AutotuneForbiddenError(
                f"the engine tried to tune under VIBESIM_AUTOTUNE={policy} and stopped "
                f"({server_log}): {line}"
            )
        return returncode
    if warm_caches:
        manifest = warm.write_manifest(
            cfg,
            config_path,
            cache_dir,
            cache_key,
            image=tag,
            image_id=image_id,
            before=before,
            seconds=time.monotonic() - started,
        )
        print(f"[warm-caches] manifest: {manifest}", flush=True)
    else:
        rebind_result(log_dir, path_info["paths"], host_devices)
    return returncode


def _own_manifest() -> dict[str, Any]:
    from profiling.exec.env import IMAGE_MANIFEST, IMAGE_MANIFEST_ENV

    path = Path(os.environ.get(IMAGE_MANIFEST_ENV) or IMAGE_MANIFEST)
    return json.loads(path.read_text(encoding="utf-8"))


def own_cache_key(manifest: dict[str, Any]) -> str:
    """`profiling/exec/local.py:image_cache_key` as seen from inside the image.

    The engine and its official base digest come from the image's manifest, the
    architecture from the GPUs this process sees; whatever mounted `/cache`
    keyed it the same way.
    """
    digest = str(manifest.get("base_image", "")).rpartition("@sha256:")[2]
    if not digest:
        raise ImageRunError(f"the image manifest names no pinned base image: {manifest}")
    caps = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if len(set(caps)) != 1:
        raise ImageRunError(f"the visible GPUs report compute capabilities {sorted(set(caps))}")
    return f"{manifest['image']}-{digest[:12]}-sm{caps[0].replace('.', '')}"


def run_profile_unhosted(config_path: Path, *, allow_autotune: bool = False) -> dict:
    """`alignment profile` in an engine image that no host-side runner started.

    A remote job or a pod runs the capture here directly, so this does what
    `run_profile_in_image` does from a host: it owns the autotune policy. (The
    image's own engine and nsys replace whatever the config names when it is
    loaded in an image, `load_profile_config`.) Without
    `allow_autotune` the capture needs a warm-caches manifest in `/cache`; when
    there is none for this deployment, a warm-up pass runs first in this same
    process (logged under `<log_dir>.warm-caches`) and records it, and the
    capture then runs with tuning forbidden.
    """
    import dataclasses

    from launcher.alignment_config import load_profile_config
    from reference_engines.autotune_guard import (
        ALLOW,
        FORBID,
        POLICY_ENV,
        PREFLIGHT_ENV,
        AutotuneForbiddenError,
    )

    from . import warm_caches as warm
    from .runner import run_profile

    cfg = load_profile_config(config_path)
    engine = ENGINE_IMAGES.get(cfg.engine)
    if engine is None:
        raise ImageRunError(f"no alignment image runs engine {cfg.engine!r}")
    manifest = _own_manifest()
    if manifest.get("image") != engine.image:
        raise ImageRunError(
            f"a {cfg.engine} capture needs the {engine.image} image; "
            f"this is {manifest.get('image')}"
        )
    if allow_autotune:
        os.environ[POLICY_ENV] = ALLOW
        return run_profile(cfg)
    cache_key = own_cache_key(manifest)
    try:
        found = warm.preflight(cfg, config_path, Path(CACHE), cache_key)
    except AutotuneForbiddenError:
        log_dir = Path(cfg.log_dir)
        warm_cfg = dataclasses.replace(
            cfg, log_dir=str(log_dir.with_name(log_dir.name + ".warm-caches"))
        )
        print(f"[profile] no warm caches in {CACHE} for this deployment: warming first", flush=True)
        os.environ[POLICY_ENV] = ALLOW
        before = warm.snapshot(Path(CACHE))
        started = time.monotonic()
        run_profile(warm_cfg, warm_only=True)
        found = warm.write_manifest(
            cfg,
            config_path,
            Path(CACHE),
            cache_key,
            image=f"{engine.image}:{manifest.get('inputs')}",
            image_id=str(manifest.get("inputs")),
            before=before,
            seconds=time.monotonic() - started,
        )
        print(f"[warm-caches] manifest: {found}", flush=True)
    os.environ[POLICY_ENV] = FORBID
    os.environ[PREFLIGHT_ENV] = f"warm-caches manifest {found}"
    print(f"[profile] warm caches: {found}", flush=True)
    return run_profile(cfg)
