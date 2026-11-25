"""Build an image of `images.py` as an enroot squashfs, on a host with no Docker.

Slurm clusters with pyxis/enroot (a GB200 NVL72 cluster, for one) run containers but cannot
build them. This interprets the image's Dockerfile step by step on enroot
containers instead: each stage starts from an imported squashfs of its `FROM`,
`RUN` executes in that stage's root filesystem as (namespaced) root, `COPY`
copies the build context (filtered by `.dockerignore`) or another stage's files,
and the last stage is exported as one squashfs. The tag, the build arguments and
the pinned base are `images.build_plan`'s, so the result is the image `docker
build` would tag, flattened (no layers, no cache).

Two halves, because the plan needs the checkout's Python (PyYAML) and the build
needs enroot, which only compute nodes have:

    # on any host with the checkout's venv
    python -m profiling.container.enroot_build plan vllm --platform linux/arm64 --out plan.json
    # on a compute node of that platform, with system python3 only
    python3 profiling/container/enroot_build.py build plan.json \
        --base /path/to/base.sqsh --out-dir /path/to/images

`--base` is the squashfs of the plan's `BASE_IMAGE` (`enroot import` of the
pinned digest's tag); the build records the reference it stands for.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# -- plan (checkout venv) ----------------------------------------------------


def plan(name: str, platform_name: str | None, root: Path = REPO_ROOT) -> dict:
    import reference_engines
    from profiling.container import images

    if platform_name is not None:
        reference_engines.host_platform = lambda: platform_name  # type: ignore[assignment]
    tag, args = images.build_plan(name, root)
    image = images.IMAGES[name]
    return {
        "image": name,
        "tag": tag,
        "platform": platform_name or reference_engines.host_platform(),
        "dockerfile": image.dockerfile,
        "root": str(root),
        "build_args": args,
    }


# -- Dockerfile ----------------------------------------------------------------


def instructions(text: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, arguments) of a Dockerfile, continuations joined, comments dropped."""

    out: list[tuple[str, str]] = []
    current = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not current and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if current and line.lstrip().startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        current += line
        keyword, _, rest = current.strip().partition(" ")
        out.append((keyword.upper(), rest.strip()))
        current = ""
    if current.strip():
        keyword, _, rest = current.strip().partition(" ")
        out.append((keyword.upper(), rest.strip()))
    return out


def substitute(text: str, values: dict[str, str]) -> str:
    def one(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return values.get(name, "")

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", one, text)


def pairs(text: str) -> dict[str, str]:
    """`K=V K2="V 2"` (ENV/LABEL/ARG syntax), values unquoted."""

    out = {}
    for token in shlex.split(text):
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {token!r}")
        out[key] = value
    return out


# -- .dockerignore --------------------------------------------------------------


class Ignore:
    def __init__(self, root: Path):
        self.rules: list[tuple[bool, str]] = []
        path = root / ".dockerignore"
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                negate = line.startswith("!")
                pattern = line[1:] if negate else line
                self.rules.append((negate, pattern.strip("/")))

    @staticmethod
    def _match(pattern: str, relative: str) -> bool:
        if pattern.startswith("**/"):
            tail = pattern[3:]
            parts = relative.split("/")
            return any(fnmatch.fnmatchcase("/".join(parts[i:]), tail) for i in range(len(parts)))
        return fnmatch.fnmatchcase(relative, pattern)

    def ignored(self, relative: str) -> bool:
        # A path is excluded if it or a parent matches the last matching rule.
        parts = relative.split("/")
        result = False
        for negate, pattern in self.rules:
            for i in range(1, len(parts) + 1):
                if self._match(pattern, "/".join(parts[:i])):
                    result = not negate
                    break
        return result


def copy_tree(source: Path, dest: Path, relative: str, ignore: Ignore | None) -> None:
    if source.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        child = f"{relative}/{entry.name}" if relative else entry.name
        if ignore is not None and ignore.ignored(child):
            continue
        if entry.is_symlink():
            target = dest / entry.name
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(os.readlink(entry), target)
        elif entry.is_dir():
            copy_tree(entry, dest / entry.name, child, ignore)
        else:
            shutil.copy2(entry, dest / entry.name)


# -- build (compute node, stdlib only) ------------------------------------------


def run(command: list[str], **kwargs) -> None:
    print(
        "+ "
        + " ".join(shlex.quote(c) for c in command[:12])
        + (" ..." if len(command) > 12 else ""),
        flush=True,
    )
    subprocess.run(command, check=True, **kwargs)


class Stage:
    def __init__(self, name: str, container: str, base: Path, data: Path):
        self.name = name
        self.container = container
        self.rootfs = data / container
        self.env: dict[str, str] = {}
        self.args: dict[str, str] = {}
        self.workdir = "/"
        self.labels: dict[str, str] = {}
        if self.rootfs.exists():
            run(["enroot", "remove", "-f", container])
        run(["enroot", "create", "-n", container, str(base)])
        environment = self.rootfs / "etc" / "environment"
        if environment.exists():
            for line in environment.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep:
                    self.env[key.strip()] = value.strip().strip('"')
        # apt drops to _apt for downloads, which a user namespace cannot do.
        sandbox = self.rootfs / "etc" / "apt" / "apt.conf.d" / "00vibesim-enroot-build"
        if sandbox.parent.exists():
            sandbox.write_text('APT::Sandbox::User "root";\n')

    def path(self, inside: str) -> Path:
        if not inside.startswith("/"):
            inside = f"{self.workdir.rstrip('/')}/{inside}"
        return self.rootfs / inside.lstrip("/")

    def values(self) -> dict[str, str]:
        return {**self.env, **self.args}

    def shell(self, script: str) -> None:
        command = ["enroot", "start", "--root", "--rw"]
        for key, value in self.values().items():
            command += ["--env", f"{key}={value}"]
        command += [self.container, "/bin/sh", "-c", f"cd {shlex.quote(self.workdir)} && {script}"]
        started = time.monotonic()
        print(f"[{self.name}] RUN {script[:160]}", flush=True)
        subprocess.run(command, check=True)
        print(f"[{self.name}]   done in {time.monotonic() - started:.0f} s", flush=True)

    def finish(self) -> None:
        (self.rootfs / "etc" / "apt" / "apt.conf.d" / "00vibesim-enroot-build").unlink(
            missing_ok=True
        )
        lines = [f'{key}="{value}"' for key, value in self.env.items()]
        (self.rootfs / "etc" / "environment").write_text("\n".join(lines) + "\n")
        (self.rootfs / "etc" / "vibesim-image-labels.json").write_text(
            json.dumps(self.labels, indent=1, sort_keys=True) + "\n"
        )


def build(plan_path: Path, base: Path, out_dir: Path, keep: bool) -> Path:
    spec = json.loads(plan_path.read_text())
    root = Path(spec["root"])
    build_args: dict[str, str] = spec["build_args"]
    data = Path(os.environ.get("ENROOT_DATA_PATH", "/nonexistent"))
    if not data.is_dir():
        raise SystemExit("set ENROOT_DATA_PATH to enroot's container directory")
    ignore = Ignore(root)
    stages: dict[str, Stage] = {}
    order: list[Stage] = []
    global_args: dict[str, str] = {}
    stage: Stage | None = None
    suffix = spec["tag"].rpartition(":")[2]
    for keyword, rest in instructions((root / spec["dockerfile"]).read_text()):
        if keyword == "ARG":
            key, sep, default = rest.partition("=")
            value = build_args.get(key, default if sep else build_args.get(key, ""))
            (stage.args if stage is not None else global_args)[key] = value
        elif keyword == "FROM":
            words = rest.split()
            reference = substitute(words[0], global_args)
            if reference != build_args.get("BASE_IMAGE"):
                raise SystemExit(
                    f"FROM {reference}: only the plan's BASE_IMAGE is available as --base"
                )
            alias = (
                words[2] if len(words) == 3 and words[1].upper() == "AS" else f"stage{len(order)}"
            )
            stage = Stage(alias, f"vibesim-build-{suffix}-{alias}", base, data)
            stages[alias] = stage
            order.append(stage)
        elif stage is None:
            raise SystemExit(f"{keyword} before FROM")
        elif keyword == "RUN":
            stage.shell(rest)
        elif keyword == "ENV":
            stage.env.update({k: substitute(v, stage.values()) for k, v in pairs(rest).items()})
        elif keyword == "WORKDIR":
            stage.workdir = substitute(rest, stage.values())
            stage.path(stage.workdir).mkdir(parents=True, exist_ok=True)
        elif keyword == "LABEL":
            stage.labels.update({k: substitute(v, stage.values()) for k, v in pairs(rest).items()})
        elif keyword == "COPY":
            words = shlex.split(substitute(rest, stage.values()))
            source_stage = None
            while words and words[0].startswith("--"):
                flag = words.pop(0)
                if flag.startswith("--from="):
                    source_stage = stages[flag.removeprefix("--from=")]
                else:
                    raise SystemExit(f"COPY {flag}: unsupported")
            *sources, dest = words
            for source in sources:
                if source_stage is not None:
                    src = source_stage.path(source)
                    use_ignore = None
                    relative = source
                else:
                    src = root / source
                    use_ignore = ignore
                    relative = source.strip("/")
                target = stage.path(dest)
                if dest.endswith("/") or (len(sources) > 1):
                    target = target / src.name
                print(f"[{stage.name}] COPY {source} -> {dest}", flush=True)
                copy_tree(src, target, relative, use_ignore)
        elif keyword in ("ENTRYPOINT", "CMD", "SHELL", "USER", "EXPOSE", "VOLUME"):
            if keyword in ("SHELL", "USER"):
                raise SystemExit(f"{keyword}: unsupported")
        else:
            raise SystemExit(f"{keyword}: unsupported")
    assert stage is not None
    stage.finish()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (spec["tag"].replace(":", "+") + ".sqsh")
    partial = out.with_suffix(".sqsh.partial")
    partial.unlink(missing_ok=True)
    run(["enroot", "export", "-f", "-o", str(partial), stage.container])
    partial.rename(out)
    (out.with_suffix(".json")).write_text(
        json.dumps({**spec, "squashfs": str(out), "base_squashfs": str(base)}, indent=1) + "\n"
    )
    if not keep:
        for s in order:
            run(["enroot", "remove", "-f", s.container])
    print(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="enroot_build")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("plan")
    p.add_argument("image")
    p.add_argument(
        "--platform", default=None, help="linux/arm64 or linux/amd64 (default: this host's)"
    )
    p.add_argument("--out", type=Path, required=True)
    b = commands.add_parser("build")
    b.add_argument("plan", type=Path)
    b.add_argument("--base", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--keep", action="store_true", help="keep the stages' containers")
    args = parser.parse_args(argv)
    if args.command == "plan":
        args.out.write_text(json.dumps(plan(args.image, args.platform), indent=1) + "\n")
        print(json.loads(args.out.read_text())["tag"])
    else:
        build(args.plan, args.base, args.out_dir, args.keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
