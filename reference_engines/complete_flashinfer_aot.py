"""Add FlashInfer modules the installed jit-cache wheel lacks to its AOT providers.

Run at image build, on the engine's interpreter:

    python complete_flashinfer_aot.py PROVIDER[,PROVIDER...] MODULE=PACKAGE.MODULE:GENERATOR ...

e.g. `sm100a,sm103a tinygemm2=flashinfer.gemm.routergemm:gen_tinygemm2_module`.

An engine that runs with FLASHINFER_DISABLE_JIT=1 can only load modules the
jit-cache providers ship; a module the wheel was not built with raises
MissingJITCacheError at first use. For each provider named, this compiles the
module for that provider's CUDA architecture (no GPU needed), installs the
library at `<provider>/jit_cache/<module>/<module>.so`, and lists the module in
the provider's manifest, so FlashInfer resolves it per architecture exactly as
it resolves the modules the wheel ships. A module a provider already has is
left alone.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _providers() -> dict[str, Path]:
    import flashinfer_jit_cache

    return {
        provider.provider_id: Path(provider.jit_cache_dir)
        for provider in flashinfer_jit_cache.get_jit_cache_providers()
    }


def _arch_list(provider_id: str) -> str:
    """`sm100a` -> `10.0a`, `sm120f` -> `12.0f` (FLASHINFER_CUDA_ARCH_LIST spelling)."""

    digits, suffix = provider_id.removeprefix("sm"), ""
    if digits[-1].isalpha():
        digits, suffix = digits[:-1], digits[-1]
    return f"{digits[:-1]}.{digits[-1]}{suffix}"


_RESOLVE = (
    "import importlib, pathlib, sys\n"
    "path, name = sys.argv[1].split(':')\n"
    "spec = getattr(importlib.import_module(path), name)()\n"
    "assert spec.is_aot and spec.aot_path == pathlib.Path(sys.argv[2]), spec.aot_path\n"
)


def _build(module: str, generator: str, arch: str) -> Path:
    """Compile ``module`` for ``arch`` in a fresh process and return its library."""

    workspace = Path(tempfile.mkdtemp(prefix=f"flashinfer-aot-{module}-"))
    code = (
        "import importlib, sys\n"
        "path, name = sys.argv[1].split(':')\n"
        "spec = getattr(importlib.import_module(path), name)()\n"
        "spec.build(verbose=False)\n"
        "print(spec.get_library_path())\n"
    )
    env = dict(os.environ, FLASHINFER_CUDA_ARCH_LIST=arch, FLASHINFER_WORKSPACE_BASE=str(workspace))
    env.pop("FLASHINFER_DISABLE_JIT", None)
    result = subprocess.run(
        [sys.executable, "-c", code, generator],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    library = Path(result.stdout.strip().splitlines()[-1])
    if not library.is_file():
        raise SystemExit(f"{module} for {arch}: no library at {library}")
    return library


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    wanted = argv[0].split(",")
    modules = dict(item.split("=", 1) for item in argv[1:])
    providers = _providers()
    missing = [provider for provider in wanted if provider not in providers]
    if missing:
        raise SystemExit(f"no jit-cache provider {missing} (installed: {sorted(providers)})")
    for provider_id in wanted:
        jit_cache = providers[provider_id]
        manifest_path = jit_cache.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for module, generator in modules.items():
            if module in manifest["modules"]:
                print(f"{provider_id}: {module} already present")
                continue
            library = _build(module, generator, _arch_list(provider_id))
            target = jit_cache / module / f"{module}.so"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(library, target)
            manifest["modules"].append(module)
            print(f"{provider_id}: added {module} ({target.stat().st_size} bytes)")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # In a fresh process, as a worker would: each module resolves to the
    # provider's copy for that provider's architecture, with JIT disabled.
    for provider_id in wanted:
        for module, generator in modules.items():
            target = providers[provider_id] / module / f"{module}.so"
            env = dict(
                os.environ,
                FLASHINFER_CUDA_ARCH_LIST=_arch_list(provider_id),
                FLASHINFER_DISABLE_JIT="1",
            )
            subprocess.run(
                [sys.executable, "-c", _RESOLVE, generator, str(target)], env=env, check=True
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
