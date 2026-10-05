"""Public presets: the deployments the public site shows.

A public preset (``presets/public/<checkpoint>/<arch type>.yaml``, by the
checkpoint's Hugging Face repo name) is one arch type on one checkpoint, with
every supported value of its parameters in the launcher's sweep language
(``sweep`` / ``compound`` / ``derived`` / ``constraints``, expanded by
``launcher.schema.expand``):

    checkpoint: Qwen/Qwen3-235B-A22B-FP8   # a key of model/catalog.yaml
    gpu: NVIDIA H200
    arch:                                  # model_config comes from the catalog
      type: qwen3_moe_fp8_dp_attn_ep_ffn
      ep_size: ${ep_size}
      ...
    sweep:
      ep_size: [4, 8]

Each expanded member is one ``{gpu, arch}`` block, the input of
``simulator cost-trees``. A preset holds no cases: a prediction's cases come
from whoever asks for it.

MoE routing is either written in ``arch`` (``routing: uniform``) or swept as a
``compound`` group named ``workload``: each row is a workload the preset
supports, binding ``routing`` and the capture it reads (a full
``hf://datasets/...`` reference). The first row is the default.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from launcher.schema.expand import expand_sweep_params, normalize_arch
from launcher.schema.loader import Registry

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOT = REPO_ROOT / "presets" / "public"
MODEL_CATALOG = REPO_ROOT / "model" / "catalog.yaml"
# The reader-facing name of each arch type, `{type: {name}}`.
ARCH_CATALOG = REPO_ROOT / "model" / "arch_catalog.yaml"
WORKLOAD = "workload"

# The launcher's sweep language a preset may use (`launcher.schema.expand`).
CONTROL = ("sweep", "compound", "derived", "constraints")


class PresetError(ValueError):
    """A public preset that does not describe a deployment."""


def preset_paths(root: Path = PRESET_ROOT) -> list[Path]:
    return sorted(root.glob("*/*.yaml"))


def load(path: Path, catalog: dict | None = None) -> dict:
    """Read one preset and check its fields and its checkpoint."""
    preset = yaml.safe_load(path.read_text())
    if not isinstance(preset, dict):
        raise PresetError(f"{path}: not a mapping")
    allowed = {"checkpoint", "gpu", "arch", *CONTROL}
    unknown = sorted(set(preset) - allowed)
    if unknown:
        raise PresetError(f"{path}: unknown keys {unknown}")
    arch = preset.get("arch")
    if not isinstance(arch, dict) or not arch.get("type"):
        raise PresetError(f"{path}: arch needs a type")
    if "model_config" in arch:
        raise PresetError(f"{path}: model_config comes from the checkpoint's catalog entry")
    if path.stem != arch["type"]:
        raise PresetError(f"{path}: name it after its arch type, {arch['type']}")
    catalog = yaml.safe_load(MODEL_CATALOG.read_text()) if catalog is None else catalog
    entry = catalog.get(preset.get("checkpoint"))
    if entry is None:
        raise PresetError(f"{path}: checkpoint {preset.get('checkpoint')!r} is not in the catalog")
    if path.parent.name != preset["checkpoint"].split("/")[-1]:
        raise PresetError(
            f"{path}: put it under its checkpoint's directory ({preset['checkpoint']})"
        )
    return preset | {"_config": entry["config"]}


def members(preset: dict, registry: Registry | None = None) -> list[dict]:
    """Every deployment the preset supports, as ``{gpu, arch, labels}``.

    ``labels`` names the member by its swept values (a compound group by its
    row label), so ``workload`` is the workload's name. Given the schema
    (``simulator list-params``), each arch is complete, as the launcher writes
    a run's: every param typed and each one the preset leaves out at its
    default, so ``cost-trees`` and ``timing-predict`` read the same block.
    """
    tree = {key: preset[key] for key in ("gpu", "arch")}
    tree["arch"] = {
        **tree["arch"],
        "model_config": str(REPO_ROOT / "model" / "config" / f"{preset['_config']}.json"),
    }
    return [
        {
            "gpu": candidate["gpu"],
            "arch": normalize_arch(candidate["arch"], registry) if registry else candidate["arch"],
            "labels": labels,
        }
        for candidate, labels in expand(preset, tree)
    ]


def expand(preset: dict, tree: dict) -> list[tuple[dict, dict]]:
    """``tree`` expanded by ``preset``'s sweep language (:data:`CONTROL`), each
    member with its labels: its swept values, a compound group by its row label."""
    control = {key: preset[key] for key in CONTROL if key in preset}
    swept = [*preset.get("sweep", {}), *preset.get("compound", {})]
    out = []
    for candidate in expand_sweep_params(tree | control, registry=None):
        env, labels = candidate["_env"], candidate.get("_sweep_labels", {})
        out.append((candidate, {name: labels.get(name, env[name]) for name in swept}))
    return out
