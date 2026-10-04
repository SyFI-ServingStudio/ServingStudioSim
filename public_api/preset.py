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

from launcher.schema.expand import expand_sweep_params

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOT = REPO_ROOT / "presets" / "public"
MODEL_CATALOG = REPO_ROOT / "model" / "catalog.yaml"
WORKLOAD = "workload"

_CONTROL = ("sweep", "compound", "derived", "constraints")


class PresetError(ValueError):
    """A public preset that does not describe a deployment."""


def preset_paths(root: Path = PRESET_ROOT) -> list[Path]:
    return sorted(root.glob("*/*.yaml"))


def load(path: Path, catalog: dict | None = None) -> dict:
    """Read one preset and check its fields and its checkpoint."""
    preset = yaml.safe_load(path.read_text())
    if not isinstance(preset, dict):
        raise PresetError(f"{path}: not a mapping")
    allowed = {"checkpoint", "gpu", "arch", *_CONTROL}
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


def members(preset: dict) -> list[dict]:
    """Every deployment the preset supports, as ``{gpu, arch, labels}``.

    ``labels`` names the member by its swept values (a compound group by its
    row label), so ``workload`` is the workload's name.
    """
    tree = {key: preset[key] for key in ("gpu", "arch")}
    tree["arch"] = {
        **tree["arch"],
        "model_config": str(REPO_ROOT / "model" / "config" / f"{preset['_config']}.json"),
    }
    control = {key: preset[key] for key in _CONTROL if key in preset}
    out = []
    for candidate in expand_sweep_params(tree | control, registry=None):
        env = candidate["_env"]
        labels = candidate.get("_sweep_labels", {})
        swept = [*preset.get("sweep", {}), *preset.get("compound", {})]
        out.append(
            {
                "gpu": candidate["gpu"],
                "arch": candidate["arch"],
                "labels": {name: labels.get(name, env[name]) for name in swept},
            }
        )
    return out
