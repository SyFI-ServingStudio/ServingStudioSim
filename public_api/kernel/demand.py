"""A reader's name for a kernel config's routing: its ``expert_demand``.

A fused-MoE kernel config carries its routed demand as ``expert_demand`` (Rust
``ExpertDemand`` in ``simulator/src/timing/expert_demand.rs``): a per-layer
popularity table or a token-corpus binding. The value does not say where it came
from (a uniform table and a measured one are both ``popularity``), so the name
comes from the arch blocks that built the config, the kernel-config registry's
sources, which name the routing and its artifact:

- ``routing``, the arch param (its ``list-params`` default where a block leaves it
  out), names a synthetic routing (``uniform``, ``random``) by itself;
- a measured routing reads the arch field ``ROUTING_ARTIFACTS`` gives it
  (``launcher/alignment_campaign/check.py``). Its ``reference`` is the ``hf://``
  reference the preset wrote, which the launcher records in place of the file
  it fetched (``launcher/corpus.py``), or the repo-relative path of a file this
  checkout tracks; any other file has none.

A measured routing's ``label`` is its name in ``presets/alignment/routings.yaml``
(:func:`load_names`), which names the artifact by content: a corpus by its
payload checksum, a popularity file by its sha256. One the table does not name
has no label; a synthetic routing's label is its kind and seed. ``routing`` is
the category a reader sees first, ordered by ``preference``.

Every name carries a fingerprint: the corpus manifest's payload checksum, or
a hash of the popularity table the config folds. ``binding`` holds how the config
reads the artifact (a corpus's verify width and layer slice, a table's layer
count), which tells apart configs of one routing built for different layers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from launcher.alignment_campaign.check import ROUTING_ARTIFACTS
from launcher.corpus import HF_SCHEME
from profiling.db.kernel_config import content_hash
from public_api.kernel.sources import REPO_ROOT

#: The config field this module names, and the arch param naming its routing.
FIELD = "expert_demand"
PARAM = "routing"

#: The categories a reader sees, in order: measured routings first, a token
#: corpus before a popularity marginal, as
#: ``skills/operate-run-simulation/references/moe-routing.md`` prefers them;
#: then the synthetic ones, which are never the page's default pick.
PREFERENCE = ("corpus", "popularity", "random", "uniform")
#: The measured routings' names.
NAMES = REPO_ROOT / "presets" / "alignment" / "routings.yaml"


def load_names(path: Path | None = None) -> dict[str, str]:
    """``{key: name}`` from a routing-name table (default :data:`NAMES`):
    ``fnv1a64:<hex>`` for a corpus (the form :func:`_fingerprint` gives it),
    ``sha256:<hex>`` for a popularity file's bytes."""

    path = NAMES if path is None else path
    table = yaml.safe_load(path.read_text()) or {}
    if table.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version {table.get('schema_version')!r}")
    names = {}
    for kind, field in (("corpus", "checksum_fnv1a64"), ("popularity", "sha256")):
        for entry in table.get(kind) or ():
            prefix = "fnv1a64" if kind == "corpus" else "sha256"
            key = f"{prefix}:{str(entry[field]).lower()}"
            if key in names:
                raise ValueError(f"{path}: {key} is named twice")
            names[key] = entry["name"]
    return names


@lru_cache(maxsize=256)
def _file_sha256(path: str, mtime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_key(path: str) -> str | None:
    """``sha256:<hex>`` of the file an arch block names (absolute, or relative
    to the repository), or None when this machine does not have it."""

    local = Path(path)
    if not local.is_absolute():
        local = REPO_ROOT / local
    if not local.is_file():
        return None
    stat = local.stat()
    return f"sha256:{_file_sha256(str(local), stat.st_mtime_ns, stat.st_size)}"


def _fingerprint(demand: dict) -> str:
    """What the demand is, independent of where its file lives."""

    if "corpus" in demand:
        return f"fnv1a64:{demand['corpus']['checksum_fnv1a64']:016x}"
    return f"sha256:{content_hash(demand)[:12]}"


def _binding(demand: dict) -> dict[str, Any]:
    if "corpus" in demand:
        corpus = demand["corpus"]
        return {k: corpus[k] for k in ("group_size", "layer_start", "layer_end")}
    return {"layers": len(demand["popularity"]["layerwise_global_ppm"])}


def artifact_paths(archs: Iterable[dict]) -> list[str]:
    """The repo-relative form of every artifact path the arch blocks name, for
    asking git which of them it tracks."""

    out = []
    for arch in archs:
        for key, _ in ROUTING_ARTIFACTS.values():
            path = arch.get(key)
            if (
                isinstance(path, str)
                and path
                and not path.startswith(HF_SCHEME)
                and (relative := _repo_relative(path))
            ):
                out.append(relative)
    return out


def _repo_relative(path: str) -> str | None:
    local = Path(path)
    if local.is_absolute():
        if not local.is_relative_to(REPO_ROOT):
            return None
        local = local.relative_to(REPO_ROOT)
    return local.as_posix()


def _artifact_reference(path: str, tracked: set[str]) -> str | None:
    """What a reader fetches or opens an artifact by, as an arch block gave it
    (an ``hf://`` reference, a repo-relative path, or an absolute one): the
    reference, or the repo path git tracks. A file only the building machine
    had has none."""

    if path.startswith(HF_SCHEME):
        return path
    relative = _repo_relative(path)
    return relative if relative in tracked else None


def demand_name(
    demand: dict,
    archs: Iterable[dict],
    routing_default: Callable[[str | None], str | None],
    tracked: set[str],
    names: Mapping[str, str],
) -> dict | None:
    """The name of one config's ``expert_demand`` from the arch blocks that
    built it, or None when none names a routing (a deployment-level source).

    ``routing_default(arch_tag)`` is the arch's ``routing`` default; ``tracked``
    holds the :func:`artifact_paths` git tracks; ``names`` is
    :func:`load_names`. Arch blocks that name the same demand differently (two
    files of equal content) share a ``label``; their ``reference`` is the
    first."""

    fingerprint = _fingerprint(demand)
    found: dict[tuple[str, str], str | None] = {}
    for arch in archs:
        routing = arch.get(PARAM, routing_default(arch.get("type")))
        if routing is None:
            continue
        key = ROUTING_ARTIFACTS.get(routing, (None,))[0]
        path = arch.get(key) if key else None
        if isinstance(path, str) and path:
            content = fingerprint if routing == "corpus" else file_key(path)
            label = names.get(content) if content else None
            reference = _artifact_reference(path, tracked)
        else:
            seed = arch.get("routing_seed")
            label = routing if seed is None else f"{routing}, seed {seed}"
            reference = None
        found.setdefault((routing, label or ""), reference)
    if not found:
        return None
    ordered = sorted(found, key=lambda n: (_preference(n[0]), n[1]))
    routing = ordered[0][0]
    labels = [label for r, label in ordered if r == routing and label]
    return {
        "routing": routing,
        "label": " | ".join(dict.fromkeys(labels)) or None,
        "reference": found[ordered[0]],
        "fingerprint": fingerprint,
        "binding": _binding(demand),
        "preference": _preference(routing),
    }


def _preference(routing: str) -> int:
    return PREFERENCE.index(routing) if routing in PREFERENCE else len(PREFERENCE)
