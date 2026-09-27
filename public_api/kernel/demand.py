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
  (``launcher/alignment_campaign/check.py``). A file this checkout tracks is named
  by its repo-relative path, a file in the local Hugging Face hub cache by the
  ``hf://`` reference that fetches it, any other file by its file name and the
  fingerprint of the demand it produced.

Every name carries that fingerprint: the corpus manifest's payload checksum, or
a hash of the popularity table the config folds. ``binding`` holds how the config
reads the artifact (a corpus's verify width and layer slice, a table's layer
count), which tells apart configs of one routing built for different layers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from launcher.alignment_campaign.check import ROUTING_ARTIFACTS
from profiling.db.kernel_config import content_hash
from public_api.kernel.sources import REPO_ROOT

#: The config field this module names, and the arch param naming its routing.
FIELD = "expert_demand"
PARAM = "routing"

#: Measured routings first, a token corpus before a popularity marginal, as
#: ``skills/operate-run-simulation/references/moe-routing.md`` prefers them; a
#: synthetic routing is never the page's default pick.
PREFERENCE = ("corpus", "popularity")


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
            if isinstance(path, str) and path and (relative := _repo_relative(path)):
                out.append(relative)
    return out


def names_local_files(archs: Iterable[dict]) -> bool:
    """Whether an arch block names an artifact outside the checkout (a hub
    cache file, say)."""

    return any(
        isinstance(path := arch.get(key), str) and path and _repo_relative(path) is None
        for arch in archs
        for key, _ in ROUTING_ARTIFACTS.values()
    )


def _repo_relative(path: str) -> str | None:
    local = Path(path)
    if local.is_absolute():
        if not local.is_relative_to(REPO_ROOT):
            return None
        local = local.relative_to(REPO_ROOT)
    return local.as_posix()


def _artifact_name(
    path: str, fingerprint: str, tracked: set[str], hub: dict[str, str]
) -> tuple[str, str | None]:
    """``(label, reference)`` of one artifact path as an arch block gave it
    (repo-relative, or absolute). The reference is what a reader fetches or
    opens it by; a file only the building machine had has none."""

    if path in hub:
        return hub[path], hub[path]
    relative = _repo_relative(path)
    if relative in tracked:
        return relative, relative
    return f"{Path(path).name} · {fingerprint}", None


def demand_name(
    demand: dict,
    archs: Iterable[dict],
    routing_default: Callable[[str | None], str | None],
    tracked: set[str],
    hub: dict[str, str],
) -> dict | None:
    """The name of one config's ``expert_demand`` from the arch blocks that
    built it, or None when none names a routing (a deployment-level source).

    ``routing_default(arch_tag)`` is the arch's ``routing`` default; ``tracked``
    holds the :func:`artifact_paths` git tracks, and ``hub`` maps hub-cache
    files to their references (``KernelSources.hub_references``). Arch blocks
    that name the same demand differently (two files of equal content) are
    listed in ``label`` in turn."""

    fingerprint = _fingerprint(demand)
    names: dict[tuple[str, str], str | None] = {}
    for arch in archs:
        routing = arch.get(PARAM, routing_default(arch.get("type")))
        if routing is None:
            continue
        key = ROUTING_ARTIFACTS.get(routing, (None,))[0]
        path = arch.get(key) if key else None
        if isinstance(path, str) and path:
            label, reference = _artifact_name(path, fingerprint, tracked, hub)
        else:
            seed = arch.get("routing_seed")
            label = routing if seed is None else f"{routing}, seed {seed}"
            reference = None
        names.setdefault((routing, label), reference)
    if not names:
        return None
    ordered = sorted(names, key=lambda n: (_preference(n[0]), n[1]))
    routing = ordered[0][0]
    return {
        "routing": routing,
        "label": " | ".join(label for _, label in ordered),
        "reference": names[ordered[0]],
        "fingerprint": fingerprint,
        "binding": _binding(demand),
        "preference": _preference(routing),
    }


def _preference(routing: str) -> int:
    return PREFERENCE.index(routing) if routing in PREFERENCE else len(PREFERENCE)
