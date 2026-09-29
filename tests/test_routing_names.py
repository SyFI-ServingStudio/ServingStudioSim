"""`presets/alignment/routings.yaml` names every measured routing a preset or run uses.

A reader picks a routing by its name (``public_api.kernel.demand``), so a
registered run whose routing the table does not name would show a path, or
nothing. The table names content: a corpus by its payload checksum, a
popularity file by its sha256.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path

import yaml

from launcher.corpus import CorpusError, resolve_reference
from profiling.db import storage
from public_api.kernel import demand

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = REPO_ROOT / "profiling" / "profile.db"


def test_the_table_loads_and_its_files_are_what_it_names() -> None:
    names = demand.load_names()
    assert names and set(names.values()) <= {"enwik9", "enwik9_short"}
    table = yaml.safe_load(demand.NAMES.read_text())
    for entry in table["popularity"]:
        path = REPO_ROOT / entry["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["file"]
    for entry in table["corpus"]:
        assert entry["file"].startswith("hf://"), entry["file"]
        # The checksum is the manifest's, when this machine's hub cache has it.
        try:
            manifest = Path(resolve_reference(entry["file"], local_only=True))
        except CorpusError:
            continue
        checksum = json.loads(manifest.read_text())["checksum_fnv1a64"]
        assert f"{checksum:016x}" == entry["checksum_fnv1a64"], entry["file"]


# A routing file field and its value, in a YAML or JSON preset.
_REFERENCE = re.compile(
    r"""["']?(expert_popularity_file|token_corpus_file)["']?\s*:\s*["']?([^"',\s}]+)"""
)


def test_every_routing_a_tracked_preset_names_is_named() -> None:
    """Before any run registers it: a preset that names a routing file names
    one the table names. A relative path resolves against the repository, or
    against the preset's own directory (an alignment pack's files)."""

    names = demand.load_names()
    table = yaml.safe_load(demand.NAMES.read_text())
    hub = {entry["file"] for entry in table["corpus"]}
    listed = subprocess.run(
        ["git", "ls-files", "-z", "presets"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout
    presets = [REPO_ROOT / p for p in listed.split("\0") if p.endswith((".yaml", ".json"))]
    seen, unnamed = 0, []
    for preset in presets:
        for field, value in _REFERENCE.findall(preset.read_text()):
            seen += 1
            where = f"{preset.relative_to(REPO_ROOT)}: {value}"
            if value.startswith("hf://"):
                if value not in hub:
                    unnamed.append(where)
                continue
            candidates = [REPO_ROOT / value, preset.parent / value]
            local = next((c for c in candidates if c.is_file()), None)
            assert local is not None, f"{where}: no such file"
            if demand.file_key(str(local.resolve())) not in names:
                unnamed.append(where)
    assert seen, "no tracked preset names a routing file"
    assert not unnamed, f"routings.yaml does not name {unnamed}"


def _repo_copies() -> dict[str, str]:
    """``{path a run recorded: repo path}`` from the provenance sidecars, as
    the public API maps a run registered on another machine."""

    out = {}
    for sidecar in (REPO_ROOT / "presets").rglob("*.provenance.json"):
        meta = json.loads(sidecar.read_text())
        target = sidecar.with_name(sidecar.name.removesuffix(".provenance.json") + ".json")
        out.update({path: str(target.relative_to(REPO_ROOT)) for path in meta["copied_from"]})
    return out


def _arch_blocks(value):
    if isinstance(value, dict):
        if "type" in value and "routing" in value:
            yield value
        for child in value.values():
            yield from _arch_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _arch_blocks(child)


def test_every_registered_measured_routing_is_named() -> None:
    names = demand.load_names()
    copies = _repo_copies()
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        sources = [
            json.loads(s) for (s,) in conn.execute("select source from _kernel_config_source")
        ]
        blobs = storage.load_blobs(conn)
        identities = [
            storage.unpack_identity(text, blobs)
            for (text,) in conn.execute("select identity from _kernel_config")
        ]
    finally:
        conn.close()

    unnamed = set()
    files = {
        block["expert_popularity_file"]
        for source in sources
        if "supported" not in source
        for block in _arch_blocks(source)
        if block["routing"] == "popularity"
    }
    assert files, "profile.db registers no popularity run"
    for path in files:
        local = copies.get(path, path)
        key = demand.file_key(local)
        assert key is not None, f"{path}: no copy in this repository"
        if key not in names:
            unnamed.add(local)

    corpora = {
        f"fnv1a64:{identity['expert_demand']['corpus']['checksum_fnv1a64']:016x}"
        for identity in identities
        if "corpus" in identity.get("expert_demand", {})
    }
    assert corpora, "profile.db registers no corpus config"
    unnamed |= corpora - set(names)
    assert not unnamed, f"routings.yaml does not name {sorted(unnamed)}"
