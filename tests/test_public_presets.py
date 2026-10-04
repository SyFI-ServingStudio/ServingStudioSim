"""The public presets: every deployment the public site shows."""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from public_api import preset as public_preset

PRESETS = public_preset.preset_paths()
CAPTURE = re.compile(
    r"^hf://datasets/UW-SyFI/servingstudio-workload@[0-9a-f]{40}/"
    r"[a-z0-9_]+/[a-z0-9_]+/[a-z0-9_]+/capture/\d{8}/(popularity|manifest)\.json$"
)
CAPTURE_FIELDS = ("expert_popularity_file", "token_corpus_file")


def _ids(paths):
    return [f"{path.parent.name}/{path.stem}" for path in paths]


def _routing_tags(sim_bin) -> set[str]:
    """Arch types that take a `routing` param: the MoE ones."""
    params = json.loads(
        subprocess.run([sim_bin, "list-params"], capture_output=True, text=True, check=True).stdout
    )
    return {
        tag
        for tags in params["providers"]["arch"].values()
        for tag, provider in tags.items()
        if any(param["name"] == "routing" for param in provider["params"])
    }


def test_there_are_public_presets():
    assert PRESETS


@pytest.mark.parametrize("path", PRESETS, ids=_ids(PRESETS))
def test_a_public_preset_loads_and_expands(path):
    members = public_preset.members(public_preset.load(path))

    assert members
    keys = [json.dumps(member["arch"], sort_keys=True) for member in members]
    assert len(set(keys)) == len(keys), "two members are the same deployment"


@pytest.mark.parametrize("path", PRESETS, ids=_ids(PRESETS))
def test_a_workload_names_its_routing_and_a_pinned_capture(path):
    preset = public_preset.load(path)
    rows = preset.get("compound", {}).get(public_preset.WORKLOAD, {})
    assert set(preset.get("compound", {})) <= {public_preset.WORKLOAD}

    for name, row in rows.items():
        assert re.fullmatch(r"[a-z0-9_]+", name)
        captures = {field: row.get(field) for field in CAPTURE_FIELDS if row.get(field)}
        assert row["routing"] in ("popularity", "corpus"), name
        expected = (
            "expert_popularity_file" if row["routing"] == "popularity" else "token_corpus_file"
        )
        assert list(captures) == [expected], name
        assert CAPTURE.match(captures[expected]), captures[expected]


@pytest.mark.needs_binary
def test_every_moe_member_states_its_routing(sim_bin):
    moe = _routing_tags(sim_bin)
    for path in PRESETS:
        for member in public_preset.members(public_preset.load(path)):
            arch = member["arch"]
            if arch["type"] not in moe:
                continue
            # There is no default routing: a capture, or uniform/random written out.
            assert arch.get("routing") in ("uniform", "random", "popularity", "corpus"), path
            if arch["routing"] in ("uniform", "random"):
                assert not any(arch.get(field) for field in CAPTURE_FIELDS), path


@pytest.mark.needs_binary
def test_every_member_builds(sim_bin):
    """Structure only: routing reads its capture at run time, not here."""
    blocks, names = [], []
    for path in PRESETS:
        for member in public_preset.members(public_preset.load(path)):
            arch = {k: v for k, v in member["arch"].items() if k not in CAPTURE_FIELDS}
            if "routing" in arch:
                arch["routing"] = "uniform"
            blocks.append({"gpu": member["gpu"], "arch": arch})
            names.append(f"{path.parent.name}/{path.stem} {member['labels']}")

    out = subprocess.run(
        [sim_bin, "cost-trees", "-"],
        input=json.dumps(blocks),
        capture_output=True,
        text=True,
        check=True,
    )
    errors = [
        (name, built["error"])
        for name, built in zip(names, json.loads(out.stdout), strict=True)
        if built.get("error")
    ]
    assert not errors
