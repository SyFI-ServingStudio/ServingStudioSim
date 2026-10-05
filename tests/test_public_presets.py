"""The public presets: every deployment the public site shows."""

from __future__ import annotations

import json
import re
import subprocess

import pytest
import yaml

from launcher.corpus import ROUTING_FILES
from profiling.perf_api import DB_PATH
from public_api import predict
from public_api import preset as public_preset
from public_api.deployments import DeploymentIndex
from public_api.sources import Sources

PRESETS = public_preset.preset_paths()
CAPTURE = re.compile(
    r"^hf://datasets/UW-SyFI/servingstudio-workload@[0-9a-f]{40}/"
    r"[a-z0-9_]+/[a-z0-9_]+/[a-z0-9_]+/capture/\d{8}/(popularity|manifest)\.json$"
)
SYNTHETIC = ("uniform", "random")


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
    """A workload is a capture, or a synthetic routing named after itself; a
    synthetic one is never the default (first) row unless there is no capture."""
    preset = public_preset.load(path)
    rows = preset.get("compound", {}).get(public_preset.WORKLOAD, {})
    assert set(preset.get("compound", {})) <= {public_preset.WORKLOAD}
    captured = [name for name, row in rows.items() if row["routing"] not in SYNTHETIC]
    if captured:
        assert next(iter(rows)) == captured[0], "a synthetic workload is the default"

    for name, row in rows.items():
        assert re.fullmatch(r"[a-z0-9_]+", name)
        captures = {field: row.get(field) for field in ROUTING_FILES.values() if row.get(field)}
        if row["routing"] in SYNTHETIC:
            assert name == row["routing"] and not captures, name
            continue
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
                assert not any(arch.get(field) for field in ROUTING_FILES.values()), path


@pytest.mark.needs_binary
def test_every_named_arch_is_one_the_simulator_has(sim_bin):
    """model/arch_catalog.yaml names real arch types, each with a name; the
    index refuses a preset whose arch has none."""
    params = json.loads(
        subprocess.run([sim_bin, "list-params"], capture_output=True, text=True, check=True).stdout
    )
    tags = {tag for tags in params["providers"]["arch"].values() for tag in tags}
    catalog = yaml.safe_load(public_preset.ARCH_CATALOG.read_text())
    assert set(catalog) <= tags
    assert all(isinstance(entry.get("name"), str) and entry["name"] for entry in catalog.values())
    used = {public_preset.load(path)["arch"]["type"] for path in PRESETS}
    assert used <= set(catalog), used - set(catalog)


@pytest.fixture(scope="module")
def index(sim_bin) -> DeploymentIndex:
    sources = Sources(db_path=DB_PATH)
    index = DeploymentIndex.build(sources.cost_trees, sources.list_params(), sim_commit=None)
    index.check(lambda member: predict.missing_specs(member))
    return index


@pytest.mark.needs_binary
@pytest.mark.needs_db
def test_every_leaf_names_a_published_config(index):
    """Every leaf names a published config: the simulator matches each slot to
    the record its kernel reads, so the Kernels page and the tree name one."""
    for preset in index.presets.values():
        for member in preset.members:
            for section in member.sections:
                for slot in section["slots"]:
                    assert slot["config"] in index.configs, (preset.id, member.params, slot)


@pytest.mark.needs_binary
@pytest.mark.needs_db
def test_every_member_is_measured(index):
    """Each member, with its captures, builds and finds every profile.db row its
    kernels read in the shipped profile.db: the site offers only deployments it
    can predict from measurements."""
    unmeasured = [
        (preset.id, member.params, member.error or member.missing)
        for preset in index.presets.values()
        for member in preset.members
        if member.error or member.missing
    ]
    assert not unmeasured
