"""The public sim presets: every deployment the public site can simulate."""

from __future__ import annotations

import json

import pytest

from launcher.schema.loader import schema_from_dict
from profiling.perf_api import DB_PATH
from public_api import preset as public_preset
from public_api import sim_preset, simulate
from public_api.deployments import DeploymentIndex
from public_api.sources import Sources

PRESETS = sim_preset.sim_preset_paths()


def _ids(paths):
    return [f"{path.parent.name}/{path.stem}" for path in paths]


def test_there_are_sim_presets():
    assert PRESETS


@pytest.mark.parametrize("path", PRESETS, ids=_ids(PRESETS))
def test_a_sim_preset_loads_and_expands(path):
    members = sim_preset.members(sim_preset.load(path))

    assert members
    keys = [json.dumps(member["pools"], sort_keys=True) for member in members]
    assert len(set(keys)) == len(keys), "two members are the same deployment"
    labels = [json.dumps(member["labels"], sort_keys=True) for member in members]
    assert len(set(labels)) == len(labels), "two members have the same axis values"
    for member in members:
        for pool in member["pools"].values():
            assert isinstance(pool["replicas"], int) and pool["replicas"] >= 1


@pytest.fixture(scope="module")
def sims(sim_bin) -> sim_preset.SimIndex:
    """Every sim preset, bound to the arch presets it references and built
    through the simulator's ``dry-run``, as the service checks it at start."""
    sources = Sources(db_path=DB_PATH)
    arch_paths = sorted(
        {
            public_preset.PRESET_ROOT / path.parent.name / f"{group['arch']['preset']}.yaml"
            for path in PRESETS
            for pool in sim_preset.load(path)["pools"].values()
            for group in pool["groups"]
        }
    )
    index = DeploymentIndex.build(
        sources.cost_trees, sources.list_params(), sim_commit=None, paths=arch_paths
    )
    registry = schema_from_dict(sources.list_params())
    sims = sim_preset.SimIndex.build(index)
    sims.check(lambda member, capture: simulate.missing_rows(index, member, capture, registry))
    return sims


@pytest.mark.needs_binary
def test_an_moe_member_replays_its_captures_never_a_synthetic_routing(sims):
    """A member's captures are its arch preset's capture rows (a dense arch,
    which routes nothing, takes every published trace); uniform is never one."""
    for preset in sims.presets.values():
        assert preset.captures, preset.id
        for capture in preset.captures:
            assert capture.trace.startswith("hf://") and capture.trace.endswith("/trace.csv")
            if capture.arch_row is not None:
                assert capture.routing in ("popularity", "corpus"), (preset.id, capture)


@pytest.mark.needs_binary
@pytest.mark.needs_db
def test_every_sim_member_builds_and_is_measured(sims):
    """Each member, on each capture it offers, builds through the deployment's
    ``build_flow`` and finds every profile.db row in the shipped profile.db:
    the site offers only simulations it can run from measurements."""
    unavailable = [
        (preset.id, member.params, member.summary(sims.index)["unavailable"])
        for preset in sims.presets.values()
        for member in preset.members
        if member.summary(sims.index)["unavailable"]
    ]
    assert not unavailable
