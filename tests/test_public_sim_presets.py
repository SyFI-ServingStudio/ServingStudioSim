"""The public sim presets: every deployment the public site can simulate."""

from __future__ import annotations

import json
import os

import pytest

from launcher.schema.loader import schema_from_dict
from profiling.perf_api import DB_PATH
from public_api import preset as public_preset
from public_api import sim_preset, simulate, workloads
from public_api.deployments import DeploymentIndex
from public_api.sources import Sources

PRESETS = sim_preset.sim_preset_paths()

# The checks run one simulator process per member; use the host's cores.
JOBS = min(64, os.cpu_count() or 8)


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
def bound(sim_bin) -> tuple[sim_preset.SimIndex, object]:
    """Every sim preset, bound to the arch presets it references, and the
    simulator's schema."""
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
        sources.cost_trees, sources.list_params(), sim_commit=None, paths=arch_paths, jobs=JOBS
    )
    return sim_preset.SimIndex.build(index), schema_from_dict(sources.list_params())


@pytest.fixture(scope="module")
def sims(bound) -> sim_preset.SimIndex:
    """:func:`bound`'s presets, built through the simulator's ``dry-run``, as
    the service checks them at start."""
    sims, registry = bound
    sims.check(
        lambda member, capture, build: simulate.check_capture(
            sims.index, member, capture, registry, build=build
        ),
        jobs=JOBS,
    )
    return sims


@pytest.mark.presets
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


@pytest.mark.presets
@pytest.mark.needs_binary
@pytest.mark.needs_db
def test_every_sim_member_builds_and_is_measured(sims):
    """Each member, on each capture it offers, builds through the deployment's
    ``build_flow`` and finds every profile.db row in the shipped profile.db:
    the site offers only simulations it can run from measurements. A capture
    longer than a member's context is a misfit of its requests, not of the
    member (the test below)."""
    unavailable = [
        (preset.id, member.params, capture, reason)
        for preset in sims.presets.values()
        for member in preset.members
        for capture, reason in member.summary(sims.index)["unavailable"].items()
        if "misfit" not in reason
    ]
    assert not unavailable


@pytest.mark.presets
@pytest.mark.needs_binary
def test_a_capture_longer_than_the_members_context_does_not_fit(bound):
    """A capture whose requests exceed a member's context is that capture's
    misfit, counted as the simulator's own check counts it; a longer context
    fits it."""
    sims, registry = bound
    preset = "GLM-5.3-Flash/glm53_flash_vllm_fp8_kda_dsa_moe_chunked_prefill"
    short = sims.member(preset, {"replicas": 1, "server": "ctx8k"})
    _, found, _ = simulate.check_capture(
        sims.index, short, short.capture("diverse_100"), registry, build=False
    )
    assert {k: found[k] for k in ("requests", "total", "max_model_len")} == {
        "requests": 46,
        "total": 100,
        "max_model_len": 8192,
    }
    long = sims.member(preset, {"replicas": 1, "server": "ctx128k"})
    _, fits, _ = simulate.check_capture(
        sims.index, long, long.capture("diverse_100"), registry, build=False
    )
    assert fits is None


def _plan_run(bound, tmp_path, preset: str, params: dict, rows: str, tags: list[str]) -> list:
    """``workloads.plan_run`` of a run of ``preset`` ``params`` on its first
    capture's routing, replaying ``rows`` (an independent trace)."""
    sims, registry = bound
    member = sims.member(preset, params)
    trace = tmp_path / "trace.csv"
    trace.write_text(rows)
    block = {
        "trace_files": [str(trace)],
        "input_file_format": "text-generation-independent",
        "input_file_tags": tags,
        "arrival_mode": "trace_timed",
        "request_rate": 1.0,
        "run_to_end": True,
    }
    tree = simulate.run_tree(sims.index, member, member.captures[0], block, tmp_path)
    return workloads.plan_run(simulate.concrete(tree, registry))


@pytest.mark.presets
@pytest.mark.needs_binary
def test_a_request_past_the_checkpoints_positions_is_refused(bound, tmp_path):
    """Llama 3.1 8B has no max_model_len param: its config's
    max_position_embeddings (131072) bounds every pool."""
    llama = ("Llama-3.1-8B/llama3_dense_tp_barebone", {"tp_size": 1, "replicas": 1})
    header = "id,arrival_time,input_len,output_len\n"
    fits = _plan_run(bound, tmp_path, *llama, header + "a,0.0,130872,200\n", [])
    assert [row["request_id"] for row in fits] == ["a"]
    rows = header + "a,0.0,100,10\nb,0.0,131000,200\n"
    with pytest.raises(workloads.BadWorkload) as error:
        _plan_run(bound, tmp_path, *llama, rows, [])
    assert str(error.value) == (
        "1 of 2 requests exceed pool main's max_model_len 131072 "
        '(prefix_len + input_len + output_len), first ids ["b"]'
    )


@pytest.mark.presets
@pytest.mark.needs_binary
def test_a_speculative_trace_must_fit_the_draft_and_its_width(bound, tmp_path):
    """GLM-5.2 NVFP4 MTP at ctx8k drafts 5 tokens past every output, and a
    per-position acceptance needs one probability for each of them."""
    spec = (
        "GLM-5.2-NVFP4/glm52_vllm_nvfp4_dsa_moe_speculative",
        {"replicas": 1, "server": "ctx8k"},
    )
    header = "id,arrival_time,input_len,output_len,accept_rate\n"
    with pytest.raises(workloads.BadWorkload) as error:
        _plan_run(bound, tmp_path, *spec, header + "a,0.0,8000,190,0.7\n", ["speculative"])
    assert "max_model_len 8192 (prefix_len + input_len + output_len + 5 draft tokens)" in str(
        error.value
    )
    vectors = header + 'a,0.0,100,10,"[0.9,0.8,0.7,0.6,0.5]"\nb,0.0,100,10,"[0.9,0.8]"\n'
    with pytest.raises(workloads.BadWorkload) as error:
        _plan_run(bound, tmp_path, *spec, vectors, ["speculative"])
    assert "1 of 2 requests give pool main's speculative worker, which drafts 5 tokens" in str(
        error.value
    )
    fits = header + 'a,0.0,100,10,"[0.9,0.8,0.7,0.6,0.5]"\nb,0.0,8000,187,0.6\n'
    assert len(_plan_run(bound, tmp_path, *spec, fits, ["speculative"])) == 2
