"""Portable campaign rendering for the bounded stock Trainium operating point."""

from __future__ import annotations

import csv
import dataclasses
import io
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from launcher.alignment_campaign import check, cli, label, render
from launcher.alignment_campaign.pack import PackError, load_host, load_pack
from launcher.alignment_config import load_profile_config, load_timing_predict_config

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "presets/alignment/llama31_8b_bf16_trainium2_tp4"


@pytest.fixture
def pack():
    return load_pack(PACK)


def test_stock_pack_has_unchanged_generic_policy(pack):
    generic = load_pack(ROOT / "presets/alignment/glm53_flash_fp8_b200_tp4_ep4")
    assert pack.acceptance["tolerances"]["default"] == generic.acceptance["tolerances"]["default"]
    assert pack.acceptance["tolerances"]["per_case"] == {}
    assert "after" in pack.cases[0].provenance["policy_timing"]


@pytest.mark.parametrize("prefix", ["/host-a", "/different-host"])
def test_stock_render_uses_host_paths_and_one_chip(pack, tmp_path, prefix):
    body = {
        "schema_version": 1, "host": "portable", "checkpoints": {"llama31_8b": prefix + "/model"},
        "text_corpus": prefix + "/museum-pool.txt", "device_roles": {"primary": "2"},
        "neuron_server": {
            "cache_path": prefix + "/cache", "image": "sha256:" + "1" * 64,
            "docker_host": "unix://" + prefix + "/docker.sock",
            "req_frontend_binary": prefix + "/session_runner",
            "accepted_forward_path": prefix + "/accepted-forward",
        },
    }
    host_file = tmp_path / "host.yaml"
    host_file.write_text(yaml.safe_dump(body))
    host = load_host(host_file)
    case = pack.cases[0]
    directory = render.render_case(pack, case, host, tmp_path / "rendered", ROOT).directory
    for name, kind in (("profile_native", "neuron"), ("profile_clean", "workload_metrics")):
        path = directory / f"{name}.yaml"
        config = load_profile_config(path, require_python_runtime=False)
        assert config.profile_kind == kind
        assert config.server.model_path == prefix + "/model"
        assert config.server.cache_path == prefix + "/cache"
        assert config.server.neuron_device == 2
        assert config.server.tp_size == 4
        assert config.server.logical_nc_config == 2
        assert config.server.token_buckets == (1, 16)
        assert config.server.max_model_len == 512
        assert config.workload.token_pool_limit == 79784
        document = yaml.safe_load(path.read_text())
        assert "cuda_visible_devices" not in document
        assert "extra_args" not in document["server"]
        # Static checking may use placeholders; actual profile preflight may not.
        with pytest.raises(ValueError, match="model_path does not exist"):
            load_profile_config(path)
    predictor = load_timing_predict_config(directory / "timing_predict.yaml")
    assert predictor.input_builder.to_mapping()["type"] == "engine_text"
    simulation = yaml.safe_load((directory / "simulation.yaml").read_text())
    group = simulation["pools"]["main"]["groups"][0]
    assert group["arch"]["type"] == "llama3_vllm_neuron"
    assert group["arch"]["decode_buckets"] == [1, 16]
    assert group["worker"]["max_batch_tokens"] == 512
    assert group["worker"]["max_resident_requests_per_partition"] == 16
    assert group["worker"]["prefix_cache_mode"] == "disabled"
    assert simulation["io"]["log_output_token_times"] is True
    assert simulation["io"]["kv_log_stride"] == 1


def test_stock_trace_is_simultaneous_and_uses_full_context(pack):
    case = pack.cases[0]
    rows = list(csv.DictReader(io.StringIO(render.case_traces(pack, case)["trace.csv"])))
    assert len(rows) == len({row["id"] for row in rows}) == 16
    assert {(int(row["input_len"]), int(row["output_len"])) for row in rows} == {(504, 8)}
    assert {float(row["arrival_time"]) for row in rows} == {0.0}
    overflow = dataclasses.replace(case.workload_trace, shapes=((505, 8),))
    with pytest.raises(PackError, match="max_model_len"):
        render.case_trace_text(pack, case, overflow)
    # Other engines retain the exclusive context limit and their arrival times.
    with pytest.raises(PackError, match="max_model_len"):
        render.trace_text(case, case.workload_trace)
    smaller = dataclasses.replace(case.workload_trace, shapes=((503, 8),))
    legacy = list(csv.DictReader(io.StringIO(render.trace_text(case, smaller))))
    assert float(legacy[1]["arrival_time"]) == 1000.0


def test_stock_render_rejects_multiple_chips_or_host_fields_in_variant(pack, tmp_path):
    case = pack.cases[0]
    variant = pack.variant_of(case)
    host = check.host_for(pack, None)
    with pytest.raises(PackError, match="one whole-chip"):
        render.case_documents(
            pack, case, dataclasses.replace(host, device_roles={"primary": "0,1,2,3"}),
            tmp_path, ROOT,
        )
    polluted = dataclasses.replace(
        variant, server={**variant.server, "image": "sha256:" + "2" * 64}
    )
    patched = dataclasses.replace(pack, variants={variant.name: polluted})
    with pytest.raises(PackError, match="host"):
        render.case_documents(patched, case, host, tmp_path, ROOT)


def test_stock_render_requires_declared_host_fields(pack, tmp_path):
    host = check.host_for(pack, None)
    with pytest.raises(PackError, match="missing"):
        render.case_documents(
            pack, pack.cases[0], dataclasses.replace(host, neuron_server={}), tmp_path, ROOT
        )


@pytest.mark.parametrize("symlink", [False, True])
def test_stock_host_paths_resolve_from_repository_at_arbitrary_render_depth(
    pack, tmp_path, symlink
):
    repository = tmp_path / "repository"
    model = repository / "models/checkpoint"
    model.mkdir(parents=True)
    checkpoint = "models/checkpoint"
    if symlink:
        alias = repository / "checkpoint-alias"
        alias.symlink_to(model, target_is_directory=True)
        checkpoint = str(alias)
    (repository / "cache").mkdir()
    (repository / "session_runner").touch()
    host = dataclasses.replace(
        check.host_for(pack, None), hf_hub_root="", checkpoints={"llama31_8b": checkpoint},
        text_corpus="museum-pool.txt",
        neuron_server={
            "cache_path": "cache", "req_frontend_binary": "session_runner",
            "accepted_forward_path": "accepted-forward", "image": "sha256:" + "1" * 64,
            "docker_host": "unix:///host-owned/docker.sock",
            "python_executable": "/image-owned/python",
        },
    )
    directory = render.render_case(
        pack, pack.cases[0], host, tmp_path / "unrelated/nested/rendered", repository
    ).directory
    for name in ("profile_native", "profile_clean"):
        config = load_profile_config(directory / f"{name}.yaml")
        assert config.server.model_path == config.workload.tokenizer == str(model)
        assert config.server.cache_path == str(repository / "cache")
        assert config.server.req_frontend_binary == str(repository / "session_runner")
        assert config.server.accepted_forward_path == str(repository / "accepted-forward")
        assert config.workload.text_file == str(repository / "museum-pool.txt")
        assert config.server.docker_host == "unix:///host-owned/docker.sock"
        assert config.server.python_executable == "/image-owned/python"


@pytest.mark.parametrize("kind", ["nsys", "neuron"])
def test_label_cli_uses_either_native_provider(pack, tmp_path, monkeypatch, kind):
    case = pack.cases[0]
    variant = pack.variant_of(case)
    capture = dataclasses.replace(variant.profile_passes[0], kind=kind)
    variant = dataclasses.replace(variant, profile_passes=(capture,))
    pack = dataclasses.replace(pack, variants={variant.name: variant})
    (tmp_path / case.slug).mkdir()
    monkeypatch.setattr(cli, "_resolve_pack", lambda _: pack)
    calls = []

    def run_label(pack, variant, directory, native_pass, *, refresh):
        calls.append((directory, native_pass, refresh))
        return label.LabelReport(case_slug=case.slug, converged=True)

    monkeypatch.setattr(cli.label_module, "label_case", run_label)
    args = Namespace(pack="stock", out_root=str(tmp_path), case=None, refresh=False)
    assert cli._label(args) == 0
    assert calls == [(tmp_path / case.slug, capture.name, False)]
