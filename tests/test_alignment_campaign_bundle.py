"""`alignment-campaign bundle` / `verify-bundle`: scrub, refuse a leak, round-trip.

Pure CPU. A rendered case of a tracked pack gets a fake capture whose records
carry host paths, a user name, a host name and GPU UUIDs the way a real capture
does (`container_run.json`, `capture_facts.json`, `profile_result.json`).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path, PurePosixPath

import pytest

from alignment.image_runtime import container_cache_path, image_tag
from launcher.alignment_campaign import bundle as bundle_module
from launcher.alignment_campaign import check as check_module
from launcher.alignment_campaign import execute
from launcher.alignment_campaign.pack import load_pack
from launcher.alignment_campaign.render import render_case
from launcher.process.markers import mark_complete

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = REPO_ROOT / "presets" / "alignment" / "glm53_flash_fp8_b200_tp4_ep4"

USER = "alicezz"
HOSTNAME = "node-alpha-7"
UUIDS = ["GPU-11111111-2222-3333-4444-555555555555", "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"]


@pytest.fixture
def identity(monkeypatch, tmp_path):
    home = tmp_path / "home" / USER
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(bundle_module.getpass, "getuser", lambda: USER)
    monkeypatch.setattr(bundle_module.socket, "gethostname", lambda: HOSTNAME)
    return home


@pytest.fixture
def campaign(tmp_path, identity):
    """(pack, host, out_root, case, workload pass name) with one capture written."""
    pack = load_pack(PACK_DIR)
    case = pack.cases[0]
    variant = pack.variant_of(case)
    workload = next(item for item in variant.profile_passes if item.kind == "workload_metrics")
    corpus = identity / "corpus" / "enwik9"
    host = dataclasses.replace(check_module.host_for(pack, None), text_corpus=str(corpus))
    out_root = tmp_path / "out"
    render_case(pack, case, host, out_root, REPO_ROOT)
    case_dir = out_root / case.slug
    _write_capture(case_dir, workload.name, engine="vllm", home=identity)
    assert execute.phase_complete(case_dir, variant, workload.name)
    return pack, host, out_root, case, workload.name


def _write_capture(case_dir: Path, pass_name: str, *, engine: str, home: Path) -> Path:
    capture = case_dir / pass_name
    (capture / "vllm").mkdir(parents=True)
    trace = case_dir / "trace.csv"
    (capture / "profile_result.json").write_text(json.dumps({
        "log_dir": str(capture),
        "engine": engine,
        "gpu": "NVIDIA B200",
        "server_log": str(capture / "vllm" / "x_server.log"),
        "drive_summary": {"source_trace": str(trace)},
    }))
    (capture / "container_run.json").write_text(json.dumps({
        "image": image_tag(engine),
        "image_id": "sha256:" + "0" * 64,
        "gpu_uuids": UUIDS,
        "cache_dir": str(home / ".cache" / "profiler"),
        "argv": [f"{capture}:/out:rw", f"{home}/.cache:/cache:rw"],
    }))
    (capture / "capture_facts.json").write_text(json.dumps({
        "hostname": "container-id",
        "gpus": [{"uuid": uuid, "name": "NVIDIA B200", "driver_version": "620.70.06"}
                 for uuid in UUIDS],
        "nvlink_status": f"GPU 0: NVIDIA B200 (UUID: {UUIDS[0]})",
    }))
    (capture / "vllm" / "x_server.log").write_text(f"started on {HOSTNAME} as {USER}\n")
    (capture / "trace.sqlite").write_bytes(b"regenerable")
    (capture / "blob.bin").write_bytes(b"\x00\x01measured\x02")
    mark_complete(capture)
    return capture


def _bundle(campaign, dest):
    pack, host, out_root, case, _ = campaign
    return bundle_module.bundle(pack, host, out_root, [case], dest)


def test_bundle_scrubs_a_relative_out_root(campaign, tmp_path, monkeypatch):
    pack, host, out_root, case, _ = campaign
    monkeypatch.chdir(out_root.parent)
    report = bundle_module.bundle(pack, host, Path(out_root.name), [case], tmp_path / "bundle")
    assert str(out_root) not in _all_text(report.dest)


def _all_text(root: Path) -> str:
    return "\n".join(path.read_text(errors="replace") for path in root.rglob("*") if path.is_file())


def test_container_cache_path_maps_a_host_cache_file_into_the_container(tmp_path):
    manifest = tmp_path / "cache" / "warm-manifests" / "k.json"
    assert container_cache_path(manifest, tmp_path / "cache") == PurePosixPath(
        "/cache/warm-manifests/k.json"
    )


def test_bundle_scrubs_host_identity_and_hashes_what_it_writes(campaign, tmp_path):
    _, _, out_root, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    text = _all_text(report.dest)
    for leaked in (USER, HOSTNAME, *UUIDS, str(out_root)):
        assert leaked not in text, leaked

    captured = report.dest / case.slug / pass_name
    result = json.loads((captured / "profile_result.json").read_text())
    assert result["log_dir"] == "{capture_dir}"
    assert "GPU-0" in (captured / "capture_facts.json").read_text()
    assert (captured / "vllm" / "x_server.log").read_text() == "started on {hostname} as {user}\n"
    assert not (captured / "trace.sqlite").exists()
    assert (captured / ".complete").is_file()

    entry = report.manifest["cases"][case.slug]["passes"][pass_name]
    assert entry["state"] == "complete"
    assert entry["conditions"]["gpu"] == "NVIDIA B200"
    assert entry["conditions"]["gpu_count"] == 2
    for relative, item in entry["files"].items():
        assert bundle_module._sha256(captured / relative) == item["sha256"]


def test_bundle_refuses_a_binary_that_names_the_user(campaign, tmp_path):
    _, _, out_root, case, pass_name = campaign
    (out_root / case.slug / pass_name / "trace.nsys-rep").write_bytes(
        b"\x00ENV=/raid/" + USER.encode() + b"/x\x00"
    )
    with pytest.raises(bundle_module.BundleError, match="trace.nsys-rep"):
        _bundle(campaign, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_verify_bundle_places_the_capture_and_restores_its_log_dir(campaign, tmp_path):
    pack, host, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = tmp_path / "maintainer"
    render_case(pack, case, host, fresh, REPO_ROOT)

    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert result.ok, result.errors
    assert f"{case.slug}/{pass_name}" in result.placed
    placed = fresh / case.slug / pass_name
    assert json.loads((placed / "profile_result.json").read_text())["log_dir"] == str(placed.resolve())
    assert execute.phase_complete(fresh / case.slug, pack.variant_of(case), pass_name)

    again = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("exists" in line for line in again.errors)


def test_verify_bundle_rejects_a_changed_or_extra_file(campaign, tmp_path):
    pack, host, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = tmp_path / "maintainer"
    render_case(pack, case, host, fresh, REPO_ROOT)

    (report.dest / case.slug / pass_name / "blob.bin").write_bytes(b"edited")
    (report.dest / "stray.txt").write_text("x")
    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("blob.bin: sha256 mismatch" in line for line in result.errors)
    assert any("stray.txt" in line for line in result.errors)
    assert not (fresh / case.slug / pass_name).exists()


def test_verify_bundle_rejects_a_capture_from_another_image(campaign, tmp_path):
    pack, host, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    manifest = json.loads((report.dest / "bundle.json").read_text())
    manifest["cases"][case.slug]["passes"][pass_name]["conditions"]["image"] = "other:tag"
    (report.dest / "bundle.json").write_text(json.dumps(manifest))
    fresh = tmp_path / "maintainer"
    render_case(pack, case, host, fresh, REPO_ROOT)

    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("image is other:tag" in line for line in result.errors)
    assert not (fresh / case.slug / pass_name).exists()
