"""`alignment-campaign bundle` / `verify-bundle`: scrub, refuse a leak, round-trip.

Pure CPU. A rendered case of a tracked pack gets a fake capture whose records
carry host paths, a user name, a host name and GPU UUIDs the way a real image
capture does (`container_run.json`, `capture_facts.json`, `profile_result.json`,
`profile.yaml`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest
import yaml

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
SNAPSHOT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def identity(monkeypatch, tmp_path):
    home = tmp_path / "home" / USER
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(bundle_module.getpass, "getuser", lambda: USER)
    monkeypatch.setattr(bundle_module.socket, "gethostname", lambda: HOSTNAME)
    monkeypatch.setattr(bundle_module.socket, "getfqdn", lambda: HOSTNAME + ".example.org")
    return home


def _host(pack, tmp_path, home):
    stub = check_module.host_for(pack, None)
    hub = tmp_path / "shared" / "hub"
    return dataclasses.replace(
        stub,
        hf_hub_root=str(hub),
        checkpoints={key: f"models--org--m/snapshots/{SNAPSHOT}" for key in stub.checkpoints},
        text_corpus=str(home / "corpus" / "enwik9"),
    )


@pytest.fixture
def campaign(tmp_path, identity):
    """(pack, host, out_root, case, workload pass name) with one capture written."""
    pack = load_pack(PACK_DIR)
    case = pack.cases[0]
    variant = pack.variant_of(case)
    workload = next(item for item in variant.profile_passes if item.kind == "workload_metrics")
    host = _host(pack, tmp_path, identity)
    out_root = tmp_path / "out"
    render_case(pack, case, host, out_root, REPO_ROOT)
    case_dir = out_root / case.slug
    _write_capture(case_dir, workload.name, host=host, home=identity)
    assert execute.phase_complete(case_dir, variant, workload.name)
    return pack, host, out_root, case, workload.name


def _write_capture(case_dir: Path, pass_name: str, *, host, home: Path) -> Path:
    config = yaml.safe_load((case_dir / f"{pass_name}.yaml").read_text())
    engine = config["engine"]
    capture = case_dir / pass_name
    (capture / "vllm").mkdir(parents=True)
    trace = case_dir / "trace.csv"
    (capture / "profile.yaml").write_text(yaml.safe_dump(config))
    (capture / "profile_result.json").write_text(json.dumps({
        "log_dir": str(capture),
        "engine": engine,
        "gpu": config["gpu"],
        "server_log": str(capture / "vllm" / "x_server.log"),
        "drive_summary": {"source_trace": str(trace)},
    }))
    (capture / "container_run.json").write_text(json.dumps({
        "image": image_tag(engine),
        "image_id": "sha256:" + "0" * 64,
        "gpu_uuids": UUIDS,
        "cache_dir": str(home / ".cache" / "profiler"),
        "argv": [f"{capture}:/out:rw", f"${home}/.cache:/cache:rw"],
        "paths": {
            "/out": str(capture),
            "/models": host.hf_hub_root,
            "/work/trace.csv": str(trace),
            "/work/text_corpus": host.text_corpus,
        },
    }))
    (capture / "capture_facts.json").write_text(json.dumps({
        "hostname": "container-id",
        "gpus": [{"uuid": uuid, "name": "NVIDIA B200", "driver_version": "620.70.06"}
                 for uuid in UUIDS],
        "nvlink_status": f"GPU 0: NVIDIA B200 (UUID: {UUIDS[0]})",
    }))
    (capture / "vllm" / "x_server.log").write_text(
        f"started on {HOSTNAME} as {USER}\n"
    )
    (capture / "trace.sqlite").write_bytes(b"regenerable")
    # nsys keeps each device's bare UUID; that is published, not refused.
    (capture / "blob.nsys-rep").write_bytes(b"\x00" + UUIDS[0][4:].encode() + b"\x01measured\x02")
    mark_complete(capture)
    return capture


def _bundle(campaign, dest):
    pack, host, out_root, case, _ = campaign
    return bundle_module.bundle(pack, host, out_root, [case], dest)


def _rendered(campaign, tmp_path):
    pack, host, _, case, _ = campaign
    fresh = tmp_path / "maintainer"
    render_case(pack, case, host, fresh, REPO_ROOT)
    return fresh


def _all_text(root: Path) -> str:
    return "\n".join(path.read_text(errors="replace") for path in root.rglob("*") if path.is_file())


def _rewrite(report, mutate) -> None:
    """Edit bundle.json in place."""
    path = report.dest / "bundle.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


def _rehash(report, slug, pass_name, relative) -> None:
    file = report.dest / slug / pass_name / relative
    digest = hashlib.sha256(file.read_bytes()).hexdigest()

    def mutate(manifest):
        manifest["cases"][slug]["passes"][pass_name]["files"][relative] = {
            "sha256": digest, "size": file.stat().st_size,
        }

    _rewrite(report, mutate)


# ── bundle ───────────────────────────────────────────────────────────────────


def test_container_cache_path_maps_a_host_cache_file_into_the_container(tmp_path):
    manifest = tmp_path / "cache" / "warm-manifests" / "k.json"
    assert container_cache_path(manifest, tmp_path / "cache") == PurePosixPath(
        "/cache/warm-manifests/k.json"
    )


def test_bundle_scrubs_host_identity_and_hashes_what_it_writes(campaign, tmp_path):
    _, host, out_root, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    text = "\n".join(
        path.read_text(errors="replace")
        for path in report.dest.rglob("*")
        if path.is_file() and path.suffix != ".nsys-rep"
    )
    for leaked in (USER, HOSTNAME, *UUIDS, str(out_root), host.text_corpus, host.hf_hub_root):
        assert leaked not in text, leaked

    captured = report.dest / case.slug / pass_name
    result = json.loads((captured / "profile_result.json").read_text())
    assert result["log_dir"] == "${capture_dir}"
    assert result["drive_summary"]["source_trace"] == "${mount:/work/trace.csv}"
    # Placeholders keep every record parseable.
    config = yaml.safe_load((captured / "profile.yaml").read_text())
    assert config["server"]["model_path"].endswith(f"/snapshots/{SNAPSHOT}")
    assert "GPU-0" in (captured / "capture_facts.json").read_text()
    assert (captured / "vllm" / "x_server.log").read_text() == "started on ${hostname} as ${user}\n"
    assert not (captured / "trace.sqlite").exists()
    assert (captured / ".complete").is_file()
    assert (report.dest / case.slug / f"{pass_name}.trace.csv").is_file()

    entry = report.manifest["cases"][case.slug]["passes"][pass_name]
    assert entry["state"] == "complete"
    assert entry["conditions"]["model_snapshot"] == SNAPSHOT
    assert entry["conditions"]["gpu_count"] == 2
    for relative, item in entry["files"].items():
        assert bundle_module._sha256(captured / relative) == item["sha256"]


def test_bundle_copies_and_never_hard_links(campaign, tmp_path):
    _, _, out_root, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    original = out_root / case.slug / pass_name / "blob.nsys-rep"
    copied = report.dest / case.slug / pass_name / "blob.nsys-rep"
    assert original.stat().st_ino != copied.stat().st_ino


def test_bundle_follows_a_symlinked_subdirectory(campaign, tmp_path):
    _, _, out_root, case, pass_name = campaign
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "trace.nsys-rep").write_bytes(b"\x00kernels\x00")
    (out_root / case.slug / pass_name / "nsys").symlink_to(real)
    report = _bundle(campaign, tmp_path / "bundle")
    assert (report.dest / case.slug / pass_name / "nsys" / "trace.nsys-rep").is_file()


def test_bundle_scrubs_a_relative_out_root(campaign, tmp_path, monkeypatch):
    pack, host, out_root, case, _ = campaign
    monkeypatch.chdir(out_root.parent)
    report = bundle_module.bundle(pack, host, Path(out_root.name), [case], tmp_path / "bundle")
    assert str(out_root) not in _all_text(report.dest)


@pytest.mark.parametrize("leak", ["/raid/${user}/x", "corpus"])
def test_bundle_refuses_a_binary_that_names_the_user_or_a_host_path(campaign, tmp_path, leak):
    _, host, out_root, case, pass_name = campaign
    value = USER if leak != "corpus" else host.text_corpus
    payload = leak.replace("${user}", value) if leak != "corpus" else value
    (out_root / case.slug / pass_name / "trace.nsys-rep").write_bytes(
        b"\x00ENV=" + payload.encode() + b"\x00"
    )
    with pytest.raises(bundle_module.BundleError, match="trace.nsys-rep"):
        _bundle(campaign, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_a_short_user_name_is_never_rewritten(campaign, tmp_path, monkeypatch):
    monkeypatch.setattr(bundle_module.getpass, "getuser", lambda: "vllm")
    _, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    result = json.loads((report.dest / case.slug / pass_name / "profile_result.json").read_text())
    assert result["engine"] == "vllm"
    assert result["server_log"] == "${capture_dir}/vllm/x_server.log"


def test_a_user_name_inside_a_longer_word_refuses_the_bundle(campaign, tmp_path):
    _, _, out_root, case, pass_name = campaign
    (out_root / case.slug / pass_name / "note.txt").write_text(f"/raid/{USER}_data/x\n")
    with pytest.raises(bundle_module.BundleError, match="note.txt"):
        _bundle(campaign, tmp_path / "bundle")


def test_download_fetches_the_bundle_directory(monkeypatch, tmp_path):
    from huggingface_hub.utils import filter_repo_objects

    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    sha = "b" * 40
    path = bundle_module.download(f"org/data@{sha}:pack/b", tmp_path)
    assert path == tmp_path / "pack" / "b"
    assert seen["revision"] == sha
    kept = list(filter_repo_objects(
        ["pack/b/bundle.json", "pack/b/c/p/x.json", "pack/b2/bundle.json"],
        allow_patterns=seen["allow_patterns"],
    ))
    assert kept == ["pack/b/bundle.json", "pack/b/c/p/x.json"]


# ── verify-bundle ────────────────────────────────────────────────────────────


def test_verify_bundle_places_the_capture_and_restores_its_paths(campaign, tmp_path):
    pack, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)

    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert result.ok, result.errors
    assert f"{case.slug}/{pass_name}" in result.placed
    placed = fresh / case.slug / pass_name
    document = json.loads((placed / "profile_result.json").read_text())
    assert document["log_dir"] == str(placed.resolve())
    assert document["drive_summary"]["source_trace"] == str((fresh / case.slug / "trace.csv").resolve())
    assert execute.phase_complete(fresh / case.slug, pack.variant_of(case), pass_name)
    assert not list((fresh / case.slug).glob(".*.verify-bundle"))

    again = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("exists" in line for line in again.errors)
    assert bundle_module.verify_bundle(pack, report.dest, fresh, replace=True).ok


def test_verify_bundle_rejects_a_changed_or_extra_file(campaign, tmp_path):
    pack, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)

    (report.dest / case.slug / pass_name / "blob.nsys-rep").write_bytes(b"edited")
    (report.dest / "stray.txt").write_text("x")
    (report.dest / case.slug / pass_name / ".cache").mkdir()
    (report.dest / case.slug / pass_name / ".cache" / "evil.json").write_text("{}")
    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("blob.nsys-rep: sha256 mismatch" in line for line in result.errors)
    assert any("stray.txt" in line for line in result.errors)
    assert any("evil.json" in line for line in result.errors)
    assert not (fresh / case.slug / pass_name).exists()


@pytest.mark.parametrize("name", ["../../escaped", ".", "simulation"])
def test_verify_bundle_accepts_only_the_cases_profile_passes(campaign, tmp_path, name):
    pack, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)
    sentinel = fresh / case.slug / "keep.txt"
    sentinel.write_text("maintainer file")

    def mutate(manifest):
        passes = manifest["cases"][case.slug]["passes"]
        passes[name] = passes.pop(pass_name)

    _rewrite(report, mutate)
    result = bundle_module.verify_bundle(pack, report.dest, fresh, replace=True)
    assert any("not a profile pass" in line for line in result.errors)
    assert sentinel.is_file()
    assert not (tmp_path / "escaped").exists()


def test_verify_bundle_rejects_an_unsafe_file_path(campaign, tmp_path):
    pack, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)

    def mutate(manifest):
        files = manifest["cases"][case.slug]["passes"][pass_name]["files"]
        files["../../../x.bin"] = files["blob.nsys-rep"]

    _rewrite(report, mutate)
    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("unsafe path" in line for line in result.errors)


def test_verify_bundle_reads_conditions_from_the_capture_not_the_manifest(campaign, tmp_path):
    pack, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)
    record = report.dest / case.slug / pass_name / "container_run.json"
    document = json.loads(record.read_text())
    document["image"] = "other:tag"
    record.write_text(json.dumps(document))
    _rehash(report, case.slug, pass_name, "container_run.json")
    _rewrite(report, lambda m: m["cases"][case.slug]["passes"][pass_name].update(conditions={}))

    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("image is other:tag" in line for line in result.errors)
    assert not (fresh / case.slug / pass_name).exists()


def test_verify_bundle_treats_a_missing_condition_as_an_error(campaign, tmp_path):
    pack, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)
    (report.dest / case.slug / pass_name / "profile.yaml").write_text("schema_version: 1\n")
    _rehash(report, case.slug, pass_name, "profile.yaml")

    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("records no model_snapshot" in line for line in result.errors)


def test_verify_bundle_places_nothing_when_one_pass_fails_a_check(campaign, tmp_path):
    pack, host, out_root, case, pass_name = campaign
    variant = pack.variant_of(case)
    other = next(item for item in variant.profile_passes if item.kind == "nsys")
    _write_capture(out_root / case.slug, other.name, host=host, home=Path.home())
    for name in execute._artifacts_for(variant, other.name):
        path = out_root / case.slug / other.name / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)
    (fresh / case.slug / other.name).mkdir()  # exists, no --replace

    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("exists" in line for line in result.errors)
    assert not (fresh / case.slug / pass_name).exists()


def test_verify_bundle_rejects_a_complete_pass_without_its_artifacts(campaign, tmp_path):
    pack, _, _, case, pass_name = campaign
    report = _bundle(campaign, tmp_path / "bundle")
    fresh = _rendered(campaign, tmp_path)

    def mutate(manifest):
        manifest["cases"][case.slug]["passes"][pass_name]["files"].pop("profile_result.json")

    (report.dest / case.slug / pass_name / "profile_result.json").unlink()
    _rewrite(report, mutate)
    result = bundle_module.verify_bundle(pack, report.dest, fresh)
    assert any("lacks ['profile_result.json']" in line for line in result.errors)


@pytest.mark.parametrize("revision", ["main", "v1", "abc123"])
def test_from_hf_requires_a_pinned_revision(revision):
    with pytest.raises(bundle_module.BundleError, match="not pinned"):
        bundle_module.parse_hf_spec(f"org/data@{revision}:pack/b")


def test_from_hf_accepts_a_commit_or_a_pr_ref():
    sha = "a" * 40
    assert bundle_module.parse_hf_spec(f"org/data@{sha}:pack/b/") == ("org/data", sha, "pack/b")
    assert bundle_module.parse_hf_spec("org/data@refs/pr/3:pack/b")[1] == "refs/pr/3"
