"""Naming a recording by repository and revision instead of carrying it in git."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from launcher import corpus as corpus_module
from launcher.corpus import CorpusError, resolve_hf_references

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A snapshot directory laid out the way the hub lays one out."""
    snapshot = tmp_path / "snapshot"
    (snapshot / "glm53").mkdir(parents=True)
    (snapshot / "glm53" / "manifest.json").write_text(json.dumps({"data_file": "routes.u16"}))
    (snapshot / "glm53" / "routes.u16").write_bytes(b"\x00\x01")
    requested: list[tuple[str, str, str]] = []

    def fake_download(repo, revision, path):
        requested.append((repo, revision, path))
        local = snapshot / path
        if not local.is_file():
            raise FileNotFoundError(path)
        return local

    monkeypatch.setattr(corpus_module, "_download", fake_download)
    return requested


def test_a_manifest_reference_also_fetches_the_payload_beside_it(hub):
    resolved = resolve_hf_references({"arch": {"token_corpus_file": f"hf://uw/corpora@{SHA}/glm53/manifest.json"}})

    assert resolved["arch"]["token_corpus_file"].endswith("glm53/manifest.json")
    # The simulator resolves `data_file` relative to the manifest, so fetching
    # the manifest alone would leave it pointing at nothing.
    assert hub == [
        ("uw/corpora", SHA, "glm53/manifest.json"),
        ("uw/corpora", SHA, "glm53/routes.u16"),
    ]


def test_everything_that_is_not_a_reference_is_left_alone(hub):
    config = {
        "arch": {"expert_popularity_file": "presets/alignment/x/expert_popularity.json"},
        "workload": {"max_concurrency": 256, "tags": ["speculative"], "rate": None},
    }

    assert resolve_hf_references(config) == config
    assert hub == []


@pytest.mark.parametrize(
    "reference",
    [
        # A branch or tag can move, and a corpus that moved under a built config
        # reprices every profiled shape instead of failing.
        "hf://uw/corpora@main/glm53/manifest.json",
        "hf://uw/corpora@v1.0/glm53/manifest.json",
        f"hf://corpora@{SHA}/glm53/manifest.json",
        "hf://uw/corpora/glm53/manifest.json",
    ],
)
def test_a_reference_without_a_pinned_commit_is_refused(hub, reference):
    with pytest.raises(CorpusError, match="commit sha"):
        resolve_hf_references({"token_corpus_file": reference})


def test_a_manifest_whose_payload_lands_elsewhere_is_refused(tmp_path, monkeypatch):
    def fake_download(repo, revision, path):
        local = tmp_path / Path(path).name if path.endswith(".u16") else tmp_path / "a" / "b.json"
        local.parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".json"):
            local.write_text(json.dumps({"data_file": "routes.u16"}))
        else:
            local.write_bytes(b"")
        return local

    monkeypatch.setattr(corpus_module, "_download", fake_download)

    with pytest.raises(CorpusError, match="not beside its manifest"):
        resolve_hf_references({"token_corpus_file": f"hf://uw/corpora@{SHA}/glm53/manifest.json"})
