"""BASE parsing and the per-platform base it pins (reference_engines/__init__.py)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from reference_engines import ENGINES, BaseError, format_base, parse_base, read_base

COMMIT = "a" * 40
OTHER = "b" * 40
AMD = "sha256:" + "1" * 64
ARM = "sha256:" + "2" * 64


def _base(**arm64: str) -> str:
    platforms = {"linux/amd64": {"digest": AMD}}
    if arm64:
        platforms["linux/arm64"] = arm64
    doc = {
        "upstream_repo": "https://example.invalid/engine.git",
        "upstream_commit": COMMIT,
        "image": "registry.invalid/engine:tag",
        "platforms": platforms,
    }
    return "# header\n" + yaml.safe_dump(doc, sort_keys=False)


def test_a_platform_published_under_another_tag_and_commit():
    base = parse_base(
        _base(
            note="why this pin",
            image="registry.invalid/engine:older-arm64",
            upstream_commit=OTHER,
            digest=ARM,
        )
    )
    assert base.image_for("linux/amd64") == f"registry.invalid/engine@{AMD}"
    assert base.image_for("linux/arm64") == f"registry.invalid/engine@{ARM}"
    assert base.commit_for("linux/amd64") == COMMIT
    assert base.commit_for("linux/arm64") == OTHER
    assert base.notes == {"linux/arm64": "why this pin"}
    # format_base writes what parse_base reads.
    assert parse_base(format_base(base, "header")) == base


def test_a_platform_without_its_own_commit_uses_the_base_commit():
    base = parse_base(_base(digest=ARM))
    assert base.commit_for("linux/arm64") == COMMIT


@pytest.mark.parametrize(
    "arm64, message",
    [
        ({"upstream_commit": OTHER}, "digest must be"),
        ({"digest": ARM, "upstream_commit": "abc"}, "40-hex"),
        ({"image": "registry.invalid/engine:x"}, "digest must be"),
        ({"digest": ARM, "tag": "x"}, "unknown key tag"),
    ],
)
def test_malformed_platform_entries_are_refused(arm64, message):
    with pytest.raises(BaseError, match=message):
        parse_base(_base(**arm64))


@pytest.mark.parametrize(
    "text, message",
    [
        ("- a list", "expected a mapping"),
        ("upstream_repo: x\n", "missing"),
        ("a: [", "not YAML"),
    ],
)
def test_malformed_documents_are_refused(text, message):
    with pytest.raises(BaseError, match=message):
        parse_base(text)


def test_a_platform_the_base_does_not_pin_is_refused():
    with pytest.raises(BaseError, match="no image for linux/arm64"):
        parse_base(_base()).image_for("linux/arm64")


@pytest.mark.parametrize("engine", ENGINES)
def test_checked_in_bases_parse(engine):
    base = read_base(engine)
    assert base.image_for("linux/amd64").startswith(base.image.rpartition(":")[0])


def _complete_aot():
    path = Path(__file__).resolve().parents[1] / "reference_engines" / "complete_flashinfer_aot.py"
    spec = importlib.util.spec_from_file_location("complete_flashinfer_aot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "provider, arch",
    [
        ("sm100a", "10.0a"),
        ("sm103a", "10.3a"),
        ("sm120f", "12.0f"),
        ("sm90a", "9.0a"),
        ("sm89", "8.9"),
    ],
)
def test_provider_ids_spell_flashinfer_arch_lists(provider, arch):
    assert _complete_aot()._arch_list(provider) == arch


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_a_tree_with_untracked_or_ignored_work_is_not_replaced(tmp_path):
    from reference_engines.__main__ import RefError, _check_replaceable

    tree = tmp_path / "src"
    tree.mkdir()
    _git("init", "-q", cwd=tree)
    (tree / ".gitignore").write_text(".venv/\n")
    _git("add", ".gitignore", cwd=tree)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base", cwd=tree)
    _check_replaceable(tree, force=True)  # only committed history: replaceable
    with pytest.raises(RefError, match="pass --force"):
        _check_replaceable(tree, force=False)

    (tree / ".venv").mkdir()
    (tree / ".venv" / "python").write_text("")
    with pytest.raises(RefError, match=r"\.venv"):
        _check_replaceable(tree, force=True)
    shutil.rmtree(tree / ".venv")

    (tree / "new_patch_file.py").write_text("")
    with pytest.raises(RefError, match="new_patch_file.py"):
        _check_replaceable(tree, force=True)

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x").write_text("")
    with pytest.raises(RefError, match="not a materialized tree"):
        _check_replaceable(plain, force=True)
