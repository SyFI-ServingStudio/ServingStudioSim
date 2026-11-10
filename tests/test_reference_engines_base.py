"""BASE parsing and the per-platform base it pins (reference_engines/__init__.py)."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

import reference_engines
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
        ({"digest": ARM, "image": f"registry.invalid/engine:x@{ARM}"}, "without a digest"),
        ({"digest": ARM, "image": "registry.invalid/engine"}, "repository:tag"),
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
    _check_replaceable(tree, force=True, patches=[])  # only committed history: replaceable
    with pytest.raises(RefError, match="pass --force"):
        _check_replaceable(tree, force=False, patches=[])

    (tree / ".venv").mkdir()
    (tree / ".venv" / "python").write_text("")
    with pytest.raises(RefError, match=r"\.venv"):
        _check_replaceable(tree, force=True, patches=[])
    shutil.rmtree(tree / ".venv")

    (tree / "new_patch_file.py").write_text("")
    with pytest.raises(RefError, match="new_patch_file.py"):
        _check_replaceable(tree, force=True, patches=[])

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x").write_text("")
    with pytest.raises(RefError, match="not a materialized tree"):
        _check_replaceable(plain, force=True, patches=[])


def test_a_tree_with_unexported_commits_a_stash_or_a_paused_am_is_not_replaced(tmp_path):
    from reference_engines.__main__ import RefError, _check_replaceable

    tree, patches = tmp_path / "src", tmp_path / "patches"
    tree.mkdir()
    commit = ("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam")
    _git("init", "-q", "-b", "reference/x", cwd=tree)
    (tree / "a.py").write_text("base\n")
    _git("add", "a.py", cwd=tree)
    _git(*commit, "base", cwd=tree)
    (tree / "a.py").write_text("patched\n")
    _git(*commit, "exported", cwd=tree)
    _git("format-patch", "-q", "--zero-commit", "-o", str(patches), "HEAD~1..HEAD", cwd=tree)
    exported = sorted(patches.glob("*.patch"))
    _check_replaceable(tree, force=True, patches=exported)

    # Commits export-patches writes nothing for: empty, or only ALIGNMENT_BASE.
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "empty", cwd=tree)
    (tree / "ALIGNMENT_BASE").write_text("x\n")
    _git("add", "ALIGNMENT_BASE", cwd=tree)
    _git(*commit, "alignment base only", cwd=tree)
    _check_replaceable(tree, force=True, patches=exported)

    # A rename exported as a rename (format-patch detects it) still matches.
    _git("mv", "a.py", "b.py", cwd=tree)
    (tree / "b.py").write_text("patched\nrenamed\n")
    _git(*commit, "rename", cwd=tree)
    _git("format-patch", "-q", "--zero-commit", "-o", str(patches), "HEAD~1..HEAD", cwd=tree)
    exported = sorted(patches.glob("*.patch"))
    assert "rename from a.py" in exported[-1].read_text()
    _check_replaceable(tree, force=True, patches=exported)

    # A commit on HEAD that export-patches has not written.
    (tree / "b.py").write_text("new work\n")
    _git(*commit, "not exported", cwd=tree)
    with pytest.raises(RefError, match="not exported"):
        _check_replaceable(tree, force=True, patches=exported)
    _git("reset", "-q", "--hard", "HEAD~1", cwd=tree)
    _check_replaceable(tree, force=True, patches=exported)

    # ... or on another branch.
    _git("switch", "-q", "-c", "side", cwd=tree)
    (tree / "b.py").write_text("side work\n")
    _git(*commit, "side work", cwd=tree)
    _git("switch", "-q", "reference/x", cwd=tree)
    with pytest.raises(RefError, match="side work"):
        _check_replaceable(tree, force=True, patches=exported)
    _git("branch", "-q", "-D", "side", cwd=tree)

    # A stash.
    (tree / "b.py").write_text("stashed\n")
    _git("-c", "user.name=t", "-c", "user.email=t@t", "stash", "-q", cwd=tree)
    with pytest.raises(RefError, match="stash"):
        _check_replaceable(tree, force=True, patches=exported)
    _git("stash", "drop", "-q", cwd=tree)

    # A `git am` paused on a patch that did not apply (the tree stays clean).
    (tree / ".git" / "rebase-apply").mkdir()
    with pytest.raises(RefError, match="rebase-apply"):
        _check_replaceable(tree, force=True, patches=exported)


def test_a_commit_on_the_base_that_export_skips_is_not_unexported(tmp_path):
    from reference_engines.__main__ import _check_replaceable

    tree = tmp_path / "src"
    tree.mkdir()
    commit = ("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm")
    _git("init", "-q", cwd=tree)
    (tree / "a.py").write_text("base\n")
    _git("add", "a.py", cwd=tree)
    _git(*commit, "base", cwd=tree)
    (tree / "ALIGNMENT_BASE").write_text("x\n")
    _git("add", "ALIGNMENT_BASE", cwd=tree)
    _git(*commit, "alignment base only", cwd=tree)
    # Not the root commit as a whole-tree patch: nothing to export, so replaceable.
    _check_replaceable(tree, force=True, patches=[])


@pytest.mark.parametrize("engine", ENGINES)
def test_checked_in_patches_touch_only_what_the_image_installs(engine):
    assert reference_engines.patched_paths_not_installed(engine) == []


def test_a_native_change_in_a_patch_is_reported(tmp_path, monkeypatch):
    patch = tmp_path / "0001-x.patch"
    patch.write_text(
        "diff --git a/vllm/v1/a.py b/vllm/v1/a.py\n"
        "diff --git a/tests/v1/test_a.py b/tests/v1/test_a.py\n"
        "diff --git a/ALIGNMENT.md b/ALIGNMENT.md\n"
        "diff --git a/csrc/moe/topk.cu b/csrc/moe/topk.cu\n"
        "diff --git a/CMakeLists.txt b/CMakeLists.txt\n"
        "diff --git a/vllm/kernel.cu b/csrc/kernel.cu\n"
        "diff --git a/vllm/x y.py b/csrc/x y.cu\n"
        'diff --git "a/vllm/caf\\303\\251.py" "b/csrc/caf\\303\\251.cu"\n'
        "similarity index 90%\n"
        'rename from "vllm/caf\\303\\251.py"\n'
        'rename to "csrc/caf\\303\\251.cu"\n'
        "diff --git a/vllm/b.py b/vllm/b.py\n"
        "--- a/vllm/b.py\n"
        "+++ b/vllm/b.py\n"
        "@@ -1 +1 @@\n"
        "--- a/csrc/not_a_header.cu\n"  # a body line removing "-- a/..."
    )
    monkeypatch.setattr(reference_engines, "patch_files", lambda engine: [patch])
    assert reference_engines.patched_paths_not_installed("vllm") == [
        "0001-x.patch: csrc/moe/topk.cu",
        "0001-x.patch: CMakeLists.txt",
        "0001-x.patch: csrc/kernel.cu",  # renamed into a native source
        "0001-x.patch: csrc/x y.cu",  # a path with a space
        "0001-x.patch: csrc/café.cu",  # a C-quoted path
    ]
