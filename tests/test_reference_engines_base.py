"""BASE parsing and the per-platform base it pins (reference_engines/__init__.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from reference_engines import ENGINES, BaseError, format_base, parse_base, read_base

COMMIT = "a" * 40
OTHER = "b" * 40
AMD = "sha256:" + "1" * 64
ARM = "sha256:" + "2" * 64


def _base(*extra: str) -> str:
    return "\n".join(
        [
            "# header",
            "upstream_repo=https://example.invalid/engine.git",
            f"upstream_commit={COMMIT}",
            "image=registry.invalid/engine:tag",
            f"digest.linux/amd64={AMD}",
            *extra,
        ]
    )


def test_a_platform_published_under_another_tag_and_commit():
    base = parse_base(
        _base(
            "image.linux/arm64=registry.invalid/engine:older-arm64",
            f"upstream_commit.linux/arm64={OTHER}",
            f"digest.linux/arm64={ARM}",
        )
    )
    assert base.image_for("linux/amd64") == f"registry.invalid/engine@{AMD}"
    assert base.image_for("linux/arm64") == f"registry.invalid/engine@{ARM}"
    assert base.commit_for("linux/amd64") == COMMIT
    assert base.commit_for("linux/arm64") == OTHER
    # format_base writes what parse_base reads.
    assert parse_base(format_base(base, "header")) == base


def test_a_platform_without_its_own_commit_uses_the_base_commit():
    base = parse_base(_base(f"digest.linux/arm64={ARM}"))
    assert base.commit_for("linux/arm64") == COMMIT


@pytest.mark.parametrize(
    "extra, message",
    [
        ((f"upstream_commit.linux/arm64={OTHER}",), "has no digest"),
        ((f"digest.linux/arm64={ARM}", "upstream_commit.linux/arm64=abc"), "40-hex"),
        (("image.linux/arm64=registry.invalid/engine:x",), "has no digest"),
    ],
)
def test_malformed_platform_entries_are_refused(extra, message):
    with pytest.raises(BaseError, match=message):
        parse_base(_base(*extra))


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
