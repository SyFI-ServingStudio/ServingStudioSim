"""cutedsl_routed_fused_fc1_fc2_sm100: the fixed per-band chain and its vendored sources (CPU)."""

from __future__ import annotations

import pytest

from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.nvfp4_fused_moe_cutedsl_fused import _VARIANTS, _VENDOR, chain_for


@pytest.mark.parametrize(
    ("t", "variant", "ocfg"),
    [
        (1, "f_rwepf2_tab", (1, 128, 0)),
        (64, "f_rwepf2_tab", (1, 128, 0)),
        (72, "f_rwepf2_tabea6d4", (1, 128, 0)),
        (128, "f_rwepf2_tabea6d4", (1, 128, 0)),
        (136, "f_rwepf2_tabea6d8", (2, 128, 0)),
        (256, "f_rwepf2_tabea6d8", (2, 128, 0)),
        (512, "f_rwepf2_tabea6d8", (1, 128, 0)),
        (1024, "f_rwepf2_tabd8", (8, 256, 7)),
        (1152, "f_rwepf2_tabd8", (16, 256, 7)),
        (2048, "f_rwepf2_tabd8", (16, 256, 7)),
        (3072, "f_rwepf_cl2tab", (2, 128, 5)),
        (65536, "f_rwepf_cl2tab", (2, 128, 5)),
    ],
)
def test_chain_per_band(t: int, variant: str, ocfg: tuple[int, int, int]) -> None:
    """The measured rows were taken with these chains; a band edge that moves
    silently re-prices every row on the other side of it."""
    ch = chain_for(t)
    assert (ch["variant"], ch["ocfg"]) == (variant, ocfg)
    assert variant in _VARIANTS
    if ch["ocfg"][2] == 7:
        tb = ch["ocfg"][0]
        assert t % tb == 0 and t // tb <= 128


def test_unsupported_tokens_raise() -> None:
    # Above 2048 tokens the router's block of two needs an even count.
    with pytest.raises(ProfilerNotImplemented):
        chain_for(3073)


def test_vendored_sources_take_their_config_from_the_runner() -> None:
    """The runner passes compile-time options through ``_PFUSED_CFG``; a vendored
    file that reads the environment instead would build a different kernel."""
    for f in ("p_fused.py", "p_fused_dyn.py", "n_fc1.py", "p_route.py", "p_route8.cuh"):
        text = (_VENDOR / f).read_text()
        assert "d3563d86" in text.splitlines()[1]  # provenance header
        assert "os.environ.get(" not in text
