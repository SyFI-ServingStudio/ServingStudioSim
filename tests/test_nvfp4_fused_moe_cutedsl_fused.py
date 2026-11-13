"""cutedsl_routed_fused_fc1_fc2_sm100: registration and the fixed per-band chain (CPU only)."""

from __future__ import annotations

import pytest

from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.moe.nvfp4_fused_moe_cutedsl_fused import _VENDOR, chain_for

BACKEND = "cutedsl_routed_fused_fc1_fc2_sm100"


def test_registered_under_routed_contract() -> None:
    spec = find_kernel_profiler_spec("nvfp4_fused_moe", BACKEND)
    routed = find_kernel_profiler_spec("nvfp4_fused_moe", "flashinfer_trtllm_routed_sm100")
    assert spec.args_schema is routed.args_schema
    assert spec.table_name == routed.table_name
    assert spec.subprocess_env == "vllm_env"
    assert spec.runner_ref.load is not None


@pytest.mark.parametrize(
    ("t", "name", "ocfg"),
    [
        (1, "tfusedrepf2", (1, 128, 0)),
        (64, "tfusedrepf2", (1, 128, 0)),
        (72, "tfusedrepf2ea6d4", (1, 128, 0)),
        (128, "tfusedrepf2ea6d4", (1, 128, 0)),
        (136, "tfusedrepf2ea6d8", (2, 128, 0)),
        (256, "tfusedrepf2ea6d8", (2, 128, 0)),
        (512, "tfusedrepf2ea6d8", (1, 128, 0)),
        (1024, "tfusedrepf2d8", (8, 256, 7)),
        (1152, "tfusedrepf2d8", (16, 256, 7)),
        (2048, "tfusedrepf2d8", (16, 256, 7)),
        (3072, "ufusedrepfcl2", (2, 128, 5)),
        (65536, "ufusedrepfcl2", (2, 128, 5)),
    ],
)
def test_chain_per_band(t: int, name: str, ocfg: tuple[int, int, int]) -> None:
    ch = chain_for(t)
    assert (ch["name"], ch["ocfg"]) == (name, ocfg)
    if ch["ocfg"][2] == 7:
        tb = ch["ocfg"][0]
        assert t % tb == 0 and t // tb <= 128


def test_unsupported_tokens_raise() -> None:
    with pytest.raises(ProfilerNotImplemented):
        chain_for(3073)


def test_vendored_sources_present() -> None:
    for f in ("p_fused.py", "p_fused_dyn.py", "n_fc1.py", "p_route.py", "p_route8.cuh"):
        text = (_VENDOR / f).read_text()
        assert "d3563d86" in text  # provenance header
        assert "os.environ.get(" not in text
