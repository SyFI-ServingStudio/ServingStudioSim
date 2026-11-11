"""The shared campaign.yaml writer changes layout only, never content."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from launcher.alignment_campaign.packfile import GENERATED_HEADER, dump_campaign, write_campaign

PACK_ROOT = Path(__file__).resolve().parents[1] / "presets" / "alignment"


def _document():
    shared = {"packages": [{"name": "nvtx", "version": "0.2.15"}]}
    return {
        "schema_version": 1,
        "variants": {
            "a": {
                "python_runtime": shared,
                "server": {
                    "extra_args": [
                        "--block-size", "128",
                        "--speculative-config", '{"method":"dspark","num_speculative_tokens":5}',
                        "--cudagraph-capture-sizes", *[str(size) for size in range(8, 2049, 8)],
                    ],
                },
                "arch": {"cudagraph_capture_sizes": list(range(8, 2049, 8)), "empty": []},
                "profile_passes": [{"kind": "nsys", "name": "profile_nsys"}],
            },
            "b": {"python_runtime": shared},
        },
        "cases": [
            {
                "index": 1,
                "workload_trace": {
                    "shapes": [list(pair) for pair in [(1024, 32), (2048, 64)] * 40],
                    "repeats": 2,
                },
                "speculative_acceptance": {
                    "value": [0.7957, 0.802, None, True],
                    "status": "measured",
                },
            }
        ],
    }


def test_round_trips_to_the_same_object():
    document = _document()
    assert yaml.safe_load(dump_campaign(document)) == document


def test_scalar_lists_are_flow_and_mappings_stay_block():
    text = dump_campaign(_document())
    assert "shapes: [[1024, 32], [2048, 64], [1024, 32]" in text
    assert "extra_args: [--block-size, '128', --speculative-config," in text
    assert "cudagraph_capture_sizes: [8, 16, 24," in text
    assert "empty: []" in text
    # Lists of mappings keep one item per block entry.
    assert "profile_passes:\n    - kind: nsys\n" in text
    # A flow list wraps at the width instead of growing one line per item.
    assert max(len(line) for line in text.splitlines()) <= 120
    block = yaml.safe_dump(_document(), sort_keys=False, width=110)
    assert text.count("\n") * 5 < block.count("\n")


def test_write_campaign_prefixes_the_generated_header(tmp_path):
    path = write_campaign(tmp_path / "campaign.yaml", _document())
    text = path.read_text()
    assert text.startswith(GENERATED_HEADER)
    assert yaml.safe_load(text) == _document()


@pytest.mark.parametrize(
    "pack_dir",
    sorted(path.parent for path in PACK_ROOT.glob("*/make_pack.py")),
    ids=lambda path: path.name,
)
def test_generated_packs_are_written_by_the_shared_writer(pack_dir):
    """A generator's committed output must be what the writer gives for its
    content, so regenerating a pack never reflows it."""
    text = (pack_dir / "campaign.yaml").read_text()
    assert text.startswith(GENERATED_HEADER)
    assert "write_campaign(" in (pack_dir / "make_pack.py").read_text()
    assert text == GENERATED_HEADER + dump_campaign(yaml.safe_load(text))
