"""Source replay refuses mutation/overwrite even with Python assertions disabled."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/trainium2/replay_decode_buffer_capacity.py"


def module():
    spec = importlib.util.spec_from_file_location("decode_capacity_replay", TOOL)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_bad_source_preserved_and_no_output(tmp_path):
    source, output = tmp_path / "stock.py", tmp_path / "new.py"
    source.write_bytes(b"unknown framework source")
    with pytest.raises(ValueError, match="exact validated stock"):
        module().replay(source, output)
    assert source.read_bytes() == b"unknown framework source" and not output.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_output_and_symlink_preserved(tmp_path, symlink):
    source, output = tmp_path / "stock.py", tmp_path / "new.py"
    source.write_bytes(b"original")
    if symlink:
        output.symlink_to(source)
    else:
        output.write_bytes(b"existing candidate")
    expected = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        module().replay(source, output)
    assert output.read_bytes() == expected and source.read_bytes() == b"original"


def test_optimized_python_cannot_bypass_source_guard(tmp_path):
    source, output = tmp_path / "stock.py", tmp_path / "new.py"
    source.write_bytes(b"unvalidated")
    result = subprocess.run(
        [sys.executable, "-O", str(TOOL), str(source), str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "exact validated stock" in result.stderr
    assert source.read_bytes() == b"unvalidated" and not output.exists()
