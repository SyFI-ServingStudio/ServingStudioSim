"""CPU test for the rocprofv3 capture driver (``alignment/profiler/rocprof_capture``).

Mirrors the nsys-capture contract tests. It proves, without rocprofv3 or a GPU:

1. executable resolution refuses a non-absolute path and records the version;
2. the assembled ``rocprofv3 ... -- <server argv>`` carries the mandatory
   ``--kernel-trace`` + ``--marker-trace`` + ``--output-format rocpd`` and the
   command after a literal ``--``;
3. ``validate_rocpd`` recognizes a complete capture and flags the two failure
   modes (no iteration roctx ranges / no dispatch rows);
4. the ``rocpd-capture`` CLI runs the whole capture -> rocpd -> parse path: with
   the rocprofv3 binary and the GPU subprocess stood in for (the fake runner
   writes a synthetic rocpd), the real ``alignment/rocpd`` producer emits
   ``parsed.json`` + ``parsed.kernels.parquet`` + ``kernel_sequences.json``.
"""

from __future__ import annotations

import sqlite3
import subprocess

import pyarrow.parquet as pq
import pytest

from alignment.nsys.parsed_io import _KERNEL_SCHEMA, kernel_rows_path
from alignment.profiler import rocprof_capture
from alignment.profiler.config import RocprofConfig
from alignment.profiler.rocprof_capture import (
    ResolvedRocprofExecutable,
    build_capture_argv,
    locate_rocpd,
    resolve_rocprof_executable,
    validate_rocpd,
)

_GUID = "0000b1c7_c35b_735b_96a7_f0a02ff013cc"
_PID = 4242
_AGENT = 1


def _write_synthetic_rocpd(path, dispatches, regions):
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE rocpd_string_{_GUID} (id INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        f"CREATE TABLE rocpd_info_kernel_symbol_{_GUID} "
        "(id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_kernel_dispatch_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, agent_id INTEGER, "
        "kernel_id INTEGER, dispatch_id INTEGER, queue_id INTEGER, stream_id INTEGER, "
        "start BIGINT, end BIGINT, region_name_id INTEGER)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_region_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, start BIGINT, end BIGINT, name_id INTEGER)"
    )
    strings: dict[str, int] = {}

    def intern(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            conn.execute(
                f"INSERT INTO rocpd_string_{_GUID} (id, string) VALUES (?, ?)",
                (strings[text], text),
            )
        return strings[text]

    symbols: dict[str, int] = {}
    for _s, _e, name, _stream in dispatches:
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)",
                (symbols[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name, stream) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, region_name_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, 55, _AGENT, symbols[name], i, 0, stream, start, end, 0),
        )
    for i, (start, end, text) in enumerate(regions, start=1):
        conn.execute(
            f"INSERT INTO rocpd_region_{_GUID} (id, pid, tid, start, end, name_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, _PID, 55, start, end, intern(text)),
        )
    conn.commit()
    conn.close()


def _complete_fixture(path):
    _write_synthetic_rocpd(
        path,
        dispatches=[
            (500, 900, "warmup_fill_kernel", 0),
            (1_100, 1_300, "rms_norm_kernel", 0),
            (1_400, 2_000, "hipblaslt_gemm_f16", 0),
            (2_100, 2_900, "flash_fwd_attn_kernel", 0),
            (3_000, 3_200, "silu_and_mul_kernel", 0),
            (3_300, 3_900, "hipblaslt_gemm_f16", 0),
            (4_000, 4_200, "vectorized_elementwise_kernel", 0),
        ],
        regions=[(1_000, 5_000, "vllm_iteration(0): forward")],
    )


def _fake_exe(tmp_path):
    return ResolvedRocprofExecutable(path=tmp_path / "rocprofv3", version="rocprofv3 1.0.0")


def test_resolve_requires_absolute_path(monkeypatch):
    monkeypatch.delenv("ROCPROF_BIN", raising=False)
    with pytest.raises(ValueError, match="not configured"):
        resolve_rocprof_executable(None)
    with pytest.raises(ValueError, match="must be an absolute path"):
        resolve_rocprof_executable("rocprofv3")


def test_resolve_records_version(tmp_path, monkeypatch):
    exe = tmp_path / "rocprofv3"
    exe.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then printf "rocprofv3 1.2.3\\n"; else exit 2; fi\n'
    )
    exe.chmod(0o755)
    monkeypatch.setenv("ROCPROF_BIN", str(exe))
    resolved = resolve_rocprof_executable(None)
    assert resolved.provenance() == {"executable": str(exe.resolve()), "version": "rocprofv3 1.2.3"}


def test_capture_argv_carries_mandatory_flags_and_command(tmp_path):
    argv = build_capture_argv(
        _fake_exe(tmp_path),
        RocprofConfig(),
        ["python", "-m", "vllm.entrypoints.cli.main", "serve", "model"],
        tmp_path / "out",
        "rank0",
    )
    assert argv[0] == str(tmp_path / "rocprofv3")
    assert "--kernel-trace" in argv
    assert "--marker-trace" in argv  # roctx iteration ranges
    assert argv[argv.index("--output-format") + 1] == "rocpd"
    assert argv[argv.index("-d") + 1] == str(tmp_path / "out")
    assert argv[argv.index("-o") + 1] == "rank0"
    cut = argv.index("--")
    assert argv[cut + 1 :] == ["python", "-m", "vllm.entrypoints.cli.main", "serve", "model"]


def test_config_requires_traces_and_rocpd_format():
    with pytest.raises(ValueError, match="kernel_trace is required"):
        RocprofConfig(kernel_trace=False).validate()
    with pytest.raises(ValueError, match="marker_trace is required"):
        RocprofConfig(marker_trace=False).validate()
    with pytest.raises(ValueError, match="output_format"):
        RocprofConfig(output_format="csv").validate()


def test_validate_rocpd_ok_and_failure_modes(tmp_path):
    good = tmp_path / "good.db"
    _complete_fixture(good)
    report = validate_rocpd(good)
    assert report["ok"] is True
    assert report["iteration_ranges"] == 1
    assert report["kernel_rows"] == 7

    no_markers = tmp_path / "no_markers.db"
    _write_synthetic_rocpd(no_markers, [(100, 200, "rms_norm_kernel", 0)], regions=[])
    assert validate_rocpd(no_markers)["ok"] is False

    no_kernels = tmp_path / "no_kernels.db"
    _write_synthetic_rocpd(no_kernels, [], regions=[(1_000, 5_000, "vllm_iteration(0): forward")])
    report = validate_rocpd(no_kernels)
    assert report["ok"] is False
    assert report["kernel_rows"] == 0


def test_locate_rocpd_accepts_results_and_plain_suffix(tmp_path):
    (tmp_path / "a_results.db").write_text("")
    assert locate_rocpd(tmp_path, "a").name == "a_results.db"
    (tmp_path / "b.db").write_text("")
    assert locate_rocpd(tmp_path, "b").name == "b.db"
    with pytest.raises(FileNotFoundError):
        locate_rocpd(tmp_path, "missing")


def test_cli_capture_then_parse_end_to_end(tmp_path):
    """The rocpd-capture CLI drives capture -> rocpd -> parse, GPU subprocess mocked."""
    out_dir = tmp_path / "cap"
    parsed_out = tmp_path / "parsed.json"
    sequences_out = tmp_path / "kernel_sequences.json"

    captured_argv = {}

    def fake_runner(argv, env=None, check=True):
        # Stand in for the real `rocprofv3 ... -- vllm serve ...` GPU subprocess:
        # write the synthetic rocpd where the driver told rocprofv3 to put it.
        captured_argv["argv"] = argv
        db_dir = argv[argv.index("-d") + 1]
        name = argv[argv.index("-o") + 1]
        from pathlib import Path

        Path(db_dir).mkdir(parents=True, exist_ok=True)
        _complete_fixture(Path(db_dir) / f"{name}_results.db")
        return subprocess.CompletedProcess(argv, 0)

    rc = rocprof_capture.main(
        [
            "--output-dir",
            str(out_dir),
            "--output-name",
            "rank0",
            "--parsed-output",
            str(parsed_out),
            "--sequences-output",
            str(sequences_out),
            "--",
            "python",
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "model",
        ],
        resolver=lambda _p: _fake_exe(out_dir),
        runner=fake_runner,
    )
    assert rc == 0
    # The GPU subprocess really was invoked as a rocprofv3 capture of the server.
    assert "--kernel-trace" in captured_argv["argv"]
    assert captured_argv["argv"][captured_argv["argv"].index("--") + 1] == "python"

    # The real producer emitted the normalized Check-1 files.
    assert parsed_out.exists()
    rows_path = kernel_rows_path(parsed_out)
    assert rows_path.exists()
    table = pq.read_table(rows_path)
    assert table.schema.names == _KERNEL_SCHEMA.names
    for name in _KERNEL_SCHEMA.names:
        assert table.column(name).null_count == 0
    assert sequences_out.exists()


def test_cli_requires_double_dash():
    with pytest.raises(SystemExit):
        rocprof_capture.main(["--output-dir", "x", "--output-name", "n"])
