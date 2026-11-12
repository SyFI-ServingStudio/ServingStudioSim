"""The optimality R4a ceilings read from profile.db: best GEMM per dtype, large copies only."""

from __future__ import annotations

import sqlite3

from profiling.db.achievable import MIN_COPY_INPUT_BYTES, achievable_peaks


def _db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE single_gemm (gpu_name TEXT, backend TEXT, m INT, n INT, k INT, "
        "dtype TEXT, tflops REAL, is_outlier INT)"
    )
    conn.execute(
        "CREATE TABLE elementwise (gpu_name TEXT, backend TEXT, input_size_bytes INT, "
        "output_size_bytes INT, memory_bandwidth_gbps REAL, is_outlier INT)"
    )
    conn.executemany(
        "INSERT INTO single_gemm VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("NVIDIA B200", "torch", 4096, 4096, 4096, "bf16", 1500.0, 0),
            ("NVIDIA B200", "vllm", 8192, 4096, 4096, "bf16", 1650.0, 0),
            ("NVIDIA B200", "vllm", 1, 1, 1, "bf16", 9000.0, 1),  # outlier
            ("NVIDIA B200", "deepgemm", 8192, 4096, 4096, "fp8_e4m3", 3200.0, 0),
            ("NVIDIA H200", "torch", 8192, 4096, 4096, "bf16", 800.0, 0),
        ],
    )
    big = MIN_COPY_INPUT_BYTES
    conn.executemany(
        "INSERT INTO elementwise VALUES (?, ?, ?, ?, ?, ?)",
        [
            # L2-resident copy: faster than HBM, must not count.
            ("NVIDIA B200", "triton", 80 << 20, 80 << 20, 8400.0, 0),
            ("NVIDIA B200", "triton", big, big, 7000.0, 0),
            ("NVIDIA B200", "triton", 2 * big, 2 * big, 7050.0, 0),
            # A reduction (fan-in 2) is not a copy.
            ("NVIDIA B200", "triton", 4 * big, 2 * big, 7900.0, 0),
            ("NVIDIA B200", "torch", 4 * big, 4 * big, 7800.0, 1),  # outlier
        ],
    )
    conn.commit()
    conn.close()


def test_best_gemm_per_dtype_and_best_large_copy(tmp_path):
    db = tmp_path / "profile.db"
    _db(db)
    peaks = achievable_peaks(db, "NVIDIA B200")

    assert peaks["gemm_by_dtype"]["bf16"] == {
        "tflops": 1650.0,
        "backend": "vllm",
        "shape": {"m": 8192, "n": 4096, "k": 4096},
    }
    assert peaks["gemm_by_dtype"]["fp8_e4m3"]["tflops"] == 3200.0
    assert peaks["copy"] == {
        "gbps": 7050.0,
        "backend": "triton",
        "input_size_bytes": 2 * MIN_COPY_INPUT_BYTES,
    }


def test_a_gpu_without_rows_reports_nothing(tmp_path):
    db = tmp_path / "profile.db"
    _db(db)
    peaks = achievable_peaks(db, "NVIDIA A100")
    assert peaks["gemm_by_dtype"] == {}
    assert peaks["copy"] is None
