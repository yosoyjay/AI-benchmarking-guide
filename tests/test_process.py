"""Tests for infra.process -- CSV writing, converters, and process_run pipeline."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from infra.capture import RunContext
from infra.process import (
    CSV_COLUMNS,
    _row,
    append_combined,
    cpu_stream_to_csv,
    fio_to_csv,
    flash_attention_to_csv,
    gemm_cublas_lt_to_csv,
    gemm_hipblas_lt_to_csv,
    hbm_bandwidth_to_csv,
    llama3_pretrain_to_csv,
    llm_benchmark_amd_to_csv,
    llm_benchmark_nv_to_csv,
    multichase_to_csv,
    nccl_bandwidth_to_csv,
    process_run,
    rccl_bandwidth_to_csv,
    transfer_bench_to_csv,
    write_processed_csv,
)


def _sample_row(**overrides: str) -> dict[str, str]:
    """Build a valid CSV row dict with sensible defaults."""
    base = {
        "datetime": "2025-06-01T12:00:00",
        "version": "abc123",
        "sku": "NVIDIA H200",
        "platform": "nvidia",
        "benchmark": "gemm_cublas_lt",
        "metric_name": "tflops",
        "metric_value": "123.4",
        "unit": "TFLOPS",
        "metadata_json": '{"m": 1024}',
    }
    base.update(overrides)
    return base


def _make_ctx(tmp_path: Path, benchmark: str = "test_bench", **extra: Any) -> RunContext:
    """Build a RunContext rooted in tmp_path."""
    session_dir = tmp_path / "session"
    run_dir = session_dir / "run"
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "processed").mkdir(parents=True, exist_ok=True)
    ts = datetime(2025, 6, 1, 12, 0, 0)
    return RunContext(
        benchmark=benchmark,
        sku="TestSKU",
        platform="nvidia",
        version="abc123",
        timestamp=ts,
        session_dir=session_dir,
        run_dir=run_dir,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# CSV_COLUMNS
# ---------------------------------------------------------------------------


class TestCSVColumns:
    def test_has_nine_columns(self) -> None:
        assert len(CSV_COLUMNS) == 9

    def test_required_columns_present(self) -> None:
        for col in ["datetime", "version", "sku", "platform", "benchmark", "metric_name", "metric_value", "unit"]:
            assert col in CSV_COLUMNS

    def test_metadata_json_present(self) -> None:
        assert "metadata_json" in CSV_COLUMNS


# ---------------------------------------------------------------------------
# _row helper
# ---------------------------------------------------------------------------


class TestRowHelper:
    def test_builds_all_columns(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        row = _row(ctx, "tflops", 42.5, "TFLOPS", {"m": 1024})
        assert set(row.keys()) == set(CSV_COLUMNS)

    def test_metadata_is_json_string(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        row = _row(ctx, "tflops", 42.5, "TFLOPS", {"m": 1024, "n": 2048})
        meta = json.loads(row["metadata_json"])
        assert meta["m"] == 1024
        assert meta["n"] == 2048

    def test_metric_value_is_string(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        row = _row(ctx, "tflops", 42.5, "TFLOPS", {})
        assert row["metric_value"] == "42.5"

    def test_timestamp_is_iso(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        row = _row(ctx, "tflops", 1, "TFLOPS", {})
        assert row["datetime"] == "2025-06-01T12:00:00"


# ---------------------------------------------------------------------------
# write_processed_csv
# ---------------------------------------------------------------------------


class TestWriteProcessedCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 6, 1, 12, 0, 0)
        rows = [_sample_row()]
        path = write_processed_csv(rows, run_dir, "gemm_cublas_lt", "abc123", ts)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_filename_format(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 6, 1, 12, 0, 0)
        path = write_processed_csv([_sample_row()], run_dir, "nccl_bandwidth", "v1", ts)
        assert path.name == "nccl_bandwidth_v1_20250601_120000.csv"

    def test_header_row(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 1, 1)
        path = write_processed_csv([_sample_row()], run_dir, "test", "v1", ts)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS

    def test_row_content(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 1, 1)
        row = _sample_row(metric_value="42.5")
        path = write_processed_csv([row], run_dir, "test", "v1", ts)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["metric_value"] == "42.5"

    def test_multiple_rows(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 1, 1)
        rows = [_sample_row(metric_value=str(i)) for i in range(5)]
        path = write_processed_csv(rows, run_dir, "test", "v1", ts)
        with open(path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 5

    def test_empty_rows(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 1, 1)
        path = write_processed_csv([], run_dir, "test", "v1", ts)
        with open(path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 0


# ---------------------------------------------------------------------------
# append_combined
# ---------------------------------------------------------------------------


class TestAppendCombined:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        combined_path = tmp_path / "combined.csv"
        append_combined([_sample_row()], combined_path)
        assert combined_path.exists()
        with open(combined_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS

    def test_appends_without_duplicate_header(self, tmp_path: Path) -> None:
        combined_path = tmp_path / "combined.csv"
        append_combined([_sample_row(metric_value="1")], combined_path)
        append_combined([_sample_row(metric_value="2")], combined_path)
        with open(combined_path) as f:
            lines = f.readlines()
        # 1 header + 2 data lines
        assert len(lines) == 3
        assert lines[0].startswith("datetime")
        assert "1" in lines[1]
        assert "2" in lines[2]

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        combined_path = tmp_path / "combined.csv"
        append_combined([_sample_row()], combined_path)
        append_combined([_sample_row(), _sample_row()], combined_path)
        with open(combined_path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 3

    def test_empty_file_gets_header(self, tmp_path: Path) -> None:
        combined_path = tmp_path / "combined.csv"
        combined_path.write_text("")
        append_combined([_sample_row()], combined_path)
        with open(combined_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS


# ---------------------------------------------------------------------------
# to_csv_rows converters
# ---------------------------------------------------------------------------


class TestGemmCublasLtToCsv:
    def test_two_rows_per_input(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "gemm_cublas_lt", datatype="fp8e4m3")
        parsed = [{"m": "1024", "n": "2048", "k": "4096", "batch_size": "1", "time_us": "10.5", "tflops": "123.4"}]
        rows = gemm_cublas_lt_to_csv(ctx, parsed)
        assert len(rows) == 2
        names = {r["metric_name"] for r in rows}
        assert names == {"tflops", "time_us"}

    def test_metadata_includes_datatype(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "gemm_cublas_lt", datatype="fp16")
        parsed = [{"m": "1024", "n": "1024", "k": "1024", "batch_size": "1", "time_us": "5", "tflops": "100"}]
        rows = gemm_cublas_lt_to_csv(ctx, parsed)
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["datatype"] == "fp16"

    def test_empty_input(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "gemm_cublas_lt")
        assert gemm_cublas_lt_to_csv(ctx, []) == []


class TestGemmHipblasLtToCsv:
    def test_one_row_per_input(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "gemm_hipblas_lt")
        parsed = [{"m": "1024", "n": "2048", "k": "4096", "tflops": 123.4}]
        rows = gemm_hipblas_lt_to_csv(ctx, parsed)
        assert len(rows) == 1
        assert rows[0]["metric_name"] == "tflops"


class TestNcclBandwidthToCsv:
    def test_output(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "nccl_bandwidth", algorithm="NVLS")
        parsed = [{"size": "8", "bandwidth": "123.4"}, {"size": "16", "bandwidth": "234.5"}]
        rows = nccl_bandwidth_to_csv(ctx, parsed)
        assert len(rows) == 2
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["algorithm"] == "NVLS"


class TestRcclBandwidthToCsv:
    def test_includes_algo(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "rccl_bandwidth")
        parsed = [{"size": "8", "bandwidth": "100"}]
        rows = rccl_bandwidth_to_csv(ctx, parsed, "Ring")
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["algorithm"] == "Ring"


class TestFlashAttentionToCsv:
    def test_two_metrics_per_input(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "flash_attention")
        parsed = [{"causal": "False", "headdim": 64, "flash2_tflops": 100.0, "pytorch_tflops": 50.0}]
        rows = flash_attention_to_csv(ctx, parsed)
        assert len(rows) == 2
        names = {r["metric_name"] for r in rows}
        assert names == {"flash2_tflops", "pytorch_tflops"}


class TestHbmBandwidthToCsv:
    def test_three_metrics_per_op(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "hbm_bandwidth", units="TB/s")
        summary = {"Copy": {"min": 1.0, "max": 2.0, "mean": 1.5}}
        rows = hbm_bandwidth_to_csv(ctx, summary)
        assert len(rows) == 3
        names = {r["metric_name"] for r in rows}
        assert names == {"bw_min", "bw_max", "bw_mean"}


class TestCpuStreamToCsv:
    def test_uses_gbs_unit(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "cpu_stream")
        summary = {"Copy": {"min": 10, "max": 20, "mean": 15}}
        rows = cpu_stream_to_csv(ctx, summary)
        assert all(r["unit"] == "GB/s" for r in rows)


class TestMultichaseToCsv:
    def test_output(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "multichase")
        node_names = ["NODE0", "NODE1"]
        parsed_rows = [{"cpu": "0", "NODE0": 3.2, "NODE1": 5.1}]
        rows = multichase_to_csv(ctx, node_names, parsed_rows)
        assert len(rows) == 2
        assert rows[0]["metric_name"] == "latency"


class TestFioToCsv:
    def test_output(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "fio")
        parsed = [("read", "1M", "1234MiB/s"), ("write", "512k", "567MiB/s")]
        rows = fio_to_csv(ctx, parsed)
        assert len(rows) == 2
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["rw"] == "read"


class TestTransferBenchToCsv:
    def test_output(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "transfer_bench")
        parsed = {"h2d": "100.5", "d2h": "95.3"}
        rows = transfer_bench_to_csv(ctx, parsed)
        assert len(rows) == 2


class TestLlmBenchmarkNvToCsv:
    def test_output(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "llm_benchmark", model="Llama-3.1-8B")
        parsed = [{"tp_size": "1", "input_len": "128", "output_len": "128", "throughput": "1000"}]
        rows = llm_benchmark_nv_to_csv(ctx, parsed)
        assert len(rows) == 1
        assert rows[0]["unit"] == "tokens/s"
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["model"] == "Llama-3.1-8B"


class TestLlmBenchmarkAmdToCsv:
    def test_output(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "llm_benchmark", model="Llama-3.1-70B")
        parsed = [("128", "1024", "8", "500")]
        rows = llm_benchmark_amd_to_csv(ctx, parsed)
        assert len(rows) == 1
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["tp_size"] == "8"


class TestLlama3PretrainToCsv:
    def test_both_values(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "llama3_pretrain", model_size="8b")
        rows = llama3_pretrain_to_csv(ctx, 3.5, 2.1)
        assert len(rows) == 2
        names = {r["metric_name"] for r in rows}
        assert names == {"time_steady_state", "loss_steady_state"}

    def test_none_values(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "llama3_pretrain")
        rows = llama3_pretrain_to_csv(ctx, None, None)
        assert len(rows) == 0

    def test_partial(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "llama3_pretrain")
        rows = llama3_pretrain_to_csv(ctx, 3.5, None)
        assert len(rows) == 1
        assert rows[0]["metric_name"] == "time_steady_state"


# ---------------------------------------------------------------------------
# process_run pipeline
# ---------------------------------------------------------------------------


class TestProcessRun:
    def test_writes_csv_and_combined(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "test_bench")
        csv_rows = [_sample_row()]
        path = process_run("test_bench", ctx, csv_rows)
        assert path is not None
        assert path.exists()
        combined = ctx.session_dir / "combined.csv"
        assert combined.exists()

    def test_returns_none_for_empty_rows(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "test_bench")
        path = process_run("test_bench", ctx, [])
        assert path is None

    def test_combined_accumulates(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path, "test_bench")
        process_run("test_bench", ctx, [_sample_row(metric_value="1")])
        process_run("test_bench", ctx, [_sample_row(metric_value="2")])
        combined = ctx.session_dir / "combined.csv"
        with open(combined) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 2
