"""Tests for infra.process -- CSV writing infrastructure."""

import csv
from datetime import datetime

from infra.process import CSV_COLUMNS, append_combined, write_processed_csv


def _sample_row(**overrides):
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


# ---------------------------------------------------------------------------
# CSV_COLUMNS
# ---------------------------------------------------------------------------


class TestCSVColumns:
    def test_has_nine_columns(self):
        assert len(CSV_COLUMNS) == 9

    def test_required_columns_present(self):
        for col in ["datetime", "version", "sku", "platform", "benchmark", "metric_name", "metric_value", "unit"]:
            assert col in CSV_COLUMNS

    def test_metadata_json_present(self):
        assert "metadata_json" in CSV_COLUMNS


# ---------------------------------------------------------------------------
# write_processed_csv
# ---------------------------------------------------------------------------


class TestWriteProcessedCsv:
    def test_creates_file(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 6, 1, 12, 0, 0)
        rows = [_sample_row()]
        path = write_processed_csv(rows, run_dir, "gemm_cublas_lt", "abc123", ts)
        assert path.exists()
        assert path.suffix == ".csv"

    def test_filename_format(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 6, 1, 12, 0, 0)
        path = write_processed_csv([_sample_row()], run_dir, "nccl_bandwidth", "v1", ts)
        assert path.name == "nccl_bandwidth_v1_20250601_120000.csv"

    def test_header_row(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 1, 1)
        path = write_processed_csv([_sample_row()], run_dir, "test", "v1", ts)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS

    def test_row_content(self, tmp_path):
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

    def test_multiple_rows(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "processed").mkdir(parents=True)
        ts = datetime(2025, 1, 1)
        rows = [_sample_row(metric_value=str(i)) for i in range(5)]
        path = write_processed_csv(rows, run_dir, "test", "v1", ts)
        with open(path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 5

    def test_empty_rows(self, tmp_path):
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
    def test_creates_file_with_header(self, tmp_path):
        combined_path = tmp_path / "combined.csv"
        append_combined([_sample_row()], combined_path)
        assert combined_path.exists()
        with open(combined_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS

    def test_appends_without_duplicate_header(self, tmp_path):
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

    def test_appends_to_existing(self, tmp_path):
        combined_path = tmp_path / "combined.csv"
        append_combined([_sample_row()], combined_path)
        append_combined([_sample_row(), _sample_row()], combined_path)
        with open(combined_path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 3

    def test_empty_file_gets_header(self, tmp_path):
        combined_path = tmp_path / "combined.csv"
        combined_path.write_text("")
        append_combined([_sample_row()], combined_path)
        with open(combined_path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CSV_COLUMNS
