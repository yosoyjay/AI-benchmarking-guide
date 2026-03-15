"""Tests for GEMM CuBLASLt benchmark helpers."""

from benchmarks.nvidia.gemm_cublas_lt import _build_table, parse_cublaslt_line


class TestParseCublasltLine:
    def test_valid_six_columns(self) -> None:
        result = parse_cublaslt_line("1024 2048 4096 1 123.45 67.89")
        assert result == {
            "m": "1024",
            "n": "2048",
            "k": "4096",
            "batch_size": "1",
            "time_us": "123.45",
            "tflops": "67.89",
        }

    def test_returns_none_for_too_few_columns(self) -> None:
        assert parse_cublaslt_line("1024 2048") is None

    def test_returns_none_for_too_many_columns(self) -> None:
        assert parse_cublaslt_line("1024 2048 4096 1 123.45 67.89 extra") is None

    def test_returns_none_for_empty(self) -> None:
        assert parse_cublaslt_line("") is None

    def test_returns_none_for_whitespace_only(self) -> None:
        assert parse_cublaslt_line("   ") is None


class TestBuildTable:
    def test_smoke(self) -> None:
        rows = [
            {"m": "1024", "n": "1024", "k": "1024", "batch_size": "1", "time_us": "50.0", "tflops": "100.0"},
            {"m": "2048", "n": "2048", "k": "2048", "batch_size": "1", "time_us": "75.0", "tflops": "200.0"},
        ]
        table = _build_table(rows)
        assert len(table.rows) == 2
        assert "M" in table.field_names
        assert "TFLOPS" in table.field_names

    def test_empty_rows(self) -> None:
        table = _build_table([])
        assert len(table.rows) == 0
