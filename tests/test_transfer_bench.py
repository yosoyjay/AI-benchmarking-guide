"""Tests for AMD TransferBench benchmark module."""

from benchmarks.amd import transfer_bench


class TestParseTransferBenchOutput:
    def test_extracts_h2d_and_d2h(self):
        text = "sum  | 48.123 |  0.000 |  0.000 |  0.000 | 32.456 |  0.000 |  0.000\n"
        result = transfer_bench.parse_transfer_bench_output(text)
        assert result["h2d"] == "48.123"
        assert result["d2h"] == "32.456"

    def test_skips_separator_lines(self):
        text = "============================\nsum  | 48.123 |  0.000 |  0.000 |  0.000 | 32.456 |  0.000\n"
        result = transfer_bench.parse_transfer_bench_output(text)
        assert result["h2d"] == "48.123"
        assert result["d2h"] == "32.456"

    def test_returns_error_for_empty(self):
        result = transfer_bench.parse_transfer_bench_output("")
        assert result["h2d"] == "error"
        assert result["d2h"] == "error"

    def test_returns_error_for_no_sum_line(self):
        result = transfer_bench.parse_transfer_bench_output("some output\nno sum here")
        assert result["h2d"] == "error"
        assert result["d2h"] == "error"

    def test_returns_error_for_short_sum_line(self):
        result = transfer_bench.parse_transfer_bench_output("sum")
        assert result["h2d"] == "error"
        assert result["d2h"] == "error"


class TestBuildTable:
    def test_smoke(self):
        parsed = {"h2d": "48.123", "d2h": "32.456"}
        table = transfer_bench._build_table(parsed)
        text = table.get_string()
        assert "48.123" in text
        assert "32.456" in text
        assert "Host to Device" in text
        assert "Device to Host" in text

    def test_error_values(self):
        parsed = {"h2d": "error", "d2h": "error"}
        table = transfer_bench._build_table(parsed)
        text = table.get_string()
        assert text.count("error") == 2


class TestModuleConstants:
    def test_repo_constant_exists(self):
        assert hasattr(transfer_bench, "_TRANSFERBENCH_REPO")
        assert "TransferBench" in transfer_bench._TRANSFERBENCH_REPO

    def test_run_is_callable(self):
        assert callable(transfer_bench.run)
