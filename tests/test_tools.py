"""Tests for BabelStream helpers, config loading, run_parallel_builds, and find_nvcc in infra/tools.py."""

import json
import logging
from unittest.mock import patch

import pytest

from infra.tools import (
    BABELSTREAM_OPS,
    find_nvcc,
    load_benchmark_config,
    parse_babelstream_output,
    run_parallel_builds,
    summarize_babelstream,
)

# Realistic BabelStream output (hip-stream / cuda-stream)
SAMPLE_BABELSTREAM_OUTPUT = """\
BabelStream
Version: 4.0
Implementation: HIP
Running kernels 100 times
Precision: double
Array size: 268.4 MB (=0.3 GB)
Total size: 805.3 MB (=0.8 GB)
Function    MBytes/sec  Min (sec)   Max (sec)   Average (sec)
Copy        1620000.000 0.00033     0.00035     0.00034
Mul         1610000.000 0.00033     0.00036     0.00034
Add         1600000.000 0.00050     0.00053     0.00051
Triad       1590000.000 0.00051     0.00054     0.00052
Dot         1580000.000 0.00034     0.00037     0.00035
"""


class TestParseBabelstreamOutput:
    def test_extracts_five_ops(self) -> None:
        results = parse_babelstream_output(SAMPLE_BABELSTREAM_OUTPUT)
        assert len(results) == 5
        names = [r[0] for r in results]
        assert names == list(BABELSTREAM_OPS)

    def test_bandwidth_values(self) -> None:
        results = parse_babelstream_output(SAMPLE_BABELSTREAM_OUTPUT)
        assert results[0] == ["Copy", "1620000.000"]
        assert results[4] == ["Dot", "1580000.000"]

    def test_empty_input_returns_empty(self) -> None:
        assert parse_babelstream_output("") == []

    def test_garbage_input_returns_empty(self) -> None:
        assert parse_babelstream_output("no ops here\njust noise\n") == []

    def test_partial_output(self) -> None:
        partial = "Copy        1620000.000 0.00033     0.00035     0.00034\n"
        results = parse_babelstream_output(partial)
        assert len(results) == 1
        assert results[0] == ["Copy", "1620000.000"]


class TestSummarizeBabelstream:
    def _make_buffer(self, n_runs: int = 3) -> list[list[list[str]]]:
        """Build a buffer of n identical runs for testing."""
        run = [
            ["Copy", "1620000"],
            ["Mul", "1610000"],
            ["Add", "1600000"],
            ["Triad", "1590000"],
            ["Dot", "1580000"],
        ]
        return [run] * n_runs

    def test_computes_min_max_mean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Suppress print and export_markdown side effects
        monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
        import infra.tools as tools_mod

        monkeypatch.setattr(tools_mod, "export_markdown", lambda *a, **kw: None)

        buffer = self._make_buffer(3)
        table = summarize_babelstream(
            buffer,
            divisor=1_000_000,
            units="TB/s",
            title="test",
            description="test",
        )
        assert table is not None
        assert len(table.rows) == 5
        # With identical runs, min == max == mean
        # Copy: 1620000 / 1_000_000 = 1.62
        copy_row = table.rows[0]
        assert copy_row == ["Copy", 1.62, 1.62, 1.62]

    def test_returns_none_on_incomplete_data(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
        import infra.tools as tools_mod

        monkeypatch.setattr(tools_mod, "export_markdown", lambda *a, **kw: None)

        # Buffer with only 2 ops per run (incomplete)
        incomplete_run = [["Copy", "100"], ["Mul", "200"]]
        with caplog.at_level(logging.WARNING):
            result = summarize_babelstream(
                [incomplete_run],
                divisor=1,
                units="MB/s",
                title="test",
                description="test",
            )
        assert result is None
        assert "incomplete" in caplog.text.lower() or "skipping" in caplog.text.lower()

    def test_warns_on_incomplete_runs(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
        import infra.tools as tools_mod

        monkeypatch.setattr(tools_mod, "export_markdown", lambda *a, **kw: None)

        incomplete_run = [["Copy", "100"]]
        with caplog.at_level(logging.WARNING):
            summarize_babelstream(
                [incomplete_run],
                divisor=1,
                units="MB/s",
                title="test",
                description="test",
            )
        assert "skipping" in caplog.text.lower() or "incomplete" in caplog.text.lower()

    def test_mixed_valid_and_incomplete_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.print", lambda *a, **kw: None)
        import infra.tools as tools_mod

        monkeypatch.setattr(tools_mod, "export_markdown", lambda *a, **kw: None)

        good_run = [
            ["Copy", "1000"],
            ["Mul", "2000"],
            ["Add", "3000"],
            ["Triad", "4000"],
            ["Dot", "5000"],
        ]
        bad_run = [["Copy", "100"]]
        table = summarize_babelstream(
            [good_run, bad_run],
            divisor=1,
            units="MB/s",
            title="test",
            description="test",
        )
        assert table is not None
        # Only 1 good run, so min == max == mean
        assert len(table.rows) == 5


# ---------------------------------------------------------------------------
# load_benchmark_config + validation
# ---------------------------------------------------------------------------


class TestLoadBenchmarkConfig:
    def test_missing_section_raises_key_error(self, tmp_path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"Other": {}}))
        with pytest.raises(KeyError, match="GEMMCublasLt"):
            load_benchmark_config(str(cfg), "GEMMCublasLt")

    def test_valid_section_returns_dict(self, tmp_path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"GEMMCublasLt": {"datatype": "fp8e4m3"}}))
        section = load_benchmark_config(str(cfg), "GEMMCublasLt")
        assert section["datatype"] == "fp8e4m3"

    def test_missing_required_key_raises_value_error(self, tmp_path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"GEMMCublasLt": {"other": 1}}))
        with pytest.raises(ValueError, match="datatype"):
            load_benchmark_config(str(cfg), "GEMMCublasLt")

    def test_unknown_section_skips_validation(self, tmp_path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"Custom": {"anything": "works"}}))
        section = load_benchmark_config(str(cfg), "Custom")
        assert section["anything"] == "works"


# ---------------------------------------------------------------------------
# run_parallel_builds
# ---------------------------------------------------------------------------


class TestRunParallelBuilds:
    def test_all_succeed(self) -> None:
        calls = []
        builds = [
            ("A", lambda: calls.append("A")),
            ("B", lambda: calls.append("B")),
        ]
        failed = run_parallel_builds(builds)
        assert failed == []
        assert set(calls) == {"A", "B"}

    def test_collects_failures(self) -> None:
        def _boom():
            raise RuntimeError("bang")

        builds = [
            ("good", lambda: None),
            ("bad", _boom),
        ]
        failed = run_parallel_builds(builds)
        assert failed == ["bad"]

    def test_max_workers(self) -> None:
        calls = []
        builds = [
            ("A", lambda: calls.append("A")),
            ("B", lambda: calls.append("B")),
        ]
        failed = run_parallel_builds(builds, max_workers=1)
        assert failed == []
        assert set(calls) == {"A", "B"}


# ---------------------------------------------------------------------------
# find_nvcc
# ---------------------------------------------------------------------------


class TestFindNvcc:
    @patch("infra.tools.os.environ", {"CUDA_HOME": "/opt/cuda"})
    @patch("infra.tools.os.path.isfile", return_value=True)
    def test_uses_cuda_home(self, mock_isfile) -> None:
        assert find_nvcc() == "/opt/cuda/bin/nvcc"

    @patch("infra.tools.os.environ", {})
    @patch("infra.tools.os.path.isfile", return_value=True)
    def test_falls_back_to_usr_local_cuda(self, mock_isfile) -> None:
        result = find_nvcc()
        assert result == "/usr/local/cuda/bin/nvcc"

    @patch("infra.tools.shutil.which", return_value="/some/path/nvcc")
    @patch("infra.tools.os.environ", {})
    @patch("infra.tools.os.path.isfile", return_value=False)
    def test_falls_back_to_which(self, mock_isfile, mock_which) -> None:
        assert find_nvcc() == "/some/path/nvcc"

    @patch("infra.tools.shutil.which", return_value=None)
    @patch("infra.tools.os.environ", {})
    @patch("infra.tools.os.path.isfile", return_value=False)
    def test_returns_nvcc_as_last_resort(self, mock_isfile, mock_which) -> None:
        assert find_nvcc() == "nvcc"
