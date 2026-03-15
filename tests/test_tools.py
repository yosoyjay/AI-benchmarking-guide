"""Tests for BabelStream helpers in infra/tools.py."""

import logging

from infra.tools import BABELSTREAM_OPS, parse_babelstream_output, summarize_babelstream

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
    def test_extracts_five_ops(self):
        results = parse_babelstream_output(SAMPLE_BABELSTREAM_OUTPUT)
        assert len(results) == 5
        names = [r[0] for r in results]
        assert names == list(BABELSTREAM_OPS)

    def test_bandwidth_values(self):
        results = parse_babelstream_output(SAMPLE_BABELSTREAM_OUTPUT)
        assert results[0] == ["Copy", "1620000.000"]
        assert results[4] == ["Dot", "1580000.000"]

    def test_empty_input_returns_empty(self):
        assert parse_babelstream_output("") == []

    def test_garbage_input_returns_empty(self):
        assert parse_babelstream_output("no ops here\njust noise\n") == []

    def test_partial_output(self):
        partial = "Copy        1620000.000 0.00033     0.00035     0.00034\n"
        results = parse_babelstream_output(partial)
        assert len(results) == 1
        assert results[0] == ["Copy", "1620000.000"]


class TestSummarizeBabelstream:
    def _make_buffer(self, n_runs=3):
        """Build a buffer of n identical runs for testing."""
        run = [
            ["Copy", "1620000"],
            ["Mul", "1610000"],
            ["Add", "1600000"],
            ["Triad", "1590000"],
            ["Dot", "1580000"],
        ]
        return [run] * n_runs

    def test_computes_min_max_mean(self, monkeypatch):
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

    def test_returns_none_on_incomplete_data(self, monkeypatch, caplog):
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

    def test_warns_on_incomplete_runs(self, monkeypatch, caplog):
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

    def test_mixed_valid_and_incomplete_runs(self, monkeypatch):
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
