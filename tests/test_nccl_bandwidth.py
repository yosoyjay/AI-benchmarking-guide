"""Tests for NCCL Bandwidth benchmark helpers."""

from benchmarks.nvidia.nccl_bandwidth import _build_table, parse_nccl_output

# Realistic all_reduce_perf output (13 columns per data line)
# Columns: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
SAMPLE_ALL_REDUCE_OUTPUT = """\
           8             2   float     sum      -1    0.01    0.00    0.00      0    0.01    0.00    0.00      0
          16             2   float     sum      -1    0.01    0.00    0.00      0    0.01    0.00    0.00      0
         256            32   float     sum      -1    0.02    0.01    0.01      0    0.02    0.01    0.01      0
     1048576        131072   float     sum      -1    0.05   19.63   18.44      0    0.05   19.50   18.32      0
  8589934592    1073741824   float     sum      -1   44.12  194.71  182.81      0   44.15  194.57  182.68      0
"""


class TestParseNcclOutput:
    def test_extracts_all_rows(self):
        rows = parse_nccl_output(SAMPLE_ALL_REDUCE_OUTPUT)
        assert len(rows) == 5

    def test_size_and_bandwidth(self):
        rows = parse_nccl_output(SAMPLE_ALL_REDUCE_OUTPUT)
        assert rows[0]["size"] == "8"
        assert rows[0]["bandwidth"] == "0.00"
        assert rows[-1]["size"] == "8589934592"
        assert rows[-1]["bandwidth"] == "182.68"

    def test_empty_input_returns_empty(self):
        assert parse_nccl_output("") == []

    def test_short_lines_ignored(self):
        assert parse_nccl_output("only three fields here\n") == []


class TestBuildTable:
    def test_smoke(self):
        rows = parse_nccl_output(SAMPLE_ALL_REDUCE_OUTPUT)
        table = _build_table(rows, "NVLS")
        assert len(table.rows) == 5
        assert "Message Size" in table.field_names
        assert "Bandwidth (NVLS)" in table.field_names

    def test_empty_rows(self):
        table = _build_table([], "Ring")
        assert len(table.rows) == 0
