"""Tests for AMD RCCL bandwidth benchmark module."""

from benchmarks.amd import rccl_bandwidth

# Realistic 13-column all_reduce_perf output (header + 3 data lines)
_SAMPLE_OUTPUT = """\
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                                (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           8             2     float     sum      -1    48.81    0.00    0.00      0    47.92    0.00    0.00      0
          16             4     float     sum      -1    47.98    0.00    0.00      0    47.73    0.00    0.00      0
     8589934592    2147483648     float     sum      -1  297816.6   28.84   50.47      0  296410.7   28.98   50.71      0
"""


class TestParseRcclOutput:
    def test_extracts_all_rows(self):
        rows = rccl_bandwidth.parse_rccl_output(_SAMPLE_OUTPUT)
        assert len(rows) == 3

    def test_size_and_bandwidth(self):
        rows = rccl_bandwidth.parse_rccl_output(_SAMPLE_OUTPUT)
        assert rows[0]["size"] == "8"
        assert rows[0]["bandwidth"] == "0.00"
        assert rows[-1]["size"] == "8589934592"
        assert rows[-1]["bandwidth"] == "50.71"

    def test_empty_input_returns_empty(self):
        assert rccl_bandwidth.parse_rccl_output("") == []

    def test_short_lines_ignored(self):
        assert rccl_bandwidth.parse_rccl_output("# header\nshort line\n") == []


class TestBuildTable:
    def test_smoke(self):
        sizes = ["8", "16"]
        bw_cols = [["0.00", "0.00"], ["0.01", "0.01"]]
        algos = ["Tree", "Ring"]
        table = rccl_bandwidth._build_table(sizes, bw_cols, algos)
        text = table.get_string()
        assert "Tree" in text
        assert "Ring" in text
        assert "Message Size" in text

    def test_empty(self):
        table = rccl_bandwidth._build_table([], [], [])
        assert table.get_string() is not None


class TestModuleConstants:
    def test_image_constant(self):
        assert hasattr(rccl_bandwidth, "_RCCL_PYTORCH_IMAGE")

    def test_repos(self):
        assert "rccl" in rccl_bandwidth._RCCL_REPO
        assert "rccl-tests" in rccl_bandwidth._RCCL_TESTS_REPO

    def test_run_is_callable(self):
        assert callable(rccl_bandwidth.run)
