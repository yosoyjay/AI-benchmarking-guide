"""Tests for FIO benchmark module."""

from benchmarks import fio


class TestParseFioOutput:
    def test_extracts_bandwidth(self) -> None:
        # fio --group_reporting output contains a line like:
        #   read: bw=2145MiB/s (2249MB/s), 2145MiB/s-2145MiB/s ...
        text = """\
test: (g=0): rw=read, bs=(R) 1024KiB-1024KiB
   read: bw=2145MiB/s (2249MB/s), 2145MiB/s-2145MiB/s (2249MB/s-2249MB/s), io=628GiB (674GB), run=300001-300001msec
"""
        assert fio.parse_fio_output(text) == "2249MB/s"

    def test_extracts_from_write_output(self) -> None:
        text = """\
test: (g=0): rw=write, bs=(R) 1024KiB-1024KiB
  write: bw=1500MiB/s (1573MB/s), 1500MiB/s-1500MiB/s (1573MB/s-1573MB/s), io=439GiB (471GB), run=300001-300001msec
"""
        assert fio.parse_fio_output(text) == "1573MB/s"

    def test_returns_error_for_empty(self) -> None:
        assert fio.parse_fio_output("") == "error"

    def test_returns_error_for_no_bw_line(self) -> None:
        assert fio.parse_fio_output("some random output\nwithout bw") == "error"

    def test_returns_error_for_short_bw_line(self) -> None:
        assert fio.parse_fio_output("  read: bw=") == "error"


class TestBuildTable:
    def test_smoke(self) -> None:
        rows = [("read", "1M", "2145MiB/s"), ("write", "512k", "1500MiB/s")]
        table = fio._build_table(rows)
        text = table.get_string()
        assert "read" in text
        assert "2145MiB/s" in text

    def test_empty_rows(self) -> None:
        table = fio._build_table([])
        assert table.get_string() is not None


class TestModuleConstants:
    def test_fio_tests_is_list(self) -> None:
        assert isinstance(fio._FIO_TESTS, list)
        assert len(fio._FIO_TESTS) == 8

    def test_run_is_callable(self) -> None:
        assert callable(fio.run)
