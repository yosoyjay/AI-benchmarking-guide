"""Tests for NV Bandwidth benchmark helpers."""

import pytest

from benchmarks.nvidia.nv_bandwidth import (
    TEST_NAMES,
    _build_tables,
    extract_summary_table,
    parse_sections,
)

SAMPLE_OUTPUT = """\
nvbandwidth Version: 0.5
Built from revision: abc123

NOTE: This is sample output

device_to_host_memcpy_ce
running device_to_host_memcpy_ce
memcpy CE GPU(row) -> CPU(column) bandwidth (GB/s)
          0        1        2        3
0         26.1     25.9     26.0     25.8
1         25.8     26.2     25.7     26.0
2         26.0     25.8     26.1     25.9
3         25.9     26.0     25.8     26.2

SUM device_to_host_memcpy_ce 207.3

host_to_device_memcpy_ce
running host_to_device_memcpy_ce
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0        1        2        3
0         30.1     30.2     30.0     29.9
1         30.0     30.3     29.8     30.1
2         30.2     30.0     30.1     29.9
3         29.9     30.1     30.0     30.2

SUM host_to_device_memcpy_ce 241.1

device_to_device_bidirectional_memcpy_read_ce
running device_to_device_bidirectional_memcpy_read_ce
memcpy CE GPU(row) <-> GPU(column) bandwidth (GB/s)
          0        1        2        3
0         0.0      45.1     45.0     44.9
1         45.2     0.0      44.8     45.1
2         45.0     44.9     0.0      45.2
3         44.9     45.0     45.1     0.0

SUM device_to_device_bidirectional_memcpy_read_ce 534.3
"""


class TestParseSections:
    def test_finds_all_three_sections(self) -> None:
        sections = parse_sections(SAMPLE_OUTPUT)
        assert len(sections) == 3
        for name in TEST_NAMES:
            assert name in sections

    def test_section_content_excludes_test_name(self) -> None:
        sections = parse_sections(SAMPLE_OUTPUT)
        for name in TEST_NAMES:
            for line in sections[name].splitlines():
                assert line.strip() != name

    def test_empty_input_returns_empty(self) -> None:
        sections = parse_sections("")
        assert sections == {}


class TestExtractSummaryTable:
    def test_returns_first_table(self) -> None:
        sections = parse_sections(SAMPLE_OUTPUT)
        table = extract_summary_table(sections["device_to_host_memcpy_ce"])
        # Header row + 4 GPU rows
        assert len(table) == 5
        assert table[0] == [0.0, 1.0, 2.0, 3.0]

    def test_numeric_cells_are_floats(self) -> None:
        sections = parse_sections(SAMPLE_OUTPUT)
        table = extract_summary_table(sections["device_to_host_memcpy_ce"])
        for row in table:
            for cell in row:
                assert isinstance(cell, float)


class TestBuildTables:
    def test_smoke(self) -> None:
        tables = _build_tables(SAMPLE_OUTPUT)
        assert len(tables) == 3
        assert "Device To Host" in tables[0][0]

    def test_missing_section_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            tables = _build_tables("some unrelated text\nwith no test names\n")
        assert "not found in nvbandwidth output" in caplog.text
        assert tables == []

    def test_empty_input(self) -> None:
        tables = _build_tables("")
        assert tables == []
