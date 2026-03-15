"""Tests for Multichase benchmark module."""

from benchmarks.nvidia import multichase

SAMPLE_OUTPUT = """\
 CPU   NODE0   NODE1
   0     3.2     5.1
  48     4.8     3.1
  95     6.0     7.2
"""


class TestParseMultichaseOutput:
    def test_extracts_rows(self) -> None:
        node_names, rows = multichase.parse_multichase_output(SAMPLE_OUTPUT)
        assert len(rows) == 3

    def test_node_names(self) -> None:
        node_names, rows = multichase.parse_multichase_output(SAMPLE_OUTPUT)
        assert node_names == ["NODE0", "NODE1"]

    def test_cpu_values(self) -> None:
        _, rows = multichase.parse_multichase_output(SAMPLE_OUTPUT)
        assert rows[0]["cpu"] == "0"
        assert rows[1]["cpu"] == "48"
        assert rows[2]["cpu"] == "95"

    def test_latency_floats(self) -> None:
        _, rows = multichase.parse_multichase_output(SAMPLE_OUTPUT)
        assert rows[0]["NODE0"] == 3.2
        assert rows[0]["NODE1"] == 5.1
        assert rows[1]["NODE1"] == 3.1

    def test_empty_input(self) -> None:
        node_names, rows = multichase.parse_multichase_output("")
        assert node_names == []
        assert rows == []

    def test_header_only(self) -> None:
        node_names, rows = multichase.parse_multichase_output(" CPU   NODE0   NODE1\n")
        assert node_names == ["NODE0", "NODE1"]
        assert rows == []

    def test_single_node(self) -> None:
        text = " CPU   NODE0\n   0     2.5\n"
        node_names, rows = multichase.parse_multichase_output(text)
        assert node_names == ["NODE0"]
        assert len(rows) == 1
        assert rows[0] == {"cpu": "0", "NODE0": 2.5}

    def test_malformed_row_skipped(self) -> None:
        text = " CPU   NODE0   NODE1\n   0     3.2\n   1     4.0     5.0\n"
        _, rows = multichase.parse_multichase_output(text)
        # First row has wrong field count and is skipped
        assert len(rows) == 1
        assert rows[0]["cpu"] == "1"


class TestModuleConstants:
    def test_repo_constant_exists(self) -> None:
        assert hasattr(multichase, "_MULTICHASE_REPO")
        assert "multichase" in multichase._MULTICHASE_REPO

    def test_run_is_callable(self) -> None:
        assert callable(multichase.run)
