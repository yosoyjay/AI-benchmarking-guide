"""Tests for flash attention output parsing (NVIDIA module)."""

import pytest

from benchmarks.nvidia.flash_attention import _build_table, parse_flash_attention_output

# Realistic excerpt: two blocks from benchmark_flash_attention.py filtered to
# batch_size=2, seqlen=8192.  The regex must span across the Flash2/Pytorch
# sub-blocks within each causal/headdim section.
SAMPLE_OUTPUT = """\
### batch_size=2, seqlen=8192 ###
### causal=False, headdim=64 ###
Flash2 fwd: 427.15
Flash2 bwd: 238.41
Flash2 fwd + bwd: 312.50
Pytorch fwd: 165.22
Pytorch bwd: 102.18
Pytorch fwd + bwd: 198.30
### causal=False, headdim=128 ###
Flash2 fwd: 510.60
Flash2 bwd: 295.10
Flash2 fwd + bwd: 380.20
Pytorch fwd: 180.40
Pytorch bwd: 110.50
Pytorch fwd + bwd: 215.70
### causal=True, headdim=64 ###
Flash2 fwd: 400.00
Flash2 bwd: 220.00
Flash2 fwd + bwd: 290.00
Pytorch fwd: 150.00
Pytorch bwd: 95.00
Pytorch fwd + bwd: 180.00
"""


class TestParseFlashAttentionOutput:
    def test_parses_all_rows(self) -> None:
        rows = parse_flash_attention_output(SAMPLE_OUTPUT)
        assert len(rows) == 3

    def test_field_values(self) -> None:
        rows = parse_flash_attention_output(SAMPLE_OUTPUT)
        first = rows[0]
        assert first["causal"] == "False"
        assert first["headdim"] == 64
        assert first["flash2_tflops"] == 312.50
        assert first["pytorch_tflops"] == 198.30

    def test_causal_true_row(self) -> None:
        rows = parse_flash_attention_output(SAMPLE_OUTPUT)
        causal_rows = [r for r in rows if r["causal"] == "True"]
        assert len(causal_rows) == 1
        assert causal_rows[0]["headdim"] == 64

    def test_headdim_is_int(self) -> None:
        rows = parse_flash_attention_output(SAMPLE_OUTPUT)
        for row in rows:
            assert isinstance(row["headdim"], int)

    def test_tflops_are_float(self) -> None:
        rows = parse_flash_attention_output(SAMPLE_OUTPUT)
        for row in rows:
            assert isinstance(row["flash2_tflops"], float)
            assert isinstance(row["pytorch_tflops"], float)

    def test_empty_input_returns_empty(self) -> None:
        rows = parse_flash_attention_output("")
        assert rows == []

    def test_unmatched_input_returns_empty(self, caplog: pytest.LogCaptureFixture) -> None:
        rows = parse_flash_attention_output("no matching content here\n")
        assert rows == []

    def test_warns_on_no_matches(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING"):
            parse_flash_attention_output("nothing useful")
        assert "no results parsed" in caplog.text.lower()


class TestBuildTable:
    def test_smoke(self) -> None:
        rows = parse_flash_attention_output(SAMPLE_OUTPUT)
        table = _build_table(rows)
        assert len(table.rows) == 3
        assert "causal" in table.field_names[0].lower()

    def test_empty_rows(self) -> None:
        table = _build_table([])
        assert len(table.rows) == 0
