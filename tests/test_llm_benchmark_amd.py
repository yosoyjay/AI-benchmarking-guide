"""Tests for AMD LLM benchmark module."""

from benchmarks.amd import llm_benchmark


class TestParseVllmThroughputOutput:
    def test_extracts_throughput(self):
        # vLLM benchmark_throughput.py prints a line like:
        # Throughput: 100 requests/s, 50000 total tokens/s, 25000 output tokens/s
        text = "Throughput: 100 requests/s, 50000 total tokens/s, 25000 output tokens/s\n"
        result = llm_benchmark.parse_vllm_throughput_output(text)
        # parts[6] from space-split is "25000"
        assert result == "25000"

    def test_returns_none_for_empty(self):
        assert llm_benchmark.parse_vllm_throughput_output("") is None

    def test_returns_none_for_no_throughput(self):
        assert llm_benchmark.parse_vllm_throughput_output("some output\nno throughput here") is None


class TestBuildTable:
    def test_smoke(self):
        rows = [("128", "128", "8", "50000")]
        table = llm_benchmark._build_table(rows)
        text = table.get_string()
        assert "128" in text
        assert "50000" in text

    def test_empty_rows(self):
        table = llm_benchmark._build_table([])
        assert table.get_string() is not None


class TestModuleConstants:
    def test_image_constant(self):
        assert hasattr(llm_benchmark, "_VLLM_IMAGE")
        assert "vllm" in llm_benchmark._VLLM_IMAGE

    def test_run_is_callable(self):
        assert callable(llm_benchmark.run)
