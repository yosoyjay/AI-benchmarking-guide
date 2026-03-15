"""Tests for NVIDIA LLM benchmark module."""

from benchmarks.nvidia import llm_benchmark


class TestParseTrtllmBenchOutput:
    def test_extracts_all_four_values(self):
        text = (
            "[TensorRT-LLM] Report\n"
            "  TP Size: 8\n"
            "  Average Input Length (tokens): 128.0\n"
            "  Average Output Length (tokens): 2048.0\n"
            "  Token Throughput (tokens/sec): 5432.1\n"
        )
        result = llm_benchmark.parse_trtllm_bench_output(text)
        assert result == {
            "tp_size": "8",
            "input_len": "128",
            "output_len": "2048",
            "throughput": "5432",
        }

    def test_returns_none_for_empty(self):
        assert llm_benchmark.parse_trtllm_bench_output("") is None

    def test_returns_none_for_partial(self):
        text = "  TP Size: 8\n  Average Input Length (tokens): 128.0\n"
        assert llm_benchmark.parse_trtllm_bench_output(text) is None

    def test_returns_none_for_no_matching_lines(self):
        text = "some random benchmark output\nno keywords here\n"
        assert llm_benchmark.parse_trtllm_bench_output(text) is None

    def test_handles_integer_values(self):
        text = (
            "  TP Size: 4\n"
            "  Average Input Length (tokens): 256\n"
            "  Average Output Length (tokens): 512\n"
            "  Token Throughput (tokens/sec): 10000\n"
        )
        result = llm_benchmark.parse_trtllm_bench_output(text)
        assert result == {
            "tp_size": "4",
            "input_len": "256",
            "output_len": "512",
            "throughput": "10000",
        }


class TestBuildTable:
    def test_smoke(self):
        rows = [
            {"tp_size": "8", "input_len": "128", "output_len": "2048", "throughput": "5432"},
        ]
        table = llm_benchmark._build_table(rows)
        text = table.get_string()
        assert "128" in text
        assert "5432" in text

    def test_empty_rows(self):
        table = llm_benchmark._build_table([])
        assert table.get_string() is not None


class TestModuleConstants:
    def test_repo_constant(self):
        assert "TensorRT-LLM" in llm_benchmark._TENSORRT_LLM_REPO

    def test_run_is_callable(self):
        assert callable(llm_benchmark.run)
