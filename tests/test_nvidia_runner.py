"""Tests for nvidia_runner -- dispatch table, BENCHMARKS dict, _make_ctx."""

from datetime import datetime
from pathlib import Path

from infra.capture import RunContext
from nvidia_runner import BENCHMARKS, _make_ctx


class TestBenchmarksDict:
    def test_all_keys_present(self) -> None:
        expected = {
            "gemm",
            "nccl",
            "hbm",
            "nv",
            "fa",
            "multichase",
            "cpustream",
            "fio",
            "llm",
            "llama_8b_pretrain",
            "llama_3b_pretrain",
        }
        assert set(BENCHMARKS.keys()) == expected

    def test_values_are_strings(self) -> None:
        for v in BENCHMARKS.values():
            assert isinstance(v, str)


class TestMakeCtx:
    def test_returns_run_context(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        ts = datetime(2025, 6, 1, 12, 0, 0)
        ctx = _make_ctx("gemm_cublas_lt", "NVIDIA H200", session_dir, "v1", ts)
        assert isinstance(ctx, RunContext)
        assert ctx.benchmark == "gemm_cublas_lt"
        assert ctx.sku == "NVIDIA H200"
        assert ctx.platform == "nvidia"

    def test_run_dir_created(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        ts = datetime(2025, 1, 1)
        ctx = _make_ctx("nccl_bandwidth", "NVIDIA H200", session_dir, "v1", ts)
        assert ctx.run_dir.name == "nccl_bandwidth"
        assert (ctx.run_dir / "raw").is_dir()
        assert (ctx.run_dir / "processed").is_dir()
