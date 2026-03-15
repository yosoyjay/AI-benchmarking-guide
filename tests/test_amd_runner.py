"""Tests for amd_runner -- dispatch table, BENCHMARKS dict, _make_ctx."""

from datetime import datetime
from pathlib import Path

from amd_runner import BENCHMARKS, _make_ctx
from infra.capture import RunContext


class TestBenchmarksDict:
    def test_all_keys_present(self) -> None:
        expected = {"gemm", "rccl", "hbm", "transfer", "fa", "fio", "llm"}
        assert set(BENCHMARKS.keys()) == expected

    def test_values_are_strings(self) -> None:
        for v in BENCHMARKS.values():
            assert isinstance(v, str)


class TestMakeCtx:
    def test_returns_run_context(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        ts = datetime(2025, 6, 1, 12, 0, 0)
        ctx = _make_ctx("gemm_hipblas_lt", "ND_MI300X_v5", session_dir, "v1", ts)
        assert isinstance(ctx, RunContext)
        assert ctx.benchmark == "gemm_hipblas_lt"
        assert ctx.platform == "amd"

    def test_run_dir_created(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        ts = datetime(2025, 1, 1)
        ctx = _make_ctx("rccl_bandwidth", "ND_MI300X_v5", session_dir, "v1", ts)
        assert ctx.run_dir.name == "rccl_bandwidth"
        assert (ctx.run_dir / "raw").is_dir()
        assert (ctx.run_dir / "processed").is_dir()
