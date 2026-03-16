"""Tests for nvidia_runner -- dispatch table, BENCHMARKS dict, _make_ctx, subcommands."""

from datetime import datetime
from pathlib import Path

from infra.capture import RunContext
from nvidia_runner import BENCHMARKS, _ensure_subcommand, _make_ctx


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


class TestEnsureSubcommand:
    def test_injects_run_for_benchmark_name(self) -> None:
        argv = ["nvidia_runner.py", "hbm"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "run", "hbm"]

    def test_injects_run_for_multiple_benchmarks(self) -> None:
        argv = ["nvidia_runner.py", "hbm", "nccl", "gemm"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "run", "hbm", "nccl", "gemm"]

    def test_preserves_run_subcommand(self) -> None:
        argv = ["nvidia_runner.py", "run", "hbm"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "run", "hbm"]

    def test_preserves_install_subcommand(self) -> None:
        argv = ["nvidia_runner.py", "install"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "install"]

    def test_preserves_install_with_force(self) -> None:
        argv = ["nvidia_runner.py", "install", "--force"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "install", "--force"]

    def test_preserves_help_flag(self) -> None:
        argv = ["nvidia_runner.py", "-h"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "-h"]

    def test_preserves_help_long(self) -> None:
        argv = ["nvidia_runner.py", "--help"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "--help"]

    def test_no_args(self) -> None:
        argv = ["nvidia_runner.py"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py"]
