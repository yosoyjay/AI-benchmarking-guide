"""Tests for amd_runner -- dispatch table, BENCHMARKS dict, _make_ctx, subcommands, _install."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from amd_runner import BENCHMARKS, _ensure_subcommand, _install, _make_ctx
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


class TestEnsureSubcommand:
    def test_injects_run_for_benchmark_name(self) -> None:
        argv = ["amd_runner.py", "hbm"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "run", "hbm"]

    def test_injects_run_for_multiple_benchmarks(self) -> None:
        argv = ["amd_runner.py", "hbm", "rccl"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "run", "hbm", "rccl"]

    def test_preserves_run_subcommand(self) -> None:
        argv = ["amd_runner.py", "run", "hbm"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "run", "hbm"]

    def test_preserves_install_subcommand(self) -> None:
        argv = ["amd_runner.py", "install"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "install"]

    def test_preserves_install_with_force(self) -> None:
        argv = ["amd_runner.py", "install", "--force"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "install", "--force"]

    def test_preserves_help_flag(self) -> None:
        argv = ["amd_runner.py", "-h"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "-h"]

    def test_no_args(self) -> None:
        argv = ["amd_runner.py"]
        assert _ensure_subcommand(argv) == ["amd_runner.py"]

    def test_install_with_jobs(self) -> None:
        argv = ["amd_runner.py", "install", "--jobs", "2"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "install", "--jobs", "2"]

    def test_install_with_j_short(self) -> None:
        argv = ["amd_runner.py", "install", "-j", "4"]
        assert _ensure_subcommand(argv) == ["amd_runner.py", "install", "-j", "4"]


class TestInstall:
    """Test _install() calls all builds and handles failures."""

    @patch("amd_runner._build_single_docker_image")
    @patch("benchmarks.amd.transfer_bench._build")
    @patch("benchmarks.amd.hbm_bandwidth._build")
    def test_all_builds_called(self, mock_hbm, mock_tb, mock_docker) -> None:
        args = argparse.Namespace(force=False, jobs=None)
        _install(args)
        assert mock_hbm.called
        assert mock_tb.called
        assert mock_docker.call_count == 2  # two docker images

    @patch("amd_runner._build_single_docker_image")
    @patch("benchmarks.amd.transfer_bench._build")
    @patch("benchmarks.amd.hbm_bandwidth._build", side_effect=RuntimeError("boom"))
    def test_collects_failures(self, mock_hbm, mock_tb, mock_docker) -> None:
        args = argparse.Namespace(force=False, jobs=None)
        with pytest.raises(SystemExit) as exc_info:
            _install(args)
        assert exc_info.value.code == 1
