"""Tests for nvidia_runner -- dispatch table, BENCHMARKS dict, _make_ctx, subcommands, _install."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from infra.capture import RunContext
from nvidia_runner import BENCHMARKS, _ensure_subcommand, _install, _make_ctx


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

    def test_install_with_jobs(self) -> None:
        argv = ["nvidia_runner.py", "install", "--jobs", "2"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "install", "--jobs", "2"]

    def test_install_with_j_short(self) -> None:
        argv = ["nvidia_runner.py", "install", "-j", "4"]
        assert _ensure_subcommand(argv) == ["nvidia_runner.py", "install", "-j", "4"]


class TestInstall:
    """Test _install() calls all builds and handles failures."""

    _BUILD_PATCHES = [
        "benchmarks.nvidia.hbm_bandwidth._build",
        "benchmarks.nvidia.cpu_stream._build",
        "benchmarks.nvidia.nv_bandwidth._build",
        "benchmarks.nvidia.nccl_bandwidth._build",
        "benchmarks.nvidia.multichase._build",
        "benchmarks.nvidia.gemm_cublas_lt._build",
    ]

    @patch("nvidia_runner._detect_gpu_name", return_value="NVIDIA H200")
    @patch("nvidia_runner.tools.load_benchmark_config", return_value={"datatype": "fp8e4m3"})
    def test_all_builds_called(self, mock_config, mock_gpu) -> None:
        mocks = {}
        patches = []
        for target in self._BUILD_PATCHES:
            p = patch(target)
            m = p.start()
            patches.append(p)
            mocks[target] = m

        try:
            args = argparse.Namespace(force=False, jobs=None)
            _install(args)
            for target, m in mocks.items():
                assert m.called, f"{target} was not called"
        finally:
            for p in patches:
                p.stop()

    @patch("nvidia_runner._detect_gpu_name", return_value="NVIDIA H200")
    @patch("nvidia_runner.tools.load_benchmark_config", return_value={"datatype": "fp8e4m3"})
    def test_collects_failures(self, mock_config, mock_gpu) -> None:
        patches = []
        for target in self._BUILD_PATCHES:
            p = patch(target)
            p.start()
            patches.append(p)

        # Make one build fail
        fail_patch = patch(
            "benchmarks.nvidia.hbm_bandwidth._build",
            side_effect=RuntimeError("boom"),
        )
        fail_patch.start()
        patches.append(fail_patch)

        try:
            args = argparse.Namespace(force=False, jobs=None)
            with pytest.raises(SystemExit) as exc_info:
                _install(args)
            assert exc_info.value.code == 1
        finally:
            for p in patches:
                p.stop()

    @patch("nvidia_runner._detect_gpu_name", return_value="NVIDIA H200")
    @patch("nvidia_runner.tools.load_benchmark_config", return_value={"datatype": "fp8e4m3"})
    def test_jobs_flag_serial(self, mock_config, mock_gpu) -> None:
        patches = []
        for target in self._BUILD_PATCHES:
            p = patch(target)
            p.start()
            patches.append(p)

        try:
            args = argparse.Namespace(force=False, jobs=1)
            _install(args)  # should not raise
        finally:
            for p in patches:
                p.stop()
