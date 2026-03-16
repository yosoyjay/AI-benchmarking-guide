"""Tests for binary-existence build guards in _build() functions.

Each benchmark's _build() should skip build commands when its target binary
already exists, and run build commands when the binary is missing.
"""

from unittest.mock import patch

from benchmarks.amd import hbm_bandwidth as amd_hbm
from benchmarks.amd import transfer_bench as amd_tb
from benchmarks.nvidia import cpu_stream as nv_cpu
from benchmarks.nvidia import gemm_cublas_lt as nv_gemm
from benchmarks.nvidia import hbm_bandwidth as nv_hbm
from benchmarks.nvidia import multichase as nv_mc
from benchmarks.nvidia import nccl_bandwidth as nv_nccl
from benchmarks.nvidia import nv_bandwidth as nv_nvb


class TestNvBandwidthBuildGuard:
    @patch("benchmarks.nvidia.nv_bandwidth.tools.run_cmd")
    @patch("benchmarks.nvidia.nv_bandwidth.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_run_cmd) -> None:
        nv_nvb._build("/work")
        mock_isfile.assert_called_once_with("/work/nvbandwidth/nvbandwidth")
        mock_run_cmd.assert_not_called()

    @patch("benchmarks.nvidia.nv_bandwidth.os.path.exists", return_value=False)
    @patch("benchmarks.nvidia.nv_bandwidth.tools.run_cmd")
    @patch("benchmarks.nvidia.nv_bandwidth.os.path.isdir", return_value=True)
    @patch("benchmarks.nvidia.nv_bandwidth.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd, mock_exists) -> None:
        nv_nvb._build("/work")
        assert mock_run_cmd.call_count >= 1


class TestGemmCublasLtBuildGuard:
    @patch("benchmarks.nvidia.gemm_cublas_lt.tools.run_cmd")
    @patch("benchmarks.nvidia.gemm_cublas_lt.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_run_cmd) -> None:
        result = nv_gemm._build("/work", "fp8e4m3")
        mock_isfile.assert_called_once_with("/work/bin/cublaslt_gemm")
        mock_run_cmd.assert_not_called()
        assert result == "/work/bin"

    @patch("benchmarks.nvidia.gemm_cublas_lt.shutil.copy2")
    @patch("benchmarks.nvidia.gemm_cublas_lt.os.makedirs")
    @patch("benchmarks.nvidia.gemm_cublas_lt.tools.run_cmd")
    @patch("benchmarks.nvidia.gemm_cublas_lt.os.path.isdir", return_value=True)
    @patch("benchmarks.nvidia.gemm_cublas_lt.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd, mock_makedirs, mock_copy2) -> None:
        result = nv_gemm._build("/work", "fp8e4m3")
        assert mock_run_cmd.call_count >= 1
        assert result == "/work/bin"


class TestCpuStreamBuildGuard:
    @patch("benchmarks.nvidia.cpu_stream.tools.run_cmd")
    @patch("benchmarks.nvidia.cpu_stream.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_run_cmd) -> None:
        result = nv_cpu._build("/work")
        mock_isfile.assert_called_once_with("/work/CPUStream/build/omp-stream")
        mock_run_cmd.assert_not_called()
        assert result == "/work/CPUStream/build"

    @patch("benchmarks.nvidia.cpu_stream.os.makedirs")
    @patch("benchmarks.nvidia.cpu_stream.tools.run_cmd")
    @patch("benchmarks.nvidia.cpu_stream.os.path.isdir", return_value=True)
    @patch("benchmarks.nvidia.cpu_stream.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd, mock_makedirs) -> None:
        result = nv_cpu._build("/work")
        assert mock_run_cmd.call_count >= 1
        assert result == "/work/CPUStream/build"


class TestNcclBandwidthBuildGuard:
    @patch("benchmarks.nvidia.nccl_bandwidth.os.environ", {"LD_LIBRARY_PATH": ""})
    @patch("benchmarks.nvidia.nccl_bandwidth.tools.run_cmd")
    @patch("benchmarks.nvidia.nccl_bandwidth.os.path.isdir", return_value=True)
    @patch("benchmarks.nvidia.nccl_bandwidth.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_isdir, mock_run_cmd) -> None:
        tests_dir, env = nv_nccl._build("/work", None)
        mock_isfile.assert_called_once_with("/work/nccl-tests/build/all_reduce_perf")
        mock_run_cmd.assert_not_called()
        assert tests_dir == "/work/nccl-tests"
        assert "NCCL_HOME" in env

    @patch("benchmarks.nvidia.nccl_bandwidth.os.environ", {"LD_LIBRARY_PATH": ""})
    @patch("benchmarks.nvidia.nccl_bandwidth.tools.run_cmd")
    @patch("benchmarks.nvidia.nccl_bandwidth.os.path.isdir", return_value=True)
    @patch("benchmarks.nvidia.nccl_bandwidth.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd) -> None:
        tests_dir, env = nv_nccl._build("/work", None)
        assert mock_run_cmd.call_count >= 1
        assert tests_dir == "/work/nccl-tests"


class TestMultichaseBuildGuard:
    @patch("benchmarks.nvidia.multichase.tools.run_cmd")
    @patch("benchmarks.nvidia.multichase.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_run_cmd) -> None:
        nv_mc._build("/work")
        mock_isfile.assert_called_once_with("/work/multichase/multichase")
        mock_run_cmd.assert_not_called()

    @patch("benchmarks.nvidia.multichase.tools.run_cmd")
    @patch("benchmarks.nvidia.multichase.os.path.isdir", return_value=True)
    @patch("benchmarks.nvidia.multichase.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd) -> None:
        nv_mc._build("/work")
        assert mock_run_cmd.call_count >= 1


class TestHbmBandwidthNvBuildGuard:
    @patch("benchmarks.nvidia.hbm_bandwidth.tools.run_cmd")
    @patch("benchmarks.nvidia.hbm_bandwidth.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_run_cmd) -> None:
        result = nv_hbm._build("/work", "NVIDIA H200")
        mock_isfile.assert_called_once_with("/work/BabelStream/build/cuda-stream")
        mock_run_cmd.assert_not_called()
        assert result == "/work/BabelStream/build"

    @patch("benchmarks.nvidia.hbm_bandwidth.os.makedirs")
    @patch("benchmarks.nvidia.hbm_bandwidth.tools.run_cmd")
    @patch("benchmarks.nvidia.hbm_bandwidth.os.path.isdir", return_value=True)
    @patch("benchmarks.nvidia.hbm_bandwidth.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd, mock_makedirs) -> None:
        result = nv_hbm._build("/work", "NVIDIA H200")
        assert mock_run_cmd.call_count >= 1
        assert result == "/work/BabelStream/build"


class TestHbmBandwidthAmdBuildGuard:
    @patch("benchmarks.amd.hbm_bandwidth.tools.run_cmd")
    @patch("benchmarks.amd.hbm_bandwidth.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_run_cmd) -> None:
        amd_hbm._build("/work")
        mock_isfile.assert_called_once_with("/work/BabelStream/build/hip-stream")
        mock_run_cmd.assert_not_called()

    @patch("benchmarks.amd.hbm_bandwidth.tools.run_cmd")
    @patch("benchmarks.amd.hbm_bandwidth.os.path.isdir", return_value=True)
    @patch("benchmarks.amd.hbm_bandwidth.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd) -> None:
        amd_hbm._build("/work")
        assert mock_run_cmd.call_count >= 1


class TestTransferBenchBuildGuard:
    @patch("benchmarks.amd.transfer_bench.tools.run_cmd")
    @patch("benchmarks.amd.transfer_bench.os.path.isfile", return_value=True)
    def test_skips_build_when_binary_exists(self, mock_isfile, mock_run_cmd) -> None:
        amd_tb._build("/work")
        mock_isfile.assert_called_once_with("/work/TransferBench/build/TransferBench")
        mock_run_cmd.assert_not_called()

    @patch("benchmarks.amd.transfer_bench.os.makedirs")
    @patch("benchmarks.amd.transfer_bench.tools.run_cmd")
    @patch("benchmarks.amd.transfer_bench.os.path.isdir", return_value=True)
    @patch("benchmarks.amd.transfer_bench.os.path.isfile", return_value=False)
    def test_builds_when_binary_missing(self, mock_isfile, mock_isdir, mock_run_cmd, mock_makedirs) -> None:
        amd_tb._build("/work")
        assert mock_run_cmd.call_count >= 1
