"""Tests for NVIDIA HBM bandwidth benchmark module."""

from benchmarks.nvidia import hbm_bandwidth


class TestGetCudaArch:
    def test_a100(self) -> None:
        assert hbm_bandwidth._get_cuda_arch("NVIDIA A100 80GB") == "sm_80"

    def test_gb200(self) -> None:
        assert hbm_bandwidth._get_cuda_arch("NVIDIA GB200") == "sm_100"

    def test_b200(self) -> None:
        assert hbm_bandwidth._get_cuda_arch("NVIDIA B200") == "sm_100"

    def test_b100(self) -> None:
        assert hbm_bandwidth._get_cuda_arch("NVIDIA B100") == "sm_100"

    def test_h100_default(self) -> None:
        assert hbm_bandwidth._get_cuda_arch("NVIDIA H100 80GB") == "sm_90"

    def test_h200_default(self) -> None:
        assert hbm_bandwidth._get_cuda_arch("NVIDIA H200") == "sm_90"

    def test_unknown_default(self) -> None:
        assert hbm_bandwidth._get_cuda_arch("Some Future GPU") == "sm_90"


class TestModuleConstants:
    def test_repo_constant_exists(self) -> None:
        assert hasattr(hbm_bandwidth, "_BABELSTREAM_REPO")
        assert "BabelStream" in hbm_bandwidth._BABELSTREAM_REPO

    def test_run_is_callable(self) -> None:
        assert callable(hbm_bandwidth.run)
