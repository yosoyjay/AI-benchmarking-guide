"""Tests for AMD HBM bandwidth benchmark module."""

from benchmarks.amd import hbm_bandwidth


class TestModuleConstants:
    def test_repo_constant_exists(self):
        assert hasattr(hbm_bandwidth, "_BABELSTREAM_REPO")
        assert "BabelStream" in hbm_bandwidth._BABELSTREAM_REPO

    def test_run_is_callable(self):
        assert callable(hbm_bandwidth.run)
