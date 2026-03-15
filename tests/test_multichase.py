"""Tests for Multichase benchmark module."""

from benchmarks.nvidia import multichase


class TestModuleConstants:
    def test_repo_constant_exists(self):
        assert hasattr(multichase, "_MULTICHASE_REPO")
        assert "multichase" in multichase._MULTICHASE_REPO

    def test_run_is_callable(self):
        assert callable(multichase.run)
