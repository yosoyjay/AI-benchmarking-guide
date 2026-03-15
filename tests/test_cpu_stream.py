"""Tests for CPU Stream benchmark module."""

from benchmarks.nvidia import cpu_stream


class TestModuleConstants:
    def test_repo_constant_exists(self):
        assert hasattr(cpu_stream, "_BABELSTREAM_REPO")
        assert "BabelStream" in cpu_stream._BABELSTREAM_REPO

    def test_run_is_callable(self):
        assert callable(cpu_stream.run)
