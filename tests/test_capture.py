"""Tests for infra.capture -- RunContext, get_version, make_run_dir, save_raw."""

from datetime import datetime
from unittest.mock import patch

from infra.capture import (
    RunContext,
    _format_timestamp,
    _sanitize_sku,
    get_version,
    make_run_dir,
    save_raw,
)

# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


class TestRunContext:
    def test_fields(self, tmp_path):
        ts = datetime(2025, 1, 15, 10, 30, 0)
        ctx = RunContext(
            benchmark="gemm_cublas_lt",
            sku="NVIDIA H200",
            platform="nvidia",
            version="abc1234",
            timestamp=ts,
            results_dir=tmp_path,
            run_dir=tmp_path / "run",
        )
        assert ctx.benchmark == "gemm_cublas_lt"
        assert ctx.sku == "NVIDIA H200"
        assert ctx.platform == "nvidia"
        assert ctx.extra == {}

    def test_extra_default(self, tmp_path):
        ts = datetime(2025, 1, 1)
        ctx = RunContext(
            benchmark="fio",
            sku="sku",
            platform="nvidia",
            version="v",
            timestamp=ts,
            results_dir=tmp_path,
            run_dir=tmp_path,
        )
        assert ctx.extra == {}
        ctx.extra["rw"] = "read"
        assert ctx.extra == {"rw": "read"}

    def test_extra_not_shared(self, tmp_path):
        """Each instance should get its own extra dict."""
        ts = datetime(2025, 1, 1)
        a = RunContext("a", "s", "nvidia", "v", ts, tmp_path, tmp_path)
        b = RunContext("b", "s", "nvidia", "v", ts, tmp_path, tmp_path)
        a.extra["x"] = 1
        assert "x" not in b.extra


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_returns_git_describe(self):
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = b"v0.1.0-3-gabc1234\n"
            assert get_version() == "v0.1.0-3-gabc1234"

    def test_returns_unknown_on_failure(self):
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 128
            mock_run.return_value.stdout = b""
            assert get_version() == "unknown"

    def test_returns_unknown_when_git_missing(self):
        with patch("infra.capture.subprocess.run", side_effect=FileNotFoundError):
            assert get_version() == "unknown"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestSanitizeSku:
    def test_spaces(self):
        assert _sanitize_sku("NVIDIA H200") == "NVIDIA_H200"

    def test_slashes(self):
        assert _sanitize_sku("ND/MI300X") == "ND_MI300X"

    def test_combined(self):
        assert _sanitize_sku("NVIDIA H200 / GB200") == "NVIDIA_H200___GB200"


class TestFormatTimestamp:
    def test_format(self):
        ts = datetime(2025, 3, 15, 14, 30, 45)
        assert _format_timestamp(ts) == "20250315_143045"


# ---------------------------------------------------------------------------
# make_run_dir
# ---------------------------------------------------------------------------


class TestMakeRunDir:
    def test_creates_directory_structure(self, tmp_path):
        ts = datetime(2025, 6, 1, 12, 0, 0)
        run_dir = make_run_dir(tmp_path, "gemm_cublas_lt", "NVIDIA H200", ts)
        assert run_dir.name == "gemm_cublas_lt_NVIDIA_H200_20250601_120000"
        assert (run_dir / "raw").is_dir()
        assert (run_dir / "processed").is_dir()

    def test_idempotent(self, tmp_path):
        ts = datetime(2025, 6, 1, 12, 0, 0)
        d1 = make_run_dir(tmp_path, "fio", "sku", ts)
        d2 = make_run_dir(tmp_path, "fio", "sku", ts)
        assert d1 == d2

    def test_parent_is_results_dir(self, tmp_path):
        ts = datetime(2025, 1, 1)
        run_dir = make_run_dir(tmp_path, "nccl_bandwidth", "H100", ts)
        assert run_dir.parent == tmp_path


# ---------------------------------------------------------------------------
# save_raw
# ---------------------------------------------------------------------------


class TestSaveRaw:
    def test_writes_files(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "raw").mkdir(parents=True)
        ts = datetime(2025, 3, 15, 10, 0, 0)

        stdout_path, stderr_path = save_raw(run_dir, "gemm_cublas_lt", "abc1234", ts, "hello stdout", "hello stderr")
        assert stdout_path.read_text() == "hello stdout"
        assert stderr_path.read_text() == "hello stderr"
        assert stdout_path.name == "gemm_cublas_lt_abc1234_20250315_100000.stdout"
        assert stderr_path.name == "gemm_cublas_lt_abc1234_20250315_100000.stderr"

    def test_suffix(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "raw").mkdir(parents=True)
        ts = datetime(2025, 1, 1, 0, 0, 0)

        stdout_path, _ = save_raw(run_dir, "fio", "v1", ts, "data", "", suffix="_read_1M")
        assert "_read_1M.stdout" in stdout_path.name

    def test_empty_strings(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "raw").mkdir(parents=True)
        ts = datetime(2025, 1, 1)

        stdout_path, stderr_path = save_raw(run_dir, "test", "v1", ts, "", "")
        assert stdout_path.read_text() == ""
        assert stderr_path.read_text() == ""
