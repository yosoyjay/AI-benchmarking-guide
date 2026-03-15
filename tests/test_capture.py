"""Tests for infra.capture -- RunContext, get_version, make_session_dir, make_run_dir, save_raw, capture_cmd, capture_docker."""

import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from infra.capture import (
    RunContext,
    _format_timestamp,
    _sanitize_sku,
    capture_cmd,
    capture_docker,
    get_version,
    make_run_dir,
    make_session_dir,
    save_raw,
)

# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


class TestRunContext:
    def test_fields(self, tmp_path: Path) -> None:
        ts = datetime(2025, 1, 15, 10, 30, 0)
        ctx = RunContext(
            benchmark="gemm_cublas_lt",
            sku="NVIDIA H200",
            platform="nvidia",
            version="abc1234",
            timestamp=ts,
            session_dir=tmp_path,
            run_dir=tmp_path / "run",
        )
        assert ctx.benchmark == "gemm_cublas_lt"
        assert ctx.sku == "NVIDIA H200"
        assert ctx.platform == "nvidia"
        assert ctx.extra == {}

    def test_extra_default(self, tmp_path: Path) -> None:
        ts = datetime(2025, 1, 1)
        ctx = RunContext(
            benchmark="fio",
            sku="sku",
            platform="nvidia",
            version="v",
            timestamp=ts,
            session_dir=tmp_path,
            run_dir=tmp_path,
        )
        assert ctx.extra == {}
        ctx.extra["rw"] = "read"
        assert ctx.extra == {"rw": "read"}

    def test_extra_not_shared(self, tmp_path: Path) -> None:
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
    def test_returns_git_describe(self) -> None:
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = b"v0.1.0-3-gabc1234\n"
            assert get_version() == "v0.1.0-3-gabc1234"

    def test_returns_unknown_on_failure(self) -> None:
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 128
            mock_run.return_value.stdout = b""
            assert get_version() == "unknown"

    def test_returns_unknown_when_git_missing(self) -> None:
        with patch("infra.capture.subprocess.run", side_effect=FileNotFoundError):
            assert get_version() == "unknown"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestSanitizeSku:
    def test_spaces(self) -> None:
        assert _sanitize_sku("NVIDIA H200") == "NVIDIA_H200"

    def test_slashes(self) -> None:
        assert _sanitize_sku("ND/MI300X") == "ND_MI300X"

    def test_combined(self) -> None:
        assert _sanitize_sku("NVIDIA H200 / GB200") == "NVIDIA_H200___GB200"


class TestFormatTimestamp:
    def test_format(self) -> None:
        ts = datetime(2025, 3, 15, 14, 30, 45)
        assert _format_timestamp(ts) == "20250315_143045"


# ---------------------------------------------------------------------------
# make_session_dir
# ---------------------------------------------------------------------------


class TestMakeSessionDir:
    def test_creates_directory(self, tmp_path: Path) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0)
        session_dir = make_session_dir(tmp_path, "NVIDIA H200", ts)
        assert session_dir.is_dir()
        assert session_dir.name == "NVIDIA_H200_20250601_120000"

    def test_parent_is_results_dir(self, tmp_path: Path) -> None:
        ts = datetime(2025, 1, 1)
        session_dir = make_session_dir(tmp_path, "sku", ts)
        assert session_dir.parent == tmp_path

    def test_idempotent(self, tmp_path: Path) -> None:
        ts = datetime(2025, 6, 1, 12, 0, 0)
        d1 = make_session_dir(tmp_path, "sku", ts)
        d2 = make_session_dir(tmp_path, "sku", ts)
        assert d1 == d2


# ---------------------------------------------------------------------------
# make_run_dir
# ---------------------------------------------------------------------------


class TestMakeRunDir:
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        run_dir = make_run_dir(session_dir, "gemm_cublas_lt")
        assert run_dir.name == "gemm_cublas_lt"
        assert (run_dir / "raw").is_dir()
        assert (run_dir / "processed").is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        d1 = make_run_dir(session_dir, "fio")
        d2 = make_run_dir(session_dir, "fio")
        assert d1 == d2

    def test_parent_is_session_dir(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "session"
        session_dir.mkdir()
        run_dir = make_run_dir(session_dir, "nccl_bandwidth")
        assert run_dir.parent == session_dir


# ---------------------------------------------------------------------------
# save_raw
# ---------------------------------------------------------------------------


class TestSaveRaw:
    def test_writes_files(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "raw").mkdir(parents=True)
        ts = datetime(2025, 3, 15, 10, 0, 0)

        stdout_path, stderr_path = save_raw(run_dir, "gemm_cublas_lt", "abc1234", ts, "hello stdout", "hello stderr")
        assert stdout_path.read_text() == "hello stdout"
        assert stderr_path.read_text() == "hello stderr"
        assert stdout_path.name == "gemm_cublas_lt_abc1234_20250315_100000.stdout"
        assert stderr_path.name == "gemm_cublas_lt_abc1234_20250315_100000.stderr"

    def test_suffix(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "raw").mkdir(parents=True)
        ts = datetime(2025, 1, 1, 0, 0, 0)

        stdout_path, _ = save_raw(run_dir, "fio", "v1", ts, "data", "", suffix="_read_1M")
        assert "_read_1M.stdout" in stdout_path.name

    def test_empty_strings(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "raw").mkdir(parents=True)
        ts = datetime(2025, 1, 1)

        stdout_path, stderr_path = save_raw(run_dir, "test", "v1", ts, "", "")
        assert stdout_path.read_text() == ""
        assert stderr_path.read_text() == ""


# ---------------------------------------------------------------------------
# capture_cmd
# ---------------------------------------------------------------------------


def _make_ctx(tmp_path: Path, benchmark: str = "test_bench") -> RunContext:
    """Build a RunContext rooted in tmp_path with raw/ dir created."""
    run_dir = tmp_path / "run"
    (run_dir / "raw").mkdir(parents=True)
    ts = datetime(2025, 6, 1, 12, 0, 0)
    return RunContext(
        benchmark=benchmark,
        sku="TestSKU",
        platform="nvidia",
        version="abc123",
        timestamp=ts,
        session_dir=tmp_path,
        run_dir=run_dir,
    )


class TestCaptureCmd:
    def test_returns_completed_process(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["echo", "hi"], returncode=0, stdout=b"hello\n", stderr=b""
            )
            result = capture_cmd(["echo", "hi"], ctx=ctx)
            assert isinstance(result, subprocess.CompletedProcess)
            assert result.returncode == 0

    def test_saves_stdout_file(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["cmd"], returncode=0, stdout=b"output data", stderr=b"err data"
            )
            capture_cmd(["cmd"], ctx=ctx)
            raw_dir = ctx.run_dir / "raw"
            stdout_files = list(raw_dir.glob("*.stdout"))
            stderr_files = list(raw_dir.glob("*.stderr"))
            assert len(stdout_files) == 1
            assert len(stderr_files) == 1
            assert stdout_files[0].read_text() == "output data"
            assert stderr_files[0].read_text() == "err data"

    def test_suffix_in_filename(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=0, stdout=b"x", stderr=b"")
            capture_cmd(["cmd"], ctx=ctx, suffix="_m1024")
            raw_dir = ctx.run_dir / "raw"
            stdout_files = list(raw_dir.glob("*.stdout"))
            assert "_m1024.stdout" in stdout_files[0].name

    def test_forwards_kwargs(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=0, stdout=b"", stderr=b"")
            capture_cmd(["cmd"], ctx=ctx, cwd="/tmp", shell=False)
            _, call_kwargs = mock_run.call_args
            assert call_kwargs["cwd"] == "/tmp"

    def test_handles_none_stdout_stderr(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=0, stdout=None, stderr=None)
            result = capture_cmd(["cmd"], ctx=ctx)
            assert result.returncode == 0
            raw_dir = ctx.run_dir / "raw"
            stdout_files = list(raw_dir.glob("*.stdout"))
            assert stdout_files[0].read_text() == ""

    def test_nonzero_returncode_logged(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        with patch("infra.capture.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout=b"", stderr=b"fail")
            result = capture_cmd(["cmd"], ctx=ctx)
            assert result.returncode == 1


# ---------------------------------------------------------------------------
# capture_docker
# ---------------------------------------------------------------------------


class TestCaptureDocker:
    def _mock_container(
        self, stdout: bytes = b"docker out", stderr: bytes = b"docker err", exit_code: int = 0
    ) -> MagicMock:
        container = MagicMock()
        container.exec_run.return_value = SimpleNamespace(
            output=(stdout, stderr),
            exit_code=exit_code,
        )
        return container

    def test_returns_tuple(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = self._mock_container()
        result = capture_docker(container, ["python", "bench.py"], ctx=ctx)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_stdout_stderr_content(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = self._mock_container(stdout=b"hello", stderr=b"warning")
        stdout, stderr, exit_code = capture_docker(container, ["cmd"], ctx=ctx)
        assert stdout == "hello"
        assert stderr == "warning"
        assert exit_code == 0

    def test_saves_raw_files(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = self._mock_container(stdout=b"out data", stderr=b"err data")
        capture_docker(container, ["cmd"], ctx=ctx)
        raw_dir = ctx.run_dir / "raw"
        stdout_files = list(raw_dir.glob("*.stdout"))
        stderr_files = list(raw_dir.glob("*.stderr"))
        assert len(stdout_files) == 1
        assert stdout_files[0].read_text() == "out data"
        assert stderr_files[0].read_text() == "err data"

    def test_suffix_in_filename(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = self._mock_container()
        capture_docker(container, ["cmd"], ctx=ctx, suffix="_ring")
        raw_dir = ctx.run_dir / "raw"
        stdout_files = list(raw_dir.glob("*.stdout"))
        assert "_ring.stdout" in stdout_files[0].name

    def test_calls_demux_true(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = self._mock_container()
        capture_docker(container, ["cmd"], ctx=ctx)
        container.exec_run.assert_called_once_with(["cmd"], demux=True)

    def test_nonzero_exit_code(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = self._mock_container(exit_code=1)
        _, _, exit_code = capture_docker(container, ["cmd"], ctx=ctx)
        assert exit_code == 1

    def test_handles_none_output(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = MagicMock()
        container.exec_run.return_value = SimpleNamespace(
            output=(None, None),
            exit_code=0,
        )
        stdout, stderr, exit_code = capture_docker(container, ["cmd"], ctx=ctx)
        assert stdout == ""
        assert stderr == ""

    def test_handles_non_demux_output(self, tmp_path: Path) -> None:
        """When demux is not supported, output is raw bytes instead of tuple."""
        ctx = _make_ctx(tmp_path)
        container = MagicMock()
        container.exec_run.return_value = SimpleNamespace(
            output=b"raw bytes",
            exit_code=0,
        )
        stdout, stderr, exit_code = capture_docker(container, ["cmd"], ctx=ctx)
        assert stdout == "raw bytes"
        assert stderr == ""

    def test_forwards_kwargs(self, tmp_path: Path) -> None:
        ctx = _make_ctx(tmp_path)
        container = self._mock_container()
        capture_docker(container, ["cmd"], ctx=ctx, stderr=True)
        container.exec_run.assert_called_once_with(["cmd"], demux=True, stderr=True)
