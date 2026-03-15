"""Raw output capture and directory scaffolding for benchmark runs."""

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunContext:
    """Metadata and paths for a single benchmark run."""

    benchmark: str  # e.g., "gemm_cublas_lt"
    sku: str  # e.g., "NVIDIA H200"
    platform: str  # "nvidia" or "amd"
    version: str  # git describe --always --dirty
    timestamp: datetime
    results_dir: Path  # project_root/results
    run_dir: Path  # results/<benchmark>_<sku>_<datetime>/
    extra: dict = field(default_factory=dict)  # benchmark-specific context


def get_version() -> str:
    """Return a short version string from *git describe --always --dirty*."""
    try:
        result = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=_PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8").strip()
    except FileNotFoundError:
        pass
    return "unknown"


def _sanitize_sku(sku: str) -> str:
    """Replace characters unsafe for directory names."""
    return sku.replace(" ", "_").replace("/", "_")


def _format_timestamp(ts: datetime) -> str:
    return ts.strftime("%Y%m%d_%H%M%S")


def make_run_dir(results_dir: Path, benchmark: str, sku: str, timestamp: datetime) -> Path:
    """Create and return ``results/<benchmark>_<sku>_<datetime>/`` with raw/ and processed/ subdirs."""
    safe_sku = _sanitize_sku(sku)
    ts_str = _format_timestamp(timestamp)
    run_dir = results_dir / f"{benchmark}_{safe_sku}_{ts_str}"
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "processed").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_raw(
    run_dir: Path,
    benchmark: str,
    version: str,
    timestamp: datetime,
    stdout: str,
    stderr: str,
    *,
    suffix: str = "",
) -> tuple[Path, Path]:
    """Write raw stdout/stderr to files under ``run_dir/raw/``.

    Returns ``(stdout_path, stderr_path)``.
    """
    ts_str = _format_timestamp(timestamp)
    base = f"{benchmark}_{version}_{ts_str}{suffix}"
    stdout_path = run_dir / "raw" / f"{base}.stdout"
    stderr_path = run_dir / "raw" / f"{base}.stderr"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path
