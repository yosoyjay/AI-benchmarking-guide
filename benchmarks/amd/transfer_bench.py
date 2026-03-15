"""TransferBench benchmark (AMD)."""

import logging
import os

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_cmd

logger = logging.getLogger(__name__)

_TRANSFERBENCH_REPO = "https://github.com/ROCm/TransferBench.git"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_transfer_bench_output(text: str) -> dict[str, str]:
    """Parse TransferBench 'sum' line for H2D and D2H bandwidth.

    Expects pipe-delimited output filtered to the ``sum`` line.
    Returns ``{"h2d": <str>, "d2h": <str>}`` with bandwidth values
    or ``"error"`` for each field if parsing fails.
    """
    for line in text.splitlines():
        if "sum" not in line:
            continue
        if "=" in line:
            continue
        parts = line.split("|")
        h2d = parts[1].strip() if len(parts) > 1 else "error"
        d2h = parts[5].strip() if len(parts) > 5 else "error"
        return {"h2d": h2d, "d2h": d2h}
    return {"h2d": "error", "d2h": "error"}


def _build_table(parsed: dict[str, str]) -> PrettyTable:
    """Format parsed H2D/D2H dict into a PrettyTable."""
    table = PrettyTable(["Test", "Result"])
    table.add_row(["Host to Device memcpy", parsed["h2d"]])
    table.add_row(["Device to Host memcpy", parsed["d2h"]])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir: str) -> None:
    """Clone and build TransferBench."""
    repo_dir = os.path.join(work_dir, "TransferBench")
    if not os.path.isdir(repo_dir):
        logger.info("Building TransferBench...")
        tools.run_cmd(
            ["git", "clone", _TRANSFERBENCH_REPO, "TransferBench"],
            cwd=work_dir,
        )
        build_dir = os.path.join(repo_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        tools.run_cmd(
            ["cmake", ".."],
            cwd=build_dir,
            env={"CXX": "/opt/rocm/bin/hipcc"},
        )
        tools.run_cmd(["make"], cwd=build_dir)


def run(
    work_dir: str, machine_name: str, config_path: str = "config.json", ctx: RunContext | None = None
) -> dict[str, str] | None:
    """Clone, build, run TransferBench, parse and report results."""
    config = tools.load_benchmark_config(config_path, "TransferBench")
    cfg_file = config.get("config_file", "benchmarks/amd/transferbench.cfg")

    _build(work_dir)

    logger.info("Running TransferBench...")
    tb_bin = os.path.join(work_dir, "TransferBench", "build", "TransferBench")
    cfg = os.path.join(work_dir, cfg_file)
    cmd = ["sudo", tb_bin, cfg]
    if ctx is not None:
        result = capture_cmd(cmd, ctx=ctx)
    else:
        result = tools.run_cmd(cmd)

    if result.returncode != 0:
        logger.warning("TransferBench failed: returncode=%s", result.returncode)
        parsed = {"h2d": "error", "d2h": "error"}
    else:
        parsed = parse_transfer_bench_output(result.stdout.decode("utf-8"))

    if ctx is not None:
        return parsed

    table = _build_table(parsed)
    print(table)
    tools.export_markdown("TransferBench", "TransferBench Results in GB/s", table)
    return None
