"""RCCL AllReduce bandwidth benchmark (AMD ROCm, Docker-based)."""

import logging

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_docker
from infra.containers import AmdContainer

logger = logging.getLogger(__name__)

_RCCL_IMAGE = "ai-bench/amd-rccl:latest"

_ALGOS = ["Tree", "Ring", "NVLS", "NVLSTree"]


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_rccl_output(text: str) -> list[dict[str, str]]:
    """Parse all_reduce_perf output into row dicts.

    Looks for 13-column whitespace-delimited lines (``float`` lines).
    Returns a list of ``{"size": str, "bandwidth": str}`` dicts where
    bandwidth is field index 11 (bus bandwidth).
    """
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 13:
            rows.append({"size": fields[0], "bandwidth": fields[11]})
    return rows


def _build_table(sizes: list[str], bandwidth_columns: list[list[str]], algos: list[str]) -> PrettyTable:
    """Build a multi-column PrettyTable: sizes + one column per algo."""
    table = PrettyTable()
    table.add_column("Message Size", sizes)
    for algo, bw_col in zip(algos, bandwidth_columns):
        table.add_column(algo, bw_col)
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "The values (in GB/s) are the bus bandwidth values obtained from the "
    "RCCL AllReduce tests with Tree, Ring, NVLS and NVLSTree algos "
    "(in-place operations), varying from 1KB to 8GB of data."
)


def run(work_dir: str, machine_name: str, ctx: RunContext | None = None) -> dict[str, list[dict[str, str]]] | None:
    """Run RCCL AllReduce inside Docker, parse and report."""
    all_parsed = {}
    with AmdContainer(_RCCL_IMAGE, work_dir, entrypoint="/bin/bash") as container:
        logger.info("Running RCCL AllReduce...")
        perf_bin = "/opt/rccl-tests/build/all_reduce_perf"
        sizes = []
        bandwidth_columns = []

        for algo in _ALGOS:
            cmd = f"NCCL_ALGO={algo} {perf_bin} " f"-b 8 -e 8G -f 2 -g 8 -n 40"
            if ctx is not None:
                stdout, stderr, exit_code = capture_docker(
                    container, ["/bin/sh", "-c", cmd], ctx=ctx, suffix=f"_{algo.lower()}"
                )
                if exit_code != 0:
                    continue
                rows = parse_rccl_output(stdout)
            else:
                res = container.exec_run(
                    ["/bin/sh", "-c", cmd],
                    stderr=True,
                )
                if res.exit_code != 0:
                    tools.write_log(res.output.decode("utf-8"))
                    continue
                rows = parse_rccl_output(res.output.decode("utf-8"))
            if not sizes:
                sizes = [r["size"] for r in rows]
            bandwidth_columns.append([r["bandwidth"] for r in rows])
            if ctx is not None:
                all_parsed[algo] = rows

    if ctx is not None:
        return all_parsed

    table = _build_table(sizes, bandwidth_columns, _ALGOS)
    print(table)
    tools.export_markdown("RCCL Bandwidth", _DESCRIPTION, table)
    return None
