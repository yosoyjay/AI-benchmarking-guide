"""RCCL AllReduce bandwidth benchmark (AMD ROCm, Docker-based)."""

import logging
import os

from prettytable import PrettyTable

from infra import tools
from infra.containers import AmdContainer

logger = logging.getLogger(__name__)

_RCCL_PYTORCH_IMAGE = "rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue"
_RCCL_REPO = "https://github.com/ROCm/rccl.git"
_RCCL_TESTS_REPO = "https://github.com/ROCm/rccl-tests.git"

_ALGOS = ["Tree", "Ring", "NVLS", "NVLSTree"]


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_rccl_output(text):
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


def _build_table(sizes, bandwidth_columns, algos):
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


def _build(container, work_dir):
    """Clone and build RCCL + rccl-tests inside the container."""
    rccl_dir = os.path.join(work_dir, "rccl")
    if not os.path.isdir(rccl_dir):
        logger.info("Building RCCL Library...")
        res = container.exec_run(
            ["git", "clone", _RCCL_REPO, rccl_dir],
            stderr=True,
        )
        if res.exit_code != 0:
            tools.write_log(res.output.decode("utf-8"))

        res = container.exec_run(
            ["/bin/sh", "-c", f"cd {rccl_dir} && cmake . && make"],
            stderr=True,
        )
        if res.exit_code != 0:
            tools.write_log(res.output.decode("utf-8"))

    tests_dir = os.path.join(work_dir, "rccl-tests")
    if not os.path.isdir(tests_dir):
        logger.info("Building RCCL Tests...")
        res = container.exec_run(
            ["git", "clone", _RCCL_TESTS_REPO, tests_dir],
            stderr=True,
        )
        if res.exit_code != 0:
            tools.write_log(res.output.decode("utf-8"))

        make_cmd = (
            f"cd {tests_dir} && "
            f"make HIP_HOME=/opt/rocm NCCL_HOME={rccl_dir} "
            f"CUSTOM_RCCL_LIB={rccl_dir}/librccl.so && "
            f"make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm "
            f"NCCL_HOME={rccl_dir}"
        )
        res = container.exec_run(
            ["/bin/sh", "-c", make_cmd],
            stderr=True,
        )
        if res.exit_code != 0:
            tools.write_log(res.output.decode("utf-8"))


def run(work_dir, machine_name):
    """Clone, build, run RCCL AllReduce inside Docker, parse and report."""
    with AmdContainer(_RCCL_PYTORCH_IMAGE, work_dir, entrypoint="/bin/bash") as container:
        _build(container, work_dir)

        logger.info("Running RCCL AllReduce...")
        perf_bin = os.path.join(work_dir, "rccl-tests", "build", "all_reduce_perf")
        sizes = []
        bandwidth_columns = []

        for algo in _ALGOS:
            cmd = f"NCCL_ALGO={algo} {perf_bin} " f"-b 8 -e 8G -f 2 -g 8 -n 40"
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

    table = _build_table(sizes, bandwidth_columns, _ALGOS)
    print(table)
    tools.export_markdown("RCCL Bandwidth", _DESCRIPTION, table)
