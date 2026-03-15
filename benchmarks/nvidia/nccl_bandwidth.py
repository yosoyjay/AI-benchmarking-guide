"""NCCL Bandwidth benchmark (NVIDIA)."""

import logging
import os

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext

logger = logging.getLogger(__name__)

_NCCL_REPO = "https://github.com/NVIDIA/nccl.git"
_NCCL_COMMIT = "361915904b456d397e6e1578f8f65ea1a45bdd28"
_NCCL_TESTS_REPO = "https://github.com/NVIDIA/nccl-tests.git"
_NCCL_TESTS_COMMIT = "af1dcac92ad7ed81ffa32b593480ab3f0f7baa01"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_nccl_output(text: str) -> list[dict[str, str]]:
    """Extract size and bandwidth from all_reduce_perf 13-column output.

    Returns a list of dicts with keys: size, bandwidth.
    """
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 13:
            rows.append({"size": fields[0], "bandwidth": fields[11]})
    return rows


def _build_table(rows: list[dict[str, str]], algo: str) -> PrettyTable:
    """Format parsed rows into a PrettyTable."""
    table = PrettyTable()
    table.add_column("Message Size", [r["size"] for r in rows])
    table.add_column(f"Bandwidth ({algo})", [r["bandwidth"] for r in rows])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _get_gpu_count() -> int:
    """Detect number of GPUs via nvidia-smi."""
    result = tools.run_cmd(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    )
    if result.returncode != 0:
        logger.warning("nvidia-smi failed to detect GPU count, defaulting to 8")
        return 8
    lines = [line for line in result.stdout.decode("utf-8").splitlines() if line.strip()]
    return len(lines) if lines else 8


def _build(work_dir: str, env: dict[str, str] | None) -> tuple[str, dict[str, str]]:
    """Clone and build NCCL and nccl-tests."""
    nccl_dir = os.path.join(work_dir, "nccl")
    if not os.path.isdir(nccl_dir):
        logger.info("Building NCCL Library...")
        tools.run_cmd(["git", "clone", _NCCL_REPO, "nccl"], cwd=work_dir)
        tools.run_cmd(["git", "checkout", _NCCL_COMMIT], cwd=nccl_dir)
        tools.run_cmd(["make", "-j", "src.build"], cwd=nccl_dir)

    nccl_home = os.path.join(nccl_dir, "build")
    ld_path = f"{os.path.join(nccl_dir, 'build', 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    env = {**os.environ, "NCCL_HOME": nccl_home, "LD_LIBRARY_PATH": ld_path}

    tests_dir = os.path.join(work_dir, "nccl-tests")
    if not os.path.isdir(tests_dir):
        logger.info("Building NCCL Test...")
        tools.run_cmd(["git", "clone", _NCCL_TESTS_REPO, "nccl-tests"], cwd=work_dir)
        tools.run_cmd(["git", "checkout", _NCCL_TESTS_COMMIT], cwd=tests_dir)
        tools.run_cmd(["make"], env=env, cwd=tests_dir)

    return tests_dir, env


def run(work_dir: str, machine_name: str, ctx: RunContext | None = None) -> list[dict[str, str]] | None:
    """Clone, build, run NCCL all-reduce, parse and report results."""
    tests_dir, env = _build(work_dir, None)

    num_gpus = _get_gpu_count()
    algo = "Ring" if num_gpus == 4 else "NVLS"
    logger.info("Running NCCL AllReduce on %s GPUs", num_gpus)

    all_reduce_bin = os.path.join(tests_dir, "build", "all_reduce_perf")
    run_env = {**env, "NCCL_ALGO": algo}
    cmd = [
        all_reduce_bin,
        "-b",
        "8",
        "-e",
        "8G",
        "-f",
        "2",
        "-g",
        str(num_gpus),
        "-n",
        "40",
    ]
    if ctx is not None:
        from infra.capture import capture_cmd

        result = capture_cmd(cmd, ctx=ctx, env=run_env)
    else:
        result = tools.run_cmd(cmd, env=run_env)
    output = result.stdout.decode("utf-8")
    # Filter to lines containing "float" (same as the old grep)
    float_lines = "\n".join(line for line in output.splitlines() if "float" in line)
    rows = parse_nccl_output(float_lines)

    if ctx is not None:
        ctx.extra["algorithm"] = algo
        return rows

    table = _build_table(rows, algo)
    print(table)
    tools.export_markdown(
        "NCCL Bandwidth",
        f"The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce ({algo} algorithm) tests in-place operations, varying from 1KB to 8GB of data.",
        table,
    )
    return None
