"""CPU Stream benchmark (NVIDIA, BabelStream-based)."""

import logging
import os
import time

from infra import tools

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/UoB-HPC/BabelStream"
_BABELSTREAM_COMMIT = "2411a9ac6832eb6562d81c81d016c53a66355eed"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir):
    """Clone and build BabelStream for OpenMP."""
    repo_dir = os.path.join(work_dir, "CPUStream")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _BABELSTREAM_REPO, "CPUStream"], cwd=work_dir)
        tools.run_cmd(["git", "checkout", _BABELSTREAM_COMMIT], cwd=repo_dir)

    build_dir = os.path.join(repo_dir, "build")
    if not os.path.isdir(build_dir):
        os.mkdir(build_dir)
        tools.run_cmd(
            ["cmake", "-DMODEL=omp", "../"],
            cwd=build_dir,
        )
        tools.run_cmd(["make"], cwd=build_dir)

    return build_dir


def run(work_dir, machine_name, config_path="config.json"):
    """Clone, build, run CPU stream, parse and report results."""
    config = tools.load_benchmark_config(config_path, "CPUStream")
    num_runs = config["inputs"]["num_runs"]
    interval = config["inputs"]["interval"]
    cpu_count = os.cpu_count() or 4

    build_dir = _build(work_dir)

    logger.info("Running CPU Stream...")
    env = {
        "OMP_NUM_THREADS": str(cpu_count),
        "OMP_PROC_BIND": "spread",
    }
    buffer = []
    for _ in range(num_runs):
        result = tools.run_cmd(
            ["taskset", "-c", f"0-{cpu_count - 1}", "./omp-stream"],
            cwd=build_dir,
            env=env,
        )
        log = tools.parse_babelstream_output(result.stdout.decode("utf-8"))
        buffer.append(log)
        time.sleep(int(interval))

    tools.summarize_babelstream(
        buffer,
        divisor=1_000,
        units="GB/s",
        title="CPU STREAM",
        description="CPU STREAM Results",
    )
