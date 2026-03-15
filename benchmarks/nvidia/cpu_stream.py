"""CPU Stream benchmark (NVIDIA, BabelStream-based)."""

import logging
import os
import time

from infra import tools
from infra.capture import RunContext

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/UoB-HPC/BabelStream"
_BABELSTREAM_COMMIT = "2411a9ac6832eb6562d81c81d016c53a66355eed"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir: str) -> str:
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


def run(
    work_dir: str, machine_name: str, config_path: str = "config.json", ctx: RunContext | None = None
) -> dict[str, dict[str, float]] | None:
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
    cmd = ["taskset", "-c", f"0-{cpu_count - 1}", "./omp-stream"]
    buffer = []
    for i in range(num_runs):
        if ctx is not None:
            from infra.capture import capture_cmd

            result = capture_cmd(cmd, ctx=ctx, suffix=f"_run{i}", cwd=build_dir, env=env)
        else:
            result = tools.run_cmd(cmd, cwd=build_dir, env=env)
        log = tools.parse_babelstream_output(result.stdout.decode("utf-8"))
        buffer.append(log)
        time.sleep(int(interval))

    if ctx is not None:
        import statistics

        ops = {name: [] for name in tools.BABELSTREAM_OPS}
        for log in buffer:
            if len(log) < 5:
                continue
            for idx, name in enumerate(tools.BABELSTREAM_OPS):
                ops[name].append(float(log[idx][1]))
        summary = {}
        for name in tools.BABELSTREAM_OPS:
            values = ops[name]
            if values:
                summary[name] = {
                    "min": round(min(values) / 1_000, 2),
                    "max": round(max(values) / 1_000, 2),
                    "mean": round(statistics.mean(values) / 1_000, 2),
                }
        return summary

    tools.summarize_babelstream(
        buffer,
        divisor=1_000,
        units="GB/s",
        title="CPU STREAM",
        description="CPU STREAM Results",
    )
    return None
