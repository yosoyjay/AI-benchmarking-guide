"""HBM Bandwidth benchmark (AMD, BabelStream-based)."""

import logging
import os
import time

from infra import tools
from infra.capture import RunContext

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/gitaumark/BabelStream"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir: str) -> None:
    """Clone and build BabelStream for HIP."""
    repo_dir = os.path.join(work_dir, "BabelStream")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(
            ["git", "clone", _BABELSTREAM_REPO, "BabelStream"],
            cwd=work_dir,
        )
        tools.run_cmd(
            [
                "cmake",
                "-Bbuild",
                "-H.",
                "-DMODEL=hip",
                "-DRELEASE_FLAGS=-O3",
                "-DCMAKE_CXX_COMPILER=hipcc",
            ],
            cwd=repo_dir,
        )
        tools.run_cmd(["cmake", "--build", "build"], cwd=repo_dir)


def run(
    work_dir: str, machine_name: str, config_path: str = "config.json", ctx: RunContext | None = None
) -> dict[str, dict[str, float]] | None:
    """Clone, build, run HBM bandwidth, parse and report results."""
    config = tools.load_benchmark_config(config_path, "HBMBandwidth")
    num_runs = config["inputs"]["num_runs"]
    interval = config["inputs"]["interval"]

    _build(work_dir)

    logger.info("Running HBM Bandwidth...")
    hip_stream_bin = os.path.join(work_dir, "BabelStream", "build", "hip-stream")
    cmd = ["sudo", hip_stream_bin]
    buffer = []
    for i in range(num_runs):
        if ctx is not None:
            from infra.capture import capture_cmd

            result = capture_cmd(cmd, ctx=ctx, suffix=f"_run{i}")
        else:
            result = tools.run_cmd(cmd)
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
                    "min": round(min(values) / 1_000_000, 2),
                    "max": round(max(values) / 1_000_000, 2),
                    "mean": round(statistics.mean(values) / 1_000_000, 2),
                }
        ctx.extra["units"] = "TB/s"
        return summary

    tools.summarize_babelstream(
        buffer,
        divisor=1_000_000,
        units="TB/s",
        title="HBM Bandwidth",
        description="HBM bandwidth Results",
    )
