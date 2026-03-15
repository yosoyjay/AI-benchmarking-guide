"""HBM Bandwidth benchmark (NVIDIA, BabelStream-based)."""

import logging
import os
import time

from infra import tools
from infra.capture import RunContext

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/gitaumark/BabelStream"
_BABELSTREAM_COMMIT = "1a1a729517df6c44bfbfe6d0db36a24fe8dc6726"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def _get_cuda_arch(machine_name: str) -> str:
    """Map machine name to CUDA architecture string."""
    if "A100" in machine_name:
        return "sm_80"
    if "GB200" in machine_name:
        return "sm_100"
    return "sm_90"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir: str, machine_name: str) -> str:
    """Clone and build BabelStream for CUDA."""
    repo_dir = os.path.join(work_dir, "BabelStream")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _BABELSTREAM_REPO, "BabelStream"], cwd=work_dir)
        tools.run_cmd(["git", "checkout", _BABELSTREAM_COMMIT], cwd=repo_dir)

    build_dir = os.path.join(repo_dir, "build")
    if not os.path.isdir(build_dir):
        os.mkdir(build_dir)
        arch = _get_cuda_arch(machine_name)
        tools.run_cmd(
            [
                "cmake",
                "../",
                "-DMODEL=cuda",
                f"-DCUDA_ARCH={arch}",
                "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
            ],
            cwd=build_dir,
        )
        tools.run_cmd(["make"], cwd=build_dir)

    return build_dir


def run(
    work_dir: str, machine_name: str, config_path: str = "config.json", ctx: RunContext | None = None
) -> dict[str, dict[str, float]] | None:
    """Clone, build, run HBM bandwidth, parse and report results."""
    config = tools.load_benchmark_config(config_path, "HBMBandwidth")
    num_runs = config["inputs"]["num_runs"]
    interval = config["inputs"]["interval"]

    build_dir = _build(work_dir, machine_name)

    logger.info("Running HBM Bandwidth...")
    cmd = ["./cuda-stream"]
    buffer = []
    for i in range(num_runs):
        if ctx is not None:
            from infra.capture import capture_cmd

            result = capture_cmd(cmd, ctx=ctx, suffix=f"_run{i}", cwd=build_dir)
        else:
            result = tools.run_cmd(cmd, cwd=build_dir)
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
    return None
