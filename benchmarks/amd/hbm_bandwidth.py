"""HBM Bandwidth benchmark (AMD, BabelStream-based)."""

import logging
import os
import time

from infra import tools

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/gitaumark/BabelStream"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir):
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


def run(work_dir, machine_name, config_path="config.json"):
    """Clone, build, run HBM bandwidth, parse and report results."""
    config = tools.load_benchmark_config(config_path, "HBMBandwidth")
    num_runs = config["inputs"]["num_runs"]
    interval = config["inputs"]["interval"]

    _build(work_dir)

    logger.info("Running HBM Bandwidth...")
    hip_stream_bin = os.path.join(work_dir, "BabelStream", "build", "hip-stream")
    buffer = []
    for _ in range(num_runs):
        result = tools.run_cmd(["sudo", hip_stream_bin])
        log = tools.parse_babelstream_output(result.stdout.decode("utf-8"))
        buffer.append(log)
        time.sleep(int(interval))

    tools.summarize_babelstream(
        buffer,
        divisor=1_000_000,
        units="TB/s",
        title="HBM Bandwidth",
        description="HBM bandwidth Results",
    )
