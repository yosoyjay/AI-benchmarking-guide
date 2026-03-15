"""HBM Bandwidth benchmark (NVIDIA, BabelStream-based)."""

import logging
import os
import time

from infra import tools

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/gitaumark/BabelStream"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def _get_cuda_arch(machine_name):
    """Map machine name to CUDA architecture string."""
    if "A100" in machine_name:
        return "sm_80"
    if "GB200" in machine_name:
        return "sm_100"
    return "sm_90"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir, machine_name):
    """Clone and build BabelStream for CUDA."""
    repo_dir = os.path.join(work_dir, "BabelStream")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _BABELSTREAM_REPO, "BabelStream"], cwd=work_dir)

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


def run(work_dir, machine_name, config_path="config.json"):
    """Clone, build, run HBM bandwidth, parse and report results."""
    config = tools.load_benchmark_config(config_path, "HBMBandwidth")
    num_runs = config["inputs"]["num_runs"]
    interval = config["inputs"]["interval"]

    build_dir = _build(work_dir, machine_name)

    logger.info("Running HBM Bandwidth...")
    buffer = []
    for _ in range(num_runs):
        result = tools.run_cmd(["./cuda-stream"], cwd=build_dir)
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
