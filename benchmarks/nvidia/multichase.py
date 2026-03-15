"""Multichase memory latency benchmark (NVIDIA)."""

import logging
import os

from infra import tools

logger = logging.getLogger(__name__)

_MULTICHASE_REPO = "https://github.com/google/multichase"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir):
    """Clone and build multichase."""
    repo_dir = os.path.join(work_dir, "multichase")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _MULTICHASE_REPO, "multichase"], cwd=work_dir)
        tools.run_cmd(["make"], cwd=repo_dir)


def run(work_dir, machine_name):
    """Clone, build, run multichase, and report raw output."""
    _build(work_dir)

    logger.info("Running Multichase...")
    script_dir = os.path.join(work_dir, "benchmarks", "nvidia")
    tools.run_cmd(["chmod", "755", "run_multichase.sh"], cwd=script_dir)
    result = tools.run_cmd(["./run_multichase.sh"], cwd=script_dir)
    output = result.stdout.decode("utf-8")
    print(output)
    tools.export_markdown("Multichase", output, None)
