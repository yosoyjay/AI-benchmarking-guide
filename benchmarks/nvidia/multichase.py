"""Multichase memory latency benchmark (NVIDIA)."""

import logging
import os

from infra import tools

logger = logging.getLogger(__name__)

_MULTICHASE_REPO = "https://github.com/google/multichase"
_MULTICHASE_COMMIT = "a1ffd89b20d033f28fa44d6ee92d7a378b1b1dca"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_multichase_output(text):
    """Parse tabular multichase output into node names and row dicts.

    Expects a header like ``CPU  NODE0  NODE1 ...`` followed by data rows
    where the first field is the CPU id and the rest are latency floats.

    Returns ``(node_names, rows)`` where *node_names* is e.g.
    ``["NODE0", "NODE1"]`` and each row is
    ``{"cpu": "0", "NODE0": 3.2, "NODE1": 5.1}``.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return [], []

    header_fields = lines[0].split()
    if not header_fields or header_fields[0] != "CPU":
        return [], []

    node_names = header_fields[1:]
    rows = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) != len(header_fields):
            continue
        row = {"cpu": fields[0]}
        for name, val in zip(node_names, fields[1:]):
            try:
                row[name] = float(val)
            except ValueError:
                continue
        rows.append(row)
    return node_names, rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir):
    """Clone and build multichase."""
    repo_dir = os.path.join(work_dir, "multichase")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _MULTICHASE_REPO, "multichase"], cwd=work_dir)
        tools.run_cmd(["git", "checkout", _MULTICHASE_COMMIT], cwd=repo_dir)
        tools.run_cmd(["make"], cwd=repo_dir)


def run(work_dir, machine_name):
    """Clone, build, run multichase, and report raw output."""
    _build(work_dir)

    logger.info("Running Multichase...")
    multichase_bin = os.path.join(work_dir, "multichase", "multichase")
    script_path = os.path.join(work_dir, "benchmarks", "nvidia", "run_multichase.sh")
    result = tools.run_cmd([script_path, multichase_bin])
    output = result.stdout.decode("utf-8")
    print(output)
    tools.export_markdown("Multichase", output, None)
