"""FIO storage benchmark."""

import glob
import logging
import os
import subprocess

from prettytable import PrettyTable

from infra import tools

logger = logging.getLogger(__name__)

_FIO_TESTS = [
    ["read", "1M"],
    ["read", "512k"],
    ["read", "1k"],
    ["write", "1M"],
    ["write", "512k"],
    ["write", "1k"],
    ["randwrite", "1k"],
    ["randread", "1k"],
]


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_fio_output(text):
    """Extract bandwidth string from fio output.

    Scans for lines containing ': bw=' and extracts the bandwidth value
    (third token on the matching line, stripped of commas/parens).
    Returns the bandwidth string or ``"error"`` if not found.
    """
    for line in text.splitlines():
        if ": bw=" in line:
            tokens = line.split()
            if len(tokens) >= 3:
                return tokens[2].strip(",()")
    return "error"


def _build_table(rows):
    """Format parsed row tuples into a PrettyTable."""
    table = PrettyTable(["Test", "Batch Size(Bytes)", "Bandwidth"])
    for rw, bs, bw in rows:
        table.add_row([rw, bs, bw])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(work_dir, machine_name):
    """Run FIO storage benchmarks, parse and report results."""
    output_dir = os.path.join(work_dir, "Outputs")

    logger.info("Running FIO Tests...")
    rows = []
    for rw, bs in _FIO_TESTS:
        result = subprocess.run(
            [
                "fio",
                f"--bs={bs}",
                "--ioengine=libaio",
                "--iodepth=255",
                f"--directory={output_dir}",
                "--direct=1",
                "--runtime=300",
                "--numjobs=4",
                f"--rw={rw}",
                "--name=test",
                "--group_reporting",
                "--gtod_reduce=1",
                "--size=10G",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            logger.warning("fio failed for %s bs=%s: returncode=%s", rw, bs, result.returncode)
            bw = "error"
        else:
            bw = parse_fio_output(result.stdout.decode("utf-8"))
        rows.append((rw, bs, bw))

    table = _build_table(rows)
    print(table)
    tools.export_markdown("FIO Tests", "", table)

    # Clean up test files
    for path in glob.glob(os.path.join(output_dir, "test*")):
        os.remove(path)
