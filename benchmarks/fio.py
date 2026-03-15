"""FIO storage benchmark."""

import glob
import logging
import os
import subprocess
import tempfile

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_cmd

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


def parse_fio_output(text: str) -> str:
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


def _build_table(rows: list[tuple[str, str, str]]) -> PrettyTable:
    """Format parsed row tuples into a PrettyTable."""
    table = PrettyTable(["Test", "Batch Size(Bytes)", "Bandwidth"])
    for rw, bs, bw in rows:
        table.add_row([rw, bs, bw])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(work_dir: str, machine_name: str, ctx: RunContext | None = None) -> list[tuple[str, str, str]] | None:
    """Run FIO storage benchmarks, parse and report results."""
    if ctx is not None:
        fio_dir = str(ctx.run_dir)
    else:
        fio_dir = tempfile.mkdtemp(prefix="fio_")

    logger.info("Running FIO Tests...")
    rows = []
    for rw, bs in _FIO_TESTS:
        cmd = [
            "fio",
            f"--bs={bs}",
            "--ioengine=libaio",
            "--iodepth=255",
            f"--directory={fio_dir}",
            "--direct=1",
            "--runtime=300",
            "--numjobs=4",
            f"--rw={rw}",
            "--name=test",
            "--group_reporting",
            "--gtod_reduce=1",
            "--size=10G",
        ]
        if ctx is not None:
            result = capture_cmd(cmd, ctx=ctx, suffix=f"_{rw}_{bs}")
        else:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        if result.returncode != 0:
            logger.warning("fio failed for %s bs=%s: returncode=%s", rw, bs, result.returncode)
            bw = "error"
        else:
            bw = parse_fio_output(result.stdout.decode("utf-8"))
        rows.append((rw, bs, bw))

    # Clean up fio test files
    for path in glob.glob(os.path.join(fio_dir, "test*")):
        os.remove(path)

    if ctx is not None:
        return rows

    table = _build_table(rows)
    print(table)
    tools.export_markdown("FIO Tests", "", table)
    return None
