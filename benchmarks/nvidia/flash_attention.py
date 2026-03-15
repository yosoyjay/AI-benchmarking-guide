"""Flash Attention 2 benchmark (NVIDIA)."""

import logging
import os
import re

from prettytable import PrettyTable

from infra import tools

logger = logging.getLogger(__name__)

_FLASH_ATTENTION_REPO = "https://github.com/Dao-AILab/flash-attention.git"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_flash_attention_output(text: str) -> list[dict]:
    """Extract causal/headdim/tflops rows from benchmark output.

    Returns a list of dicts with keys: causal, headdim, flash2_tflops,
    pytorch_tflops.
    """
    if not text:
        return []

    rows = []
    for m in re.findall(
        r"causal=(\w+), headdim=(\d+).*?fwd \+ bwd: ([\d.]+).*?fwd \+ bwd: ([\d.]+)",
        text,
        re.DOTALL,
    ):
        rows.append(
            {
                "causal": m[0],
                "headdim": int(m[1]),
                "flash2_tflops": float(m[2]),
                "pytorch_tflops": float(m[3]),
            }
        )

    if not rows:
        logger.warning("Flash Attention: no results parsed from output")
    return rows


def _build_table(rows: list[dict]) -> PrettyTable:
    """Format parsed rows into a PrettyTable."""
    table = PrettyTable(["causal", "headdim", "Flash2 total (TFLOPs)", "Pytorch total (TFLOPs)"])
    for r in rows:
        table.add_row([r["causal"], r["headdim"], r["flash2_tflops"], r["pytorch_tflops"]])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "The performance (in TFLOPS), in table below, represents the "
    "performance for a batch size of 2, and a sequence length of 8192."
)


def run(work_dir: str, machine_name: str) -> list[dict]:
    """Clone repo (if needed), run benchmark, parse and report results."""
    repo_dir = os.path.join(work_dir, "flash-attention")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _FLASH_ATTENTION_REPO], cwd=work_dir)

    bench_dir = os.path.join(repo_dir, "benchmarks")
    logger.info("Running Flash Attention with batch size=2, seqlen=8192...")
    result = tools.run_cmd(["python3", "benchmark_flash_attention.py"], cwd=bench_dir)

    rows = parse_flash_attention_output(result.stdout.decode("utf-8"))
    table = _build_table(rows)
    print(table)
    tools.export_markdown("Flash Attention 2", _DESCRIPTION, table)
    return rows
