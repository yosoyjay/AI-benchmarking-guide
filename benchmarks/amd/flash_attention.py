"""Flash Attention 2 benchmark (AMD ROCm, Docker-based)."""

import logging
import os
import re

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_docker
from infra.containers import AmdContainer

logger = logging.getLogger(__name__)

_FLASH_ATTENTION_IMAGE = "powderluv/vllm_dev_channel:20240927"
_FLASH_ATTENTION_REPO = "https://github.com/Dao-AILab/flash-attention.git"
_FLASH_ATTENTION_CHECKOUT = "418d677"


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


def run(work_dir: str, machine_name: str, ctx: RunContext | None = None) -> list[dict] | None:
    """Clone repo, run benchmark inside Docker, parse and report results."""
    repo_dir = os.path.join(work_dir, "flash-attention")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _FLASH_ATTENTION_REPO], cwd=work_dir)

    tools.run_cmd(["git", "checkout", _FLASH_ATTENTION_CHECKOUT], cwd=repo_dir)

    bench_script = os.path.join(work_dir, "flash-attention", "benchmarks", "benchmark_flash_attention.py")

    logger.info("Running Flash Attention...")
    with AmdContainer(_FLASH_ATTENTION_IMAGE, work_dir) as container:
        if ctx is not None:
            stdout, stderr, exit_code = capture_docker(container, ["python3", bench_script], ctx=ctx)
            output_text = stdout
        else:
            res = container.exec_run(["python3", bench_script])
            tools.write_log(res.output.decode("utf-8"))
            output_text = res.output.decode("utf-8")

    rows = parse_flash_attention_output(output_text)

    if ctx is not None:
        return rows

    table = _build_table(rows)
    print(table)
    tools.export_markdown("Flash Attention 2", _DESCRIPTION, table)
    return rows
