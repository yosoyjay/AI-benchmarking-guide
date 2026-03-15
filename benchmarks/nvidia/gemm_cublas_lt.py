"""GEMM CuBLASLt benchmark (NVIDIA)."""

import logging
import os
import subprocess

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_cmd

logger = logging.getLogger(__name__)

_SUPERBENCHMARK_REPO = "https://github.com/gitaumark/superbenchmark"
_SUPERBENCHMARK_COMMITS = {
    "main": "ece4c2a05d20aa7200ed24a25baa6051d9148c8b",
    "fp4": "d08fffbbc363f6840420730ee0dd7ad469f3d9de",
}


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_cublaslt_line(text: str) -> dict[str, str] | None:
    """Parse a single whitespace-delimited output line into a dict.

    Expects 6 columns: m, n, k, batch_size, time_us, tflops.
    Returns a dict or None if malformed.
    """
    tokens = text.split()
    if len(tokens) != 6:
        return None
    return {
        "m": tokens[0],
        "n": tokens[1],
        "k": tokens[2],
        "batch_size": tokens[3],
        "time_us": tokens[4],
        "tflops": tokens[5],
    }


def _build_table(rows: list[dict[str, str]]) -> PrettyTable:
    """Format parsed row dicts into a PrettyTable."""
    table = PrettyTable(["M", "N", "K", "Batch Size", "Time(us)", "TFLOPS"])
    for r in rows:
        table.add_row([r["m"], r["n"], r["k"], r["batch_size"], r["time_us"], r["tflops"]])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir: str, datatype: str) -> str:
    """Clone superbenchmark repo, checkout correct branch, build binary."""
    repo_dir = os.path.join(work_dir, "superbenchmark")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _SUPERBENCHMARK_REPO, "superbenchmark"], cwd=work_dir)

    branch = "fp4" if datatype == "fp4e2m1" else "main"
    tools.run_cmd(["git", "checkout", _SUPERBENCHMARK_COMMITS[branch]], cwd=repo_dir)

    build_path = os.path.join(
        repo_dir,
        "superbench",
        "benchmarks",
        "micro_benchmarks",
        "cublaslt_gemm",
    )
    tools.run_cmd(["cmake", "-S", "./"], cwd=build_path)
    tools.run_cmd(["make"], cwd=build_path)

    bindir = os.path.join(work_dir, "bin")
    os.makedirs(bindir, exist_ok=True)
    subprocess.run(
        ["mv", os.path.join(build_path, "cublaslt_gemm"), bindir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return bindir


def run(
    work_dir: str, machine_name: str, config_path: str = "config.json", ctx: RunContext | None = None
) -> list[dict[str, str]] | None:
    """Clone, build, run CuBLASLt GEMM, parse and report results."""
    config = tools.load_benchmark_config(config_path, "GEMMCublasLt")
    datatype = config["datatype"]

    # A100 does not support fp8
    if "A100" in machine_name:
        logger.warning("A100 does not support %s, using fp16 instead", datatype)
        datatype = "fp16"

    bindir = _build(work_dir, datatype)

    b = 1
    i = 1000
    w = 10000

    logger.info("Running CublasLt with datatype %s...", datatype)
    if datatype == "fp8e4m3":
        m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
        n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]
    elif datatype == "fp4e2m1":
        m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 802816]
        n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 768]
    else:
        m_dims = [1024, 2048, 4096, 8192, 16384, 1024, 6144, 802816]
        n_dims = [1024, 2048, 4096, 8192, 16384, 2145, 12288, 192]
        k_dims = [1024, 2048, 4096, 8192, 16384, 1024, 12288, 768]

    cublaslt_bin = os.path.join(bindir, "cublaslt_gemm")
    rows = []
    for m, n, k in zip(m_dims, n_dims, k_dims):
        cmd = [
            cublaslt_bin,
            "-m",
            str(m),
            "-n",
            str(n),
            "-k",
            str(k),
            "-b",
            str(b),
            "-i",
            str(i),
            "-w",
            str(w),
            "-t",
            datatype,
        ]
        if ctx is not None:
            result = capture_cmd(cmd, ctx=ctx, suffix=f"_m{m}_n{n}_k{k}")
        else:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            logger.warning(
                "cublaslt_gemm failed for M=%s N=%s K=%s: returncode=%s",
                m,
                n,
                k,
                result.returncode,
            )
            tools.write_log(tools.check_error(result))
            continue
        parsed = parse_cublaslt_line(result.stdout.decode("utf-8"))
        if parsed:
            rows.append(parsed)
        else:
            logger.warning(
                "Skipping cublaslt_gemm result with unexpected format: %s",
                result.stdout.decode("utf-8").strip(),
            )
        tools.write_log(tools.check_error(result))

    if ctx is not None:
        ctx.extra["datatype"] = datatype
        return rows

    table = _build_table(rows)
    print(table)
    tools.export_markdown(
        "GEMM CuBLASLt",
        f"The results shown below are with random initialization (best representation of real-life workloads) {datatype}, and {w} warmup iterations.",
        table,
    )
    return None
