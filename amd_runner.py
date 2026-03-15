import argparse
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from benchmarks import fio as FIO
from benchmarks.amd import flash_attention as FA
from benchmarks.amd import gemm_hipblas_lt as GEMM
from benchmarks.amd import hbm_bandwidth as HBM
from benchmarks.amd import llm_benchmark as llmb
from benchmarks.amd import rccl_bandwidth as RCCL
from benchmarks.amd import transfer_bench as TB
from infra import process, tools
from infra.capture import RunContext, get_version, make_run_dir, make_session_dir

logger = logging.getLogger(__name__)

_PLATFORM = "amd"

_SKU_MAP = {
    "MI300X": "ND_MI300X_v5",
    "MI300": "ND_MI300X_v5",
    "MI250X": "ND_MI250X_v4",
    "MI250": "ND_MI250_v4",
}


def _detect_sku() -> str:
    try:
        results = subprocess.run(
            "rocminfo | grep 'Marketing Name'",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if results.returncode == 0 and results.stdout:
            name = results.stdout.decode("utf-8").strip()
            for gpu_name, sku in _SKU_MAP.items():
                if gpu_name in name:
                    return sku
    except Exception:
        pass
    logger.warning("could not detect AMD GPU SKU, falling back to ND_MI300X_v5")
    return "ND_MI300X_v5"


def get_system_specs(session_dir: Path) -> str:
    specs_path = session_dir / "system_specs.txt"
    with open(specs_path, "w") as file:

        results = subprocess.run(
            "rocminfo | grep 'ROCk module version'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if results.returncode == 0:
            parts = results.stdout.decode("utf-8").strip().split(" ")
            rocm_version = parts[3] if len(parts) >= 4 else "unknown"
        else:
            rocm_version = "unknown"
        file.write(f"ROCm version     : {rocm_version}\n")

        results = subprocess.run(
            "lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if results.returncode == 0:
            parts = results.stdout.decode("utf-8").strip().split("\t")
            ubuntu = parts[1] if len(parts) >= 2 else "unknown"
        else:
            ubuntu = "unknown"
        file.write(f"ubuntu version   : {ubuntu}\n")

        results = subprocess.run(
            "grep 'stepping\\|model\\|microcode' /proc/cpuinfo | grep microcode",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if results.returncode == 0:
            lines = results.stdout.decode("utf-8").split("\n")
            microcode = lines[0] if lines else ""
        else:
            microcode = ""
        file.write(f"{microcode}\n")

        results = subprocess.run(
            "grep 'stepping\\|model\\|microcode' /proc/cpuinfo | grep name",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if results.returncode == 0:
            lines = results.stdout.decode("utf-8").split("\n")
            file.write(f"{lines[0] if lines else ''}\n")
        else:
            file.write("\n")

        results = subprocess.run(
            "grep 'cores\\|model\\|microcode' /proc/cpuinfo | grep cores",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if results.returncode == 0:
            lines = results.stdout.decode("utf-8").split("\n")
            file.write(lines[0] if lines else "")
        else:
            file.write("")
    return _detect_sku()


def _make_ctx(benchmark: str, sku: str, session_dir: Path, version: str, timestamp: datetime) -> RunContext:
    """Create a RunContext for a single benchmark."""
    run_dir = make_run_dir(session_dir, benchmark)
    return RunContext(
        benchmark=benchmark,
        sku=sku,
        platform=_PLATFORM,
        version=version,
        timestamp=timestamp,
        session_dir=session_dir,
        run_dir=run_dir,
    )


def run_transfer_bench(machine_name: str, current: str, ctx: RunContext | None = None) -> None:
    parsed = TB.run(work_dir=current, machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.transfer_bench_to_csv(ctx, parsed)
        process.process_run("transfer_bench", ctx, csv_rows)


def run_gemm_hipblas(machine_name: str, current: str, ctx: RunContext | None = None) -> None:
    parsed = GEMM.run(work_dir=current, machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.gemm_hipblas_lt_to_csv(ctx, parsed)
        process.process_run("gemm_hipblas_lt", ctx, csv_rows)


def run_rccl_bandwidth(machine_name: str, current: str, ctx: RunContext | None = None) -> None:
    all_parsed = RCCL.run(work_dir=current, machine_name=machine_name, ctx=ctx)
    if ctx is not None and all_parsed is not None:
        csv_rows = []
        for algo, rows in all_parsed.items():
            csv_rows.extend(process.rccl_bandwidth_to_csv(ctx, rows, algo))
        process.process_run("rccl_bandwidth", ctx, csv_rows)


def run_flash_attention(machine_name: str, current: str, ctx: RunContext | None = None) -> None:
    parsed = FA.run(work_dir=current, machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.flash_attention_to_csv(ctx, parsed)
        process.process_run("flash_attention", ctx, csv_rows)


def run_fio(machine_name: str, current: str, ctx: RunContext | None = None) -> None:
    parsed = FIO.run(work_dir=current, machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.fio_to_csv(ctx, parsed)
        process.process_run("fio", ctx, csv_rows)


def run_hbm_bandwidth(machine_name: str, current: str, ctx: RunContext | None = None) -> None:
    parsed = HBM.run(work_dir=current, machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.hbm_bandwidth_to_csv(ctx, parsed)
        process.process_run("hbm_bandwidth", ctx, csv_rows)


def run_llm_benchmark(machine_name: str, current: str, ctx: RunContext | None = None) -> None:
    parsed = llmb.run(work_dir=current, machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.llm_benchmark_amd_to_csv(ctx, parsed)
        process.process_run("llm_benchmark", ctx, csv_rows)


BENCHMARKS = {
    "gemm": "GEMMHipBLAS",
    "rccl": "RCCLBandwidth",
    "hbm": "HBMBandwidth",
    "transfer": "TransferBench",
    "fa": "FlashAttention",
    "fio": "FIO",
    "llm": "LLMBenchmark",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="AMD GPU Benchmark Suite")
    parser.add_argument(
        "benchmarks",
        nargs="+",
        choices=[*BENCHMARKS, "all"],
        type=str.lower,
        help="Benchmarks to run",
    )
    args = parser.parse_args()

    current = os.getcwd()

    # Structured output pipeline
    version = get_version()
    results_dir = Path(current) / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now()

    # Detect SKU first so we can name the session dir
    machine_name = _detect_sku()
    session_dir = make_session_dir(results_dir, machine_name, timestamp)

    tools.set_log_path(str(session_dir / "log.txt"))
    tools.set_summary_path(str(session_dir / "summary.md"))

    get_system_specs(session_dir)

    def _ctx(benchmark):
        return _make_ctx(benchmark, machine_name, session_dir, version, timestamp)

    dispatch = {
        "gemm": lambda: run_gemm_hipblas(machine_name, current, ctx=_ctx("gemm_hipblas_lt")),
        "rccl": lambda: run_rccl_bandwidth(machine_name, current, ctx=_ctx("rccl_bandwidth")),
        "hbm": lambda: run_hbm_bandwidth(machine_name, current, ctx=_ctx("hbm_bandwidth")),
        "transfer": lambda: run_transfer_bench(machine_name, current, ctx=_ctx("transfer_bench")),
        "fa": lambda: run_flash_attention(machine_name, current, ctx=_ctx("flash_attention")),
        "fio": lambda: run_fio(machine_name, current, ctx=_ctx("fio")),
        "llm": lambda: run_llm_benchmark(machine_name, current, ctx=_ctx("llm_benchmark")),
    }

    selected = list(dispatch.keys()) if "all" in args.benchmarks else args.benchmarks
    for key in selected:
        name = BENCHMARKS[key]
        try:
            dispatch[key]()
        except Exception as e:
            logger.warning("%s benchmark failed: %s", name, e)


if __name__ == "__main__":
    main()
