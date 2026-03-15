import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from prettytable import PrettyTable

from benchmarks import fio as FIO
from benchmarks.nvidia import cpu_stream as CPU
from benchmarks.nvidia import flash_attention as FA
from benchmarks.nvidia import gemm_cublas_lt as gemm
from benchmarks.nvidia import hbm_bandwidth as HBM
from benchmarks.nvidia import llama3_run as llama3pre
from benchmarks.nvidia import llm_benchmark as llmb
from benchmarks.nvidia import multichase as Multichase
from benchmarks.nvidia import nccl_bandwidth as NCCL
from benchmarks.nvidia import nv_bandwidth as NV
from infra import process, tools
from infra.capture import RunContext, get_version, make_run_dir, make_session_dir

logger = logging.getLogger(__name__)

_PLATFORM = "nvidia"


def _detect_gpu_name() -> str:
    """Query nvidia-smi and return the GPU marketing name (e.g. 'NVIDIA H200')."""
    results = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name,vbios_version,driver_version,memory.total", "--format=csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if results.returncode != 0:
        logger.error("nvidia-smi failed. Is an NVIDIA GPU present and driver installed?")
        sys.exit(1)
    lines = results.stdout.decode("utf-8").split("\n")
    if len(lines) < 2 or not lines[1].strip():
        logger.error("nvidia-smi returned no GPU data")
        sys.exit(1)
    output = lines[1].split(",")
    return output[0].strip()


def _write_system_specs(session_dir: Path, gpu_name: str) -> None:
    """Write system specs summary.md into session_dir (only if it doesn't exist yet)."""
    summary_path = session_dir / "summary.md"
    if summary_path.exists():
        return
    results = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name,vbios_version,driver_version,memory.total", "--format=csv"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if results.returncode != 0:
        return
    lines = results.stdout.decode("utf-8").split("\n")
    if len(lines) < 2 or not lines[1].strip():
        return
    output = lines[1].split(",")

    table = PrettyTable([" ", gpu_name])
    if len(output) > 1:
        table.add_row(["VBIOS", output[1]])
    if len(output) > 2:
        table.add_row(["driver version", output[2]])
    if len(output) > 3:
        table.add_row(["GPU memory capacity", output[3]])

    results = subprocess.run(
        "nvcc --version | grep release", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if results.returncode == 0 and results.stdout:
        parts = results.stdout.decode("utf-8").split(",")
        cuda_parts = parts[1].strip().split(" ") if len(parts) > 1 else []
        cuda_version = cuda_parts[1] if len(cuda_parts) > 1 else "unknown"
    else:
        cuda_version = "unknown"
    table.add_row(["CUDA version", cuda_version])

    if gpu_name != "NVIDIA Graphics Device":
        results = subprocess.run(
            "lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if results.returncode == 0 and results.stdout:
            parts = results.stdout.decode("utf-8").strip().split("\t")
            ubuntu = parts[1] if len(parts) > 1 else "unknown"
        else:
            ubuntu = "unknown"
        table.add_row(["ubuntu version", ubuntu])
        results = subprocess.run("pip list | grep 'torch '", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if results.returncode == 0 and results.stdout:
            parts = results.stdout.decode("utf-8").strip().split()
            pyt = parts[-1] if parts else "unknown"
        else:
            pyt = "unknown"
        table.add_row(["pytorch", pyt])
    print(table)
    md_table = tools.prettytable_to_markdown(table)
    with open(summary_path, "a") as f:
        f.write(f"## {gpu_name} Benchmarking Guide\n\n")
        f.write(md_table)
        f.write("\n\n")


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


def run_cublas_lt(machine_name: str, ctx: RunContext | None = None) -> None:
    parsed = gemm.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.gemm_cublas_lt_to_csv(ctx, parsed)
        process.process_run("gemm_cublas_lt", ctx, csv_rows)


def run_hbm_bandwidth(machine_name: str, ctx: RunContext | None = None) -> None:
    if "GB200" in machine_name:
        logger.warning("HBM bandwidth Test not supported on GB200 yet")
        return
    parsed = HBM.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.hbm_bandwidth_to_csv(ctx, parsed)
        process.process_run("hbm_bandwidth", ctx, csv_rows)


def run_nv_bandwidth(machine_name: str, ctx: RunContext | None = None) -> None:
    parsed = NV.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.nv_bandwidth_to_csv(ctx, parsed)
        process.process_run("nv_bandwidth", ctx, csv_rows)


def run_nccl_bandwidth(machine_name: str, ctx: RunContext | None = None) -> None:
    parsed = NCCL.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.nccl_bandwidth_to_csv(ctx, parsed)
        process.process_run("nccl_bandwidth", ctx, csv_rows)


def run_flash_attention(machine_name: str, ctx: RunContext | None = None) -> None:
    parsed = FA.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.flash_attention_to_csv(ctx, parsed)
        process.process_run("flash_attention", ctx, csv_rows)


def run_multichase(machine_name: str, ctx: RunContext | None = None) -> None:
    result = Multichase.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and result is not None:
        node_names, rows = result
        csv_rows = process.multichase_to_csv(ctx, node_names, rows)
        process.process_run("multichase", ctx, csv_rows)


def run_cpu_stream(machine_name: str, ctx: RunContext | None = None) -> None:
    parsed = CPU.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.cpu_stream_to_csv(ctx, parsed)
        process.process_run("cpu_stream", ctx, csv_rows)


def run_fio(machine_name: str, ctx: RunContext | None = None) -> None:
    parsed = FIO.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.fio_to_csv(ctx, parsed)
        process.process_run("fio", ctx, csv_rows)


def run_llm_benchmark(machine_name: str, ctx: RunContext | None = None) -> None:
    parsed = llmb.run(work_dir=os.getcwd(), machine_name=machine_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.llm_benchmark_nv_to_csv(ctx, parsed)
        process.process_run("llm_benchmark", ctx, csv_rows)


def run_llama3_pretrain(machine_name: str, model_size: str = "8b", ctx: RunContext | None = None) -> None:
    if "GB200" not in machine_name and "H200" not in machine_name:
        logger.warning("LLAMA3 Pretraining not supported on %s yet", machine_name)
        return
    result = llama3pre.run(work_dir=os.getcwd(), machine_name=machine_name, model_size=model_size, ctx=ctx)
    if ctx is not None and result is not None:
        time_ss, loss_ss = result
        csv_rows = process.llama3_pretrain_to_csv(ctx, time_ss, loss_ss)
        process.process_run("llama3_pretrain", ctx, csv_rows)


BENCHMARKS = {
    "gemm": "CublasLt",
    "nccl": "NCCLBandwidth",
    "hbm": "HBMBandwidth",
    "nv": "NVBandwidth",
    "fa": "FlashAttention",
    "multichase": "Multichase",
    "cpustream": "CPUStream",
    "fio": "FIO",
    "llm": "LLMBenchmark",
    "llama_8b_pretrain": "LLAMA3 8b Pretrain",
    "llama_3b_pretrain": "LLAMA3 3b Pretrain",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="NVIDIA GPU Benchmark Suite")
    parser.add_argument(
        "benchmarks",
        nargs="+",
        choices=[*BENCHMARKS, "all"],
        type=str.lower,
        help="Benchmarks to run",
    )
    args = parser.parse_args()

    current = os.getcwd()
    machine_name = _detect_gpu_name()

    # Structured output pipeline
    version = get_version()
    results_dir = Path(current) / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now()
    session_dir = make_session_dir(results_dir, machine_name, timestamp)

    tools.set_log_path(str(session_dir / "log.txt"))
    tools.set_summary_path(str(session_dir / "summary.md"))

    _write_system_specs(session_dir, machine_name)

    def _ctx(benchmark):
        return _make_ctx(benchmark, machine_name, session_dir, version, timestamp)

    dispatch = {
        "gemm": lambda: run_cublas_lt(machine_name, ctx=_ctx("gemm_cublas_lt")),
        "nccl": lambda: run_nccl_bandwidth(machine_name, ctx=_ctx("nccl_bandwidth")),
        "hbm": lambda: run_hbm_bandwidth(machine_name, ctx=_ctx("hbm_bandwidth")),
        "nv": lambda: run_nv_bandwidth(machine_name, ctx=_ctx("nv_bandwidth")),
        "fa": lambda: run_flash_attention(machine_name, ctx=_ctx("flash_attention")),
        "multichase": lambda: run_multichase(machine_name, ctx=_ctx("multichase")),
        "cpustream": lambda: run_cpu_stream(machine_name, ctx=_ctx("cpu_stream")),
        "fio": lambda: run_fio(machine_name, ctx=_ctx("fio")),
        "llm": lambda: run_llm_benchmark(machine_name, ctx=_ctx("llm_benchmark")),
        "llama_8b_pretrain": lambda: run_llama3_pretrain(machine_name, "8b", ctx=_ctx("llama3_pretrain")),
        "llama_3b_pretrain": lambda: run_llama3_pretrain(machine_name, "3b", ctx=_ctx("llama3_pretrain")),
    }

    selected = list(dispatch.keys()) if "all" in args.benchmarks else args.benchmarks
    failed = []
    for key in selected:
        name = BENCHMARKS[key]
        try:
            dispatch[key]()
        except Exception:
            logger.exception("%s benchmark failed", name)
            failed.append(name)

    if failed:
        logger.error("Failed benchmarks: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
