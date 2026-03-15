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
from infra.capture import RunContext, get_version, make_run_dir

logger = logging.getLogger(__name__)

_PLATFORM = "nvidia"


def get_system_specs(current: str, host_name: str) -> str:
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
    if not os.path.exists(os.path.join(current, "Outputs", f"{host_name}_summary.md")):
        table = PrettyTable([" ", output[0]])
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

        if output[0].strip() != "NVIDIA Graphics Device":
            results = subprocess.run(
                "lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if results.returncode == 0 and results.stdout:
                parts = results.stdout.decode("utf-8").strip().split("\t")
                ubuntu = parts[1] if len(parts) > 1 else "unknown"
            else:
                ubuntu = "unknown"
            table.add_row(["ubuntu version", ubuntu])
            results = subprocess.run(
                "pip list | grep 'torch '", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if results.returncode == 0 and results.stdout:
                parts = results.stdout.decode("utf-8").strip().split()
                pyt = parts[-1] if parts else "unknown"
            else:
                pyt = "unknown"
            table.add_row(["pytorch", pyt])
        print(table)
        tools.export_markdown(f"{output[0].strip()} Benchmarking Guide", "", table)
    return output[0].strip()


def _make_ctx(benchmark: str, sku: str, results_dir: Path, version: str, timestamp: datetime) -> RunContext:
    """Create a RunContext for a single benchmark."""
    run_dir = make_run_dir(results_dir, benchmark, sku, timestamp)
    return RunContext(
        benchmark=benchmark,
        sku=sku,
        platform=_PLATFORM,
        version=version,
        timestamp=timestamp,
        results_dir=results_dir,
        run_dir=run_dir,
    )


def run_CublasLt(sku_name: str, ctx: RunContext | None = None) -> None:
    parsed = gemm.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.gemm_cublas_lt_to_csv(ctx, parsed)
        process.process_run("gemm_cublas_lt", ctx, csv_rows)


def run_HBMBandwidth(sku_name: str, ctx: RunContext | None = None) -> None:
    if "GB200" in sku_name:
        logger.warning("HBM bandwidth Test not supported on GB200 yet")
        return
    parsed = HBM.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.hbm_bandwidth_to_csv(ctx, parsed)
        process.process_run("hbm_bandwidth", ctx, csv_rows)


def run_NVBandwidth(sku_name: str, ctx: RunContext | None = None) -> None:
    parsed = NV.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.nv_bandwidth_to_csv(ctx, parsed)
        process.process_run("nv_bandwidth", ctx, csv_rows)


def run_NCCLBandwidth(sku_name: str, ctx: RunContext | None = None) -> None:
    parsed = NCCL.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.nccl_bandwidth_to_csv(ctx, parsed)
        process.process_run("nccl_bandwidth", ctx, csv_rows)


def run_FlashAttention(sku_name: str, ctx: RunContext | None = None) -> None:
    parsed = FA.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.flash_attention_to_csv(ctx, parsed)
        process.process_run("flash_attention", ctx, csv_rows)


def run_Multichase(sku_name: str, ctx: RunContext | None = None) -> None:
    result = Multichase.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and result is not None:
        node_names, rows = result
        csv_rows = process.multichase_to_csv(ctx, node_names, rows)
        process.process_run("multichase", ctx, csv_rows)


def run_CPUStream(sku_name: str, ctx: RunContext | None = None) -> None:
    parsed = CPU.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.cpu_stream_to_csv(ctx, parsed)
        process.process_run("cpu_stream", ctx, csv_rows)


def run_FIO(sku_name: str, ctx: RunContext | None = None) -> None:
    parsed = FIO.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.fio_to_csv(ctx, parsed)
        process.process_run("fio", ctx, csv_rows)


def run_LLMBenchmark(sku_name: str, ctx: RunContext | None = None) -> None:
    parsed = llmb.run(work_dir=os.getcwd(), machine_name=sku_name, ctx=ctx)
    if ctx is not None and parsed is not None:
        csv_rows = process.llm_benchmark_nv_to_csv(ctx, parsed)
        process.process_run("llm_benchmark", ctx, csv_rows)


def run_LLAMA3Pretrain(sku_name: str, model_size: str = "8b", ctx: RunContext | None = None) -> None:
    if "GB200" not in sku_name and "H200" not in sku_name:
        logger.warning("LLAMA3 Pretraining not supported on %s yet", sku_name)
        return
    result = llama3pre.run(work_dir=os.getcwd(), machine_name=sku_name, model_size=model_size, ctx=ctx)
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
    host_name = tools.get_hostname()
    tools.create_dir("Outputs")
    sku_name = get_system_specs(current, host_name)

    # Structured output pipeline
    version = get_version()
    results_dir = Path(current) / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now()

    def _ctx(benchmark):
        return _make_ctx(benchmark, sku_name, results_dir, version, timestamp)

    dispatch = {
        "gemm": lambda: run_CublasLt(sku_name, ctx=_ctx("gemm_cublas_lt")),
        "nccl": lambda: run_NCCLBandwidth(sku_name, ctx=_ctx("nccl_bandwidth")),
        "hbm": lambda: run_HBMBandwidth(sku_name, ctx=_ctx("hbm_bandwidth")),
        "nv": lambda: run_NVBandwidth(sku_name, ctx=_ctx("nv_bandwidth")),
        "fa": lambda: run_FlashAttention(sku_name, ctx=_ctx("flash_attention")),
        "multichase": lambda: run_Multichase(sku_name, ctx=_ctx("multichase")),
        "cpustream": lambda: run_CPUStream(sku_name, ctx=_ctx("cpu_stream")),
        "fio": lambda: run_FIO(sku_name, ctx=_ctx("fio")),
        "llm": lambda: run_LLMBenchmark(sku_name, ctx=_ctx("llm_benchmark")),
        "llama_8b_pretrain": lambda: run_LLAMA3Pretrain(sku_name, "8b", ctx=_ctx("llama3_pretrain")),
        "llama_3b_pretrain": lambda: run_LLAMA3Pretrain(sku_name, "3b", ctx=_ctx("llama3_pretrain")),
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
