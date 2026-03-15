import argparse
import logging
import os
import subprocess
import sys

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
from infra import tools

logger = logging.getLogger(__name__)


def get_system_specs(current, host_name):
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


def run_CublasLt(sku_name):
    gemm.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_HBMBandwidth(sku_name):
    if "GB200" in sku_name:
        logger.warning("HBM bandwidth Test not supported on GB200 yet")
        return
    HBM.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_NVBandwidth(sku_name):
    NV.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_NCCLBandwidth(sku_name):
    NCCL.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_FlashAttention(sku_name):
    FA.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_Multichase(sku_name):
    Multichase.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_CPUStream(sku_name):
    CPU.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_FIO(sku_name):
    FIO.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_LLMBenchmark(sku_name):
    llmb.run(work_dir=os.getcwd(), machine_name=sku_name)


def run_LLAMA3Pretrain(sku_name, model_size="8b"):
    if "GB200" in sku_name or "H200" in sku_name:
        test = llama3pre.LLAMA3Pretraining("config.json", sku_name, model_size)
    else:
        logger.warning("LLAMA3 Pretraining not supported on %s yet", sku_name)
        return
    test.run()


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


def main():
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

    dispatch = {
        "gemm": lambda: run_CublasLt(sku_name),
        "nccl": lambda: run_NCCLBandwidth(sku_name),
        "hbm": lambda: run_HBMBandwidth(sku_name),
        "nv": lambda: run_NVBandwidth(sku_name),
        "fa": lambda: run_FlashAttention(sku_name),
        "multichase": lambda: run_Multichase(sku_name),
        "cpustream": lambda: run_CPUStream(sku_name),
        "fio": lambda: run_FIO(sku_name),
        "llm": lambda: run_LLMBenchmark(sku_name),
        "llama_8b_pretrain": lambda: run_LLAMA3Pretrain(sku_name, "8b"),
        "llama_3b_pretrain": lambda: run_LLAMA3Pretrain(sku_name, "3b"),
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
