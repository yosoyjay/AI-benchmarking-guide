import argparse
import logging
import os
import sys
import subprocess
from Benchmarks.AMD import RCCLBandwidth as RCCL
from Benchmarks.AMD import FlashAttention as FA
from Benchmarks.AMD import HBMBandwidth as HBM
from Benchmarks.AMD import TransferBench as TB
from Benchmarks.AMD import GEMMHipblasLt as GEMM
from Benchmarks.AMD import FIO
from Infra import tools
from Benchmarks.AMD import LLMBenchmark as llmb

logger = logging.getLogger(__name__)

_SKU_MAP = {
    "MI300X": "ND_MI300X_v5",
    "MI300": "ND_MI300X_v5",
    "MI250X": "ND_MI250X_v4",
    "MI250": "ND_MI250_v4",
}


def _detect_sku():
    try:
        results = subprocess.run(
            "rocminfo | grep 'Marketing Name'",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


def get_system_specs():
    with open(os.path.join("Outputs", "system_specs.txt"), "w") as file:

        results = subprocess.run("rocminfo | grep 'ROCk module version'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            parts = results.stdout.decode('utf-8').strip().split(" ")
            rocm_version = parts[3] if len(parts) >= 4 else "unknown"
        else:
            rocm_version = "unknown"
        file.write(f"ROCm version     : {rocm_version}\n")

        results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            parts = results.stdout.decode('utf-8').strip().split("\t")
            ubuntu = parts[1] if len(parts) >= 2 else "unknown"
        else:
            ubuntu = "unknown"
        file.write(f"ubuntu version   : {ubuntu}\n")

        results = subprocess.run("grep 'stepping\\|model\\|microcode' /proc/cpuinfo | grep microcode", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            lines = results.stdout.decode('utf-8').split("\n")
            microcode = lines[0] if lines else ""
        else:
            microcode = ""
        file.write(f"{microcode}\n")

        results = subprocess.run("grep 'stepping\\|model\\|microcode' /proc/cpuinfo | grep name", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            lines = results.stdout.decode('utf-8').split("\n")
            file.write(f"{lines[0] if lines else ''}\n")
        else:
            file.write("\n")

        results = subprocess.run("grep 'cores\\|model\\|microcode' /proc/cpuinfo | grep cores", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            lines = results.stdout.decode('utf-8').split("\n")
            file.write(lines[0] if lines else "")
        else:
            file.write("")
    return _detect_sku()


def run_TransferBench(machine_name, current):
    test = TB.TransferBench("config.json", current, machine_name)
    test.build()
    test.run()

def run_GEMMHipBLAS(machine_name, current):
    test = GEMM.GEMMHipBLAS("config.json", current, machine_name)
    test.create_container()
    test.build()
    test.run_model_sizes()

def run_RCCLBandwidth(machine_name, current):
    test = RCCL.RCCLBandwidth("config.json", current, machine_name)
    test.create_container()
    test.build()
    test.run()

def run_FlashAttention(machine_name, current):
    test = FA.FlashAttention(current, machine_name)
    test.run()
    os.chdir(current)

def run_FIO(machine_name, current):
    test = FIO.FIO(current, machine_name)
    test.run()

def run_HBMBandwidth(machine_name, current):
    test = HBM.HBMBandwidth("config.json", current, machine_name)
    test.build()
    test.run()

def run_LLMBenchmark(machine_name, current):
    test = llmb.LLMBenchmark("config.json", current, machine_name)
    test.create_container()
    test.run_benchmark()


BENCHMARKS = {
    "gemm": "GEMMHipBLAS",
    "rccl": "RCCLBandwidth",
    "hbm": "HBMBandwidth",
    "transfer": "TransferBench",
    "fa": "FlashAttention",
    "fio": "FIO",
    "llm": "LLMBenchmark",
}


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="AMD GPU Benchmark Suite")
    parser.add_argument(
        "benchmarks", nargs="+",
        choices=[*BENCHMARKS, "all"],
        type=str.lower,
        help="Benchmarks to run",
    )
    args = parser.parse_args()

    current = os.getcwd()
    tools.create_dir("Outputs")
    machine_name = get_system_specs()

    dispatch = {
        "gemm": lambda: run_GEMMHipBLAS(machine_name, current),
        "rccl": lambda: run_RCCLBandwidth(machine_name, current),
        "hbm": lambda: run_HBMBandwidth(machine_name, current),
        "transfer": lambda: run_TransferBench(machine_name, current),
        "fa": lambda: run_FlashAttention(machine_name, current),
        "fio": lambda: run_FIO(machine_name, current),
        "llm": lambda: run_LLMBenchmark(machine_name, current),
    }

    selected = list(dispatch.keys()) if "all" in args.benchmarks else args.benchmarks
    for key in selected:
        name = BENCHMARKS[key]
        try:
            dispatch[key]()
        except Exception as e:
            logger.warning("%s benchmark failed: %s", name, e)
        os.chdir(current)


if __name__ == "__main__":
    main()
