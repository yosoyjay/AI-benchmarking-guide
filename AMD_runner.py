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

current = os.getcwd()
tools.create_dir("Outputs")

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
    print("Warning: could not detect AMD GPU SKU, falling back to ND_MI300X_v5")
    return "ND_MI300X_v5"

def get_system_specs():
    with open("Outputs/system_specs.txt", "w") as file:

        results = subprocess.run("rocminfo | grep 'ROCk module version'", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            parts = results.stdout.decode('utf-8').strip().split(" ")
            rocm_version = parts[3] if len(parts) >= 4 else "unknown"
        else:
            rocm_version = "unknown"
        file.write("ROCm version     : "+rocm_version+"\n")

        results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            parts = results.stdout.decode('utf-8').strip().split("\t")
            ubuntu = parts[1] if len(parts) >= 2 else "unknown"
        else:
            ubuntu = "unknown"
        file.write("ubuntu version   : "+ubuntu+"\n")

        results = subprocess.run("grep 'stepping\\|model\\|microcode' /proc/cpuinfo | grep microcode", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            lines = results.stdout.decode('utf-8').split("\n")
            microcode = lines[0] if lines else ""
        else:
            microcode = ""
        file.write(microcode+"\n")

        results = subprocess.run("grep 'stepping\\|model\\|microcode' /proc/cpuinfo | grep name", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            lines = results.stdout.decode('utf-8').split("\n")
            file.write((lines[0] if lines else "")+"\n")
        else:
            file.write("\n")

        results = subprocess.run("grep 'cores\\|model\\|microcode' /proc/cpuinfo | grep cores", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if results.returncode == 0:
            lines = results.stdout.decode('utf-8').split("\n")
            file.write(lines[0] if lines else "")
        else:
            file.write("")
    return _detect_sku()

def run_TransferBench():
    test = TB.TransferBench("config.json", current, machine_name)
    test.build()
    test.run()

def run_GEMMHipBLAS():
    test = GEMM.GEMMHipBLAS("config.json", current, machine_name)
    test.create_container()
    test.build()
    test.run_model_sizes()

def run_RCCLBandwidth():
    test = RCCL.RCCLBandwidth("config.json", current, machine_name)
    test.create_container()
    test.build()
    test.run()

def run_FlashAttention():
    test = FA.FlashAttention(current, machine_name)
    test.run()
    os.chdir(current)

def run_FIO():
    test = FIO.FIO(current, machine_name)
    test.run()

def run_HBMBandwidth():
    test = HBM.HBMBandwidth("config.json", current, machine_name)
    test.build()
    test.run()

def run_LLMBenchmark():
    test = llmb.LLMBenchmark("config.json", current, machine_name)
    test.create_container()
    test.run_benchmark()

machine_name = get_system_specs()
arguments = []
match = False
for arg in sys.argv[1:]:
    arguments.append(arg.lower())

if ("gemm" in arguments):
    match = True
    try:
        run_GEMMHipBLAS()
    except Exception as e:
        print(f"Warning: GEMMHipBLAS benchmark failed: {e}")
    os.chdir(current)

if ("rccl" in arguments):
    match = True
    try:
        run_RCCLBandwidth()
    except Exception as e:
        print(f"Warning: RCCLBandwidth benchmark failed: {e}")
    os.chdir(current)

if ("hbm" in arguments):
    match = True
    try:
        run_HBMBandwidth()
    except Exception as e:
        print(f"Warning: HBMBandwidth benchmark failed: {e}")
    os.chdir(current)

if ("transfer" in arguments):
    match = True
    try:
        run_TransferBench()
    except Exception as e:
        print(f"Warning: TransferBench benchmark failed: {e}")
    os.chdir(current)

if ("fa" in arguments):
    match = True
    try:
        run_FlashAttention()
    except Exception as e:
        print(f"Warning: FlashAttention benchmark failed: {e}")
    os.chdir(current)

if ("fio" in arguments):
    match = True
    try:
        run_FIO()
    except Exception as e:
        print(f"Warning: FIO benchmark failed: {e}")
    os.chdir(current)

if ("llm" in arguments):
    match = True
    try:
        run_LLMBenchmark()
    except Exception as e:
        print(f"Warning: LLMBenchmark failed: {e}")
    os.chdir(current)

if ("all" in arguments):
    match = True
    for _name, _fn in [
        ("HBMBandwidth", run_HBMBandwidth),
        ("TransferBench", run_TransferBench),
        ("RCCLBandwidth", run_RCCLBandwidth),
        ("FIO", run_FIO),
        ("FlashAttention", run_FlashAttention),
        ("LLMBenchmark", run_LLMBenchmark),
        ("GEMMHipBLAS", run_GEMMHipBLAS),
    ]:
        try:
            _fn()
        except Exception as e:
            print(f"Warning: {_name} benchmark failed: {e}")
        os.chdir(current)
if not match:
    print("Usage: python3 AMD_runner.py [arg]\n   or: python3 AMD_runner.py [arg1] [arg2] ... to run more than one test e.g python3 AMD_runner.py hbm nccl\nArguments are as follows, and are case insensitive:\nAll tests:  all\nROCBLAS GEMM:  gemm\nRCCL Bandwidth: rccl\nHBMBandwidth:   hbm\nTransferbench:   transfer\nFlash Attention: fa\nFIO Tests:   fio\nLLM Inference Workloads: llm")
    
