# Azure AI Benchmarking Guide

A CLI tool that runs GPU microbenchmarks and end-to-end LLM workloads on Azure GPU VMs, producing structured results for performance analysis. Use it to identify bottlenecks, validate hardware, and compare configurations across Azure GPU SKUs.

## Supported SKUs

| NVIDIA | AMD |
|---|---|
| ND A100 v4 | ND MI300X v5 |
| ND H100 v5 | |
| ND H200 v5 | |
| ND GB200 v6 | |
| ND GB300 v6 | |

## Quick start

```bash
git clone https://github.com/Azure/AI-benchmarking-guide.git
cd AI-benchmarking-guide
uv venv && source .venv/bin/activate
./install-dependencies.sh
python3 nvidia_runner.py gemm nccl   # run two NVIDIA benchmarks
python3 amd_runner.py hbm rccl       # or two AMD benchmarks
python3 nvidia_runner.py all          # run everything
```

Results are written to `results/<sku>_<timestamp>/` -- see [Output](#output) for the directory layout.

## Prerequisites

- An Azure GPU VM running one of the [supported SKUs](#supported-skus)
- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- `sudo` access (Docker commands require it)
- Docker (required for pretraining workloads and AMD container-based benchmarks)
- `fio` system package for the FIO benchmark (`sudo apt install fio`)

Additional requirements for specific benchmarks:

| Benchmark | Extra requirement |
|---|---|
| LLM inference | HuggingFace account + token; 5 TB disk for model weights |
| LLAMA3 pretraining | NVIDIA NGC credentials to pull the NeMo container |
| AMD container benchmarks | Docker `data-root` on a disk with >= 1 TB free (see [Storage](#storage)) |

## Installation

### Using the install script (recommended)

The script auto-detects your GPU platform, installs the correct dependencies, and handles special cases like GB200/GB300 (core deps only) and AMD ROCm torch.

```bash
uv venv && source .venv/bin/activate
./install-dependencies.sh
```

To use a different pip command:

```bash
./install-dependencies.sh 'python3 -m pip'
```

### Manual installation

```bash
uv venv && source .venv/bin/activate

# NVIDIA (non-GB200/GB300)
uv pip install -e ".[nvidia]"
uv pip install --no-build-isolation flash-attn==2.8.1

# NVIDIA GB200/GB300 (flash-attn not needed)
uv pip install -e .
uv pip install torch prettytable

# AMD (ROCm torch must be installed first)
uv pip install --index-url https://download.pytorch.org/whl/rocm6.2.4 torch torchvision torchaudio
uv pip install -e ".[amd]"
```

### HuggingFace setup (LLM benchmarks only)

Set the model cache directory and authenticate:

```bash
export HF_HOME=$PWD
huggingface-cli login
```

## Usage

### NVIDIA

```bash
python3 nvidia_runner.py <benchmark> [<benchmark> ...]
```

| Argument | Benchmark |
|---|---|
| `all` | Run all benchmarks |
| `gemm` | CuBLASLt GEMM |
| `nccl` | NCCL bandwidth |
| `hbm` | HBM bandwidth |
| `nv` | NV bandwidth (PCIe/NVLink) |
| `fa` | Flash Attention |
| `cpustream` | CPU STREAM |
| `multichase` | Multichase (memory latency) |
| `fio` | FIO (storage I/O) |
| `llm` | LLM inference (TensorRT-LLM) |
| `llama_8b_pretrain` | LLAMA3 8B pretraining (H200, GB200 only) |
| `llama_3b_pretrain` | LLAMA3 3B pretraining (H200, GB200 only) |

Arguments are case-insensitive. Multiple benchmarks can be specified:

```bash
python3 nvidia_runner.py hbm nccl gemm
```

### AMD

```bash
python3 amd_runner.py <benchmark> [<benchmark> ...]
```

| Argument | Benchmark |
|---|---|
| `all` | Run all benchmarks |
| `gemm` | hipBLASLt GEMM |
| `rccl` | RCCL bandwidth |
| `hbm` | HBM bandwidth |
| `transfer` | TransferBench (PCIe bandwidth) |
| `fa` | Flash Attention |
| `fio` | FIO (storage I/O) |
| `llm` | LLM inference (vLLM) |

## Output

All output from a single runner invocation is written to a session directory under `results/`:

```
results/<sku>_<timestamp>/
    log.txt                     # timestamped console output and errors
    summary.md                  # markdown tables with benchmark results
    combined.csv                # long-format CSV of all benchmark metrics
    system_specs.txt            # system info (AMD only)
    LLAMA3_*_Results.png        # training plots (LLAMA3 only)
    <benchmark>/
        raw/                    # raw stdout/stderr from each command
        processed/              # per-benchmark CSV files
```

For example, running `python3 nvidia_runner.py gemm nccl` on an H200 produces:

```
results/NVIDIA_H200_20250315_143000/
    log.txt
    summary.md
    combined.csv
    gemm_cublas_lt/
        raw/
        processed/
    nccl_bandwidth/
        raw/
        processed/
```

## Configuration

Benchmark settings are in [`config.json`](config.json). Key sections:

### GEMM datatype

```json
"GEMMCublasLt": {
    "datatype": "fp8e4m3"
}
```

Supported values: `fp8e4m3`, `fp4e2m1`, `fp16`.

### HBM / CPU STREAM repetitions

```json
"HBMBandwidth": { "inputs": { "num_runs": 5, "interval": 10 } },
"CPUStream":    { "inputs": { "num_runs": 4, "interval": 4 } }
```

### LLM inference models

The `LLMBenchmark.models` section lists available models. Set `"use_model": true` to enable a model. Models are tagged with `"type": "nvidia"` or `"type": "amd"` -- each runner only uses models matching its platform.

```json
"meta-llama/Llama-3.1-8B": {
    "use_model": true,
    "type": "nvidia",
    "input_sizes": [128, 128, 128, 500, 1024, 2048],
    "output_sizes": [128, 1024, 2048, 2000, 1024, 2048],
    "tp_size": 1,
    "num_requests": 1000,
    "precision": "FP8"
}
```

### LLAMA3 pretraining

The `LLAMA3Pretraining` section configures the NeMo container image, training script, and per-SKU model parallelism settings.

## Storage

- Clone this repository onto a disk with at least **5 TB** of free space if you plan to run LLM benchmarks (model weights are large).
- Some AMD benchmarks run inside Docker containers that are automatically created and removed. To prevent Docker from filling your OS disk, point Docker's storage to a large volume:

  Edit `/etc/docker/daemon.json`:

  ```json
  {
      "data-root": "/mnt/resource_nvme/docker"
  }
  ```

  The NVMe mount path varies by VM -- verify with `lsblk` or `df -h`. You may need to complete the Docker [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) afterward.

## Benchmark reference

### NVIDIA benchmarks

**CuBLASLt GEMM** ([source](benchmarks/nvidia/gemm_cublas_lt.py)) -- Measures matrix multiplication throughput using the cuBLAS library. Tests varying matrix sizes (m, n, k) with random initialization to represent realistic workloads. Configurable datatype (FP8, FP4, FP16) via `config.json`.

**NCCL Bandwidth** ([source](benchmarks/nvidia/nccl_bandwidth.py)) -- Measures inter-GPU communication bandwidth using NVIDIA's NCCL library. Tests collective operations (e.g., all-reduce) across all GPUs in a node.
> [Debugging Multi-Node NCCL Performance](https://dev.azure.com/msazure/AzureWiki/_wiki/wikis/AzureWiki.wiki/781566/Debugging-NCCL-Performance-Issues?anchor=recommended-command-lines-for-ndv6-infiniband-skus-(assuming-hpcx-or-openmpi-for-mpi)%3A) (Azure internal only)

**HBM Bandwidth** ([source](benchmarks/nvidia/hbm_bandwidth.py)) -- Measures GPU high-bandwidth memory throughput using BabelStream. Runs multiple iterations and reports steady-state bandwidth.

**NV Bandwidth** ([source](benchmarks/nvidia/nv_bandwidth.py)) -- Measures CPU-to-GPU and GPU-to-CPU bandwidth over PCIe, and GPU-to-GPU bandwidth over NVLink.

**Flash Attention** ([source](benchmarks/nvidia/flash_attention.py)) -- Benchmarks the FlashAttention algorithm, which speeds up attention computation and reduces memory usage from quadratic to linear in sequence length.

**CPU STREAM** ([source](benchmarks/nvidia/cpu_stream.py)) -- Measures CPU-to-RAM memory bandwidth using BabelStream. Evaluates how efficiently the system moves data between CPU and main memory.

**Multichase** ([source](benchmarks/nvidia/multichase.py)) -- Measures pointer-chasing memory latency. Unlike bandwidth benchmarks, this targets random access latency, relevant for workloads with irregular memory access patterns.

**LLM Inference** ([source](benchmarks/nvidia/llm_benchmark.py)) -- Runs end-to-end LLM inference using TensorRT-LLM with Llama 3 models (8B, 70B, 405B). Measures throughput in tokens per second. Requires HuggingFace credentials for model weight download.

**LLAMA3 Pretraining** ([source](benchmarks/nvidia/llama3_run.py)) -- Runs LLAMA3 pretraining (3B, 8B) in a NeMo Docker container. Measures per-step training time. Supported on H200 and GB200 only. Requires NGC credentials.

### AMD benchmarks

**hipBLASLt GEMM** ([source](benchmarks/amd/gemm_hipblas_lt.py)) -- Measures matrix multiplication throughput using the hipBLAS library. Equivalent to the NVIDIA CuBLASLt benchmark for AMD GPUs.

**RCCL Bandwidth** ([source](benchmarks/amd/rccl_bandwidth.py)) -- Measures inter-GPU communication bandwidth using ROCm's RCCL library. Equivalent to the NVIDIA NCCL benchmark.

**HBM Bandwidth** ([source](benchmarks/amd/hbm_bandwidth.py)) -- Measures GPU high-bandwidth memory throughput using BabelStream.

**TransferBench** ([source](benchmarks/amd/transfer_bench.py)) -- Measures CPU-to-GPU and GPU-to-CPU transfer bandwidth.

**Flash Attention** ([source](benchmarks/amd/flash_attention.py)) -- Benchmarks FlashAttention on AMD GPUs using the Triton-based implementation.

**LLM Inference** ([source](benchmarks/amd/llm_benchmark.py)) -- Runs end-to-end LLM inference using vLLM with Llama 3 models. Requires HuggingFace credentials.

## Reference results

Benchmark results for each supported SKU are in the [`Azure_Results/`](Azure_Results/) directory:

- [ND A100 v4](Azure_Results/ND_A100_v4_results.md)
- [ND H100 v5](Azure_Results/ND_H100_v5_results.md)
- [ND H200 v5](Azure_Results/ND_H200_v5_results.md)
- [ND GB200 v6](Azure_Results/ND_GB200_v6_results.md)
- [ND GB300 v6](Azure_Results/ND_GB300_v6_results.md)
- [ND MI300X v5](Azure_Results/ND_MI300X_v5_results.md)

## Development

```bash
uv pip install -e ".[dev]"
pre-commit install
```

Run checks:

```bash
ruff check .
black --check .
uv run --with mypy mypy --ignore-missing-imports <file>
pytest
```
