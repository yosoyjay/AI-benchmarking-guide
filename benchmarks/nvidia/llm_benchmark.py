"""LLM throughput benchmark (NVIDIA, TensorRT-LLM)."""

import logging
import os

from huggingface_hub import snapshot_download
from prettytable import PrettyTable

from infra import tools

logger = logging.getLogger(__name__)

_TENSORRT_LLM_REPO = "https://github.com/NVIDIA/TensorRT-LLM.git"
_TENSORRT_LLM_VERSION = "v0.18.2"
_TENSORRT_LLM_PIP_VERSION = "0.18.2"
_NVIDIA_PYPI_URL = "https://pypi.nvidia.com"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_trtllm_bench_output(text):
    """Extract benchmark metrics from trtllm-bench output text.

    Scans for lines containing TP Size, Average Input/Output Length,
    and Token Throughput.  Returns a dict with keys ``tp_size``,
    ``input_len``, ``output_len``, ``throughput`` (all strings) or
    ``None`` if the expected four values are not found.
    """
    keywords = [
        "TP Size:",
        "Average Input Length (tokens):",
        "Average Output Length (tokens):",
        "Token Throughput (tokens/sec):",
    ]
    values = []
    for line in text.splitlines():
        for keyword in keywords:
            if keyword in line:
                parts = line.split(":")
                if len(parts) < 2:
                    continue
                try:
                    values.append(str(int(float(parts[1].strip()))))
                except (ValueError, IndexError):
                    logger.warning("could not parse value from line: %s", line.strip())
                break

    if len(values) != 4:
        return None
    return {
        "tp_size": values[0],
        "input_len": values[1],
        "output_len": values[2],
        "throughput": values[3],
    }


def _build_table(rows):
    """Format parsed row dicts into a PrettyTable."""
    table = PrettyTable(["tp size", "input len", "output len", "throughput(tokens/s)"])
    for r in rows:
        table.add_row([r["tp_size"], r["input_len"], r["output_len"], r["throughput"]])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _make_env(work_dir):
    """Build environment dict with HF_HOME set."""
    return {**os.environ, "HF_HOME": work_dir}


def _install_requirements(work_dir, env):
    """Clone TensorRT-LLM and install packages if not in Docker."""
    trt_dir = os.path.join(work_dir, "TensorRT-LLM")
    if not os.path.isdir(trt_dir):
        logger.info("Cloning TensorRT-LLM repository from %s", _TENSORRT_LLM_REPO)
        tools.run_cmd(
            ["git", "clone", _TENSORRT_LLM_REPO],
            cwd=work_dir,
            env=env,
        )
        tools.run_cmd(
            ["git", "checkout", _TENSORRT_LLM_VERSION],
            cwd=trt_dir,
            env=env,
        )

        if not os.path.exists("/.dockerenv"):
            logger.info("No Docker container detected. Installing tensorrt-llm")
            tools.run_cmd(
                ["pip", "install", f"tensorrt-llm=={_TENSORRT_LLM_PIP_VERSION}"],
                env=env,
            )
            tools.run_cmd(
                ["sudo", "apt", "update"],
                env=env,
            )
            tools.run_cmd(
                ["sudo", "apt-get", "-y", "install", "libopenmpi-dev"],
                env=env,
            )
            tools.run_cmd(
                ["pip3", "install", "--no-cache-dir", "--extra-index-url", _NVIDIA_PYPI_URL, "tensorrt-libs"],
                env=env,
            )


def _download_models(config, work_dir):
    """Download HuggingFace models used by nvidia-typed benchmarks."""
    for model_name, model_cfg in config["models"].items():
        if model_cfg["use_model"] and model_cfg["type"] == "nvidia":
            snapshot_download(repo_id=model_name, cache_dir=os.path.join(work_dir, "hub"))


def _prepare_datasets(config, work_dir, env):
    """Generate synthetic datasets and build TRT-LLM engines."""
    for model_name, model_cfg in config["models"].items():
        if not model_cfg["use_model"] or model_cfg["type"] != "nvidia":
            continue

        max_sum = 0
        max_dataset_path = ""
        name = model_name.split("/")[1]

        for isl, osl in zip(model_cfg["input_sizes"], model_cfg["output_sizes"]):
            if isl + osl > max_sum:
                max_sum = isl + osl
                max_dataset_path = os.path.join(work_dir, "datasets", f"{name}_synthetic_{isl}_{osl}.txt")

            dataset_path = os.path.join(work_dir, "datasets", f"{name}_synthetic_{isl}_{osl}.txt")
            if not os.path.exists(dataset_path):
                prepare_script = os.path.join(work_dir, "TensorRT-LLM", "benchmarks", "cpp", "prepare_dataset.py")
                result = tools.run_cmd(
                    [
                        "python3",
                        prepare_script,
                        "--stdout",
                        "--tokenizer",
                        model_name,
                        "token-norm-dist",
                        "--num-requests",
                        str(model_cfg["num_requests"]),
                        "--input-mean",
                        str(isl),
                        "--output-mean",
                        str(osl),
                        "--input-stdev=0",
                        "--output-stdev=0",
                    ],
                    env=env,
                )
                with open(dataset_path, "w") as f:
                    f.write(result.stdout.decode("utf-8"))

        engines_dir = os.path.join(work_dir, "engines", model_name)
        if not os.path.isdir(engines_dir):
            logger.info("Building engine for %s", model_name)
            tools.run_cmd(
                [
                    "trtllm-bench",
                    "--workspace",
                    os.path.join(work_dir, "engines"),
                    "--model",
                    model_name,
                    "build",
                    "--tp_size",
                    str(model_cfg["tp_size"]),
                    "--dataset",
                    max_dataset_path,
                    "--quantization",
                    model_cfg["precision"],
                ],
                env=env,
            )


def _run_benchmarks(config, work_dir, env):
    """Run trtllm-bench throughput for each model/size combo."""
    for model_name, model_cfg in config["models"].items():
        if not model_cfg["use_model"] or model_cfg["type"] != "nvidia":
            continue

        tp = model_cfg["tp_size"]
        name = model_name.split("/")[1]
        logger.info("Benchmarking %s with tp size %s", model_name, tp)

        rows = []
        for isl, osl in zip(model_cfg["input_sizes"], model_cfg["output_sizes"]):
            logger.info("input/output: %s/%s...", isl, osl)
            dataset_path = os.path.join(work_dir, "datasets", f"{name}_synthetic_{isl}_{osl}.txt")
            results_path = os.path.join(work_dir, "Outputs", f"results_{name}_{isl}_{osl}.txt")

            result = tools.run_cmd(
                [
                    "trtllm-bench",
                    "--model",
                    model_name,
                    "throughput",
                    "--dataset",
                    dataset_path,
                    "--engine_dir",
                    os.path.join(work_dir, "engines", model_name, f"tp_{tp}_pp_1"),
                ],
                env=env,
            )
            with open(results_path, "w", encoding="utf-8") as f:
                f.write(result.stdout.decode("utf-8"))

            try:
                with open(results_path, encoding="utf-8") as f:
                    text = f.read()
                parsed = parse_trtllm_bench_output(text)
                if parsed:
                    rows.append(parsed)
                else:
                    logger.warning("expected 4 values from %s, skipping", results_path)
            except FileNotFoundError:
                logger.warning("benchmark output file not found: %s", results_path)

        table = _build_table(rows)
        print(table)
        tools.export_markdown(model_name, "Performance results with FP8 quantization, 1000 requests.", table)


def run(work_dir, machine_name, config_path="config.json"):
    """Install deps, download models, prepare datasets, run benchmarks."""
    config = tools.load_benchmark_config(config_path, "LLMBenchmark")
    env = _make_env(work_dir)

    tools.create_dir(os.path.join(work_dir, "datasets"))
    tools.create_dir(os.path.join(work_dir, "engines"))
    tools.create_dir(os.path.join(work_dir, "hub"))

    _install_requirements(work_dir, env)
    _download_models(config, work_dir)
    _prepare_datasets(config, work_dir, env)
    _run_benchmarks(config, work_dir, env)
