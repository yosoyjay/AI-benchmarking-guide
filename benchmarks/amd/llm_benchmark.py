"""LLM throughput benchmark (AMD ROCm, Docker-based, vLLM)."""

import logging

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_docker
from infra.containers import AmdContainer

logger = logging.getLogger(__name__)

_VLLM_IMAGE = "rocm/vllm-dev:20241121-tuned"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_vllm_throughput_output(text: str) -> str | None:
    """Extract throughput value from vLLM benchmark output.

    Scans for a line containing ``"Throughput: "`` and extracts the
    tokens/s value (7th token, index 6).  Returns the value string
    or ``None`` if not found.
    """
    for line in text.splitlines():
        if "Throughput: " in line:
            parts = line.split(" ")
            if len(parts) > 6:
                return parts[6]
    return None


def _build_table(rows: list[tuple[str, str, str, str]]) -> PrettyTable:
    """Format (input_len, output_len, tp_size, throughput) tuples into a PrettyTable."""
    table = PrettyTable(["input len", "output len", "tp size", "throughput(tokens/s)"])
    for input_len, output_len, tp_size, throughput in rows:
        table.add_row([input_len, output_len, tp_size, throughput])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    work_dir: str, machine_name: str, config_path: str = "config.json", ctx: RunContext | None = None
) -> list[tuple[str, str, str, str]] | None:
    """Run vLLM throughput benchmarks inside Docker, parse and report."""
    config = tools.load_benchmark_config(config_path, "LLMBenchmark")

    with AmdContainer(
        _VLLM_IMAGE,
        work_dir,
        entrypoint="/bin/bash",
        environment={"HF_HOME": work_dir},
    ) as container:
        all_rows = []
        for model_name, model_cfg in config["models"].items():
            if not model_cfg["use_model"] or model_cfg["type"] != "amd":
                continue

            rows = []
            for tp_size in model_cfg["tp_sizes"]:
                logger.info("Benchmarking %s with TP Size: %s", model_name, tp_size)
                for max_num_seq in model_cfg["max_num_seqs"]:
                    for input_size, output_size in zip(
                        model_cfg["input_length"],
                        model_cfg["output_length"],
                    ):
                        for request in model_cfg["num_requests"]:
                            logger.info(" Input Size: %s, Output Size: %s...", input_size, output_size)
                            cmd = (
                                f"python /app/vllm/benchmarks/benchmark_throughput.py "
                                f"--model amd/{model_name} "
                                f"--quantization fp8 "
                                f"--kv-cache-dtype fp8 "
                                f"--dtype half "
                                f"--gpu-memory-utilization 0.90 "
                                f"--distributed-executor-backend mp "
                                f"--num-scheduler-steps 10 "
                                f"--tensor-parallel-size {tp_size} "
                                f"--enable-chunked-prefill false "
                                f"--max-seq-len-to-capture 131072 "
                                f"--max-num-batched-tokens 131072 "
                                f"--max-model-len 8192 "
                                f"--max-num-seqs {max_num_seq} "
                                f"--num-prompts {request} "
                                f"--input-len {input_size} "
                                f"--output-len {output_size}"
                            )
                            if ctx is not None:
                                name = model_name.replace("/", "_")
                                stdout, stderr, exit_code = capture_docker(
                                    container,
                                    ["/bin/bash", "-c", cmd],
                                    ctx=ctx,
                                    suffix=f"_{name}_{input_size}_{output_size}",
                                )
                                output = stdout
                            else:
                                res = container.exec_run(["/bin/bash", "-c", cmd])
                                output = res.output.decode("utf-8")
                                tools.write_log(output)
                            throughput = parse_vllm_throughput_output(output)
                            result = throughput if throughput is not None else "unknown"
                            rows.append((str(input_size), str(output_size), str(tp_size), str(result)))

            if ctx is not None:
                ctx.extra["model"] = model_name
                all_rows.extend(rows)
            else:
                table = _build_table(rows)
                print(table)
                tools.export_markdown(model_name, "Performance results with FP8 quantization.", table)

    if ctx is not None:
        return all_rows
    return None
