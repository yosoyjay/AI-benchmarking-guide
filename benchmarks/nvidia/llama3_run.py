import logging
import os
import re
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_cmd

logger = logging.getLogger(__name__)

_DEFAULT_NEMO_IMAGE = "nvcr.io/nvidia/nemo:25.04"


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_llama3_output(text: str) -> tuple[list[int], list[float], list[float]]:
    """Extract global_step, train_loss, train_time triples from log text.

    Returns ``(global_steps, train_losses, train_times)`` lists.
    """
    global_steps, train_losses, train_times = [], [], []
    for line in text.splitlines():
        match = re.search(
            r"global_step: (\d+) .* reduced_train_loss: ([\d.]+) .* train_step_timing in s: ([\d.]+)",
            line,
        )
        if match:
            global_steps.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            train_times.append(float(match.group(3)))
    return global_steps, train_losses, train_times


def compute_steady_state(
    arr: list[float] | npt.NDArray[np.floating[Any]],
    window_size: int = 10,
    std_thresh: float = 0.1,
    min_windows: int = 3,
) -> tuple[int | None, float | None]:
    """Detect steady-state in a numeric array.

    Returns ``(start_idx, steady_value)`` or ``(None, None)`` if not found.
    """
    consistent = 0
    start_idx = None
    for i in range(len(arr) - window_size + 1):
        window = arr[i : i + window_size]
        if np.std(window) < std_thresh:
            consistent += 1
            if consistent >= min_windows:
                start_idx = i - (min_windows - 1)
                break
        else:
            consistent = 0
    if start_idx is None:
        return None, None
    steady: float = float(np.mean(arr[start_idx:]))
    return start_idx, steady


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _plot_results(
    text: str,
    model_size: str,
    machine_name: str,
) -> tuple[float | None, float | None]:
    """Parse training output, generate loss/time plots, return steady-state values."""
    global_steps, train_losses, train_times = parse_llama3_output(text)

    time_idx, time_ss = compute_steady_state(train_times, std_thresh=1)
    loss_idx, loss_ss = compute_steady_state(train_losses, std_thresh=0.1)

    if time_ss is not None and time_idx is not None:
        tools.write_log(f"Time steady-state: {time_ss:.4f}s starting at step {global_steps[time_idx]}")
    else:
        tools.write_log("No steady-state found for time.")

    if loss_ss is not None and loss_idx is not None:
        tools.write_log(f"Loss steady-state: {loss_ss:.4f} starting at step {global_steps[loss_idx]}")
    else:
        tools.write_log("No steady-state found for loss.")

    # create grid for both loss and time plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.tight_layout(pad=4.0)

    # plot loss
    ax1.plot(global_steps, train_losses, marker="o", label="Training Loss")
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training Loss per Global Step for LLAMA3Pretraining on {machine_name}")
    ax1.grid(True)
    ax1.legend()

    # plot time
    ax2.plot(global_steps, train_times, marker="o", color="orange", label="Training Time (s)")
    ax2.set_xlabel("Global Step")
    ax2.set_ylabel("Time (s)")
    ax2.set_title(f"Training Time per Global Step for LLAMA3Pretraining on {machine_name}")
    ax2.grid(True)
    ax2.legend()

    # add text annotation for the steady state value
    annot = []
    if loss_ss is not None:
        annot.append(f"The loss steady state is {loss_ss:.2f}")
    if time_ss is not None:
        annot.append(f"The time steady state is {time_ss:.2f}s")
    fig.text(0.5, 0.01, ";  ".join(annot), ha="center", fontsize=12, style="italic")

    # save to outputs folder
    plot_path = os.path.join("Outputs", f"LLAMA3_{model_size}_Pretrain_Results")
    plt.savefig(plot_path, dpi=300)
    logger.info("Training loss and time plot with steady state saved to %s", plot_path)
    tools.write_log(f"Training loss and time plot with steady state saved to {plot_path}")
    plt.close()

    return time_ss, loss_ss


def run(
    work_dir: str,
    machine_name: str,
    model_size: str = "8b",
    config_path: str = "config.json",
    ctx: RunContext | None = None,
) -> tuple[float | None, float | None] | None:
    """Run LLAMA3 pretraining benchmark.

    Returns ``(time_ss, loss_ss)`` when *ctx* is provided, else ``None``.
    """
    config = tools.load_benchmark_config(config_path, "LLAMA3Pretraining")
    mount_path = config.get("mount_path", ".")
    training_script = config.get("training_script", "benchmarks/nvidia/llama3_recipe.py")
    docker_image = config.get("docker_image", _DEFAULT_NEMO_IMAGE)

    log_path = os.path.join("Outputs", "llama3_docker_output.txt")
    tools.write_log(f"Pulling and launching NeMo container for {machine_name}.")
    logger.info("Pulling and launching NeMo docker container for %s and logging at 'Outputs/log.txt'.", machine_name)

    if model_size == "3b":
        time = 2
    else:
        time = 4

    tools.write_log(f"Pretraining will finish in {time} hours.")
    logger.info("Pretraining will finish in %d hours.", time)

    command = [
        "sudo",
        "docker",
        "run",
        "--rm",
        "-i",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-v",
        f"{mount_path}:/workspace/nemo-run",
        docker_image,
        "bash",
        "-c",
        f"cd /workspace/nemo-run && python {training_script} --model_size {model_size} --machine_name {(machine_name.split() or [machine_name])[-1]}",
    ]

    if ctx is not None:
        result = capture_cmd(command, ctx=ctx)
        stdout_text = result.stdout.decode("utf-8") if result.stdout else ""

        if result.returncode != 0:
            logger.warning("Docker pretraining command failed with exit code %d", result.returncode)
            return None

        _, train_losses, train_times = parse_llama3_output(stdout_text)
        _, time_ss = compute_steady_state(train_times, std_thresh=1)
        _, loss_ss = compute_steady_state(train_losses, std_thresh=0.1)
        ctx.extra["model_size"] = model_size
        return time_ss, loss_ss

    # Legacy path
    with open(os.path.join("Outputs", "llama3_docker_output.txt"), "w") as file:
        proc = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        logger.warning("Docker pretraining command failed with exit code %d, skipping plot", proc.returncode)
        tools.write_log(f"LLAMA3 pretraining failed with exit code {proc.returncode}")
        return None

    # now plot the results
    logger.info("Pretraining has finished with output saved to: %s. Now plotting.", log_path)
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()
    time_ss, loss_ss = _plot_results(text, model_size, machine_name)

    # add summary to markdown
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Model Size", model_size])
    table.add_row(["Pretrain Time Steady State", time_ss if time_ss is not None else "None"])

    tools.export_markdown(
        f"LLAMA3 {model_size} Pretraining Summary",
        f"Pretraining results for LLAMA3 {model_size} on {machine_name}.",
        table,
    )
    return None
