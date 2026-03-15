import logging
import os
import re
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from Infra import tools
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

_DEFAULT_NEMO_IMAGE = "nvcr.io/nvidia/nemo:25.04"

class LLAMA3Pretraining:
    def __init__(self, config_path: str, machine_name: str, model_size: str = "8b"):
        self.name = "LLAMA3Pretraining"
        self.machine_name = machine_name
        self.config = tools.load_benchmark_config(config_path, self.name)
        self.mount_path = self.config.get("mount_path", ".")
        self.training_script = self.config.get("training_script", "Training/LLAMA3Recipe.py")
        self.container = self.config.get("docker_image", _DEFAULT_NEMO_IMAGE)
        self.model_size = model_size

    def plot_results(self, file_path: str = None):
        # extract values from the output file
        global_steps, train_losses, train_times = [], [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.search(r"global_step: (\d+) .* reduced_train_loss: ([\d.]+) .* train_step_timing in s: ([\d.]+)", line)
                if match:
                    global_steps.append(int(match.group(1)))
                    train_losses.append(float(match.group(2)))
                    train_times.append(float(match.group(3)))

        # calculate steady state time value
        def detect_steady(arr, window_size=10, std_thresh=0.1, min_windows=3):
            consistent = 0
            start_idx = None
            for i in range(len(arr) - window_size + 1):
                window = arr[i:i + window_size]
                if np.std(window) < std_thresh:
                    consistent += 1
                    if consistent >= min_windows:
                        start_idx = i - (min_windows - 1)
                        break
                else:
                    consistent = 0
            if start_idx is None:
                return None, None
            steady = np.mean(arr[start_idx:])
            return start_idx, steady

        time_idx, time_ss = detect_steady(train_times, std_thresh=1)
        loss_idx, loss_ss = detect_steady(train_losses, std_thresh=0.1)

        if time_ss is not None:
            tools.write_log(f"Time steady-state: {time_ss:.4f}s starting at step {global_steps[time_idx]}")
        else:
            tools.write_log("No steady-state found for time.")

        if loss_ss is not None:
            tools.write_log(f"Loss steady-state: {loss_ss:.4f} starting at step {global_steps[loss_idx]}")
        else:
            tools.write_log("No steady-state found for loss.")

        # create grid for both loss and time plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        fig.tight_layout(pad=4.0)

        # plot loss
        ax1.plot(global_steps, train_losses, marker='o', label="Training Loss")
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Training Loss per Global Step for {self.name} on {self.machine_name}")
        ax1.grid(True)
        ax1.legend()

        # plot time
        ax2.plot(global_steps, train_times, marker='o', color='orange', label="Training Time (s)")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Time (s)")
        ax2.set_title(f"Training Time per Global Step for {self.name} on {self.machine_name}")
        ax2.grid(True)
        ax2.legend()

        # add text annotation for the steady state value
        annot = []
        if loss_ss is not None: annot.append(f"The loss steady state is {loss_ss:.2f}")
        if time_ss is not None: annot.append(f"The time steady state is {time_ss:.2f}s")
        fig.text(0.5, 0.01, ";  ".join(annot), ha='center', fontsize=12, style='italic')

        # save to outputs folder
        plot_path = os.path.join("Outputs", f"LLAMA3_{self.model_size}_Pretrain_Results")
        plt.savefig(plot_path, dpi=300)
        logger.info("Training loss and time plot with steady state saved to %s", plot_path)
        tools.write_log(f"Training loss and time plot with steady state saved to {plot_path}") # print to log.txt
        plt.close()

        return time_ss


    def run(self):
        log_path = os.path.join("Outputs", "llama3_docker_output.txt")
        tools.write_log(f"Pulling and launching NeMo container for {self.machine_name}.") # write to log file
        logger.info("Pulling and launching NeMo docker container for %s and logging at 'Outputs/log.txt'.", self.machine_name)

        if self.model_size == "3b":
            time = 2
        else:
            time = 4

        tools.write_log(f"Pretraining will finish in {time} hours.")
        logger.info("Pretraining will finish in %d hours.", time)

        command = [
            "sudo", "docker", "run", "--rm", "-i",
            "--gpus", "all",
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "-v", f"{self.mount_path}:/workspace/nemo-run",
            self.container,
            "bash", "-c", f"cd /workspace/nemo-run && python {self.training_script} --model_size {self.model_size} --machine_name {(self.machine_name.split() or [self.machine_name])[-1]}"
        ]

        # launch command and write to log file (this shows all info about epoch, training time, etc.)
        with open(os.path.join("Outputs", "llama3_docker_output.txt"), "w") as file:
            result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, text=True)

        if result.returncode != 0:
            logger.warning("Docker pretraining command failed with exit code %d, skipping plot", result.returncode)
            tools.write_log(f"LLAMA3 pretraining failed with exit code {result.returncode}")
            return

        # now plot the results
        logger.info("Pretraining has finished with output saved to: %s. Now plotting.", log_path)
        time_ss = self.plot_results(log_path)

        # add summary to markdown
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["Model Size", self.model_size])
        table.add_row(["Pretrain Time Steady State", time_ss if time_ss is not None else "None"])

        # Export the table to markdown
        tools.export_markdown(f"LLAMA3 {self.model_size} Pretraining Summary", f"Pretraining results for LLAMA3 {self.model_size} on {self.machine_name}.", table)
