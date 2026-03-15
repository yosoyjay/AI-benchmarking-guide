import logging
import os
import time
from infra import tools

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/gitaumark/BabelStream"

class HBMBandwidth:
    def __init__(self, path: str, machine: str):
        self.name = "HBMBandwidth"
        self.machine_name = machine
        config = tools.load_benchmark_config(path, self.name)
        self.num_runs, self.interval = self.config_conversion(config)
        self.buffer = []

    def config_conversion(self, config) -> tuple[int, int]:
        return config["inputs"]["num_runs"], config["inputs"]["interval"]

    def build(self):
        current = os.getcwd()
        path = "BabelStream"
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_cmd(
                ["git", "clone", _BABELSTREAM_REPO,  path],
            )

        build_path = os.path.join(current, "BabelStream")
        babelstream_build_path = os.path.join(build_path, "build")

        arch ="sm_90"
        if "A100" in self.machine_name:
            arch = "sm_80"
        if "GB200" in self.machine_name:
            arch = "sm_100"

        if not os.path.isdir(babelstream_build_path):
            os.mkdir(babelstream_build_path)
            results = tools.run_cmd(
                [
                    "cmake",
                    "../",
                    "-DMODEL=cuda",
                    f"-DCUDA_ARCH={arch}",
                    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
                ],
                cwd=babelstream_build_path,
            )

            results = tools.run_cmd(
                ["make"],
                cwd=babelstream_build_path,
            )

        self.build_dir = babelstream_build_path

    def run(self):
        logger.info("Running HBM Bandwidth...")
        runs_executed = 0
        buffer = []
        while runs_executed < self.num_runs:
            results = tools.run_cmd(
                ["./cuda-stream"],
                cwd=self.build_dir,
            )
            log = tools.parse_babelstream_output(results.stdout.decode("utf-8"))
            buffer.append(log)
            runs_executed += 1
            time.sleep(int(self.interval))
        self.buffer = buffer
        self.save_results()

    def save_results(self):
        tools.summarize_babelstream(
            self.buffer,
            divisor=1_000_000,
            units="TB/s",
            title="HBM Bandwidth",
            description="HBM bandwidth Results",
        )
