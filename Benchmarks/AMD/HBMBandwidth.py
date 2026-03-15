import logging
import os
import time
from Infra import tools

logger = logging.getLogger(__name__)

_BABELSTREAM_REPO = "https://github.com/gitaumark/BabelStream"

class HBMBandwidth:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "HBMBandwidth"
        self.machine_name = machine
        config = tools.load_benchmark_config(os.path.join(dir_path, config_path), self.name)
        self.num_runs, self.interval = self.config_conversion(config)
        self.dir_path = dir_path
        self.container = None
        self.buffer = []

    def parse_json(self, config):
        return config["inputs"]["num_runs"], config["inputs"]["interval"]

    def config_conversion(self, config) -> tuple[int, int]:
        return self.parse_json(config)

    def build(self):
        path = "BabelStream"
        isdir = os.path.isdir(path)
        if not isdir:
            clone_cmd = f"git clone {_BABELSTREAM_REPO} {self.dir_path}/BabelStream"
            results = tools.run_cmd(clone_cmd, shell=True)
            results = tools.run_cmd(f'cd {self.dir_path}/BabelStream && cmake -Bbuild -H. -DMODEL=hip -DRELEASE_FLAGS="-O3" -DCMAKE_CXX_COMPILER=hipcc && cmake --build build', shell=True)

    def run(self):
        logger.info("Running HBM Bandwidth...")
        runs_executed = 0
        buffer = []
        while runs_executed < self.num_runs:
            run_cmd = f'sudo "{self.dir_path}/BabelStream/build/hip-stream"'
            results = tools.run_cmd(run_cmd, shell=True)
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
