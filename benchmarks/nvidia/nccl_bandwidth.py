import logging
import os
import subprocess
from infra import tools
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

_NCCL_REPO = "https://github.com/NVIDIA/nccl.git"
_NCCL_TESTS_REPO = "https://github.com/NVIDIA/nccl-tests.git"

class NCCLBandwidth:
    def __init__(self, path:str, machine: str):
        self.name='NCCLBandwidth'
        self.machine_name = machine
        self.algo = "NVLS"
        self.env = dict(os.environ)

    def build(self):
        current = os.getcwd()
        path ='nccl'
        isdir = os.path.isdir(path)
        if not isdir:
            logger.info("Building NCCL Library...")
            results = subprocess.run(['git', 'clone', _NCCL_REPO, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nccl')
            results = tools.run_cmd('make -j src.build', shell=True, cwd=build_path)

        nccl_home = os.path.join(current, "nccl", "build")
        ld_path = f"{os.path.join(current, 'nccl', 'build', 'lib')}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        self.env = {**os.environ, 'NCCL_HOME': nccl_home, 'LD_LIBRARY_PATH': ld_path}

        path ='nccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            logger.info("Building NCCL Test...")
            results = subprocess.run(['git', 'clone', _NCCL_TESTS_REPO, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nccl-tests')
            results = tools.run_cmd(['make'], env=self.env, cwd=build_path)
        self.build_dir = os.path.join(current, 'nccl-tests')

    def run(self):
        num_gpus_result = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if num_gpus_result.returncode != 0 or not num_gpus_result.stdout.decode('utf-8').strip():
            logger.warning("nvidia-smi failed to detect GPU count, defaulting to 8")
            num_gpus = "8"
        else:
            num_gpus = num_gpus_result.stdout.decode('utf-8').strip()
        if num_gpus == '4':
            self.algo = "Ring"
        logger.info("Running NCCL AllReduce on %s GPUs", num_gpus)

        all_reduce_bin = os.path.join(self.build_dir, "build", "all_reduce_perf")
        results = tools.run_cmd(f'NCCL_ALGO={self.algo} {all_reduce_bin} -b 8 -e 8G -f 2 -g {num_gpus} -n 40 | grep float', shell=True, env=self.env)
        res = results.stdout.decode('utf-8').split('\n')
        sizes = []
        log = []
        for line in res:
            fields = line.split()
            if len(fields) == 13:
                sizes.append(fields[0])
                log.append(fields[11])

        table1 = PrettyTable()
        runs = ["Message Size", f"Bandwidth ({self.algo})"]
        table1.add_column(runs[0], sizes)
        table1.add_column(runs[1], log)
        print(table1)
        tools.export_markdown("NCCL Bandwidth", f"The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce ({self.algo} algorithm) tests in-place operations, varying from 1KB to 8GB of data.", table1)
