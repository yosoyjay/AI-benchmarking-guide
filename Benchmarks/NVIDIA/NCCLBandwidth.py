import logging
import os
import subprocess
from Infra import tools
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

class NCCLBandwidth:
    def __init__(self, path:str, machine: str):
        self.name='NCCLBandwidth'
        self.machine_name = machine
        self.buffer = []
        self.algo = "NVLS"
        self.env = dict(os.environ)

    def build(self):
        current = os.getcwd()
        path ='nccl'
        isdir = os.path.isdir(path)
        if not isdir:
            logger.info("Building NCCL Library...")
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nccl.git', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nccl')
            os.chdir(build_path)
            results = tools.run_cmd('make -j src.build', shell=True)
            os.chdir(current)

        nccl_home = current + '/nccl/build'
        ld_path = current + '/nccl/build/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        self.env = {**os.environ, 'NCCL_HOME': nccl_home, 'LD_LIBRARY_PATH': ld_path}

        path ='nccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            logger.info("Building NCCL Test...")
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nccl-tests.git', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nccl-tests')
            os.chdir(build_path)
            results = tools.run_cmd(['make'], env=self.env)
        else:
            build_path = os.path.join(current, 'nccl-tests')
            os.chdir(build_path)

    def run(self):
        current = os.getcwd()
        num_gpus_result = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if num_gpus_result.returncode != 0 or not num_gpus_result.stdout.decode('utf-8').strip():
            logger.warning("nvidia-smi failed to detect GPU count, defaulting to 8")
            num_gpus = "8"
        else:
            num_gpus = num_gpus_result.stdout.decode('utf-8').strip()
        if num_gpus == '4':
            self.algo = "Ring"
        logger.info("Running NCCL AllReduce on " + num_gpus + " GPUs")

        results = tools.run_cmd('NCCL_ALGO='+ self.algo +' ./build/all_reduce_perf -b 8 -e 8G -f 2 -g ' + num_gpus + ' -n 40 | grep float', shell=True, env=self.env)
        res = results.stdout.decode('utf-8').split('\n')
        sizes = []
        log = []
        for line in res:
            fields = line.split()
            if len(fields) == 13:
                sizes.append(fields[0])
                log.append(fields[11])

        table1 = PrettyTable()
        runs = ["Message Size", "Bandwidth (" + self.algo + ")"]
        table1.add_column(runs[0], sizes)
        table1.add_column(runs[1], log)
        print(table1)
        tools.export_markdown("NCCL Bandwidth", f"The values (in GB/s) are the bus bandwidth values obtained from the NCCL AllReduce ({self.algo} algorithm) tests in-place operations, varying from 1KB to 8GB of data.", table1)
        os.chdir(current)
