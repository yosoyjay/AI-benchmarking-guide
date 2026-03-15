import logging
import docker
import os
from prettytable import PrettyTable
from Infra import tools

logger = logging.getLogger(__name__)

_RCCL_PYTORCH_IMAGE = "rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue"
_RCCL_REPO = "https://github.com/ROCm/rccl.git"
_RCCL_TESTS_REPO = "https://github.com/ROCm/rccl-tests.git"

class RCCLBandwidth:
    def __init__(self, config_path:str, dir_path:str, machine: str):
        self.name='RCCLBandwidth'
        self.machine_name = machine
        self.dir_path = dir_path
        self.container = None
        self.buffer = []

    def create_container(self):
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'entrypoint': '/bin/bash',
            'network': 'host',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'tty': True,
            'detach': True
        }

        # Creates new Docker container from https://hub.docker.com/r/rocm/pytorch/tags
        logger.info("Pulling docker container %s...", _RCCL_PYTORCH_IMAGE)
        self.container = client.containers.run(_RCCL_PYTORCH_IMAGE, **docker_run_options)
        logger.info("Docker Container ID: %s", self.container.id)

    def build(self):
        path ='rccl'
        isdir = os.path.isdir(path)
        if not isdir:
            logger.info("Building RCCL Library...")
            clone_cmd = "git clone " + _RCCL_REPO + " " + self.dir_path + "/rccl"
            results = self.container.exec_run(clone_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/rccl && cmake . && make"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

        path ='rccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            logger.info("Building RCCL Tests...")
            clone_cmd = "git clone " + _RCCL_TESTS_REPO + " " + self.dir_path + "/rccl-tests"
            results = self.container.exec_run(clone_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/rccl-tests && make HIP_HOME=/opt/rocm NCCL_HOME={self.dir_path}/rccl CUSTOM_RCCL_LIB={self.dir_path}/rccl/librccl.so && make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm NCCL_HOME={self.dir_path}/rccl"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

    def run(self):
        sizes = []
        bandwidth_columns = []
        runs = ["Tree", "Ring", "NVLS", "NVLSTree"]
        logger.info("Running RCCL AllReduce...")
        try:
            for run in runs:
                run_cmd = "NCCL_ALGO=" + run + " " + self.dir_path +"/rccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 40 | grep float"
                run_cmd = '/bin/sh -c "' + run_cmd + '"'
                results = self.container.exec_run(run_cmd, stderr=True)
                if results.exit_code != 0:
                    tools.write_log(results.output.decode('utf-8'))
                    return
                res = results.output.decode('utf-8').split('\n')
                log = []
                for line in res:
                    fields = line.split()
                    if len(fields) == 13:
                        if not bandwidth_columns:
                            sizes.append(fields[0])
                        log.append(fields[11])
                bandwidth_columns.append(log)
        finally:
            self.container.kill()
        table1 = PrettyTable()
        col_names = ["Message Size"] + runs

        table1.add_column(col_names[0], sizes)
        for i, col in enumerate(bandwidth_columns):
            table1.add_column(col_names[i + 1], col)
        print(table1)
        tools.export_markdown("RCCL Bandwidth", "The values (in GB/s) are the bus bandwidth values obtained from the RCCL AllReduce tests with Tree, Ring, NVLS and NVLSTree algos (in-place operations), varying from 1KB to 8GB of data.", table1)
