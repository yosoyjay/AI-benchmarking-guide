import json
import docker
import os
import csv
from prettytable import PrettyTable
from Infra import tools

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
        print("Pulling docker container rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue...")
        self.container = client.containers.run('rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0_triton_llvm_reg_issue', **docker_run_options)
        print(f"Docker Container ID: {self.container.id}")

    def build(self):
        path ='rccl'
        isdir = os.path.isdir(path)
        if not isdir:
            print("Building RCCL Library...")
            clone_cmd = "git clone https://github.com/ROCm/rccl.git " + self.dir_path + "/rccl"
            results = self.container.exec_run(clone_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/rccl && cmake . && make"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

            results = self.container.exec_run(f'/bin/sh -c "cd .."', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

        path ='rccl-tests'
        isdir = os.path.isdir(path)
        if not isdir:
            print("Building RCCL Tests...")
            clone_cmd = "git clone https://github.com/ROCm/rccl-tests.git " + self.dir_path + "/rccl-tests"
            results = self.container.exec_run(clone_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

            results = self.container.exec_run(f'/bin/sh -c "cd {self.dir_path}/rccl-tests && make HIP_HOME=/opt/rocm NCCL_HOME={self.dir_path}/rccl CUSTOM_RCCL_LIB={self.dir_path}/rccl/librccl.so && make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm NCCL_HOME={self.dir_path}/rccl"', stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))

    def run(self):
        buffer=[["8 ","16 ","32 ","64 ","128 ","256 ","512 ","1K","2K","4K","8K","16K","32K","65K","132K","256K", "524K","1M","2M","4M","8M","16M","33M","67M","134M","268M","536M","1G","2G","4G","8G"]]
        runs = ["Tree", "Ring", "NVLS", "NVLSTree"]
        print("Running RCCL AllReduce...")
        for run in runs:
            run_cmd = "NCCL_ALGO=" + run + " " + self.dir_path +"/rccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 8 -n 40 | grep float"
            run_cmd = '/bin/sh -c "' + run_cmd + '"'
            results = self.container.exec_run(run_cmd, stderr=True)
            if results.exit_code != 0:
                tools.write_log(results.output.decode('utf-8'))
                self.container.kill()
                return
            res = results.output.decode('utf-8').split('\n')
            log = []
            for line in res:
                line = line.split()
                if len(line) == 13:
                    log.append(line[11])
            buffer.append(log)
        self.container.kill()
        table1 = PrettyTable()
        runs = ["Message Size", "Tree", "Ring", "NVLS", "NVLSTree"]

        for i in range(len(buffer)):
            table1.add_column(runs[i], buffer[i])
        print(table1)
        tools.export_markdown("RCCL Bandwidth", "The values (in GB/s) are the bus bandwidth values obtained from the RCCL AllReduce tests with Tree, Ring, NVLS and NVLSTree algos (in-place operations), varying from 1KB to 8GB of data.", table1)
