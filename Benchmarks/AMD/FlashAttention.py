import logging
import os
import docker
import re
from prettytable import PrettyTable
from Infra import tools

logger = logging.getLogger(__name__)

_FLASH_ATTENTION_IMAGE = "powderluv/vllm_dev_channel:20240927"
_FLASH_ATTENTION_REPO = "https://github.com/Dao-AILab/flash-attention.git"
_FLASH_ATTENTION_CHECKOUT = "418d677"

class FlashAttention:
    def __init__(self, path:str, machine: str):
        self.name='FlashAttention'
        self.machine_name = machine
        self.dir_path = path
        self.container = None

    def create_container(self):
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'name': 'flash_attention',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'tty': True,
            'detach': True,
            'auto_remove': True
        }

        # Creates new Docker container
        logger.info("Pulling docker container %s...", _FLASH_ATTENTION_IMAGE)
        self.container = client.containers.run(_FLASH_ATTENTION_IMAGE, **docker_run_options)
        logger.info("Created Docker Container ID: %s", self.container.id)

    def run(self):
        current = os.getcwd()
        path ='flash-attention'
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_cmd(f'git clone {_FLASH_ATTENTION_REPO}',shell=True)

        build_path = os.path.join(current, 'flash-attention')

        results = tools.run_cmd(f'git checkout {_FLASH_ATTENTION_CHECKOUT}',shell=True, cwd=build_path)

        self.create_container()
        logger.info("Running Flash Attention...")
        try:
            res = self.container.exec_run(f"bash -c 'python3 {self.dir_path}/flash-attention/benchmarks/benchmark_flash_attention.py | grep -A 2 \"batch_size=2, seqlen=8192 ###\"'")
            tools.write_log(res.output.decode('utf-8'))
        finally:
            try:
                self.container.kill()
            except docker.errors.NotFound:
                pass  # auto_remove already cleaned up

        table = PrettyTable(["causal", "headdim", "Flash2 total (TFLOPs)", "Pytorch total (TFLOPs)"])
        for m in re.findall(r"causal=(\w+), headdim=(\d+).*?fwd \+ bwd: ([\d.]+).*?fwd \+ bwd: ([\d.]+)", res.output.decode('utf-8'), re.DOTALL):
            table.add_row([m[0], int(m[1]), float(m[2]), float(m[3])])
        print(table)
        tools.export_markdown("Flash Attention 2", "The performance (in TFLOPS), in table below, represents the performance for a batch size of 2, and a sequence length of 8192.", table)
