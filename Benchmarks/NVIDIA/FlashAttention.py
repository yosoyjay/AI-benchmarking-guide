import logging
import subprocess
import re
import os
from prettytable import PrettyTable
from Infra import tools

logger = logging.getLogger(__name__)

_FLASH_ATTENTION_REPO = "https://github.com/Dao-AILab/flash-attention.git"

class FlashAttention:
    def __init__(self, path:str, machine: str):
        self.name='FlashAttention'
        self.machine_name = machine
        self.buffer = []

    def run(self):
        current = os.getcwd()
        path ='flash-attention'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(f'git clone {_FLASH_ATTENTION_REPO}',shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        build_path = os.path.join(current, 'flash-attention/benchmarks')

        logger.info("Running Flash Attention with batch size=2, seqlen=8192...")
        results = tools.run_cmd('python3 benchmark_flash_attention.py | grep -A 2 "batch_size=2, seqlen=8192 ###"',shell=True, cwd=build_path)

        table = PrettyTable(["causal", "headdim", "Flash2 total (TFLOPs)", "Pytorch total (TFLOPs)"])
        for m in re.findall(r"causal=(\w+), headdim=(\d+).*?fwd \+ bwd: ([\d.]+).*?fwd \+ bwd: ([\d.]+)", results.stdout.decode('utf-8'), re.DOTALL):
            table.add_row([m[0], int(m[1]), float(m[2]), float(m[3])])
        print(table)
        tools.export_markdown("Flash Attention 2", "The performance (in TFLOPS), in table below, represents the performance for a batch size of 2, and a sequence length of 8192.", table)
