import os
import shlex
import subprocess
from prettytable import PrettyTable
from Infra import tools

class FIO:
    def __init__(self, path: str, machine: str):
        self.name = "FIO"
        self.machine_name = machine
        
    def run(self):
        current = os.getcwd()
        print("Running FIO Tests...")
        tests = [
            ["read", "1M"],
            ["read", "512k"],
            ["read", "1k"],
            ["write", "1M"],
            ["write", "512k"],
            ["write", "1k"],
            ["randwrite", "1k"],
            ["randread", "1k"]
        ]
        table = PrettyTable(["Test", "Batch Size(Bytes)", "Bandwidth"])
        for test in tests:
            results = subprocess.run(
                "fio --bs=" + test[1] +  " --ioengine=libaio --iodepth=255 --directory=" + shlex.quote(current + "/Outputs") + " --direct=1 --runtime=300 --numjobs=4 --rw=" +test[0]+ " --name=test --group_reporting --gtod_reduce=1 --size=10G | grep -A 1 ': bw='",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if results.returncode != 0:
                print(f"Warning: fio failed for {test[0]} bs={test[1]}: returncode={results.returncode}")
                res = "error"
            else:
                tokens = results.stdout.decode('utf-8').split()
                res = tokens[2].strip(",()") if len(tokens) >= 3 else "error"
            table.add_row([test[0], test[1], res])
        print(table)
        tools.export_markdown("FIO Tests", "", table)
           
        results = subprocess.run(
            "rm Outputs/test*",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
