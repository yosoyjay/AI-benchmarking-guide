import os
from prettytable import PrettyTable
from Infra import tools

class TransferBench:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "TransferBench"
        self.machine_name = machine
        self.dir_path = dir_path

    def build(self):
        path = "TransferBench"
        isdir = os.path.isdir(path)
        if not isdir:
            print("Building TransferBench...")
            clone_cmd = 'git clone https://github.com/ROCm/TransferBench.git "' + self.dir_path + '/TransferBench"'
            results = tools.run_cmd(clone_cmd, shell=True)
            results = tools.run_cmd('mkdir -p "' + self.dir_path + '/TransferBench/build"', shell=True)

            results = tools.run_cmd('cd "' + self.dir_path + '/TransferBench/build" && CXX=/opt/rocm/bin/hipcc cmake .. && make', shell=True)

    def run(self):
        print("Running TransferBench...")
        run_cmd = 'sudo "' + self.dir_path + '/TransferBench/build/TransferBench" "' + self.dir_path + '/Benchmarks/AMD/transferbench.cfg" | grep -v \'=\' | grep \'sum\''
        results = tools.run_cmd(run_cmd, shell=True)
        table = PrettyTable(["Test", "Result"])
        if results.returncode != 0:
            print(f"Warning: TransferBench failed: returncode={results.returncode}")
            table.add_row(["Host to Device memcpy", "error"])
            table.add_row(["Device to Host memcpy", "error"])
        else:
            log = results.stdout.decode("utf-8").split("|")
            h2d = log[1].strip() if len(log) > 1 else "error"
            d2h = log[5].strip() if len(log) > 5 else "error"
            table.add_row(["Host to Device memcpy", h2d])
            table.add_row(["Device to Host memcpy", d2h])
        print(table)
        tools.export_markdown("TransferBench", "TransferBench Results in GB/s", table)
