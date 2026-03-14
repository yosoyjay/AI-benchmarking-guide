import json
import os
import statistics
import subprocess
from Infra import tools
from prettytable import PrettyTable

class Multichase:
    def __init__(self, path:str, machine: str):
        self.name = "Multichase"
        self.machine_name = machine
    
    def build(self):
        current = os.getcwd()
        path = "multichase"
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(
                ["git", "clone", "https://github.com/google/multichase",  path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tools.write_log(tools.check_error(results))

            build_path = os.path.join(current, "multichase")
            os.chdir(build_path)
            
            results = subprocess.run("make", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            tools.write_log(tools.check_error(results))
            os.chdir(current)

    def run(self):
        current = os.getcwd()
        print("Running Multichase...")

        results = subprocess.run("cd Benchmarks/NVIDIA && sudo chmod 755 run_multichase.sh && ./run_multichase.sh",shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tools.write_log(tools.check_error(results))
        print(results.stdout.decode("utf-8"))
        tools.export_markdown("Multichase", results.stdout.decode("utf-8"), None)
