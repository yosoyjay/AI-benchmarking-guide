import logging
import os
from Infra import tools
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

class Multichase:
    def __init__(self, path:str, machine: str):
        self.name = "Multichase"
        self.machine_name = machine
    
    def build(self):
        current = os.getcwd()
        path = "multichase"
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_cmd(
                ["git", "clone", "https://github.com/google/multichase",  path],
            )

            build_path = os.path.join(current, "multichase")
            os.chdir(build_path)

            results = tools.run_cmd("make", shell=True)
            os.chdir(current)

    def run(self):
        current = os.getcwd()
        logger.info("Running Multichase...")

        results = tools.run_cmd("cd Benchmarks/NVIDIA && sudo chmod 755 run_multichase.sh && ./run_multichase.sh",shell=True)
        print(results.stdout.decode("utf-8"))
        tools.export_markdown("Multichase", results.stdout.decode("utf-8"), None)
