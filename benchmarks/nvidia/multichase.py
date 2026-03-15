import logging
import os
from infra import tools
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

_MULTICHASE_REPO = "https://github.com/google/multichase"

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
                ["git", "clone", _MULTICHASE_REPO,  path],
            )

            build_path = os.path.join(current, "multichase")

            results = tools.run_cmd("make", shell=True, cwd=build_path)

    def run(self):
        logger.info("Running Multichase...")

        results = tools.run_cmd("sudo chmod 755 run_multichase.sh && ./run_multichase.sh", shell=True, cwd="benchmarks/nvidia")
        print(results.stdout.decode("utf-8"))
        tools.export_markdown("Multichase", results.stdout.decode("utf-8"), None)
