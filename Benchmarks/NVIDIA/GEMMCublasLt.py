import logging
import os
import subprocess
from Infra import tools
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

_SUPERBENCHMARK_REPO = "https://github.com/gitaumark/superbenchmark"

class GEMMCublastLt:
    def __init__(self, path: str, machine: str, b: int = 1, i: int = 1000, w: int = 10000):
        self.name = "GEMMCublasLt"
        config = tools.load_benchmark_config(path, self.name)
        self.datatype = self.config_conversion(config)
        self.b = b
        self.i = i
        self.w = w
        self.bindir = ''
        self.machine_name = machine
       
        # A100 does not support fp8
        if "A100" in machine:
            logger.warning("A100 does not support %s, using fp16 instead", self.datatype)
            self.datatype = "fp16"

    def config_conversion(self, config):
        return config["datatype"]

    def build(self):
        bindir = tools.create_dir("bin")
        self.bindir = bindir
        path = "superbenchmark"
        isdir = os.path.isdir(path)
        if not isdir:
            results = tools.run_cmd(
                [
                    "git",
                    "clone",
                    _SUPERBENCHMARK_REPO,
                    path,
                ],
            )
            
        if self.datatype == "fp4e2m1":
            results = subprocess.run("cd superbenchmark && git checkout fp4", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            results = subprocess.run("cd superbenchmark && git checkout main", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        current = os.getcwd()
        build_path = os.path.join(
            current,
            "superbenchmark/superbench/benchmarks/micro_benchmarks/cublaslt_gemm",
        )
        os.chdir(build_path)

        results = tools.run_cmd(
            ["cmake", "-S", "./"],
        )

        results = tools.run_cmd(
            ["make"],
        )
        logger.debug(results.stderr.decode('utf-8'))
        results = subprocess.run(
            ["mv", "cublaslt_gemm", bindir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        os.chdir(current)

    # run GEMM with predetermined matrix sizes that are commonly used in transformers
    def run_model_sizes(self):
        logger.info("Running CublasLt with datatype %s...", self.datatype)
        current = os.getcwd()
        if self.datatype == "fp8e4m3":
            m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]
        elif self.datatype == "fp4e2m1":
            m_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 768]
        else:
            m_dims = [1024, 2048, 4096, 8192, 16384, 1024, 6144, 802816]
            n_dims = [1024, 2048, 4096, 8192, 16384, 2145, 12288, 192]
            k_dims = [1024, 2048, 4096, 8192, 16384, 1024, 12288, 768]
        os.chdir(self.bindir)
        buffer = []
        for m, n, k in zip(m_dims, n_dims, k_dims):
            results = subprocess.run(
                [
                    "./cublaslt_gemm",
                    "-m",
                    str(m),
                    "-n",
                    str(n),
                    "-k",
                    str(k),
                    "-b",
                    str(self.b),
                    "-i",
                    str(self.i),
                    "-w",
                    str(self.w),
                    "-t",
                    self.datatype,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if results.returncode != 0:
                logger.warning("cublaslt_gemm failed for M=%s N=%s K=%s: returncode=%s", m, n, k, results.returncode)
                tools.write_log(tools.check_error(results))
                continue
            log = results.stdout.decode('utf-8').split()
            buffer.append(log)
            tools.write_log(tools.check_error(results))
        table1 = PrettyTable()
        table1.field_names = ["M", "N", "K", "Batch Size", "Time(us)", "TFLOPS"]
        for item in buffer:
            if len(item) == 6:
                table1.add_row(item)
            else:
                logger.warning("Skipping cublaslt_gemm result with %d columns (expected 6): %s", len(item), item)
        print(table1)
        tools.export_markdown("GEMM CuBLASLt", f"The results shown below are with random initialization (best representation of real-life workloads) {self.datatype}, and {self.w} warmup iterations.", table1)
        os.chdir(current)
