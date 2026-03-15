import logging
import os
from infra import tools
from prettytable import PrettyTable
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

_TENSORRT_LLM_REPO = "https://github.com/NVIDIA/TensorRT-LLM.git"
_TENSORRT_LLM_VERSION = "v0.18.2"
_TENSORRT_LLM_PIP_VERSION = "0.18.2"
_NVIDIA_PYPI_URL = "https://pypi.nvidia.com"

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = tools.load_benchmark_config(config_path, self.name)
        self.dir_path = dir_path
        self.machine = machine
        self.table = None
        self.env = {**os.environ, 'HF_HOME': dir_path}

        tools.create_dir(os.path.join(self.dir_path, "datasets"))
        tools.create_dir(os.path.join(self.dir_path, "engines"))
        tools.create_dir(os.path.join(self.dir_path, "hub"))

    def install_requirements(self):
        # Clone TensorRT-LLM repo
        if not os.path.exists(os.path.join(self.dir_path, 'TensorRT-LLM')):
            logger.info("Cloning TensorRT-LLM repository from %s", _TENSORRT_LLM_REPO)
            i4 = tools.run_cmd(f"git clone {_TENSORRT_LLM_REPO} && cd TensorRT-LLM && git checkout {_TENSORRT_LLM_VERSION}", shell=True, env=self.env)

            if not os.path.exists("/.dockerenv"):
                # Install required packages
                logger.info("No Docker container detected. Installing tensorrt-llm")
                i2 = tools.run_cmd(f"pip install tensorrt-llm=={_TENSORRT_LLM_PIP_VERSION}", shell=True, env=self.env)
                i2 = tools.run_cmd("sudo apt update && sudo apt-get -y install libopenmpi-dev", shell=True, env=self.env)
                i2 = tools.run_cmd(f"pip3 install --no-cache-dir --extra-index-url {_NVIDIA_PYPI_URL} tensorrt-libs", shell=True, env=self.env)

    def download_models(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "nvidia":
                snapshot_download(repo_id=model_name, cache_dir=os.path.join(self.dir_path, "hub"))

    def prepare_datasets(self):
        for model_name in self.config['models']:
            max_isl = 0
            max_osl = 0
            max_sum = 0
            max_dataset_path = ""
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "nvidia":
                for isl, osl in zip(
                    self.config['models'][model_name]['input_sizes'],
                    self.config['models'][model_name]['output_sizes'],
                ):
                    name = model_name.split('/')[1]
                    if (isl + osl > max_sum):
                        max_sum = isl + osl
                        max_isl = isl
                        max_osl = osl
                        max_dataset_path = f"{self.dir_path}/datasets/{name}_synthetic_{max_isl}_{max_osl}.txt"

                    dataset_path = os.path.join(self.dir_path, "datasets", f"{name}_synthetic_{isl}_{osl}.txt")
                    if not os.path.exists(dataset_path):
                        prepare_dataset_command = f'''
                            python3 {self.dir_path}/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py \
                            --stdout \
                            --tokenizer {model_name} \
                            token-norm-dist \
                            --num-requests {self.config['models'][model_name]['num_requests']} \
                            --input-mean {isl} \
                            --output-mean {osl} \
                            --input-stdev=0 \
                            --output-stdev=0 > {dataset_path}
                            '''
    
                        be2 = tools.run_cmd(prepare_dataset_command, shell=True, env=self.env)

                if not os.path.exists(os.path.join(self.dir_path, "engines", model_name)):
                    logger.info("Building engine for %s", model_name)
                    build_engine_command = f'''
                        trtllm-bench \
                        --workspace {self.dir_path}/engines \
                        --model {model_name} build \
                        --tp_size {self.config['models'][model_name]['tp_size']} \
                        --dataset {max_dataset_path} \
                        --quantization {self.config['models'][model_name]['precision']}
                        '''

                    be2 = tools.run_cmd(build_engine_command, shell=True, env=self.env)

    def run_benchmark(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "nvidia":
                logger.info("Benchmarking %s with tp size %s", model_name, self.config['models'][model_name]['tp_size'])
                self.table = PrettyTable(["tp size", "input len", "output len", "throughput(tokens/s)"])
                for isl, osl in zip(
                    self.config['models'][model_name]['input_sizes'],
                    self.config['models'][model_name]['output_sizes'],
                ):
                    tp = self.config['models'][model_name]['tp_size']
                    name = model_name.split('/')[1]

                    logger.info("input/output: %s/%s...", isl, osl)
                    dataset_path = f"{self.dir_path}/datasets/{name}_synthetic_{isl}_{osl}.txt"
                    results_path = os.path.join(self.dir_path, "Outputs", f"results_{name}_{isl}_{osl}.txt")

                    run_benchmark_command = f'''
                        trtllm-bench \
                        --model {model_name} throughput\
                        --dataset {dataset_path} \
                        --engine_dir {self.dir_path}/engines/{model_name}/tp_{tp}_pp_1 > {results_path}
                        '''

                    be2 = tools.run_cmd(run_benchmark_command, shell=True, env=self.env)
                    self.extract_benchmark_info(results_path)
                print(self.table)
                tools.export_markdown(model_name, "Performance results with FP8 quantization, 1000 requests.", self.table)

    def extract_benchmark_info(self, file_path):
        keywords = [
            "TP Size:",
            "Average Input Length (tokens):",
            "Average Output Length (tokens):",
            "Token Throughput (tokens/sec):"
        ]

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                row = []
                for line in file:
                    for keyword in keywords:
                        if keyword in line:
                            try:
                                parts = line.split(":")
                                if len(parts) < 2:
                                    continue
                                row.append(str(int(float(parts[1].strip()))))
                            except (ValueError, IndexError):
                                logger.warning("could not parse value from line: %s", line.strip())
                            break

                if len(row) == 4:
                    self.table.add_row(row)
                else:
                    logger.warning("expected 4 values from %s, got %d, skipping", file_path, len(row))
        except FileNotFoundError:
            logger.warning("benchmark output file not found: %s", file_path)
