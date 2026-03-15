import os
from Infra import tools
from prettytable import PrettyTable
from huggingface_hub import snapshot_download

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = tools.load_benchmark_config(config_path, self.name)
        self.dir_path = dir_path
        self.machine = machine
        self.table = None
        self.env = {**os.environ, 'HF_HOME': dir_path}

        tools.create_dir(self.dir_path + "/datasets")
        tools.create_dir(self.dir_path + "/engines")
        tools.create_dir(self.dir_path + "/hub")

    def install_requirements(self):
        # Clone TensorRT-LLM repo
        if not os.path.exists(os.path.join(self.dir_path, 'TensorRT-LLM')):
            print("Cloning TensorRT-LLM repository from https://github.com/NVIDIA/TensorRT-LLM.git")
            i4 = tools.run_cmd("git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM && git checkout v0.18.2", shell=True, env=self.env)

            if not os.path.exists("/.dockerenv"):
                # Install required packages
                print("No Docker container detected. Installing tensorrt-llm")
                i2 = tools.run_cmd("pip install tensorrt-llm==0.18.2", shell=True, env=self.env)
                i2 = tools.run_cmd("sudo apt update && sudo apt-get -y install libopenmpi-dev", shell=True, env=self.env)
                i2 = tools.run_cmd("pip3 install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs", shell=True, env=self.env)

    def download_models(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "nvidia":
                snapshot_download(repo_id=model_name, cache_dir=self.dir_path+"/hub")

    def prepare_datasets(self):
        for model_name in self.config['models']:
            max_isl = 0
            max_osl = 0
            max_sum = 0
            max_dataset_path = ""
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "nvidia":
                for i in range(len(self.config['models'][model_name]['input_sizes'])):
                    isl = self.config['models'][model_name]['input_sizes'][i]
                    osl = self.config['models'][model_name]['output_sizes'][i]
                    name = model_name.split('/')[1]
                    if (isl + osl > max_sum):
                        max_sum = isl + osl
                        max_isl = isl
                        max_osl = osl
                        max_dataset_path = self.dir_path + "/datasets/" + name + "_synthetic_" + str(max_isl) + "_" + str(max_osl) + ".txt"

                    dataset_path = self.dir_path + "/datasets/" + name + "_synthetic_" + str(isl) + "_" + str(osl) + ".txt"
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

                if not os.path.exists(self.dir_path + "/engines/" + model_name):
                    print("Building engine for ", model_name)
                    build_engine_command = f'''
                        trtllm-bench \
                        --workspace {self.dir_path + "/engines"} \
                        --model {model_name} build \
                        --tp_size {self.config['models'][model_name]['tp_size']} \
                        --dataset {max_dataset_path} \
                        --quantization {self.config['models'][model_name]['precision']}
                        '''

                    be2 = tools.run_cmd(build_engine_command, shell=True, env=self.env)

    def run_benchmark(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "nvidia":
                print("Benchmarking ", model_name, " with tp size ", self.config['models'][model_name]['tp_size'])
                self.table = PrettyTable(["tp size", "input len", "output len", "throughput(tokens/s)"])
                for i in range(len(self.config['models'][model_name]['input_sizes'])):
                    isl = self.config['models'][model_name]['input_sizes'][i]
                    osl = self.config['models'][model_name]['output_sizes'][i]
                    tp = self.config['models'][model_name]['tp_size']
                    name = model_name.split('/')[1]

                    print("input/output: " + str(isl) + "/" + str(osl) + "...")
                    dataset_path = self.dir_path + "/datasets/" + name + "_synthetic_" + str(isl) + "_" + str(osl) + ".txt"
                    results_path = self.dir_path + "/Outputs/results_" + name + "_" + str(isl) + "_" + str(osl) + ".txt"

                    run_benchmark_command = f'''
                        trtllm-bench \
                        --model {model_name} throughput\
                        --dataset {dataset_path} \
                        --engine_dir {self.dir_path + "/engines/" + model_name + "/tp_" + str(tp) + "_pp_1"} > {results_path}
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
                                print(f"Warning: could not parse value from line: {line.strip()}")
                            break

                if len(row) == 4:
                    self.table.add_row(row)
                else:
                    print(f"Warning: expected 4 values from {file_path}, got {len(row)}, skipping")
        except FileNotFoundError:
            print(f"Warning: benchmark output file not found: {file_path}")
