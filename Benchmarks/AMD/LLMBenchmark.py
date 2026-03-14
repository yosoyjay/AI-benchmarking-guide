import docker
import os
import json
import csv
from prettytable import PrettyTable
from Infra import tools

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = self.get_config(config_path)
        self.dir_path = dir_path
        self.precision = "half"
        self.table = None
        self.container = None
        self.machine = machine

    def get_config(self, path: str):
        with open(path) as file:
            data = json.load(file)
        try:
            return data[self.name]
        except KeyError:
            raise KeyError("no value found")

    def create_container(self):
        client = docker.from_env()
        # Define the Docker run options
        docker_run_options = {
            'ipc_mode':'host',
            'network': 'host',
            'entrypoint':'/bin/bash',
            'group_add': ['render'],
            'privileged': True,
            'security_opt': ['seccomp=unconfined'],
            'cap_add': ['CAP_SYS_ADMIN', 'SYS_PTRACE'],
            'devices': ['/dev/kfd', '/dev/dri', '/dev/mem'],
            'volumes': {str(self.dir_path): {'bind': str(self.dir_path), 'mode': 'rw'}},
            'environment': {'HF_HOME': str(self.dir_path)},
            'tty': True,
            'detach': True
        }

        # Creates new Docker container
        print("Pulling docker container rocm/vllm-dev:20241121-tuned")
        self.container = client.containers.run('rocm/vllm-dev:20241121-tuned', **docker_run_options)
        print(f"Docker Container ID: {self.container.id}")

    def run_benchmark(self):
        for model_name in self.config['models']:
            if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "amd":
                self.table = PrettyTable(["input len", "output len", "tp size", "throughput(tokens/s)"])
                for tp_size in self.config['models'][model_name]['tp_sizes']:
                    print(f"Benchmarking {model_name} with TP Size: {tp_size}")
                    for max_num_seq in self.config['models'][model_name]['max_num_seqs']:
                        for i in range(len(self.config['models'][model_name]['input_length'])):
                            for request in self.config['models'][model_name]['num_requests']:
                                input_size = self.config['models'][model_name]['input_length'][i]
                                output_size = self.config['models'][model_name]['output_length'][i]
                                print(f" Input Size: {input_size}, Output Size: {output_size}...")
                                run_benchmark_command = f'''
                                    /bin/bash -c \
                                    "python /app/vllm/benchmarks/benchmark_throughput.py \
                                        --model amd/{model_name} \
                                        --quantization fp8 \
                                        --kv-cache-dtype fp8 \
                                        --dtype half \
                                        --gpu-memory-utilization 0.90 \
                                        --distributed-executor-backend mp \
                                        --num-scheduler-steps 10 \
                                        --tensor-parallel-size {tp_size} \
                                        --enable-chunked-prefill false \
                                        --max-seq-len-to-capture 131072 \
                                        --max-num-batched-tokens 131072 \
                                        --max-model-len 8192 \
                                        --max-num-seqs {max_num_seq} \
                                        --num-prompts {request} \
                                        --input-len {input_size} \
                                        --output-len {output_size}"
                                    '''
                                rb1 = self.container.exec_run(run_benchmark_command)
                                tools.write_log(rb1.output.decode('utf-8'))
                                temp = rb1.output.decode('utf-8').split('\n')
                                for line in temp:
                                    if "Throughput: " in line:
                                        result = line.split(' ')[6]
                                        self.table.add_row([str(input_size), str(output_size), str(tp_size), str(result)])
                print(self.table)
                tools.export_markdown(model_name, "Performance results with FP8 quantization.", self.table)
        self.container.kill()
