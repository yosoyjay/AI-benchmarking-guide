import logging
import docker
from prettytable import PrettyTable
from Infra import tools

logger = logging.getLogger(__name__)

_VLLM_IMAGE = "rocm/vllm-dev:20241121-tuned"

class LLMBenchmark:
    def __init__(self, config_path: str, dir_path: str, machine: str):
        self.name = "LLMBenchmark"
        self.config = tools.load_benchmark_config(config_path, self.name)
        self.dir_path = dir_path
        self.precision = "half"
        self.table = None
        self.container = None
        self.machine = machine

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
        logger.info("Pulling docker container %s", _VLLM_IMAGE)
        self.container = client.containers.run(_VLLM_IMAGE, **docker_run_options)
        logger.info("Docker Container ID: %s", self.container.id)

    def run_benchmark(self):
        if self.container is None:
            logger.warning("no container created, skipping benchmark run")
            return
        try:
            for model_name in self.config['models']:
                if self.config['models'][model_name]['use_model'] and self.config['models'][model_name]['type'] == "amd":
                    self.table = PrettyTable(["input len", "output len", "tp size", "throughput(tokens/s)"])
                    for tp_size in self.config['models'][model_name]['tp_sizes']:
                        logger.info("Benchmarking %s with TP Size: %s", model_name, tp_size)
                        for max_num_seq in self.config['models'][model_name]['max_num_seqs']:
                            for input_size, output_size in zip(
                                self.config['models'][model_name]['input_length'],
                                self.config['models'][model_name]['output_length'],
                            ):
                                for request in self.config['models'][model_name]['num_requests']:
                                    logger.info(" Input Size: %s, Output Size: %s...", input_size, output_size)
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
                                            parts = line.split(' ')
                                            result = parts[6] if len(parts) > 6 else "unknown"
                                            self.table.add_row([str(input_size), str(output_size), str(tp_size), str(result)])
                    print(self.table)
                    tools.export_markdown(model_name, "Performance results with FP8 quantization.", self.table)
        finally:
            self.container.kill()
