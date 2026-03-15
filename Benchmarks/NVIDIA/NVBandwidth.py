import subprocess
import os
from Infra import tools
from prettytable import PrettyTable

class NVBandwidth:
    TEST_NAMES = [
        "device_to_host_memcpy_ce",
        "host_to_device_memcpy_ce",
        "device_to_device_bidirectional_memcpy_read_ce",
    ]
    LABELS = [
        "Device to Host memcpy",
        "Host to Device memcpy",
        "Device to Device Bidirectional memcpy Total",
    ]

    def __init__(self, path:str, machine: str):
        self.name='NVBandwidth'
        self.machine_name = machine

    def build(self):
        current = os.getcwd()
        path ='nvbandwidth'
        isdir = os.path.isdir(path)
        if not isdir:
            results = subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/nvbandwidth', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            build_path = os.path.join(current, 'nvbandwidth')
            os.chdir(build_path)
            results = subprocess.run(['sed', '-i', r'2i\set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)', 'CMakeLists.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            build_path = os.path.join(current, 'nvbandwidth')
            os.chdir(build_path)

        if os.path.exists("/.dockerenv"):
            results = tools.run_cmd('apt update && ./debian_install.sh', shell=True)
        else:
            results = tools.run_cmd('sudo apt update && sudo ./debian_install.sh', shell=True)
        os.chdir(current)

    def run(self):
        current = os.getcwd()
        os.chdir(os.path.join(current, 'nvbandwidth'))
        print("Running NVBandwidth...")
        results = tools.run_cmd('./nvbandwidth -t device_to_host_memcpy_ce host_to_device_memcpy_ce device_to_device_bidirectional_memcpy_read_ce', shell=True)
        log = results.stdout.decode('utf-8')
        os.chdir(current)

        self.format_output(log)
        os.chdir(current)

    @staticmethod
    def _parse_sections(text):
        sections = {}
        current_name = None
        current_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            # Check if this line is a test name header
            found = None
            for name in NVBandwidth.TEST_NAMES:
                if stripped == name:
                    found = name
                    break
            if found:
                if current_name is not None:
                    sections[current_name] = "\n".join(current_lines)
                current_name = found
                current_lines = []
            elif current_name is not None:
                current_lines.append(line)
        if current_name is not None:
            sections[current_name] = "\n".join(current_lines)
        return sections

    @staticmethod
    def _extract_summary_table(section_text):
        table_rows = []
        for line in section_text.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                if table_rows:
                    break
                continue
            if 'memcpy' in stripped or stripped.startswith('running') or stripped.startswith('SUM'):
                continue
            tokens = stripped.split()
            row = [round(float(x), 1) if x.replace('.', '', 1).isdigit() else x for x in tokens]
            table_rows.append(row)
        return table_rows

    def format_output(self, text):
        sections = self._parse_sections(text)
        results = []
        result_labels = []
        for name, label in zip(self.TEST_NAMES, self.LABELS):
            if name not in sections:
                print(f"Warning: section '{name}' not found in nvbandwidth output")
                continue
            table = self._extract_summary_table(sections[name])
            results.append(table)
            result_labels.append(label)

        for i, table in enumerate(results):
            if not table:
                continue
            table[0].insert(0, " ")
            t = PrettyTable(table[0])
            for j in range(1, len(table)):
                t.add_row(table[j])
            print(result_labels[i])
            print(t)

            if i == 0:
                tools.export_markdown("NV Bandwidth", result_labels[i], t)
            else:
                tools.export_markdown(None, result_labels[i], t)
