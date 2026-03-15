import os
import datetime
import logging
import subprocess
import json

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pwd = os.path.join(_PROJECT_ROOT, "Outputs", "log.txt")
curr = _PROJECT_ROOT

def run_cmd(cmd, *, shell=False, env=None, cwd=None, **kwargs):
    """Run a command, log output, and return the CompletedProcess."""
    result = subprocess.run(
        cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=cwd, **kwargs,
    )
    write_log(check_error(result))
    return result


def load_benchmark_config(path, section_name):
    """Load a JSON config file and return the section for a benchmark."""
    with open(path) as f:
        data = json.load(f)
    try:
        return data[section_name]
    except KeyError:
        raise KeyError(f"'{section_name}' section not found in {path}")


def create_dir(name: str):
    current = os.getcwd()
    outdir = os.path.join(str(current), name)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def write_log(message: str, filename: str = pwd):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]\n {message}\n"

    with open(filename, "a") as file:
        file.write(log_entry)

def check_error(results):
    stdout = results.stdout.decode("utf-8") if results.stdout else ""
    stderr = results.stderr.decode("utf-8") if results.stderr else ""
    if results.returncode != 0:
        return f"[ERROR] returncode={results.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    if stderr:
        return f"{stdout}\n[stderr]\n{stderr}"
    return stdout

def get_os_version():
    results = subprocess.run("lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if results.returncode != 0:
        return "unknown"
    parts = results.stdout.decode('utf-8').strip().split("\t")
    if len(parts) < 2:
        return "unknown"
    return f"Ubuntu {parts[1]}"

def get_hostname():
    results = subprocess.run(["hostname"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if results.returncode != 0:
        return ""
    return results.stdout.decode("utf-8").strip()

def prettytable_to_markdown(table):
    if table is None:
        return ""
    header = f"| {' | '.join(table.field_names)} |"
    separator = f"| {' | '.join('---' for _ in table.field_names)} |"
    rows = [f"| {' | '.join(str(cell) for cell in row)} |" for row in table.rows]
    return "\n".join([header, separator] + rows) 

def export_markdown(title, description, table = None):
    table = prettytable_to_markdown(table)
    filename = os.path.join(curr, "Outputs", f"{get_hostname()}_summary.md")
    with open(filename, "a") as file:
        if title is not None:
            file.write(f"## {title}\n\n")
        file.write(f"{description}\n")
        file.write(table)
        file.write("\n\n")
        
def create_bm_entry(bmName, appName, sku, result):
    id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    ubuntu = get_os_version()
    return {
        "jobId": id,
        "appName": appName,
        "bmName": bmName,
        "nodes": "1",
        "cores": "",
        "sockets": "",
        "result": result,
        "totalRunTime": "",
        "skuGen": "",
        "sku": sku,
        "os": ubuntu,
        "BIOS": "",
        "user": "azureuser",
        "runCategory": "best",
        "Run Date": "",
        "notes": "",
        "appVersion": "string"
    }

def post_benchmark_entry(entry, url):
    json_data = json.dumps(entry)
    curl_command = [
        "curl",
        "-X", "POST",
        url,
        "-H", "Content-Type: application/json",
        "-d", json_data
    ]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    return result.stdout, result.stderr

BABELSTREAM_OPS = ("Copy", "Mul", "Add", "Triad", "Dot")

def parse_babelstream_output(raw_output):
    results = []
    for line in raw_output.strip().split("\n"):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] in BABELSTREAM_OPS:
            results.append([tokens[0], tokens[1]])
    return results


def summarize_babelstream(buffer, *, divisor, units, title, description):
    """Build a PrettyTable from BabelStream run buffers.

    Returns the table, or None if no valid data was collected.
    """
    import statistics
    from prettytable import PrettyTable

    ops = {name: [] for name in BABELSTREAM_OPS}
    for log in buffer:
        if len(log) < 5:
            logger.warning("BabelStream returned %d operations (expected 5), skipping run", len(log))
            continue
        for idx, name in enumerate(BABELSTREAM_OPS):
            ops[name].append(float(log[idx][1]))

    if not ops["Copy"]:
        logger.warning("All BabelStream runs produced incomplete output, no results to report")
        return None

    table = PrettyTable()
    table.field_names = ["Operation", f"Min ({units})", f"Max ({units})", f"Mean ({units})"]
    for name in BABELSTREAM_OPS:
        values = ops[name]
        mn = round(min(values) / divisor, 2)
        mx = round(max(values) / divisor, 2)
        mean = round(statistics.mean(values) / divisor, 2)
        table.add_row([name, mn, mx, mean])

    print(table)
    export_markdown(title, description, table)
    return table
