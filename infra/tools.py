import datetime
import json
import logging
import os
import subprocess
import warnings
from typing import Any

from prettytable import PrettyTable

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curr = _PROJECT_ROOT

_log_path: str | None = None
_summary_path: str | None = None


def set_log_path(path: str) -> None:
    """Set the destination for write_log(). Called once from each runner's main()."""
    global _log_path
    _log_path = path


def set_summary_path(path: str) -> None:
    """Set the destination for export_markdown(). Called once from each runner's main()."""
    global _summary_path
    _summary_path = path


def run_cmd(
    cmd: list[str] | str,
    *,
    shell: bool = False,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command, log output, and return the CompletedProcess."""
    result = subprocess.run(
        cmd,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        **kwargs,
    )
    write_log(check_error(result))
    return result


def load_benchmark_config(path: str, section_name: str) -> dict[str, Any]:
    """Load a JSON config file and return the section for a benchmark."""
    with open(path) as f:
        data = json.load(f)
    try:
        return data[section_name]
    except KeyError:
        raise KeyError(f"'{section_name}' section not found in {path}")


def create_dir(name: str) -> str:
    current = os.getcwd()
    outdir = os.path.join(str(current), name)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def write_log(message: str, filename: str | None = None) -> None:
    target = filename or _log_path
    if target is None:
        logger.info(message)
        return
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]\n {message}\n"

    with open(target, "a") as file:
        file.write(log_entry)


def check_error(results: subprocess.CompletedProcess[bytes]) -> str:
    stdout = results.stdout.decode("utf-8") if results.stdout else ""
    stderr = results.stderr.decode("utf-8") if results.stderr else ""
    if results.returncode != 0:
        return f"[ERROR] returncode={results.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    if stderr:
        return f"{stdout}\n[stderr]\n{stderr}"
    return stdout


def get_os_version() -> str:
    results = subprocess.run(
        "lsb_release -a | grep Release", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if results.returncode != 0:
        return "unknown"
    parts = results.stdout.decode("utf-8").strip().split("\t")
    if len(parts) < 2:
        return "unknown"
    return f"Ubuntu {parts[1]}"


def get_hostname() -> str:
    results = subprocess.run(["hostname"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if results.returncode != 0:
        return ""
    return results.stdout.decode("utf-8").strip()


def prettytable_to_markdown(table: PrettyTable | None) -> str:
    if table is None:
        return ""
    header = f"| {' | '.join(table.field_names)} |"
    separator = f"| {' | '.join('---' for _ in table.field_names)} |"
    rows = [f"| {' | '.join(str(cell) for cell in row)} |" for row in table.rows]
    return "\n".join([header, separator] + rows)


def export_markdown(title: str | None, description: str, table: PrettyTable | None = None) -> None:
    warnings.warn(
        "export_markdown is deprecated; use infra.process for structured CSV output",
        DeprecationWarning,
        stacklevel=2,
    )
    md_table = prettytable_to_markdown(table)
    if _summary_path is not None:
        filename = _summary_path
    else:
        filename = os.path.join(curr, "Outputs", f"{get_hostname()}_summary.md")
    with open(filename, "a") as file:
        if title is not None:
            file.write(f"## {title}\n\n")
        file.write(f"{description}\n")
        file.write(md_table)
        file.write("\n\n")


def create_bm_entry(bmName: str, appName: str, sku: str, result: str) -> dict[str, str]:
    entry_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    ubuntu = get_os_version()
    return {
        "jobId": entry_id,
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
        "appVersion": "string",
    }


def post_benchmark_entry(entry: dict[str, str], url: str) -> tuple[str, str]:
    json_data = json.dumps(entry)
    curl_command = ["curl", "-X", "POST", url, "-H", "Content-Type: application/json", "-d", json_data]

    result = subprocess.run(curl_command, capture_output=True, text=True)
    return result.stdout, result.stderr


BABELSTREAM_OPS = ("Copy", "Mul", "Add", "Triad", "Dot")


def parse_babelstream_output(raw_output: str) -> list[list[str]]:
    results = []
    for line in raw_output.strip().split("\n"):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] in BABELSTREAM_OPS:
            results.append([tokens[0], tokens[1]])
    return results


def aggregate_babelstream_runs(buffer: list[list[list[str]]], divisor: float) -> dict[str, dict[str, float]]:
    """Compute min/max/mean across BabelStream runs, scaled by *divisor*.

    Each element of *buffer* is the output of ``parse_babelstream_output``.
    Returns ``{op_name: {"min": ..., "max": ..., "mean": ...}}``.
    """
    import statistics

    ops: dict[str, list[float]] = {name: [] for name in BABELSTREAM_OPS}
    for log in buffer:
        if len(log) < 5:
            continue
        for idx, name in enumerate(BABELSTREAM_OPS):
            ops[name].append(float(log[idx][1]))
    summary: dict[str, dict[str, float]] = {}
    for name in BABELSTREAM_OPS:
        values = ops[name]
        if values:
            summary[name] = {
                "min": round(min(values) / divisor, 2),
                "max": round(max(values) / divisor, 2),
                "mean": round(statistics.mean(values) / divisor, 2),
            }
    return summary


def summarize_babelstream(
    buffer: list[list[list[str]]], *, divisor: float, units: str, title: str, description: str
) -> PrettyTable | None:
    """Build a PrettyTable from BabelStream run buffers.

    Returns the table, or None if no valid data was collected.
    """
    warnings.warn(
        "summarize_babelstream is deprecated; use infra.process for structured CSV output",
        DeprecationWarning,
        stacklevel=2,
    )
    import statistics

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
