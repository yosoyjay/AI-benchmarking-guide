"""NV Bandwidth benchmark (NVIDIA)."""

import logging
import os

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_cmd

logger = logging.getLogger(__name__)

_NVBANDWIDTH_REPO = "https://github.com/NVIDIA/nvbandwidth"
_NVBANDWIDTH_COMMIT = "66746a3bef61c8c2e12ab34955310da70b9e38cb"

TEST_NAMES = [
    "device_to_host_memcpy_ce",
    "host_to_device_memcpy_ce",
    "device_to_device_bidirectional_memcpy_read_ce",
]
_LABELS = [
    "Device to Host memcpy",
    "Host to Device memcpy",
    "Device to Device Bidirectional memcpy Total",
]


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def parse_sections(text: str) -> dict[str, str]:
    """Split nvbandwidth output into {test_name: section_text} dict."""
    sections = {}
    current_name = None
    current_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        found = None
        for name in TEST_NAMES:
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


def extract_summary_table(section_text: str) -> list[list[str | float]]:
    """Extract numeric table rows from a single section."""
    table_rows = []
    for line in section_text.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            if table_rows:
                break
            continue
        if "memcpy" in stripped or stripped.startswith("running") or stripped.startswith("SUM"):
            continue
        tokens = stripped.split()
        row = [round(float(x), 1) if x.replace(".", "", 1).isdigit() else x for x in tokens]
        table_rows.append(row)
    return table_rows


def _build_tables(text: str) -> list[tuple[str, PrettyTable]]:
    """Parse output and return list of (label, PrettyTable) pairs."""
    sections = parse_sections(text)
    tables = []
    for name, label in zip(TEST_NAMES, _LABELS):
        if name not in sections:
            logger.warning("section '%s' not found in nvbandwidth output", name)
            continue
        raw = extract_summary_table(sections[name])
        if not raw:
            continue
        raw[0].insert(0, " ")
        t = PrettyTable(raw[0])
        for row in raw[1:]:
            t.add_row(row)
        tables.append((label, t))
    return tables


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build(work_dir: str) -> None:
    """Clone and build nvbandwidth."""
    repo_dir = os.path.join(work_dir, "nvbandwidth")
    if not os.path.isdir(repo_dir):
        tools.run_cmd(["git", "clone", _NVBANDWIDTH_REPO, "nvbandwidth"], cwd=work_dir)
        tools.run_cmd(["git", "checkout", _NVBANDWIDTH_COMMIT], cwd=repo_dir)
        tools.run_cmd(
            ["sed", "-i", r"2i\set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)", "CMakeLists.txt"],
            cwd=repo_dir,
        )

    if os.path.exists("/.dockerenv"):
        tools.run_cmd(["apt", "update"], cwd=repo_dir)
        tools.run_cmd(["./debian_install.sh"], cwd=repo_dir)
    else:
        tools.run_cmd(["sudo", "apt", "update"], cwd=repo_dir)
        tools.run_cmd(["sudo", "./debian_install.sh"], cwd=repo_dir)


def run(work_dir: str, machine_name: str, ctx: RunContext | None = None) -> list[tuple[str, PrettyTable]] | None:
    """Clone, build, run nvbandwidth, parse and report results."""
    _build(work_dir)

    repo_dir = os.path.join(work_dir, "nvbandwidth")
    logger.info("Running NVBandwidth...")
    cmd = [
        "./nvbandwidth",
        "-t",
        "device_to_host_memcpy_ce",
        "host_to_device_memcpy_ce",
        "device_to_device_bidirectional_memcpy_read_ce",
    ]
    if ctx is not None:
        result = capture_cmd(cmd, ctx=ctx, cwd=repo_dir)
    else:
        result = tools.run_cmd(cmd, cwd=repo_dir)
    text = result.stdout.decode("utf-8")

    tables = _build_tables(text)

    if ctx is not None:
        return tables

    for i, (label, table) in enumerate(tables):
        print(label)
        print(table)
        if i == 0:
            tools.export_markdown("NV Bandwidth", label, table)
        else:
            tools.export_markdown(None, label, table)
    return None
