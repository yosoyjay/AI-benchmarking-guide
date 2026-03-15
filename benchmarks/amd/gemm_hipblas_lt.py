"""GEMM HipBLASLt benchmark (AMD ROCm, Docker-based)."""

import logging
import os

from prettytable import PrettyTable

from infra import tools
from infra.containers import AmdContainer

logger = logging.getLogger(__name__)

_HIPBLAS_IMAGE = "rocm/vllm-dev:main"
_HIPBLASLT_REPO = "https://github.com/ROCm/hipBLASLt"
_HIPBLASLT_COMMIT = "a11ccf64efcd818106dbe37768f69dfcc0a7ff22"

_M_DIMS = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
_N_DIMS = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
_K_DIMS = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def _build_hipblas_yaml(m, n, k):
    """Construct the YAML config line for hipblaslt-bench.

    Produces the single-line YAML that hipblaslt-bench expects via
    ``--yaml -``, with lda=k, ldb=k, ldc=m, ldd=m.
    """
    return (
        f"- {{function: matmul, transA: T, transB: N, "
        f"a_type: f8_r, b_type: f8_r, c_type: f16_r, d_type: f16_r, "
        f"compute_type: c_f32_r, "
        f"M: {m}, N: {n}, K: {k}, lda: {k}, ldb: {k}, ldc: {m}, ldd: {m}, "
        f"alpha: 1, beta: 0, scale_type: f32_r, "
        f"iters: 2000, cold_iters: 100, "
        f"initialization: trig_float,rotating: 512}}"
    )


def parse_hipblas_results(text):
    """Parse HipBLASLt results file content into row dicts.

    Each result line starts with ``"T"`` and is comma-delimited.
    Extracts M (field 4), N (field 5), K (field 6), and TFLOPS
    (``field[-3] / 1000``).  Returns a list of dicts.
    """
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] != "T":
            continue
        fields = stripped.split(",")
        if len(fields) < 7:
            logger.warning("unexpected format in GEMMHipBLAS results: %s", stripped)
            continue
        try:
            tflops = float(fields[-3]) / 1000
        except (ValueError, IndexError):
            logger.warning("cannot parse TFLOPS from: %s", stripped)
            continue
        rows.append(
            {
                "m": fields[4],
                "n": fields[5],
                "k": fields[6],
                "tflops": tflops,
            }
        )
    return rows


def _build_table(rows):
    """Format parsed row dicts into a PrettyTable."""
    table = PrettyTable(["M", "N", "K", "TFLOPS"])
    for r in rows:
        table.add_row([r["m"], r["n"], r["k"], r["tflops"]])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_DATATYPE = "FP8"
_WARMUP = 10000


def _build(container, work_dir):
    """Clone hipBLASLt, install deps, and build inside the container."""
    repo_dir = os.path.join(work_dir, "hipBLASLt")
    if not os.path.isdir(repo_dir):
        res = container.exec_run(
            ["git", "clone", _HIPBLASLT_REPO, repo_dir],
            stderr=True,
        )
        tools.write_log(res.output.decode("utf-8"))

        res = container.exec_run(
            ["git", "checkout", _HIPBLASLT_COMMIT],
            workdir=repo_dir,
            stderr=True,
        )
        if res.exit_code != 0:
            tools.write_log(res.output.decode("utf-8"))
            return

        res = container.exec_run(["sudo", "apt-get", "-y", "update"], stderr=True)
        tools.write_log(res.output.decode("utf-8"))

        res = container.exec_run(["sudo", "apt", "-y", "install", "llvm-dev"], stderr=True)
        tools.write_log(res.output.decode("utf-8"))

        logger.info("Building hipBLAS Library...")
        res = container.exec_run(
            ["/bin/sh", "-c", f"cd {repo_dir} && ./install.sh -dc -a gfx942"],
            stderr=True,
        )
        tools.write_log(res.output.decode("utf-8"))


def run(work_dir, machine_name):
    """Clone, build, run HipBLASLt GEMM inside Docker, parse and report."""
    with AmdContainer(_HIPBLAS_IMAGE, work_dir, entrypoint="/bin/bash") as container:
        _build(container, work_dir)

        logger.info("Running HipBLAS...")
        bench_bin = os.path.join(work_dir, "hipBLASLt", "build", "release", "clients", "staging", "hipblaslt-bench")
        rows = []
        for m, n, k in zip(_M_DIMS, _N_DIMS, _K_DIMS):
            yaml_cfg = _build_hipblas_yaml(m, n, k)
            cmd = f'{bench_bin} --device 0 --flush --yaml - <<< "{yaml_cfg}"' f' | grep -B 1 "T,N,0"'
            res = container.exec_run(["/bin/bash", "-c", cmd])
            output = res.output.decode("utf-8")
            tools.write_log(output)
            rows.extend(parse_hipblas_results(output))

    table = _build_table(rows)
    print(table)
    tools.export_markdown(
        "GEMM HipBLASLt",
        f"The results shown below are with random initialization (best representation of real-life workloads) {_DATATYPE}, and {_WARMUP} warmup iterations.",
        table,
    )
