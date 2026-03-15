"""GEMM HipBLASLt benchmark (AMD ROCm, Docker-based)."""

import logging

from prettytable import PrettyTable

from infra import tools
from infra.containers import AmdContainer

logger = logging.getLogger(__name__)

_HIPBLAS_IMAGE = "ai-bench/amd-hipblas:latest"

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


def run(work_dir, machine_name, ctx=None):
    """Run HipBLASLt GEMM inside Docker, parse and report."""
    with AmdContainer(_HIPBLAS_IMAGE, work_dir, entrypoint="/bin/bash") as container:
        logger.info("Running HipBLAS...")
        bench_bin = "/opt/hipBLASLt/build/release/clients/staging/hipblaslt-bench"
        rows = []
        for m, n, k in zip(_M_DIMS, _N_DIMS, _K_DIMS):
            yaml_cfg = _build_hipblas_yaml(m, n, k)
            cmd = f'{bench_bin} --device 0 --flush --yaml - <<< "{yaml_cfg}"' f' | grep -B 1 "T,N,0"'
            if ctx is not None:
                from infra.capture import capture_docker

                stdout, stderr, exit_code = capture_docker(
                    container, ["/bin/bash", "-c", cmd], ctx=ctx, suffix=f"_m{m}_n{n}_k{k}"
                )
                output = stdout
            else:
                res = container.exec_run(["/bin/bash", "-c", cmd])
                output = res.output.decode("utf-8")
                tools.write_log(output)
            rows.extend(parse_hipblas_results(output))

    if ctx is not None:
        ctx.extra["datatype"] = _DATATYPE
        return rows

    table = _build_table(rows)
    print(table)
    tools.export_markdown(
        "GEMM HipBLASLt",
        f"The results shown below are with random initialization (best representation of real-life workloads) {_DATATYPE}, and {_WARMUP} warmup iterations.",
        table,
    )
