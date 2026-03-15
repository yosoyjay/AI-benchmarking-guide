"""GEMM HipBLASLt benchmark (AMD ROCm, Docker-based)."""

import logging

from prettytable import PrettyTable

from infra import tools
from infra.capture import RunContext, capture_docker
from infra.containers import AmdContainer

logger = logging.getLogger(__name__)

_HIPBLAS_IMAGE = "ai-bench/amd-hipblas:latest"

_DEFAULT_M_DIMS = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 6144, 802816]
_DEFAULT_N_DIMS = [1024, 2048, 4096, 8192, 16384, 32768, 2145, 12288, 192]
_DEFAULT_K_DIMS = [1024, 2048, 4096, 8192, 16384, 32768, 1024, 12288, 768]


# ---------------------------------------------------------------------------
# Pure helpers -- no side effects, fully testable
# ---------------------------------------------------------------------------


def _build_hipblas_yaml(m: int, n: int, k: int, iters: int = 2000, cold_iters: int = 100) -> str:
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
        f"iters: {iters}, cold_iters: {cold_iters}, "
        f"initialization: trig_float,rotating: 512}}"
    )


def parse_hipblas_results(text: str) -> list[dict[str, str | float]]:
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


def _build_table(rows: list[dict[str, str | float]]) -> PrettyTable:
    """Format parsed row dicts into a PrettyTable."""
    table = PrettyTable(["M", "N", "K", "TFLOPS"])
    for r in rows:
        table.add_row([r["m"], r["n"], r["k"], r["tflops"]])
    return table


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_DEFAULT_DATATYPE = "FP8"
_DEFAULT_WARMUP = 10000


def run(
    work_dir: str, machine_name: str, config_path: str = "config.json", ctx: RunContext | None = None
) -> list[dict[str, str | float]] | None:
    """Run HipBLASLt GEMM inside Docker, parse and report."""
    config = tools.load_benchmark_config(config_path, "GEMMHipBLAS")
    m_dims = config.get("m_dims", _DEFAULT_M_DIMS)
    n_dims = config.get("n_dims", _DEFAULT_N_DIMS)
    k_dims = config.get("k_dims", _DEFAULT_K_DIMS)
    datatype = config.get("datatype", _DEFAULT_DATATYPE)
    warmup = config.get("warmup", _DEFAULT_WARMUP)
    iters = config.get("iters", 2000)
    cold_iters = config.get("cold_iters", 100)

    with AmdContainer(_HIPBLAS_IMAGE, work_dir, entrypoint="/bin/bash") as container:
        logger.info("Running HipBLAS...")
        bench_bin = "/opt/hipBLASLt/build/release/clients/staging/hipblaslt-bench"
        rows = []
        for m, n, k in zip(m_dims, n_dims, k_dims):
            yaml_cfg = _build_hipblas_yaml(m, n, k, iters=iters, cold_iters=cold_iters)
            cmd = f'{bench_bin} --device 0 --flush --yaml - <<< "{yaml_cfg}"' f' | grep -B 1 "T,N,0"'
            if ctx is not None:
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
        ctx.extra["datatype"] = datatype
        return rows

    table = _build_table(rows)
    print(table)
    tools.export_markdown(
        "GEMM HipBLASLt",
        f"The results shown below are with random initialization (best representation of real-life workloads) {datatype}, and {warmup} warmup iterations.",
        table,
    )
    return None
