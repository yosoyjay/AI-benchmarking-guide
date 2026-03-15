"""Parse benchmark output, produce structured CSV, and display results."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from prettytable import PrettyTable

from infra.capture import RunContext

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "datetime",
    "version",
    "sku",
    "platform",
    "benchmark",
    "metric_name",
    "metric_value",
    "unit",
    "metadata_json",
]


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def write_processed_csv(
    rows: list[dict[str, str]],
    run_dir: Path,
    benchmark: str,
    version: str,
    timestamp: datetime,
) -> Path:
    """Write long-format CSV rows to ``run_dir/processed/<benchmark>_<version>_<ts>.csv``.

    Each dict in *rows* must contain all keys from :data:`CSV_COLUMNS`.
    Returns the path to the written file.
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    csv_path = run_dir / "processed" / f"{benchmark}_{version}_{ts_str}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def append_combined(rows: list[dict[str, str]], combined_path: Path) -> None:
    """Append long-format rows to *combined_path*, writing a header if the file is new."""
    write_header = not combined_path.exists() or combined_path.stat().st_size == 0
    with open(combined_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Row builder helper
# ---------------------------------------------------------------------------


def _row(
    ctx: RunContext, metric_name: str, metric_value: str | int | float, unit: str, metadata: dict[str, Any]
) -> dict[str, str]:
    """Build a single long-format CSV row dict from context + metric."""
    return {
        "datetime": ctx.timestamp.isoformat(),
        "version": ctx.version,
        "sku": ctx.sku,
        "platform": ctx.platform,
        "benchmark": ctx.benchmark,
        "metric_name": metric_name,
        "metric_value": str(metric_value),
        "unit": unit,
        "metadata_json": json.dumps(metadata, separators=(",", ":")),
    }


# ---------------------------------------------------------------------------
# to_csv_rows converters -- one per benchmark
# ---------------------------------------------------------------------------


def gemm_cublas_lt_to_csv(ctx: RunContext, parsed: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert gemm_cublas_lt parsed rows to long-format CSV rows."""
    rows = []
    datatype = ctx.extra.get("datatype", "")
    for r in parsed:
        meta = {"m": r["m"], "n": r["n"], "k": r["k"], "batch_size": r["batch_size"], "datatype": datatype}
        rows.append(_row(ctx, "tflops", r["tflops"], "TFLOPS", meta))
        rows.append(_row(ctx, "time_us", r["time_us"], "us", meta))
    return rows


def gemm_hipblas_lt_to_csv(ctx: RunContext, parsed: list[dict[str, str | float]]) -> list[dict[str, str]]:
    """Convert gemm_hipblas_lt parsed rows to long-format CSV rows."""
    rows = []
    datatype = ctx.extra.get("datatype", "FP8")
    for r in parsed:
        meta = {"m": r["m"], "n": r["n"], "k": r["k"], "datatype": datatype}
        rows.append(_row(ctx, "tflops", r["tflops"], "TFLOPS", meta))
    return rows


def nccl_bandwidth_to_csv(ctx: RunContext, parsed: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert nccl_bandwidth parsed rows to long-format CSV rows."""
    algo = ctx.extra.get("algorithm", "")
    rows = []
    for r in parsed:
        meta = {"message_size": r["size"], "algorithm": algo}
        rows.append(_row(ctx, "bandwidth", r["bandwidth"], "GB/s", meta))
    return rows


def rccl_bandwidth_to_csv(ctx: RunContext, parsed: list[dict[str, str]], algo: str) -> list[dict[str, str]]:
    """Convert rccl_bandwidth parsed rows for one algorithm to long-format CSV rows."""
    rows = []
    for r in parsed:
        meta = {"message_size": r["size"], "algorithm": algo}
        rows.append(_row(ctx, "bandwidth", r["bandwidth"], "GB/s", meta))
    return rows


def flash_attention_to_csv(ctx: RunContext, parsed: list[dict[str, str | int | float]]) -> list[dict[str, str]]:
    """Convert flash_attention parsed rows to long-format CSV rows."""
    rows = []
    for r in parsed:
        meta = {"causal": r["causal"], "headdim": r["headdim"]}
        rows.append(_row(ctx, "flash2_tflops", r["flash2_tflops"], "TFLOPS", meta))
        rows.append(_row(ctx, "pytorch_tflops", r["pytorch_tflops"], "TFLOPS", meta))
    return rows


def hbm_bandwidth_to_csv(ctx: RunContext, summary: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    """Convert BabelStream summary (min/max/mean per operation) to CSV rows.

    *summary* is ``{op_name: {"min": v, "max": v, "mean": v}}``.
    """
    rows = []
    for op_name, stats in summary.items():
        meta = {"operation": op_name}
        rows.append(_row(ctx, "bw_min", stats["min"], ctx.extra.get("units", "TB/s"), meta))
        rows.append(_row(ctx, "bw_max", stats["max"], ctx.extra.get("units", "TB/s"), meta))
        rows.append(_row(ctx, "bw_mean", stats["mean"], ctx.extra.get("units", "TB/s"), meta))
    return rows


def cpu_stream_to_csv(ctx: RunContext, summary: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    """Convert CPU BabelStream summary to CSV rows. Same shape as hbm_bandwidth."""
    rows = []
    for op_name, stats in summary.items():
        meta = {"operation": op_name}
        rows.append(_row(ctx, "bw_min", stats["min"], "GB/s", meta))
        rows.append(_row(ctx, "bw_max", stats["max"], "GB/s", meta))
        rows.append(_row(ctx, "bw_mean", stats["mean"], "GB/s", meta))
    return rows


def nv_bandwidth_to_csv(ctx: RunContext, tables: list[tuple[str, PrettyTable]]) -> list[dict[str, str]]:
    """Convert nv_bandwidth (label, PrettyTable) pairs to CSV rows.

    Each table has columns like ``[" ", "GPU0", "GPU1", ...]`` with numeric
    cells.  The first column of data rows contains src labels (e.g. GPU ids).
    """
    from benchmarks.nvidia.nv_bandwidth import _LABELS, TEST_NAMES

    rows = []
    for label, table in tables:
        test_name = ""
        for tn, lb in zip(TEST_NAMES, _LABELS):
            if lb == label:
                test_name = tn
                break
        for row_data in table.rows:
            src = str(row_data[0])
            for col_idx, dst_name in enumerate(table.field_names[1:], 1):
                meta = {"test_name": test_name, "src": src, "dst": str(dst_name)}
                rows.append(_row(ctx, "bandwidth", row_data[col_idx], "GB/s", meta))
    return rows


def multichase_to_csv(
    ctx: RunContext, node_names: list[str], parsed_rows: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Convert multichase parsed rows to long-format CSV rows."""
    rows = []
    for r in parsed_rows:
        for node in node_names:
            if node in r:
                meta = {"cpu": r["cpu"], "node": node}
                rows.append(_row(ctx, "latency", r[node], "ns", meta))
    return rows


def fio_to_csv(ctx: RunContext, parsed: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    """Convert fio parsed rows (rw, bs, bw) to long-format CSV rows."""
    rows = []
    for rw, bs, bw in parsed:
        meta = {"rw": rw, "block_size": bs}
        rows.append(_row(ctx, "bandwidth", bw, "", meta))
    return rows


def transfer_bench_to_csv(ctx: RunContext, parsed: dict[str, str]) -> list[dict[str, str]]:
    """Convert transfer_bench parsed dict to long-format CSV rows."""
    rows = []
    for direction, bw in parsed.items():
        meta = {"direction": direction}
        rows.append(_row(ctx, "bandwidth", bw, "GB/s", meta))
    return rows


def llm_benchmark_nv_to_csv(ctx: RunContext, parsed: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert NVIDIA LLM benchmark parsed rows to long-format CSV rows."""
    rows = []
    model = ctx.extra.get("model", "")
    for r in parsed:
        meta = {"model": model, "input_len": r["input_len"], "output_len": r["output_len"], "tp_size": r["tp_size"]}
        rows.append(_row(ctx, "throughput", r["throughput"], "tokens/s", meta))
    return rows


def llm_benchmark_amd_to_csv(ctx: RunContext, parsed: list[tuple[str, str, str, str]]) -> list[dict[str, str]]:
    """Convert AMD LLM benchmark parsed rows to long-format CSV rows.

    *parsed* is list of (input_len, output_len, tp_size, throughput) tuples.
    """
    rows = []
    model = ctx.extra.get("model", "")
    for input_len, output_len, tp_size, throughput in parsed:
        meta = {"model": model, "input_len": input_len, "output_len": output_len, "tp_size": tp_size}
        rows.append(_row(ctx, "throughput", throughput, "tokens/s", meta))
    return rows


def llama3_pretrain_to_csv(ctx: RunContext, time_ss: float | None, loss_ss: float | None) -> list[dict[str, str]]:
    """Convert LLAMA3 pretraining steady-state results to CSV rows."""
    rows = []
    model_size = ctx.extra.get("model_size", "")
    meta = {"model_size": model_size}
    if time_ss is not None:
        rows.append(_row(ctx, "time_steady_state", f"{time_ss:.4f}", "s", meta))
    if loss_ss is not None:
        rows.append(_row(ctx, "loss_steady_state", f"{loss_ss:.4f}", "", meta))
    return rows


# ---------------------------------------------------------------------------
# process_run pipeline
# ---------------------------------------------------------------------------


def process_run(
    benchmark: str,
    ctx: RunContext,
    csv_rows: list[dict[str, str]],
) -> Path | None:
    """Write per-benchmark CSV, append to combined, return processed CSV path.

    Callers are responsible for parsing raw output and calling the
    appropriate ``*_to_csv`` converter to produce *csv_rows*.
    """
    if not csv_rows:
        logger.warning("No CSV rows for %s, skipping CSV output", benchmark)
        return None

    csv_path = write_processed_csv(csv_rows, ctx.run_dir, benchmark, ctx.version, ctx.timestamp)
    combined_path = ctx.results_dir / "combined.csv"
    append_combined(csv_rows, combined_path)
    return csv_path
