"""Parse benchmark output, produce structured CSV, and display results."""

import csv
import logging
from datetime import datetime
from pathlib import Path

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


def write_processed_csv(
    rows: list[dict],
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


def append_combined(rows: list[dict], combined_path: Path) -> None:
    """Append long-format rows to *combined_path*, writing a header if the file is new."""
    write_header = not combined_path.exists() or combined_path.stat().st_size == 0
    with open(combined_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
