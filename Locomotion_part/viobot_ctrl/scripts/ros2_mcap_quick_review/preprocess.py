from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from review_tool.config import load_config
from review_tool.dataset import load_dataset
from review_tool.io_utils import atomic_write_dataframe
from review_tool.processor import process_bag

def _worker(row, cfg, force):
    return process_bag(row, cfg, force=force)

def _apply_result(df: pd.DataFrame, result: dict):
    mask = df["bag_id"] == result["bag_id"]
    for key, value in result.items():
        if key != "bag_id" and key in df.columns:
            df.loc[mask, key] = str(value)
    df.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser(description="Generate synchronized composite review videos.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_path = Path(cfg["paths"]["work_dir"]) / "dataset.csv"
    df = load_dataset(dataset_path)
    if df.empty:
        raise SystemExit("dataset.csv is empty. Run scan_files.py first.")

    candidates = df.copy()
    if not args.force:
        # Ready rows are cheap signature checks. Including them makes input or
        # config changes rebuild automatically instead of requiring --force.
        allowed = {"pending", "ready"} | ({"error"} if args.retry_errors else set())
        candidates = candidates[candidates["status"].isin(allowed)]
        priority = candidates["status"].map({"pending": 0, "error": 1, "ready": 2})
        candidates = candidates.assign(_priority=priority).sort_values(
            "_priority", kind="stable"
        ).drop(columns=["_priority"])
    if args.limit > 0:
        candidates = candidates.head(args.limit)
    rows = [row.to_dict() for _, row in candidates.iterrows()]
    print(f"Preprocessing {len(rows)} bag(s) with {args.workers} worker(s)")

    if args.workers <= 1:
        for idx, row in enumerate(rows, 1):
            result = _worker(row, cfg, args.force)
            _apply_result(df, result)
            atomic_write_dataframe(dataset_path, df)
            print(f"[{idx}/{len(rows)}] {row['bag_path']} -> {result['status']} {result.get('error', '')}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, row, cfg, args.force): row for row in rows}
            for idx, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "bag_id": row["bag_id"], "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _apply_result(df, result)
                atomic_write_dataframe(dataset_path, df)
                print(f"[{idx}/{len(rows)}] {row['bag_path']} -> {result['status']} {result.get('error', '')}")

if __name__ == "__main__":
    main()
