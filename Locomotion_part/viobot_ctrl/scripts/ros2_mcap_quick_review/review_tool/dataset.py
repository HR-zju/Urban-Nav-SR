from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml
from .io_utils import atomic_write_dataframe, stable_id

DATASET_COLUMNS = [
    "bag_id", "bag_path", "storage_id", "cache_dir", "status", "error",
    "camera_topic", "gps_topic", "vio_topic", "time_source", "duration_sec", "updated_at",
]

def _storage_id_from_metadata(metadata_path: Path) -> str:
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        info = data.get("rosbag2_bagfile_information", data)
        return str(info.get("storage_identifier", "") or "")
    except Exception:
        return ""

def discover_bags(
    data_root: str | Path,
    include_standalone_mcap: bool = True,
    include_standalone_db3: bool = True,
):
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    bag_dirs = [p.parent.resolve() for p in root.rglob("metadata.yaml")]
    discovered = []
    for bag_dir in sorted(set(bag_dirs)):
        storage_id = _storage_id_from_metadata(bag_dir / "metadata.yaml")
        has_mcap = any(bag_dir.glob("*.mcap"))
        # Python 3.8 compatibility build intentionally supports MCAP only.
        if storage_id == "mcap" or has_mcap:
            discovered.append((bag_dir, "mcap"))
    bag_dir_set = set(bag_dirs)
    if include_standalone_mcap:
        for mcap in root.rglob("*.mcap"):
            resolved = mcap.resolve()
            if any(parent in bag_dir_set for parent in resolved.parents):
                continue
            discovered.append((resolved, "mcap"))
    # SQLite3 rosbag2 (.db3) is intentionally not scanned in this build.
    del include_standalone_db3
    unique = {str(path): (path, storage) for path, storage in discovered}
    return [unique[k] for k in sorted(unique)]

def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=DATASET_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in DATASET_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[DATASET_COLUMNS]

def scan_to_dataset(cfg: dict) -> pd.DataFrame:
    work_dir = Path(cfg["paths"]["work_dir"])
    dataset_path = work_dir / "dataset.csv"
    existing = load_dataset(dataset_path)
    existing_by_id = {row["bag_id"]: row.to_dict() for _, row in existing.iterrows()}
    rows = []
    for bag_path, storage_id in discover_bags(
        cfg["paths"]["data_root"],
        bool(cfg["scan"].get("include_standalone_mcap", True)),
        bool(cfg["scan"].get("include_standalone_db3", True)),
    ):
        bag_id = stable_id(bag_path)
        cache_dir = work_dir / "cache" / bag_id
        old = existing_by_id.get(bag_id, {})
        row = {col: old.get(col, "") for col in DATASET_COLUMNS}
        row.update({
            "bag_id": bag_id, "bag_path": str(bag_path),
            "storage_id": storage_id, "cache_dir": str(cache_dir),
        })
        if not row["status"]:
            row["status"] = "pending"
        if (cache_dir / "review.mp4").exists() and row["status"] != "error":
            row["status"] = "ready"
        rows.append(row)
    df = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    atomic_write_dataframe(dataset_path, df)
    return df
