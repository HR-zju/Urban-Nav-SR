from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from filelock import FileLock
from .io_utils import atomic_write_dataframe

ANNOTATION_COLUMNS = [
    "bag_id", "bag_path", "label", "bad_reason", "comment", "reviewed_at",
]

def load_annotations(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[ANNOTATION_COLUMNS]

def save_annotation(path, bag_id, bag_path, label, bad_reason="", comment=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=10):
        df = load_annotations(path)
        row = {
            "bag_id": bag_id, "bag_path": bag_path, "label": label,
            "bad_reason": bad_reason if label == "Bad" else "",
            "comment": comment, "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        mask = df["bag_id"] == bag_id
        if mask.any():
            for key, value in row.items():
                df.loc[mask, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        atomic_write_dataframe(path, df)

def build_manifest(df: pd.DataFrame, label: str) -> str:
    paths = [str(x) for x in df[df["label"] == label]["bag_path"].tolist()]
    return "\n".join(paths) + ("\n" if paths else "")
