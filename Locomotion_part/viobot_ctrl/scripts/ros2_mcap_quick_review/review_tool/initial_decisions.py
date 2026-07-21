from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


INITIAL_COLUMNS = [
    "initial_record_id",
    "initial_match_status",
    "initial_manual_decision",
    "initial_auto_recommendation",
    "initial_issues",
    "initial_source_name",
]

MATCHED = "matched"
_RECORD_ID_RE = re.compile(r"rec_([^_.]+)")


def extract_record_id(value: str | Path) -> str:
    """Return the token after the final ``rec_`` in a path or file name."""
    name = Path(str(value)).name
    matches = _RECORD_ID_RE.findall(name)
    return matches[-1] if matches else ""


def load_initial_decisions(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Initial decisions CSV does not exist: {path}")

    # utf-8-sig accepts both BOM and plain UTF-8 files.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"name", "manual_decision", "auto_recommendation", "issues"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Initial decisions CSV is missing column(s): " + ", ".join(missing)
        )

    result = pd.DataFrame({
        "initial_record_id": df["name"].map(extract_record_id),
        "initial_manual_decision": df["manual_decision"].str.strip().str.upper(),
        "initial_auto_recommendation": (
            df["auto_recommendation"].str.strip().str.upper()
        ),
        "initial_issues": df["issues"],
        "initial_source_name": df["name"],
    })
    result["initial_match_status"] = "available"

    missing_id = result["initial_record_id"] == ""
    duplicate_id = (
        result["initial_record_id"].ne("")
        & result["initial_record_id"].duplicated(keep=False)
    )
    result.loc[missing_id, "initial_match_status"] = "no_record_id"
    result.loc[duplicate_id, "initial_match_status"] = "decision_id_ambiguous"
    return result[INITIAL_COLUMNS]


def _record_ids_in_bag_path(value: str | Path) -> list[str]:
    path = Path(str(value))
    direct_id = extract_record_id(path.name)
    if direct_id:
        return [direct_id]
    if not path.is_dir():
        return []
    ids = {
        record_id
        for child in path.glob("*.mcap")
        for record_id in [extract_record_id(child.name)]
        if record_id
    }
    return sorted(ids)


def attach_initial_decisions(
    dataset: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach first-pass decisions in memory without changing dataset.csv."""
    result = dataset.copy()
    for column in INITIAL_COLUMNS:
        result[column] = ""

    bag_ids = result["bag_path"].map(_record_ids_in_bag_path)
    result["initial_record_id"] = bag_ids.map(
        lambda values: values[0] if len(values) == 1 else ""
    )
    result["initial_match_status"] = bag_ids.map(
        lambda values: "dataset_id_ambiguous" if len(values) > 1 else (
            "no_record_id" if not values else "pending"
        )
    )

    dataset_duplicate = (
        result["initial_record_id"].ne("")
        & result["initial_record_id"].duplicated(keep=False)
    )
    result.loc[dataset_duplicate, "initial_match_status"] = "dataset_id_ambiguous"

    usable = decisions[decisions["initial_match_status"] == "available"]
    decision_map = {
        row["initial_record_id"]: row.to_dict()
        for _, row in usable.iterrows()
    }
    ambiguous_decision_ids = set(
        decisions.loc[
            decisions["initial_match_status"] == "decision_id_ambiguous",
            "initial_record_id",
        ]
    )

    for index, row in result.iterrows():
        if row["initial_match_status"] != "pending":
            continue
        record_id = row["initial_record_id"]
        if record_id in ambiguous_decision_ids:
            result.at[index, "initial_match_status"] = "decision_id_ambiguous"
            continue
        initial = decision_map.get(record_id)
        if initial is None:
            result.at[index, "initial_match_status"] = "not_in_decisions"
            continue
        result.at[index, "initial_match_status"] = MATCHED
        for column in (
            "initial_manual_decision",
            "initial_auto_recommendation",
            "initial_issues",
            "initial_source_name",
        ):
            result.at[index, column] = initial[column]
    return result


def is_selected_manual_decision(
    dataset: pd.DataFrame,
    selected_values: Iterable[str],
) -> pd.Series:
    selected = {str(value).strip().upper() for value in selected_values}
    return (
        dataset["initial_match_status"].eq(MATCHED)
        & dataset["initial_manual_decision"].isin(selected)
    )
