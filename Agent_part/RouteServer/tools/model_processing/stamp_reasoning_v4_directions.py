#!/usr/bin/env python3
"""Validate existing direction labels and add complete Reasoning V4 stamps.

This tool does not read a trajectory or derive a new direction.  It only
checks that the legacy index, name, choice, angle, source, and bin mode already
match one approved direction profile before adding the missing version fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    DirectionLabelStamp,
    canonical_direction_derivation_version,
    canonical_direction_source_codes,
    direction_source_requires_stop,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionSupervisionV4,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic


_SELECTION_MODES = (
    "sequential",
    "random",
    "coarse_direction",
    "interesting_diverse",
)
_COARSE_BUCKETS = ("straight", "left", "right", "stop")
_DEFAULT_SMOKE_RANDOM_SEED = 20260728


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse direction-stamping arguments.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument(
        "--profile-id",
        default=FPV_DEGREE10_TERMINAL_CENTROID_V2,
    )
    parser.add_argument(
        "--selection",
        choices=_SELECTION_MODES,
        default="sequential",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--interesting-jsonl",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=_DEFAULT_SMOKE_RANDOM_SEED,
    )
    parser.add_argument(
        "--consensus-min-votes",
        type=int,
        default=2,
    )
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args(argv)


def stamp_reasoning_v4_direction_row(
    row: Mapping[str, Any],
    *,
    profile_id: str,
    dataset_source: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Validate and stamp one pre-existing direction label.

    Args:
        row: Legacy or already stamped source row.
        profile_id: Exact approved profile expected for this manifest.
        dataset_source: Optional explicit V4 dataset family.

    Returns:
        A copied row containing ``direction_supervision`` and whether a new
        stamp was added.

    Raises:
        ValueError: If any existing direction field is missing, conflicts with
        the selected profile, or claims an unapproved derivation source.
    """

    profile = resolve_stage_a_direction_profile(profile_id)
    normalized_dataset = str(dataset_source or "").strip().lower()
    existing_dataset = str(
        row.get("dataset_source") or ""
    ).strip().lower()
    if (
        normalized_dataset
        and existing_dataset
        and existing_dataset != normalized_dataset
    ):
        raise ValueError(
            "existing dataset_source does not match requested dataset: "
            f"{existing_dataset!r}!={normalized_dataset!r}"
        )
    existing = row.get("direction_supervision")
    if existing is not None:
        if not isinstance(existing, Mapping):
            raise ValueError("direction_supervision must be an object")
        direction = DirectionSupervisionV4.model_validate(existing)
        if direction.profile_id != profile_id:
            raise ValueError(
                "existing direction profile does not match requested profile"
            )
        stamped = dict(row)
        stamped["direction_supervision"] = direction.model_dump(mode="json")
        if normalized_dataset:
            stamped["dataset_source"] = normalized_dataset
        return stamped, False

    conflicts = {
        "direction_profile": row.get("direction_profile"),
        "direction_profile_id": row.get("direction_profile_id"),
    }
    for field, value in conflicts.items():
        if value is not None and str(value).strip() != profile_id:
            raise ValueError(
                f"{field} conflicts with requested profile: {value}"
            )

    label_index = _required_value(
        row,
        "direction_choice_label",
        "direction_label_index",
    )
    label_name = _required_value(
        row,
        "direction_name",
        "direction_label_name",
    )
    choice = _required_value(row, "direction_choice")
    source = str(
        _required_value(row, "direction_source")
    ).strip()
    bin_mode = str(
        _required_value(row, "direction_bin_mode")
    ).strip()
    try:
        index = int(label_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"direction label index is not an integer: {label_index}"
        ) from exc
    labels = tuple(str(value) for value in profile["labels"])
    if index < 0 or index >= len(labels):
        raise ValueError(
            f"direction label index is outside profile: {index}"
        )
    expected_name = labels[index]
    if str(label_name).strip() != expected_name:
        raise ValueError(
            "legacy direction name/index mismatch: "
            f"index={index}, expected={expected_name}, actual={label_name}"
        )

    canonical_derivation = canonical_direction_derivation_version(profile_id)
    allowed_sources = set(
        canonical_direction_source_codes(
            profile_id,
            include_stop_rule=expected_name == "stop",
        )
    )
    if source not in allowed_sources:
        raise ValueError(
            "legacy direction source cannot justify the requested stamp: "
            f"label={expected_name}, source={source}"
        )
    if direction_source_requires_stop(source) and expected_name != "stop":
        raise ValueError(
            "legacy stop-only direction source requires label=stop: "
            f"label={expected_name}, source={source}"
        )

    angle = row.get("direction_angle_deg")
    if angle is not None:
        try:
            angle = float(angle)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"direction_angle_deg is not numeric: {angle}"
            ) from exc
    direction = DirectionLabelStamp(
        profile_id=profile_id,
        profile_version=str(profile["profile_version"]),
        label_index=index,
        label_name=str(label_name).strip(),
        choice=str(choice).strip(),
        angle_deg=angle,
        source=source,
        bin_mode=bin_mode,
        derivation_version=canonical_derivation,
    )
    supervision = DirectionSupervisionV4.model_validate(
        direction.model_dump(mode="json")
    )
    stamped = dict(row)
    stamped["direction_supervision"] = supervision.model_dump(mode="json")
    if normalized_dataset:
        stamped["dataset_source"] = normalized_dataset
    return stamped, True


def stamp_reasoning_v4_direction_manifest(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    report_json: str | Path,
    dataset_source: str | None = None,
    profile_id: str = FPV_DEGREE10_TERMINAL_CENTROID_V2,
    selection: str = "sequential",
    limit: int | None = None,
    interesting_jsonls: Sequence[str | Path] = (),
    random_seed: int = _DEFAULT_SMOKE_RANDOM_SEED,
    consensus_min_votes: int = 2,
    skip_errors: bool = False,
) -> dict[str, Any]:
    """Stamp one manifest while preserving an auditable conversion report.

    Args:
        input_jsonl: Legacy or already stamped source rows.
        output_jsonl: Atomically replaced stamped manifest.
        report_json: Atomically replaced JSON report.
        dataset_source: Optional explicit V4 dataset family written to rows.
        profile_id: Exact approved direction profile.
        selection: Sequential full pass, natural random pilot,
            coarse-direction smoke, or historical-interesting diverse smoke.
        limit: Optional maximum rows.  ``random`` requires an exact positive
            output size; ``sequential`` treats it as an input prefix.
        interesting_jsonls: Historical VLM-interesting manifests used only
            for diverse smoke selection.
        random_seed: Stable seed for deterministic smoke selection.
        consensus_min_votes: Number of historical manifests required for the
            consensus half of an interesting-diverse smoke.
        skip_errors: Continue past invalid rows when true.

    Returns:
        Conversion counts and bounded validation errors.

    Raises:
        ValueError: If a row is invalid and ``skip_errors`` is false.
    """

    profile = resolve_stage_a_direction_profile(profile_id)
    normalized_dataset = str(dataset_source or "").strip().lower()
    if selection not in _SELECTION_MODES:
        raise ValueError(f"unsupported selection mode: {selection}")
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive")
    if int(consensus_min_votes) <= 0:
        raise ValueError("consensus_min_votes must be positive")
    interesting_paths = [
        Path(value).expanduser() for value in interesting_jsonls
    ]
    if selection == "interesting_diverse" and not interesting_paths:
        raise ValueError(
            "interesting_diverse selection requires interesting_jsonls"
        )
    counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    selection_details: list[dict[str, Any]] = []

    def rows() -> Any:
        """Yield validated copied rows for atomic JSONL output."""

        if selection == "coarse_direction":
            yield from _coarse_direction_rows(
                input_jsonl,
                profile_id=profile_id,
                dataset_source=normalized_dataset,
                counts=counts,
                errors=errors,
                skip_errors=skip_errors,
                limit=limit,
            )
            return
        if selection == "random":
            yield from _random_rows(
                input_jsonl,
                profile_id=profile_id,
                dataset_source=normalized_dataset,
                counts=counts,
                errors=errors,
                skip_errors=skip_errors,
                limit=limit,
                random_seed=int(random_seed),
                selection_details=selection_details,
            )
            return
        if selection == "interesting_diverse":
            yield from _interesting_diverse_rows(
                input_jsonl,
                profile_id=profile_id,
                dataset_source=normalized_dataset,
                counts=counts,
                errors=errors,
                skip_errors=skip_errors,
                limit=limit,
                interesting_jsonls=interesting_paths,
                random_seed=int(random_seed),
                consensus_min_votes=int(consensus_min_votes),
                selection_details=selection_details,
            )
            return
        for index, row in enumerate(iter_jsonl(input_jsonl)):
            if limit is not None and index >= max(0, int(limit)):
                break
            counts["input_rows"] += 1
            try:
                stamped, added = stamp_reasoning_v4_direction_row(
                    row,
                    profile_id=profile_id,
                    dataset_source=normalized_dataset,
                )
            except (TypeError, ValueError) as exc:
                counts["invalid_rows"] += 1
                if len(errors) < 100:
                    errors.append(
                        {
                            "index": index,
                            "id": row.get("id"),
                            "error": str(exc),
                        }
                    )
                if not skip_errors:
                    raise
                continue
            counts["output_rows"] += 1
            counts[
                "newly_stamped_rows"
                if added
                else "already_stamped_rows"
            ] += 1
            yield stamped

    write_jsonl_atomic(output_jsonl, rows())
    report = {
        "schema_version": (
            "urbannav_reasoning_v4_direction_stamp_report_v1"
        ),
        "operation": "validate_existing_direction_and_add_stamp",
        "direction_rederived": False,
        "trajectory_read": False,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "dataset_source": normalized_dataset or None,
        "profile_id": profile_id,
        "profile_version": str(profile["profile_version"]),
        "bin_mode": str(profile["bin_mode"]),
        "derivation_version": (
            canonical_direction_derivation_version(profile_id)
        ),
        "selection": selection,
        "limit": limit,
        "random_seed": (
            int(random_seed)
            if selection in {"random", "interesting_diverse"}
            else None
        ),
        "consensus_min_votes": (
            int(consensus_min_votes)
            if selection == "interesting_diverse"
            else None
        ),
        "interesting_manifests": [
            {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
            for path in interesting_paths
        ],
        "selected_samples": selection_details,
        "counts": dict(counts),
        "errors": errors,
    }
    destination = Path(report_json).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report


def _random_rows(
    input_jsonl: str | Path,
    *,
    profile_id: str,
    dataset_source: str,
    counts: Counter[str],
    errors: list[dict[str, Any]],
    skip_errors: bool,
    limit: int | None,
    random_seed: int,
    selection_details: list[dict[str, Any]],
) -> Any:
    """Yield a seeded uniform sample over all valid input rows.

    Reservoir sampling scans the complete manifest and assigns every valid
    row the same inclusion probability.  It intentionally does not stratify
    by direction, clip, or historical-interesting membership, so pilot rates
    estimate the natural row distribution.

    Args:
        input_jsonl: Input manifest to scan completely.
        profile_id: Exact direction profile used for validation/stamping.
        dataset_source: Explicit dataset family written to selected rows.
        counts: Mutable conversion and selection counters.
        errors: Mutable bounded invalid-row records.
        skip_errors: Continue after invalid rows when true.
        limit: Exact requested output size.
        random_seed: Stable reservoir-sampling seed.
        selection_details: Mutable per-sample provenance records.

    Returns:
        An iterator of selected stamped rows in original input order.

    Raises:
        ValueError: If no exact positive limit is provided or the valid input
            contains fewer rows than requested.
    """

    if limit is None or int(limit) <= 0:
        raise ValueError("random selection requires a positive limit")
    requested = int(limit)
    rng = random.Random(int(random_seed))
    reservoir: list[tuple[int, dict[str, Any], bool]] = []
    valid_rows = 0
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        counts["input_rows"] += 1
        try:
            stamped, added = stamp_reasoning_v4_direction_row(
                row,
                profile_id=profile_id,
                dataset_source=dataset_source,
            )
        except (TypeError, ValueError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "index": index,
                        "id": row.get("id"),
                        "error": str(exc),
                    }
                )
            if not skip_errors:
                raise
            continue
        valid_rows += 1
        candidate = (index, stamped, added)
        if len(reservoir) < requested:
            reservoir.append(candidate)
            continue
        slot = rng.randrange(valid_rows)
        if slot < requested:
            reservoir[slot] = candidate
    counts["valid_input_rows"] = valid_rows
    if len(reservoir) != requested:
        raise ValueError(
            "random selection cannot satisfy requested rows: "
            f"{len(reservoir)}/{requested}"
        )

    selected_clips: set[str] = set()
    for input_index, stamped, added in sorted(
        reservoir,
        key=lambda item: item[0],
    ):
        counts["output_rows"] += 1
        counts[
            "newly_stamped_rows"
            if added
            else "already_stamped_rows"
        ] += 1
        counts["selection_group:natural_random"] += 1
        direction = stamped["direction_supervision"]
        label_name = str(direction["label_name"])
        counts[f"label:{label_name}"] += 1
        counts[f"source:{direction['source']}"] += 1
        try:
            clip_id: str | None = _sample_clip_id(stamped)
        except ValueError:
            clip_id = None
        if clip_id is not None:
            selected_clips.add(clip_id)
        selection_details.append(
            {
                "id": str(stamped["id"]),
                "clip_id": clip_id,
                "input_index": input_index,
                "direction_label": label_name,
                "coarse_bucket": _coarse_bucket(label_name),
                "selection_group": "natural_random",
                "interesting_votes": None,
            }
        )
        yield stamped
    counts["selected_unique_clips"] = len(selected_clips)


def _interesting_diverse_rows(
    input_jsonl: str | Path,
    *,
    profile_id: str,
    dataset_source: str,
    counts: Counter[str],
    errors: list[dict[str, Any]],
    skip_errors: bool,
    limit: int | None,
    interesting_jsonls: Sequence[Path],
    random_seed: int,
    consensus_min_votes: int,
    selection_details: list[dict[str, Any]],
) -> Any:
    """Yield a seeded interesting smoke with direction and clip diversity.

    Half of each coarse-direction quota is sampled from IDs marked interesting
    by at least ``consensus_min_votes`` historical manifests.  The remainder
    is sampled from the full union of historically interesting IDs.  A clip
    may contribute at most one row to the entire smoke.
    """

    requested = 16 if limit is None else int(limit)
    if requested < len(_COARSE_BUCKETS):
        raise ValueError(
            "interesting_diverse selection requires at least one row per "
            f"bucket: limit >= {len(_COARSE_BUCKETS)}"
        )
    if consensus_min_votes > len(interesting_jsonls):
        raise ValueError(
            "consensus_min_votes exceeds interesting manifest count"
        )
    votes = _load_interesting_votes(
        interesting_jsonls,
        counts=counts,
    )
    candidates: dict[
        str,
        list[tuple[dict[str, Any], bool, int]],
    ] = {bucket: [] for bucket in _COARSE_BUCKETS}
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        counts["input_rows"] += 1
        sample_id = str(row.get("id") or "").strip()
        vote_count = len(votes.get(sample_id, ()))
        if not sample_id or vote_count == 0:
            continue
        try:
            stamped, added = stamp_reasoning_v4_direction_row(
                row,
                profile_id=profile_id,
                dataset_source=dataset_source,
            )
            label_name = str(
                stamped["direction_supervision"]["label_name"]
            )
            bucket = _coarse_bucket(label_name)
            _sample_clip_id(stamped)
        except (TypeError, ValueError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "index": index,
                        "id": row.get("id"),
                        "error": str(exc),
                    }
                )
            if not skip_errors:
                raise
            continue
        candidates[bucket].append((stamped, added, vote_count))

    bucket_base, bucket_remainder = divmod(
        requested,
        len(_COARSE_BUCKETS),
    )
    quotas = {
        bucket: bucket_base + int(index < bucket_remainder)
        for index, bucket in enumerate(_COARSE_BUCKETS)
    }
    rng = random.Random(int(random_seed))
    selected: dict[
        str,
        list[tuple[dict[str, Any], bool, int, str]],
    ] = {bucket: [] for bucket in _COARSE_BUCKETS}
    used_ids: set[str] = set()
    used_clips: set[str] = set()
    for bucket in _COARSE_BUCKETS:
        quota = quotas[bucket]
        consensus_quota = quota // 2
        consensus = [
            candidate
            for candidate in candidates[bucket]
            if candidate[2] >= consensus_min_votes
        ]
        consensus.sort(key=lambda item: str(item[0].get("id") or ""))
        rng.shuffle(consensus)
        _take_diverse_candidates(
            consensus,
            selected=selected[bucket],
            requested=consensus_quota,
            selection_group="interesting_consensus",
            used_ids=used_ids,
            used_clips=used_clips,
        )
        if len(selected[bucket]) != consensus_quota:
            raise ValueError(
                "interesting-diverse consensus quota is incomplete: "
                f"{bucket}:{len(selected[bucket])}/{consensus_quota}"
            )
        remaining = quota - len(selected[bucket])
        union = sorted(
            candidates[bucket],
            key=lambda item: str(item[0].get("id") or ""),
        )
        rng.shuffle(union)
        _take_diverse_candidates(
            union,
            selected=selected[bucket],
            requested=remaining,
            selection_group="interesting_random",
            used_ids=used_ids,
            used_clips=used_clips,
        )
        if len(selected[bucket]) != quota:
            raise ValueError(
                "interesting-diverse bucket quota is incomplete: "
                f"{bucket}:{len(selected[bucket])}/{quota}"
            )

    for bucket in _COARSE_BUCKETS:
        for stamped, added, vote_count, selection_group in selected[
            bucket
        ]:
            counts["output_rows"] += 1
            counts[
                "newly_stamped_rows"
                if added
                else "already_stamped_rows"
            ] += 1
            counts[f"bucket:{bucket}"] += 1
            counts[f"selection_group:{selection_group}"] += 1
            counts[f"interesting_votes:{vote_count}"] += 1
            direction = stamped["direction_supervision"]
            counts[f"label:{direction['label_name']}"] += 1
            counts[f"source:{direction['source']}"] += 1
            selection_details.append(
                {
                    "id": str(stamped["id"]),
                    "clip_id": _sample_clip_id(stamped),
                    "direction_label": str(direction["label_name"]),
                    "coarse_bucket": bucket,
                    "selection_group": selection_group,
                    "interesting_votes": vote_count,
                }
            )
            yield stamped


def _load_interesting_votes(
    paths: Sequence[Path],
    *,
    counts: Counter[str],
) -> dict[str, set[int]]:
    """Load valid historical interesting identities with per-manifest votes."""

    votes: dict[str, set[int]] = {}
    for manifest_index, path in enumerate(paths):
        if not path.is_file():
            raise ValueError(f"interesting manifest is missing: {path}")
        manifest_ids: set[str] = set()
        for row in iter_jsonl(path):
            behavior = row.get("behavior_filter")
            if isinstance(behavior, Mapping) and (
                int(behavior.get("label") or 0) != 1
                or behavior.get("valid") is False
            ):
                continue
            sample_id = str(row.get("id") or "").strip()
            if sample_id:
                manifest_ids.add(sample_id)
        counts[
            f"interesting_manifest:{manifest_index}:unique_ids"
        ] = len(manifest_ids)
        for sample_id in manifest_ids:
            votes.setdefault(sample_id, set()).add(manifest_index)
    counts["interesting_union_ids"] = len(votes)
    return votes


def _take_diverse_candidates(
    candidates: Sequence[tuple[dict[str, Any], bool, int]],
    *,
    selected: list[tuple[dict[str, Any], bool, int, str]],
    requested: int,
    selection_group: str,
    used_ids: set[str],
    used_clips: set[str],
) -> None:
    """Append up to ``requested`` candidates with globally unique clips."""

    taken = 0
    for stamped, added, vote_count in candidates:
        if taken >= requested:
            break
        sample_id = str(stamped.get("id") or "").strip()
        clip_id = _sample_clip_id(stamped)
        if sample_id in used_ids or clip_id in used_clips:
            continue
        used_ids.add(sample_id)
        used_clips.add(clip_id)
        selected.append(
            (
                stamped,
                added,
                vote_count,
                selection_group,
            )
        )
        taken += 1


def _coarse_direction_rows(
    input_jsonl: str | Path,
    *,
    profile_id: str,
    dataset_source: str,
    counts: Counter[str],
    errors: list[dict[str, Any]],
    skip_errors: bool,
    limit: int | None,
) -> Any:
    """Yield a deterministic, near-balanced coarse-direction sample."""

    requested = 4 if limit is None else int(limit)
    if requested < len(_COARSE_BUCKETS):
        raise ValueError(
            "coarse_direction selection requires at least one row per "
            f"bucket: limit >= {len(_COARSE_BUCKETS)}"
        )
    base_quota, remainder = divmod(requested, len(_COARSE_BUCKETS))
    quotas = {
        bucket: base_quota + int(index < remainder)
        for index, bucket in enumerate(_COARSE_BUCKETS)
    }
    selected: dict[str, list[tuple[dict[str, Any], bool]]] = {
        bucket: [] for bucket in _COARSE_BUCKETS
    }
    for index, row in enumerate(iter_jsonl(input_jsonl)):
        counts["input_rows"] += 1
        try:
            stamped, added = stamp_reasoning_v4_direction_row(
                row,
                profile_id=profile_id,
                dataset_source=dataset_source,
            )
            label_name = str(
                stamped["direction_supervision"]["label_name"]
            )
            bucket = _coarse_bucket(label_name)
        except (TypeError, ValueError) as exc:
            counts["invalid_rows"] += 1
            if len(errors) < 100:
                errors.append(
                    {
                        "index": index,
                        "id": row.get("id"),
                        "error": str(exc),
                    }
                )
            if not skip_errors:
                raise
            continue
        if len(selected[bucket]) < quotas[bucket]:
            selected[bucket].append((stamped, added))
        if all(
            len(selected[bucket]) >= quotas[bucket]
            for bucket in _COARSE_BUCKETS
        ):
            break
    missing = [
        f"{bucket}:{len(selected[bucket])}/{quotas[bucket]}"
        for bucket in _COARSE_BUCKETS
        if len(selected[bucket]) < quotas[bucket]
    ]
    if missing:
        raise ValueError(
            "coarse-direction smoke bucket quotas are incomplete: "
            f"{missing}"
        )
    for bucket in _COARSE_BUCKETS:
        for stamped, added in selected[bucket]:
            counts["output_rows"] += 1
            counts[
                "newly_stamped_rows"
                if added
                else "already_stamped_rows"
            ] += 1
            direction = stamped["direction_supervision"]
            counts[f"label:{direction['label_name']}"] += 1
            counts[f"source:{direction['source']}"] += 1
            yield stamped


def _coarse_bucket(label_name: str) -> str:
    """Map one canonical label to a smoke-review direction bucket."""

    if label_name == "straight":
        return "straight"
    if label_name == "stop":
        return "stop"
    if label_name.startswith("left_"):
        return "left"
    if label_name.startswith("right_"):
        return "right"
    raise ValueError(f"unsupported direction label: {label_name!r}")


def _sample_clip_id(row: Mapping[str, Any]) -> str:
    """Return the stable clip identity required for diverse smoke sampling."""

    sidewalk_meta = row.get("sidewalk_meta")
    if not isinstance(sidewalk_meta, Mapping):
        sidewalk_meta = {}
    clip_id = str(
        row.get("clip_id")
        or sidewalk_meta.get("clip_id")
        or ""
    ).strip()
    if not clip_id:
        raise ValueError(
            f"sample is missing clip_id: {row.get('id')!r}"
        )
    return clip_id


def _file_sha256(path: Path) -> str:
    """Return the SHA256 of one selection-provenance manifest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_value(
    row: Mapping[str, Any],
    *names: str,
) -> Any:
    """Return the first present legacy value or fail closed."""

    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    raise ValueError(
        "legacy direction field is missing: "
        + "|".join(names)
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the direction-stamping CLI.

    Args:
        argv: Optional explicit CLI arguments.

    Returns:
        ``None``.
    """

    args = parse_args(argv)
    report = stamp_reasoning_v4_direction_manifest(
        args.input_jsonl,
        args.output_jsonl,
        report_json=args.report_json,
        dataset_source=args.dataset_source,
        profile_id=args.profile_id,
        selection=args.selection,
        limit=args.limit,
        interesting_jsonls=args.interesting_jsonl,
        random_seed=args.random_seed,
        consensus_min_votes=args.consensus_min_votes,
        skip_errors=args.skip_errors,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
