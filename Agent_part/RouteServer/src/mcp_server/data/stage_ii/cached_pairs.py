"""Materialize causal cached-query/action pairs from validated Stage II views."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from mcp_server.contracts.stage_i_sft import (
    StageIFullCoTViewV2,
    StageIReasoningDropoutViewV2,
    parse_stage_i_training_view_v2,
)
from mcp_server.contracts.stage_ii_cfm import (
    StageIICFMTrainingViewV2,
    StageIICachedTrajQuerySourceViewV2,
    StageIICachedTrajQueryTrainingViewV1,
    StageIITrajectoryTargetV2,
    StageIITrajectorySidecarV2,
    stage_ii_digest_v1,
)
from mcp_server.data.io.jsonl import write_jsonl_atomic
from mcp_server.data.stage_ii.sources import (
    load_stage_ii_trajectory_source_registry_v1,
    stage_ii_source_row_identity_v1,
)
from mcp_server.data.stage_ii.views import (
    build_stage_ii_cached_traj_query_source_view_v2,
    build_stage_ii_cached_traj_query_training_view_v1,
    build_stage_ii_cached_traj_query_training_view_v2,
)


CACHED_PAIR_MATERIALIZATION_REPORT_VERSION = (
    "urbannav_stage_ii_cached_traj_query_materialization_v1"
)
CACHED_PAIR_REJECT_SCHEMA_VERSION = "urbannav_stage_ii_cached_traj_query_reject_v1"
CACHED_PAIR_MATERIALIZATION_REPORT_VERSION_V2 = (
    "urbannav_stage_ii_cached_traj_query_materialization_v2"
)
CACHED_SOURCE_VIEW_MATERIALIZATION_REPORT_VERSION_V2 = (
    "urbannav_stage_ii_cached_traj_query_source_materialization_v2"
)
CACHED_SOURCE_VIEW_REJECT_SCHEMA_VERSION_V2 = (
    "urbannav_stage_ii_cached_traj_query_source_reject_v2"
)


class CachedPairMaterializationError(ValueError):
    """Describe one expected source/timing rejection with a stable reason code.

    Args:
        code: Stable lower-snake-case reject reason.
        detail: Bounded diagnostic for the reject report.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def build_stage_ii_cached_traj_query_pairs_v1(
    *,
    stage_ii_views_jsonl: str | Path,
    dataset_manifests: Mapping[str, Sequence[str | Path]],
    trajectory_source_registry_json: str | Path,
    output_pairs_jsonl: str | Path,
    output_reject_jsonl: str | Path,
    output_report_json: str | Path,
    query_period_sec: float,
    max_query_age_sec: float,
    progress_interval: int = 10000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    r"""Build causal temporal pairs using the existing Stage II V1 builder.

    Query observations are refreshed on the first action, after a visible
    condition change, or after ``query_period_sec``.  For action time ``t_a``
    the selected query time satisfies:

    \\[
    t_q = \max\{t \mid t \leq t_a,\; t \text{ is a selected refresh}\}.
    \\]

    Args:
        stage_ii_views_jsonl: Validated ``StageIICFMTrainingViewV2`` rows.
        dataset_manifests: Dataset name to one or more raw source manifests.
        trajectory_source_registry_json: Frozen source identity registry.
        output_pairs_jsonl: Atomic destination for accepted pair views.
        output_reject_jsonl: Atomic destination for expected row rejects.
        output_report_json: Atomic materialization report destination.
        query_period_sec: Requested slow-path refresh period.
        max_query_age_sec: Maximum permitted action-time query age.
        progress_interval: Progress callback interval in input view rows.
        progress_callback: Optional progress sink.

    Returns:
        A deterministic count, fingerprint, and query-age report.

    Raises:
        ValueError: If a canonical view, source registry, identity, or global
            timing invariant is invalid.
    """

    period = _positive_finite(query_period_sec, "query_period_sec")
    max_age = _positive_finite(max_query_age_sec, "max_query_age_sec")
    if period > max_age:
        raise ValueError("query_period_sec cannot exceed max_query_age_sec")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    output_pairs = Path(output_pairs_jsonl).expanduser()
    output_rejects = Path(output_reject_jsonl).expanduser()
    output_report = Path(output_report_json).expanduser()
    for destination in (output_pairs, output_rejects, output_report):
        destination.parent.mkdir(parents=True, exist_ok=True)

    registry = load_stage_ii_trajectory_source_registry_v1(
        trajectory_source_registry_json
    )
    counters: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    source_digests: dict[str, str] = {}
    age_values: list[float] = []

    with tempfile.TemporaryDirectory(
        prefix=".urbannav_cached_pairs_",
        dir=str(output_pairs.parent),
    ) as temp_dir:
        database = Path(temp_dir) / "index.sqlite3"
        connection = sqlite3.connect(database)
        try:
            _create_index(connection)
            _index_stage_ii_views(
                connection,
                stage_ii_views_jsonl,
                input_digest=input_digest,
                counters=counters,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
            if counters["input_views"] == 0:
                raise ValueError("stage_ii_views_jsonl contains no input views")
            _index_source_manifests(
                connection,
                registry,
                dataset_manifests,
                source_digests=source_digests,
                counters=counters,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
            connection.commit()

            pair_rows = _iter_pairs(
                connection,
                period=period,
                max_age=max_age,
                counters=counters,
                age_values=age_values,
            )
            written_pairs = write_jsonl_atomic(output_pairs, pair_rows)
            reject_rows = _iter_reject_rows(connection)
            written_rejects = write_jsonl_atomic(output_rejects, reject_rows)
            connection.commit()
        finally:
            connection.close()

    if written_pairs + written_rejects != counters["input_views"]:
        raise ValueError(
            "cached pair materialization did not account for every input view: "
            f"input={counters['input_views']} pairs={written_pairs} rejects={written_rejects}"
        )

    age_report = _age_report(age_values)
    report = {
        "schema_version": CACHED_PAIR_MATERIALIZATION_REPORT_VERSION,
        "inputs": {
            "stage_ii_views_jsonl": str(Path(stage_ii_views_jsonl).expanduser()),
            "trajectory_source_registry_json": str(
                Path(trajectory_source_registry_json).expanduser()
            ),
            "dataset_manifests": {
                dataset: [str(Path(path).expanduser()) for path in paths]
                for dataset, paths in dataset_manifests.items()
            },
        },
        "config": {
            "query_period_sec": period,
            "max_query_age_sec": max_age,
            "progress_interval": int(progress_interval),
        },
        "fingerprints": {
            "stage_ii_views_sha256": input_digest.hexdigest(),
            "source_manifest_sha256": source_digests,
        },
        "counts": dict(counters),
        "query_age_sec": age_report,
        "outputs": {
            "pairs_jsonl": str(output_pairs),
            "reject_jsonl": str(output_rejects),
            "report_json": str(output_report),
        },
        "complete": True,
    }
    _write_json_atomic(output_report, report)
    return report


def build_stage_ii_cached_traj_query_pairs_v2(
    *,
    source_views_jsonl: str | Path,
    dataset_manifests: Mapping[str, Sequence[str | Path]],
    trajectory_source_registry_json: str | Path,
    output_pairs_jsonl: str | Path,
    output_reject_jsonl: str | Path,
    output_report_json: str | Path,
    query_period_sec: float,
    max_query_age_sec: float,
    progress_interval: int = 10000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build pure-latent V2 pairs from visible source views and sidecars.

    ``source_views_jsonl`` contains
    ``StageIICachedTrajQuerySourceViewV2`` rows.  No Pass-R or Pass-D cache is
    parsed while selecting the slow query; only visible-condition identity,
    source timing, and the action-side trajectory target are used.
    """

    period = _positive_finite(query_period_sec, "query_period_sec")
    max_age = _positive_finite(max_query_age_sec, "max_query_age_sec")
    if period > max_age:
        raise ValueError("query_period_sec cannot exceed max_query_age_sec")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    output_pairs = Path(output_pairs_jsonl).expanduser()
    output_rejects = Path(output_reject_jsonl).expanduser()
    output_report = Path(output_report_json).expanduser()
    for destination in (output_pairs, output_rejects, output_report):
        destination.parent.mkdir(parents=True, exist_ok=True)

    registry = load_stage_ii_trajectory_source_registry_v1(
        trajectory_source_registry_json
    )
    counters: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    source_digests: dict[str, str] = {}
    age_values: list[float] = []

    with tempfile.TemporaryDirectory(
        prefix=".urbannav_cached_pairs_v2_",
        dir=str(output_pairs.parent),
    ) as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "index.sqlite3")
        try:
            _create_index(connection)
            _index_cached_source_views_v2(
                connection,
                source_views_jsonl,
                input_digest=input_digest,
                counters=counters,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
            if counters["input_views"] == 0:
                raise ValueError("source_views_jsonl contains no input views")
            _index_source_manifests(
                connection,
                registry,
                dataset_manifests,
                source_digests=source_digests,
                counters=counters,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
            connection.commit()
            pair_rows = _iter_pairs_v2(
                connection,
                period=period,
                max_age=max_age,
                counters=counters,
                age_values=age_values,
            )
            written_pairs = write_jsonl_atomic(output_pairs, pair_rows)
            written_rejects = write_jsonl_atomic(
                output_rejects, _iter_reject_rows(connection)
            )
        finally:
            connection.close()

    if written_pairs + written_rejects != counters["input_views"]:
        raise ValueError(
            "cached V2 pair materialization did not account for every source view: "
            f"input={counters['input_views']} pairs={written_pairs} rejects={written_rejects}"
        )
    report = {
        "schema_version": CACHED_PAIR_MATERIALIZATION_REPORT_VERSION_V2,
        "inputs": {
            "source_views_jsonl": str(Path(source_views_jsonl).expanduser()),
            "trajectory_source_registry_json": str(
                Path(trajectory_source_registry_json).expanduser()
            ),
            "dataset_manifests": {
                dataset: [str(Path(path).expanduser()) for path in paths]
                for dataset, paths in dataset_manifests.items()
            },
        },
        "config": {
            "condition_mode": "cached_traj_query_v2",
            "query_period_sec": period,
            "max_query_age_sec": max_age,
            "progress_interval": int(progress_interval),
        },
        "fingerprints": {
            "source_views_sha256": input_digest.hexdigest(),
            "source_manifest_sha256": source_digests,
        },
        "counts": dict(counters),
        "query_age_sec": _age_report(age_values),
        "outputs": {
            "pairs_jsonl": str(output_pairs),
            "reject_jsonl": str(output_rejects),
            "report_json": str(output_report),
        },
        "complete": True,
    }
    _write_json_atomic(output_report, report)
    return report


def build_stage_ii_cached_traj_query_source_views_v2(
    *,
    stage_i_views_jsonl: str | Path,
    trajectory_sidecars_jsonl: Sequence[str | Path],
    dataset_manifests: Mapping[str, Sequence[str | Path]],
    trajectory_source_registry_json: str | Path,
    output_source_views_jsonl: str | Path,
    output_reject_jsonl: str | Path,
    output_report_json: str | Path,
    limit: int | None = None,
    progress_interval: int = 10000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Bind Stage-I visible rows to accepted V2 trajectory sidecars.

    The output is the only input needed by the V2 temporal pair materializer.
    This step does not read or create Pass-R/Pass-D text caches.
    """

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    output_source_views = Path(output_source_views_jsonl).expanduser()
    output_rejects = Path(output_reject_jsonl).expanduser()
    output_report = Path(output_report_json).expanduser()
    for destination in (output_source_views, output_rejects, output_report):
        destination.parent.mkdir(parents=True, exist_ok=True)

    registry = load_stage_ii_trajectory_source_registry_v1(
        trajectory_source_registry_json
    )
    counters: Counter[str] = Counter()
    stage_i_digest = hashlib.sha256()
    sidecar_digests: dict[str, str] = {}
    source_digests: dict[str, str] = {}

    with tempfile.TemporaryDirectory(
        prefix=".urbannav_cached_source_views_v2_",
        dir=str(output_source_views.parent),
    ) as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "index.sqlite3")
        try:
            _create_index(connection)
            _index_trajectory_sidecars_v2(
                connection,
                trajectory_sidecars_jsonl,
                sidecar_digests=sidecar_digests,
                counters=counters,
            )
            _index_source_manifests(
                connection,
                registry,
                dataset_manifests,
                source_digests=source_digests,
                counters=counters,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
            connection.commit()
            source_rows = _iter_cached_source_views_v2(
                connection,
                stage_i_views_jsonl,
                stage_i_digest=stage_i_digest,
                counters=counters,
                limit=limit,
                progress_interval=progress_interval,
                progress_callback=progress_callback,
            )
            written_sources = write_jsonl_atomic(output_source_views, source_rows)
            written_rejects = write_jsonl_atomic(
                output_rejects, _iter_reject_rows(connection)
            )
        finally:
            connection.close()

    if written_sources + written_rejects != counters["input_views"]:
        raise ValueError(
            "cached source-view materialization did not account for every Stage-I row: "
            f"input={counters['input_views']} accepted={written_sources} "
            f"rejected={written_rejects}"
        )
    report = {
        "schema_version": CACHED_SOURCE_VIEW_MATERIALIZATION_REPORT_VERSION_V2,
        "inputs": {
            "stage_i_views_jsonl": str(Path(stage_i_views_jsonl).expanduser()),
            "trajectory_sidecars_jsonl": [
                str(Path(path).expanduser()) for path in trajectory_sidecars_jsonl
            ],
            "trajectory_source_registry_json": str(
                Path(trajectory_source_registry_json).expanduser()
            ),
            "dataset_manifests": {
                dataset: [str(Path(path).expanduser()) for path in paths]
                for dataset, paths in dataset_manifests.items()
            },
        },
        "config": {
            "condition_mode": "cached_traj_query_v2",
            "limit": limit,
            "progress_interval": int(progress_interval),
        },
        "fingerprints": {
            "stage_i_views_sha256": stage_i_digest.hexdigest(),
            "trajectory_sidecar_sha256": sidecar_digests,
            "source_manifest_sha256": source_digests,
        },
        "counts": dict(counters),
        "outputs": {
            "source_views_jsonl": str(output_source_views),
            "reject_jsonl": str(output_rejects),
            "report_json": str(output_report),
        },
        "complete": True,
    }
    _write_json_atomic(output_report, report)
    return report


def _create_index(connection: sqlite3.Connection) -> None:
    """Create the compact temporary timing/view index."""

    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.executescript(
        """
        CREATE TABLE views (
            sample_id TEXT PRIMARY KEY,
            source_group_id TEXT NOT NULL,
            split_group_id TEXT NOT NULL,
            dataset_source TEXT NOT NULL,
            context_mode TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE source_timing (
            source_group_id TEXT PRIMARY KEY,
            split_group_id TEXT NOT NULL,
            dataset_source TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE trajectory_sidecars (
            source_group_id TEXT PRIMARY KEY,
            split_group_id TEXT NOT NULL,
            dataset_source TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE frame_times (
            dataset_source TEXT NOT NULL,
            clip_id TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            timestamp_raw INTEGER NOT NULL,
            PRIMARY KEY (dataset_source, clip_id, frame_idx)
        );
        CREATE TABLE rejects (
            sequence_id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL
        );
        CREATE INDEX views_source_group_idx ON views(source_group_id);
        CREATE INDEX views_order_idx ON views(dataset_source, split_group_id, source_group_id);
        CREATE INDEX source_split_idx ON source_timing(dataset_source, split_group_id);
        """
    )


def _index_stage_ii_views(
    connection: sqlite3.Connection,
    path: str | Path,
    *,
    input_digest: Any,
    counters: Counter[str],
    progress_interval: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    """Validate and index Stage II views without retaining their JSON in RAM."""

    with Path(path).expanduser().open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            input_digest.update(line)
            if not line.strip():
                continue
            raw = _json_object(line, path, line_number)
            view = StageIICFMTrainingViewV2.model_validate(raw)
            if view.system1_condition_mode != "traj_query_only":
                raise ValueError(
                    f"cached query materialization requires traj_query_only: {view.sample_id}"
                )
            payload = json.dumps(
                view.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            duplicate = connection.execute(
                "SELECT payload FROM views WHERE sample_id = ?", (view.sample_id,)
            ).fetchone()
            if duplicate is not None:
                if duplicate[0] != payload:
                    raise ValueError(f"conflicting duplicate Stage II view: {view.sample_id}")
                counters["equivalent_duplicate_views"] += 1
                continue
            connection.execute(
                "INSERT INTO views VALUES (?, ?, ?, ?, ?, ?)",
                (
                    view.sample_id,
                    view.source_group_id,
                    view.split_group_id,
                    view.dataset_source,
                    view.context_mode,
                    payload,
                ),
            )
            counters["input_views"] += 1
            counters[f"input_dataset:{view.dataset_source}"] += 1
            if counters["input_views"] % 1000 == 0:
                connection.commit()
            _emit_progress(
                progress_callback,
                counters["input_views"],
                progress_interval,
                "index_views",
                counters,
            )


def _index_cached_source_views_v2(
    connection: sqlite3.Connection,
    path: str | Path,
    *,
    input_digest: Any,
    counters: Counter[str],
    progress_interval: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    """Index pure-latent V2 source views without retaining them in RAM."""

    with Path(path).expanduser().open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            input_digest.update(line)
            if not line.strip():
                continue
            raw = _json_object(line, path, line_number)
            view = StageIICachedTrajQuerySourceViewV2.model_validate(raw)
            payload = json.dumps(
                view.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            duplicate = connection.execute(
                "SELECT payload FROM views WHERE sample_id = ?", (view.sample_id,)
            ).fetchone()
            if duplicate is not None:
                if duplicate[0] != payload:
                    raise ValueError(f"conflicting duplicate cached V2 source: {view.sample_id}")
                counters["equivalent_duplicate_views"] += 1
                continue
            connection.execute(
                "INSERT INTO views VALUES (?, ?, ?, ?, ?, ?)",
                (
                    view.sample_id,
                    view.source_group_id,
                    view.split_group_id,
                    view.dataset_source,
                    "cached_traj_query_v2",
                    payload,
                ),
            )
            counters["input_views"] += 1
            counters[f"input_dataset:{view.dataset_source}"] += 1
            if counters["input_views"] % 1000 == 0:
                connection.commit()
            _emit_progress(
                progress_callback,
                counters["input_views"],
                progress_interval,
                "index_v2_sources",
                counters,
            )


def _index_trajectory_sidecars_v2(
    connection: sqlite3.Connection,
    paths: Sequence[str | Path],
    *,
    sidecar_digests: dict[str, str],
    counters: Counter[str],
) -> None:
    """Index accepted V2 sidecars by their stable source-group identity."""

    if not paths:
        raise ValueError("trajectory_sidecars_jsonl must contain at least one path")
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                digest.update(line)
                if not line.strip():
                    continue
                sidecar = StageIITrajectorySidecarV2.model_validate(
                    _json_object(line, path, line_number)
                )
                payload = json.dumps(
                    sidecar.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                existing = connection.execute(
                    "SELECT payload FROM trajectory_sidecars "
                    "WHERE source_group_id = ?",
                    (sidecar.source_group_id,),
                ).fetchone()
                if existing is not None:
                    if existing[0] != payload:
                        raise ValueError(
                            "conflicting trajectory sidecars: "
                            f"{sidecar.source_group_id}"
                        )
                    counters["equivalent_duplicate_sidecars"] += 1
                    continue
                connection.execute(
                    "INSERT INTO trajectory_sidecars VALUES (?, ?, ?, ?)",
                    (
                        sidecar.source_group_id,
                        sidecar.split_group_id,
                        sidecar.dataset_source,
                        payload,
                    ),
                )
                counters["sidecar_rows"] += 1
        sidecar_digests[str(path)] = digest.hexdigest()
    if counters["sidecar_rows"] == 0:
        raise ValueError("trajectory_sidecars_jsonl contains no sidecars")


def _iter_cached_source_views_v2(
    connection: sqlite3.Connection,
    path: str | Path,
    *,
    stage_i_digest: Any,
    counters: Counter[str],
    limit: int | None,
    progress_interval: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> Iterator[dict[str, Any]]:
    """Join Stage-I V2 rows to sidecars and emit target-private source views."""

    seen_source_groups: set[str] = set()
    with Path(path).expanduser().open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and counters["input_views"] >= limit:
                break
            stage_i_digest.update(line)
            if not line.strip():
                continue
            raw = _json_object(line, path, line_number)
            counters["input_views"] += 1
            dataset_source = str(raw.get("dataset_source") or "unknown")
            counters[f"input_dataset:{dataset_source}"] += 1

            typed = parse_stage_i_training_view_v2(raw)
            if not isinstance(typed, (StageIFullCoTViewV2, StageIReasoningDropoutViewV2)):
                _record_raw_source_reject(
                    connection,
                    raw,
                    "stage_i_task_not_reasoning",
                    f"task_type={typed.task_type}",
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue

            if typed.source_group_id in seen_source_groups:
                _record_raw_source_reject(
                    connection,
                    raw,
                    "duplicate_stage_i_source_group",
                    f"source_group_id={typed.source_group_id}",
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue

            sidecar_row = connection.execute(
                "SELECT split_group_id, dataset_source, payload "
                "FROM trajectory_sidecars WHERE source_group_id = ?",
                (typed.source_group_id,),
            ).fetchone()
            if sidecar_row is None:
                _record_raw_source_reject(
                    connection,
                    raw,
                    "trajectory_sidecar_missing",
                    f"source_group_id={typed.source_group_id}",
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue

            sidecar = StageIITrajectorySidecarV2.model_validate(
                json.loads(sidecar_row[2])
            )
            if sidecar.dataset_source != typed.dataset_source:
                _record_raw_source_reject(
                    connection,
                    raw,
                    "trajectory_dataset_mismatch",
                    f"stage_i={typed.dataset_source} sidecar={sidecar.dataset_source}",
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue
            if _dataset_qualified_split_group(
                typed.dataset_source, typed.split_group_id
            ) != _dataset_qualified_split_group(
                sidecar.dataset_source, sidecar.split_group_id
            ):
                _record_raw_source_reject(
                    connection,
                    raw,
                    "trajectory_split_group_mismatch",
                    f"stage_i={typed.split_group_id} sidecar={sidecar.split_group_id}",
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue

            timing_row = connection.execute(
                "SELECT payload FROM source_timing WHERE source_group_id = ?",
                (typed.source_group_id,),
            ).fetchone()
            if timing_row is None:
                _record_raw_source_reject(
                    connection,
                    raw,
                    "source_timing_missing",
                    f"source_group_id={typed.source_group_id}",
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue

            timing = json.loads(timing_row[0])
            if sidecar.source_manifest_sha256 not in timing["source_manifest_sha256s"]:
                _record_raw_source_reject(
                    connection,
                    raw,
                    "source_manifest_mismatch",
                    (
                        f"stage_i={sidecar.source_manifest_sha256} "
                        f"source={timing['source_manifest_sha256s']}"
                    ),
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue

            try:
                frame_timestamps, observation_timestamp, bev_timestamp = (
                    _resolve_source_role_timestamps(
                        connection,
                        image_roles=typed.image_roles,
                        raw_timestamp_unit=(
                            sidecar.trajectory_target.policy.raw_timestamp_unit
                        ),
                        timing=timing,
                        sample_id=typed.sample_id,
                    )
                )
                source_view = build_stage_ii_cached_traj_query_source_view_v2(
                    typed,
                    trajectory_target=sidecar.trajectory_target,
                    observation_timestamp_sec=observation_timestamp,
                    frame_timestamps_sec=frame_timestamps,
                    bev_timestamp_sec=bev_timestamp,
                    provenance={
                        "trajectory_sidecar_digest": sidecar.sidecar_digest,
                        "trajectory_source_group_id": sidecar.source_group_id,
                        "source_manifest_name": sidecar.source_manifest_name,
                        "source_manifest_sha256": sidecar.source_manifest_sha256,
                        "source_line": sidecar.source_line,
                    },
                )
            except CachedPairMaterializationError as exc:
                _record_raw_source_reject(
                    connection,
                    raw,
                    exc.code,
                    str(exc),
                    counters,
                )
                _emit_progress(
                    progress_callback,
                    counters["input_views"],
                    progress_interval,
                    "materialize_source_views",
                    counters,
                )
                continue

            seen_source_groups.add(typed.source_group_id)
            counters["accepted_source_views"] += 1
            counters[f"accepted_dataset:{typed.dataset_source}"] += 1
            yield source_view.model_dump(mode="json", exclude_none=True)
            _emit_progress(
                progress_callback,
                counters["input_views"],
                progress_interval,
                "materialize_source_views",
                counters,
            )


def _dataset_qualified_split_group(dataset_source: str, split_group_id: str) -> str:
    """Normalize native and dataset-qualified split-group identities."""

    prefix = f"{dataset_source}:"
    return split_group_id if split_group_id.startswith(prefix) else prefix + split_group_id


def _resolve_source_role_timestamps(
    connection: sqlite3.Connection,
    *,
    image_roles: Sequence[str],
    raw_timestamp_unit: str,
    timing: Mapping[str, Any],
    sample_id: str,
) -> tuple[list[float], float, float | None]:
    """Resolve source clocks in the exact order declared by Stage-I roles."""

    scale = {"nanosecond": 1e-9, "millisecond": 1e-3}[raw_timestamp_unit]
    if timing["kind"] == "multiframe":
        fpv_raw = list(timing["fpv_timestamps_raw"])
    else:
        fpv_raw = []
        for index in timing["frame_indices"]:
            row = connection.execute(
                "SELECT timestamp_raw FROM frame_times "
                "WHERE dataset_source = ? AND clip_id = ? AND frame_idx = ?",
                (timing["dataset_source"], timing["clip_id"], int(index)),
            ).fetchone()
            if row is None:
                raise CachedPairMaterializationError(
                    "egowalk_history_timestamp_missing",
                    f"clip={timing['clip_id']} frame={index}",
                )
            fpv_raw.append(int(row[0]))
    fpv_roles = [role for role in image_roles if role.startswith("fpv.")]
    if len(fpv_raw) < len(fpv_roles):
        raise CachedPairMaterializationError(
            "fpv_timestamp_count_mismatch",
            f"roles={len(fpv_roles)} timestamps={len(fpv_raw)}",
        )
    fpv_raw = fpv_raw[-len(fpv_roles):]
    observation_sec = float(timing["observation_raw"]) * scale
    fpv_seconds = [float(value) * scale for value in fpv_raw]
    if any(value > observation_sec + 1e-6 for value in fpv_seconds):
        raise CachedPairMaterializationError(
            "future_fpv_timestamp",
            f"sample_id={sample_id}",
        )
    if abs(float(fpv_raw[-1]) * scale - observation_sec) > 1e-6:
        raise CachedPairMaterializationError(
            "current_fpv_timestamp_mismatch", sample_id
        )
    bev_roles = [role for role in image_roles if role == "bev.current"]
    if len(bev_roles) > 1:
        raise CachedPairMaterializationError("duplicate_bev_current", sample_id)
    bev_sec: float | None = None
    if bev_roles:
        raw_bev = timing.get("bev_timestamp_raw")
        if raw_bev is None:
            raise CachedPairMaterializationError("bev_timestamp_missing", sample_id)
        bev_sec = float(raw_bev) * scale
        if bev_sec > observation_sec + 1e-6:
            raise CachedPairMaterializationError(
                "future_bev_timestamp",
                f"sample_id={sample_id}",
            )
    timestamps: list[float] = []
    fpv_index = 0
    for role in image_roles:
        if role.startswith("fpv."):
            timestamps.append(fpv_seconds[fpv_index])
            fpv_index += 1
        elif role == "bev.current":
            assert bev_sec is not None
            timestamps.append(bev_sec)
        else:
            raise CachedPairMaterializationError(
                "image_role_timestamp_unsupported", role
            )
    return timestamps, observation_sec, bev_sec


def _index_source_manifests(
    connection: sqlite3.Connection,
    registry: Any,
    dataset_manifests: Mapping[str, Sequence[str | Path]],
    *,
    source_digests: dict[str, str],
    counters: Counter[str],
    progress_interval: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None,
) -> None:
    """Index only compact timing payloads from the configured raw sources."""

    for dataset, paths in dataset_manifests.items():
        spec = registry.dataset(dataset)
        if spec.status != "enabled":
            raise ValueError(f"cached pair source dataset is disabled: {dataset}")
        for source_path in paths:
            path = Path(source_path).expanduser()
            manifest_sha256 = _sha256_file(path)
            with path.open("rb") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    raw = _json_object(line, path, line_number)
                    source_group_id, split_group_id = stage_ii_source_row_identity_v1(raw, spec)
                    timing = _source_timing(
                        raw,
                        dataset,
                        source_group_id,
                        split_group_id,
                        source_manifest_sha256=manifest_sha256,
                    )
                    _insert_source_timing(connection, timing, counters)
                    counters[f"source_rows:{dataset}"] += 1
                    _emit_progress(
                        progress_callback,
                        counters[f"source_rows:{dataset}"],
                        progress_interval,
                        f"index_source:{dataset}",
                        counters,
                    )
            source_digests[str(path)] = manifest_sha256
        connection.commit()


def _source_timing(
    row: Mapping[str, Any],
    dataset: str,
    source_group_id: str,
    split_group_id: str,
    *,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Extract explicit source clocks for the supported dataset contracts."""

    if dataset in {"sidewalk", "foxglove"}:
        metadata = row.get("sidewalk_meta")
        if not isinstance(metadata, Mapping):
            raise CachedPairMaterializationError(
                "sidewalk_meta_missing", f"source_group={source_group_id}"
            )
        frames = metadata.get("multiframe_frames")
        if not isinstance(frames, list) or not frames:
            raise CachedPairMaterializationError(
                "multiframe_timestamps_missing", f"source_group={source_group_id}"
            )
        fpv_timestamps = [
            _raw_timestamp(item.get("timestamp"), "multiframe frame timestamp")
            for item in frames
            if isinstance(item, Mapping)
        ]
        if len(fpv_timestamps) != len(frames):
            raise CachedPairMaterializationError(
                "multiframe_timestamp_invalid", f"source_group={source_group_id}"
            )
        observation = _raw_timestamp(
            metadata.get("timestamp", fpv_timestamps[-1]),
            "observation timestamp",
        )
        bev = metadata.get("bev_timestamp")
        return {
            "kind": "multiframe",
            "observation_raw": observation,
            "fpv_timestamps_raw": fpv_timestamps,
            "bev_timestamp_raw": (
                _raw_timestamp(bev, "BEV timestamp") if bev is not None else None
            ),
            "source_group_id": source_group_id,
            "split_group_id": split_group_id,
            "dataset_source": dataset,
            "source_manifest_sha256s": [source_manifest_sha256],
        }
    if dataset == "egowalk":
        clip_id = str(row.get("clip_id") or "")
        frame_value = row.get("frame_idx")
        history = row.get("history_frame_indices")
        if not clip_id or not isinstance(frame_value, int) or not isinstance(history, list):
            raise CachedPairMaterializationError(
                "egowalk_timeline_missing", f"source_group={source_group_id}"
            )
        frame_indices = [int(item) for item in history] + [int(frame_value)]
        return {
            "kind": "egowalk",
            "clip_id": clip_id,
            "current_frame_idx": int(frame_value),
            "frame_indices": frame_indices,
            "observation_raw": _raw_timestamp(row.get("timestamp"), "EgoWalk timestamp"),
            "source_group_id": source_group_id,
            "split_group_id": split_group_id,
            "dataset_source": dataset,
            "source_manifest_sha256s": [source_manifest_sha256],
        }
    raise CachedPairMaterializationError(
        "dataset_timing_contract_unsupported", f"dataset={dataset}"
    )


def _insert_source_timing(
    connection: sqlite3.Connection,
    timing: Mapping[str, Any],
    counters: Counter[str],
) -> None:
    """Insert one timing payload and collapse only byte-identical duplicates."""

    payload = json.dumps(timing, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    existing = connection.execute(
        "SELECT payload FROM source_timing WHERE source_group_id = ?",
        (timing["source_group_id"],),
    ).fetchone()
    if existing is not None:
        existing_timing = json.loads(existing[0])
        existing_hashes = set(existing_timing.pop("source_manifest_sha256s"))
        incoming_timing = dict(timing)
        incoming_hashes = set(incoming_timing.pop("source_manifest_sha256s"))
        if existing_timing != incoming_timing:
            raise ValueError(f"conflicting source timing: {timing['source_group_id']}")
        merged_hashes = sorted(existing_hashes | incoming_hashes)
        if merged_hashes != sorted(existing_hashes):
            existing_timing["source_manifest_sha256s"] = merged_hashes
            connection.execute(
                "UPDATE source_timing SET payload = ? WHERE source_group_id = ?",
                (
                    json.dumps(
                        existing_timing,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    timing["source_group_id"],
                ),
            )
        counters["equivalent_duplicate_sources"] += 1
        return
    connection.execute(
        "INSERT INTO source_timing VALUES (?, ?, ?, ?)",
        (
            timing["source_group_id"],
            timing["split_group_id"],
            timing["dataset_source"],
            payload,
        ),
    )
    if timing["kind"] == "egowalk":
        index = int(timing["current_frame_idx"])
        timestamp = int(timing["observation_raw"])
        existing_frame = connection.execute(
            "SELECT timestamp_raw FROM frame_times "
            "WHERE dataset_source = ? AND clip_id = ? AND frame_idx = ?",
            (timing["dataset_source"], timing["clip_id"], index),
        ).fetchone()
        if existing_frame is not None and int(existing_frame[0]) != timestamp:
            raise ValueError(
                "conflicting EgoWalk frame timestamp: "
                f"clip={timing['clip_id']} frame={index}"
            )
        if existing_frame is None:
            connection.execute(
                "INSERT INTO frame_times VALUES (?, ?, ?, ?)",
                (timing["dataset_source"], timing["clip_id"], index, timestamp),
            )


def _iter_pairs(
    connection: sqlite3.Connection,
    *,
    period: float,
    max_age: float,
    counters: Counter[str],
    age_values: list[float],
) -> Iterator[dict[str, Any]]:
    """Join ordered physical views and maintain one latest query per split."""

    query = """
        SELECT v.payload, t.payload
        FROM views AS v
        LEFT JOIN source_timing AS t ON t.source_group_id = v.source_group_id
        ORDER BY v.dataset_source, v.split_group_id,
                 json_extract(t.payload, '$.observation_raw'),
                 v.source_group_id, v.context_mode
    """
    current_split: tuple[str, str] | None = None
    cached_view: StageIICFMTrainingViewV2 | None = None
    cached_timing: Mapping[str, Any] | None = None
    cached_signature: str | None = None
    cached_timestamp: float | None = None

    for view_payload, timing_payload in connection.execute(query):
        action_view = StageIICFMTrainingViewV2.model_validate(json.loads(view_payload))
        split_key = (action_view.dataset_source, action_view.split_group_id)
        if split_key != current_split:
            current_split = split_key
            cached_view = None
            cached_timing = None
            cached_signature = None
            cached_timestamp = None
        if timing_payload is None:
            _record_reject(
                connection,
                action_view,
                "source_timing_missing",
                "no source timing matched source_group_id",
                counters,
            )
            continue
        timing = json.loads(timing_payload)
        try:
            action_timestamps, action_timestamp, action_bev = _resolve_timestamps(
                connection, action_view, timing
            )
            signature = _condition_signature(action_view)
            if (
                cached_view is None
                or cached_timing is None
                or cached_signature != signature
                or cached_timestamp is None
                or action_timestamp - cached_timestamp >= period - 1e-9
            ):
                cached_view = action_view
                cached_timing = timing
                cached_signature = signature
                cached_timestamp = action_timestamp
                counters["query_refreshes"] += 1
            query_timestamps, query_timestamp, query_bev = _resolve_timestamps(
                connection, cached_view, cached_timing
            )
            age = action_timestamp - query_timestamp
            if age < -1e-6:
                raise CachedPairMaterializationError(
                    "query_timestamp_after_action", action_view.sample_id
                )
            if age > max_age + 1e-6:
                raise CachedPairMaterializationError(
                    "query_age_exceeds_limit",
                    f"age={age:.6f} max={max_age:.6f}",
                )
            pair = build_stage_ii_cached_traj_query_training_view_v1(
                action_view,
                cached_view,
                query_source_timestamp_sec=query_timestamp,
                action_observation_timestamp_sec=action_timestamp,
                query_source_frame_timestamps_sec=query_timestamps,
                action_frame_timestamps_sec=action_timestamps,
                query_source_bev_timestamp_sec=query_bev,
                action_bev_timestamp_sec=action_bev,
            )
        except CachedPairMaterializationError as exc:
            _record_reject(connection, action_view, exc.code, str(exc), counters)
            continue
        age_values.append(float(age))
        counters["accepted_pairs"] += 1
        counters[f"accepted_dataset:{action_view.dataset_source}"] += 1
        yield pair.model_dump(mode="json", exclude_none=True)


def _iter_pairs_v2(
    connection: sqlite3.Connection,
    *,
    period: float,
    max_age: float,
    counters: Counter[str],
    age_values: list[float],
) -> Iterator[dict[str, Any]]:
    """Join visible V2 source views and emit Pass-R/Pass-D-free pairs."""

    query = """
        SELECT v.payload, t.payload
        FROM views AS v
        LEFT JOIN source_timing AS t ON t.source_group_id = v.source_group_id
        WHERE v.context_mode = 'cached_traj_query_v2'
        ORDER BY v.dataset_source, v.split_group_id,
                 json_extract(t.payload, '$.observation_raw'), v.source_group_id
    """
    current_split: tuple[str, str] | None = None
    cached_view: StageIICachedTrajQuerySourceViewV2 | None = None
    cached_timing: Mapping[str, Any] | None = None
    cached_signature: str | None = None
    cached_timestamp: float | None = None

    for view_payload, timing_payload in connection.execute(query):
        action_view = StageIICachedTrajQuerySourceViewV2.model_validate(
            json.loads(view_payload)
        )
        split_key = (action_view.dataset_source, action_view.split_group_id)
        if split_key != current_split:
            current_split = split_key
            cached_view = None
            cached_timing = None
            cached_signature = None
            cached_timestamp = None
        if timing_payload is None:
            _record_source_reject(
                connection,
                action_view,
                "source_timing_missing",
                "no source timing matched source_group_id",
                counters,
            )
            continue
        timing = json.loads(timing_payload)
        try:
            action_timestamps, action_timestamp, action_bev = (
                _resolve_source_timestamps(
                    connection,
                    action_view,
                    timing,
                )
            )
            signature = _condition_signature_observation(action_view.observation)
            if (
                cached_view is None
                or cached_timing is None
                or cached_signature != signature
                or cached_timestamp is None
                or action_timestamp - cached_timestamp >= period - 1e-9
            ):
                cached_view = action_view
                cached_timing = timing
                cached_signature = signature
                cached_timestamp = action_timestamp
                counters["query_refreshes"] += 1
            query_timestamps, query_timestamp, query_bev = _resolve_source_timestamps(
                connection,
                cached_view,
                cached_timing,
            )
            age = action_timestamp - query_timestamp
            if age < -1e-6:
                raise CachedPairMaterializationError(
                    "query_timestamp_after_action", action_view.sample_id
                )
            if age > max_age + 1e-6:
                raise CachedPairMaterializationError(
                    "query_age_exceeds_limit",
                    f"age={age:.6f} max={max_age:.6f}",
                )
            query_observation = cached_view.observation
            action_observation = action_view.observation
            # The raw source view is target-private; the pair carries only the
            # action-side target and target-free observations.
            pair = build_stage_ii_cached_traj_query_training_view_v2(
                query_observation,
                action_observation,
                trajectory_target=action_view.trajectory_target,
                condition_digest=signature,
                provenance={
                    "trajectory_source_group_id": action_view.source_group_id,
                    "query_source_view_digest": cached_view.view_digest,
                    "action_source_view_digest": action_view.view_digest,
                },
            )
        except CachedPairMaterializationError as exc:
            _record_source_reject(connection, action_view, exc.code, str(exc), counters)
            continue
        age_values.append(float(age))
        counters["accepted_pairs"] += 1
        counters[f"accepted_dataset:{action_view.dataset_source}"] += 1
        yield pair.model_dump(mode="json", exclude_none=True)


def _resolve_timestamps(
    connection: sqlite3.Connection,
    view: StageIICFMTrainingViewV2,
    timing: Mapping[str, Any],
) -> tuple[list[float], float, float | None]:
    """Resolve timestamps in the exact order of one view's image roles."""

    unit = view.trajectory_target.policy.raw_timestamp_unit
    scale = {"nanosecond": 1e-9, "millisecond": 1e-3}[unit]
    if timing["kind"] == "multiframe":
        fpv_raw = list(timing["fpv_timestamps_raw"])
    else:
        fpv_raw = []
        for index in timing["frame_indices"]:
            row = connection.execute(
                "SELECT timestamp_raw FROM frame_times "
                "WHERE dataset_source = ? AND clip_id = ? AND frame_idx = ?",
                (timing["dataset_source"], timing["clip_id"], int(index)),
            ).fetchone()
            if row is None:
                raise CachedPairMaterializationError(
                    "egowalk_history_timestamp_missing",
                    f"clip={timing['clip_id']} frame={index}",
                )
            fpv_raw.append(int(row[0]))
    fpv_roles = [role for role in view.image_roles if role.startswith("fpv.")]
    if len(fpv_raw) < len(fpv_roles):
        raise CachedPairMaterializationError(
            "fpv_timestamp_count_mismatch",
            f"roles={len(fpv_roles)} timestamps={len(fpv_raw)}",
        )
    fpv_raw = fpv_raw[-len(fpv_roles):]
    observation_sec = float(timing["observation_raw"]) * scale
    if abs(float(fpv_raw[-1]) * scale - observation_sec) > 1e-6:
        raise CachedPairMaterializationError(
            "current_fpv_timestamp_mismatch", view.sample_id
        )
    bev_indices = [
        index for index, role in enumerate(view.image_roles) if role == "bev.current"
    ]
    if len(bev_indices) > 1:
        raise CachedPairMaterializationError("duplicate_bev_current", view.sample_id)
    bev_sec: float | None = None
    if bev_indices:
        raw_bev = timing.get("bev_timestamp_raw")
        if raw_bev is None:
            raise CachedPairMaterializationError("bev_timestamp_missing", view.sample_id)
        bev_sec = float(raw_bev) * scale
    elif timing.get("bev_timestamp_raw") is not None and any(
        role.startswith("bev.") for role in view.image_roles
    ):
        raise CachedPairMaterializationError("unsupported_bev_role", view.sample_id)
    timestamps: list[float] = []
    fpv_index = 0
    for role in view.image_roles:
        if role.startswith("fpv."):
            timestamps.append(float(fpv_raw[fpv_index]) * scale)
            fpv_index += 1
        elif role == "bev.current":
            assert bev_sec is not None
            timestamps.append(bev_sec)
        else:
            raise CachedPairMaterializationError(
                "image_role_timestamp_unsupported", role
            )
    return timestamps, observation_sec, bev_sec


def _resolve_source_timestamps(
    connection: sqlite3.Connection,
    view: StageIICachedTrajQuerySourceViewV2,
    timing: Mapping[str, Any],
) -> tuple[list[float], float, float | None]:
    """Resolve timing for the pure-latent V2 source observation."""

    unit = view.trajectory_target.policy.raw_timestamp_unit
    scale = {"nanosecond": 1e-9, "millisecond": 1e-3}[unit]
    if timing["kind"] == "multiframe":
        fpv_raw = list(timing["fpv_timestamps_raw"])
    else:
        fpv_raw = []
        for index in timing["frame_indices"]:
            row = connection.execute(
                "SELECT timestamp_raw FROM frame_times "
                "WHERE dataset_source = ? AND clip_id = ? AND frame_idx = ?",
                (timing["dataset_source"], timing["clip_id"], int(index)),
            ).fetchone()
            if row is None:
                raise CachedPairMaterializationError(
                    "egowalk_history_timestamp_missing",
                    f"clip={timing['clip_id']} frame={index}",
                )
            fpv_raw.append(int(row[0]))
    fpv_roles = [
        role for role in view.observation.image_roles if role.startswith("fpv.")
    ]
    if len(fpv_raw) < len(fpv_roles):
        raise CachedPairMaterializationError(
            "fpv_timestamp_count_mismatch",
            f"roles={len(fpv_roles)} timestamps={len(fpv_raw)}",
        )
    fpv_raw = fpv_raw[-len(fpv_roles):]
    observation_sec = float(timing["observation_raw"]) * scale
    if abs(float(fpv_raw[-1]) * scale - observation_sec) > 1e-6:
        raise CachedPairMaterializationError(
            "current_fpv_timestamp_mismatch", view.sample_id
        )
    bev_indices = [
        index
        for index, role in enumerate(view.observation.image_roles)
        if role == "bev.current"
    ]
    if len(bev_indices) > 1:
        raise CachedPairMaterializationError("duplicate_bev_current", view.sample_id)
    bev_sec: float | None = None
    if bev_indices:
        raw_bev = timing.get("bev_timestamp_raw")
        if raw_bev is None:
            raise CachedPairMaterializationError("bev_timestamp_missing", view.sample_id)
        bev_sec = float(raw_bev) * scale
    elif timing.get("bev_timestamp_raw") is not None and any(
        role.startswith("bev.") for role in view.observation.image_roles
    ):
        raise CachedPairMaterializationError("unsupported_bev_role", view.sample_id)
    timestamps: list[float] = []
    fpv_index = 0
    for role in view.observation.image_roles:
        if role.startswith("fpv."):
            timestamps.append(float(fpv_raw[fpv_index]) * scale)
            fpv_index += 1
        elif role == "bev.current":
            assert bev_sec is not None
            timestamps.append(bev_sec)
        else:
            raise CachedPairMaterializationError(
                "image_role_timestamp_unsupported", role
            )
    return timestamps, observation_sec, bev_sec


def _condition_signature(view: StageIICFMTrainingViewV2) -> str:
    """Digest stable visible conditions, excluding per-frame media and evidence."""

    payload = {
        "dataset_source": view.dataset_source,
        "split_group_id": view.split_group_id,
        "modality_profile_id": view.modality_profile_id,
        "image_roles": list(view.image_roles),
        "instruction": view.instruction,
        "model_visible_context": view.model_visible_context,
        "input_modalities": view.input_modalities.model_dump(mode="json"),
    }
    return stage_ii_digest_v1(payload)


def _condition_signature_observation(observation: Any) -> str:
    """Digest stable visible conditions for a pure-latent source row."""

    payload = {
        "dataset_source": observation.dataset_source,
        "split_group_id": observation.split_group_id,
        "modality_profile_id": observation.modality_profile_id,
        "image_roles": list(observation.image_roles),
        "instruction": observation.instruction,
        "model_visible_context": observation.model_visible_context,
        "input_modalities": observation.input_modalities.model_dump(mode="json"),
    }
    return stage_ii_digest_v1(payload)


def _record_reject(
    connection: sqlite3.Connection,
    view: StageIICFMTrainingViewV2,
    reason_code: str,
    detail: str,
    counters: Counter[str],
) -> None:
    """Store one expected data rejection for the atomic reject output."""

    payload = {
        "schema_version": CACHED_PAIR_REJECT_SCHEMA_VERSION,
        "sample_id": view.sample_id,
        "source_group_id": view.source_group_id,
        "split_group_id": view.split_group_id,
        "dataset_source": view.dataset_source,
        "reason_code": reason_code,
        "detail": detail[:1000],
    }
    connection.execute(
        "INSERT INTO rejects(payload) VALUES (?)",
        (json.dumps(payload, ensure_ascii=False, separators=(",", ":")),),
    )
    counters["rejected_pairs"] += 1
    counters[f"reject:{reason_code}"] += 1


def _record_source_reject(
    connection: sqlite3.Connection,
    view: StageIICachedTrajQuerySourceViewV2,
    reason_code: str,
    detail: str,
    counters: Counter[str],
) -> None:
    """Store one pure-latent V2 source rejection."""

    payload = {
        "schema_version": CACHED_PAIR_REJECT_SCHEMA_VERSION,
        "sample_id": view.sample_id,
        "source_group_id": view.source_group_id,
        "split_group_id": view.split_group_id,
        "dataset_source": view.dataset_source,
        "reason_code": reason_code,
        "detail": detail[:1000],
        "condition_mode": "cached_traj_query_v2",
    }
    connection.execute(
        "INSERT INTO rejects(payload) VALUES (?)",
        (json.dumps(payload, ensure_ascii=False, separators=(",", ":")),),
    )
    counters["rejected_pairs"] += 1
    counters[f"reject:{reason_code}"] += 1


def _record_raw_source_reject(
    connection: sqlite3.Connection,
    raw: Mapping[str, Any],
    reason_code: str,
    detail: str,
    counters: Counter[str],
) -> None:
    """Store one Stage-I source rejection before a typed source view exists."""

    payload = {
        "schema_version": CACHED_SOURCE_VIEW_REJECT_SCHEMA_VERSION_V2,
        "sample_id": str(raw.get("sample_id") or "unknown"),
        "source_group_id": str(raw.get("source_group_id") or "unknown"),
        "split_group_id": str(raw.get("split_group_id") or "unknown"),
        "dataset_source": str(raw.get("dataset_source") or "unknown"),
        "reason_code": reason_code,
        "detail": detail[:1000],
        "condition_mode": "cached_traj_query_v2",
    }
    connection.execute(
        "INSERT INTO rejects(payload) VALUES (?)",
        (json.dumps(payload, ensure_ascii=False, separators=(",", ":")),),
    )
    counters["rejected_source_views"] += 1
    counters[f"reject:{reason_code}"] += 1


def _iter_reject_rows(connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Yield rejects in the same deterministic order they were observed."""

    for (payload,) in connection.execute(
        "SELECT payload FROM rejects ORDER BY sequence_id"
    ):
        yield json.loads(payload)


def _json_object(line: bytes, path: str | Path, line_number: int) -> dict[str, Any]:
    """Decode one JSONL object and attach its source location to failures."""

    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
    return value


def _raw_timestamp(value: Any, field_name: str) -> int:
    """Require one finite integer source clock value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CachedPairMaterializationError("timestamp_invalid", field_name)
    if isinstance(value, int):
        return value
    if not math.isfinite(value) or not value.is_integer():
        raise CachedPairMaterializationError("timestamp_invalid", field_name)
    return int(value)


def _positive_finite(value: float, field_name: str) -> float:
    """Validate one positive duration argument."""

    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _age_report(values: Sequence[float]) -> dict[str, Any]:
    """Return deterministic query-age summary and millisecond buckets."""

    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "histogram_ms": {}}
    histogram = Counter(str(round(value * 1000.0)) for value in values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "histogram_ms": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


def _emit_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    count: int,
    interval: int,
    phase: str,
    counters: Counter[str],
) -> None:
    """Emit progress only at explicit intervals."""

    if callback is not None and count > 0 and count % interval == 0:
        callback({"phase": phase, "count": count, "counts": dict(counters)})


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish one JSON report."""

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    """Return the SHA256 fingerprint used by source/sidecar provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CACHED_PAIR_MATERIALIZATION_REPORT_VERSION",
    "CACHED_PAIR_MATERIALIZATION_REPORT_VERSION_V2",
    "CACHED_SOURCE_VIEW_MATERIALIZATION_REPORT_VERSION_V2",
    "CACHED_SOURCE_VIEW_REJECT_SCHEMA_VERSION_V2",
    "CachedPairMaterializationError",
    "build_stage_ii_cached_traj_query_pairs_v1",
    "build_stage_ii_cached_traj_query_pairs_v2",
    "build_stage_ii_cached_traj_query_source_views_v2",
]
