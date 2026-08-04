from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


RECORD_ID_RE = re.compile(r"rec_([^/_.]+)")
DEFAULT_DECISIONS = ("REVIEW", "DROP")


class TransferSelectionError(ValueError):
    """Raised when the transfer selection cannot be built unambiguously."""


@dataclass(frozen=True)
class SelectedDecision:
    record_id: str
    decision: str
    source_name: str


@dataclass(frozen=True)
class LocalMatch:
    record_id: str
    decision: str
    source_name: str
    relative_path: Path
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class ReportPaths:
    files_from: Path
    manifest: Path
    missing: Path
    summary: Path


def extract_record_id(value: str | Path) -> str:
    """Extract the token following the final ``rec_`` marker."""
    matches = RECORD_ID_RE.findall(str(value))
    return matches[-1] if matches else ""


def load_selected_decisions(
    csv_path: Path,
    decision_column: str,
    selected_values: Iterable[str] = DEFAULT_DECISIONS,
) -> dict[str, SelectedDecision]:
    selected_set = {
        str(value).strip().upper()
        for value in selected_values
        if str(value).strip()
    }
    if not selected_set:
        raise TransferSelectionError("At least one decision value must be selected")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = {"name", decision_column}
        missing_columns = sorted(required - columns)
        if missing_columns:
            raise TransferSelectionError(
                "Decisions CSV is missing column(s): " + ", ".join(missing_columns)
            )

        all_record_ids: dict[str, int] = {}
        selected: dict[str, SelectedDecision] = {}
        for line_number, row in enumerate(reader, start=2):
            source_name = row["name"].strip()
            record_id = extract_record_id(source_name)
            if not record_id:
                raise TransferSelectionError(
                    f"Decisions CSV line {line_number} has no rec_ identifier: "
                    f"{source_name!r}"
                )
            previous_line = all_record_ids.get(record_id)
            if previous_line is not None:
                raise TransferSelectionError(
                    f"Duplicate rec_{record_id} in decisions CSV lines "
                    f"{previous_line} and {line_number}"
                )
            all_record_ids[record_id] = line_number

            decision = row[decision_column].strip().upper()
            if decision in selected_set:
                selected[record_id] = SelectedDecision(
                    record_id=record_id,
                    decision=decision,
                    source_name=source_name,
                )

    if not selected:
        values = ", ".join(sorted(selected_set))
        raise TransferSelectionError(
            f"No rows selected from {csv_path} using "
            f"{decision_column} in {{{values}}}"
        )
    return selected


def match_local_mcap_files(
    local_root: Path,
    selected: dict[str, SelectedDecision],
    *,
    allow_multiple: bool = False,
) -> tuple[list[LocalMatch], list[SelectedDecision]]:
    paths_by_record_id: dict[str, list[Path]] = defaultdict(list)
    for path in local_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".mcap":
            continue
        relative_path = path.relative_to(local_root)
        record_id = extract_record_id(relative_path.as_posix())
        if record_id in selected:
            paths_by_record_id[record_id].append(relative_path)

    ambiguous = {
        record_id: sorted(paths)
        for record_id, paths in paths_by_record_id.items()
        if len(paths) > 1
    }
    if ambiguous and not allow_multiple:
        details = []
        for record_id, paths in sorted(ambiguous.items()):
            rendered_paths = ", ".join(str(path) for path in paths[:5])
            if len(paths) > 5:
                rendered_paths += f", ... ({len(paths)} files total)"
            details.append(f"rec_{record_id}: {rendered_paths}")
        raise TransferSelectionError(
            "Multiple local MCAP files matched the same record identifier. "
            "Resolve the duplicates or pass --allow-multiple:\n"
            + "\n".join(details[:20])
        )

    matches: list[LocalMatch] = []
    for record_id, relative_paths in paths_by_record_id.items():
        selected_decision = selected[record_id]
        for relative_path in relative_paths:
            stat = (local_root / relative_path).stat()
            matches.append(
                LocalMatch(
                    record_id=record_id,
                    decision=selected_decision.decision,
                    source_name=selected_decision.source_name,
                    relative_path=relative_path,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                )
            )

    matches.sort(key=lambda match: match.relative_path.as_posix())
    matched_ids = set(paths_by_record_id)
    missing = sorted(
        (
            decision
            for record_id, decision in selected.items()
            if record_id not in matched_ids
        ),
        key=lambda decision: decision.record_id,
    )
    return matches, missing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_reports(
    report_dir: Path,
    csv_path: Path,
    decision_column: str,
    selected: dict[str, SelectedDecision],
    matches: Sequence[LocalMatch],
    missing: Sequence[SelectedDecision],
) -> ReportPaths:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(
        files_from=report_dir / "files-from.bin",
        manifest=report_dir / "manifest.tsv",
        missing=report_dir / "missing.tsv",
        summary=report_dir / "summary.json",
    )

    with paths.files_from.open("wb") as stream:
        for match in matches:
            stream.write(os.fsencode(str(match.relative_path)))
            stream.write(b"\0")

    with paths.manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "record_id",
                "decision",
                "relative_path",
                "size_bytes",
                "mtime_ns",
                "decisions_source_name",
            ]
        )
        for match in matches:
            writer.writerow(
                [
                    f"rec_{match.record_id}",
                    match.decision,
                    match.relative_path,
                    match.size_bytes,
                    match.mtime_ns,
                    match.source_name,
                ]
            )

    with paths.missing.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["record_id", "decision", "decisions_source_name"])
        for decision in missing:
            writer.writerow(
                [
                    f"rec_{decision.record_id}",
                    decision.decision,
                    decision.source_name,
                ]
            )

    selected_counts = Counter(item.decision for item in selected.values())
    matched_id_counts = Counter()
    for match in matches:
        matched_id_counts[match.decision] += 1
    summary = {
        "decisions_csv": str(csv_path.resolve()),
        "decisions_csv_sha256": _sha256(csv_path),
        "decision_column": decision_column,
        "selected_record_ids": len(selected),
        "selected_counts": dict(sorted(selected_counts.items())),
        "matched_record_ids": len({match.record_id for match in matches}),
        "matched_files": len(matches),
        "matched_file_counts": dict(sorted(matched_id_counts.items())),
        "missing_record_ids": len(missing),
        "total_bytes": sum(match.size_bytes for match in matches),
    }
    paths.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def build_rsync_command(
    *,
    rsync_path: str,
    local_root: Path,
    remote_destination: str,
    files_from: Path,
    ssh_port: int,
    identity_file: Path | None,
    execute: bool,
    compress: bool,
    checksum: bool,
    bandwidth_limit_kbps: int | None,
) -> list[str]:
    archive_options = "-avR"
    if compress:
        archive_options += "z"

    ssh_command = ["ssh", "-p", str(ssh_port)]
    if identity_file is not None:
        ssh_command.extend(["-i", str(identity_file)])

    command = [
        rsync_path,
        archive_options,
        "--from0",
        f"--files-from={files_from}",
        "--partial",
        "--partial-dir=.rsync-partial",
        "--itemize-changes",
        "--info=progress2",
        "--human-readable",
        "--protect-args",
        "-e",
        shlex.join(ssh_command),
    ]
    if not execute:
        command.append("--dry-run")
    if checksum:
        command.append("--checksum")
    if bandwidth_limit_kbps is not None:
        command.append(f"--bwlimit={bandwidth_limit_kbps}")

    command.extend(
        [
            f"{local_root}{os.sep}",
            remote_destination.rstrip("/") + "/",
        ]
    )
    return command


def build_csv_rsync_command(
    *,
    rsync_path: str,
    csv_path: Path,
    remote_destination: str,
    ssh_port: int,
    identity_file: Path | None,
    execute: bool,
    compress: bool,
) -> list[str]:
    archive_options = "-av"
    if compress:
        archive_options += "z"
    ssh_command = ["ssh", "-p", str(ssh_port)]
    if identity_file is not None:
        ssh_command.extend(["-i", str(identity_file)])

    command = [
        rsync_path,
        archive_options,
        "--itemize-changes",
        "--protect-args",
        "-e",
        shlex.join(ssh_command),
    ]
    if not execute:
        command.append("--dry-run")
    command.extend(
        [
            str(csv_path),
            remote_destination.rstrip("/") + "/decisions.csv",
        ]
    )
    return command


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select REVIEW/DROP MCAP files by rec_ identifier and transfer them "
            "incrementally with rsync. The default is a non-mutating dry-run."
        )
    )
    parser.add_argument(
        "--decisions-csv",
        type=Path,
        required=True,
        help="Path to decisions.csv (UTF-8 with or without BOM)",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        required=True,
        help="Common local root containing the MCAP files",
    )
    parser.add_argument(
        "--remote",
        required=True,
        help="Rsync destination directory, for example user@host:/data/rebuild/",
    )
    parser.add_argument(
        "--decision-column",
        choices=("manual_decision", "auto_recommendation"),
        default="manual_decision",
        help="CSV column used for selection (default: manual_decision)",
    )
    parser.add_argument(
        "--decisions",
        nargs="+",
        default=list(DEFAULT_DECISIONS),
        help="Decision values to select (default: REVIEW DROP)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "rsync-decision-transfer",
        help="Directory for files-from, manifest, missing list, and summary",
    )
    parser.add_argument("--ssh-port", type=_positive_int, default=10522)
    parser.add_argument(
        "--identity-file",
        type=Path,
        help="Optional SSH private-key path; the key is never copied or logged",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the transfer; without this flag rsync runs with --dry-run",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Refuse to run rsync if any selected record ID is absent locally",
    )
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Transfer every local MCAP when one record ID matches multiple files",
    )
    parser.add_argument(
        "--include-decisions-csv",
        action="store_true",
        help="Also update decisions.csv at the remote destination",
    )
    parser.add_argument(
        "--checksum",
        action="store_true",
        help=(
            "Compare file contents instead of size and modification time; slower "
            "but useful when timestamps are unreliable"
        ),
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable rsync compression (often faster for already-compressed MCAP)",
    )
    parser.add_argument(
        "--bwlimit",
        type=_positive_int,
        metavar="KBPS",
        help="Optional rsync bandwidth limit in KiB/s",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = args.decisions_csv.expanduser().resolve()
    local_root = args.local_root.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    identity_file = (
        args.identity_file.expanduser().resolve()
        if args.identity_file is not None
        else None
    )

    if not csv_path.is_file():
        raise TransferSelectionError(f"Decisions CSV does not exist: {csv_path}")
    if not local_root.is_dir():
        raise TransferSelectionError(f"Local data root does not exist: {local_root}")
    if identity_file is not None and not identity_file.is_file():
        raise TransferSelectionError(
            f"SSH identity file does not exist: {identity_file}"
        )
    if ":" not in args.remote:
        raise TransferSelectionError(
            "--remote must be an rsync SSH destination such as "
            "user@host:/data/rebuild/"
        )

    rsync_path = shutil.which("rsync")
    if rsync_path is None:
        raise TransferSelectionError("rsync is not installed or not on PATH")

    selected = load_selected_decisions(
        csv_path,
        args.decision_column,
        args.decisions,
    )
    matches, missing = match_local_mcap_files(
        local_root,
        selected,
        allow_multiple=args.allow_multiple,
    )
    reports = write_reports(
        report_dir,
        csv_path,
        args.decision_column,
        selected,
        matches,
        missing,
    )

    selected_counts = Counter(item.decision for item in selected.values())
    print(f"Selected record IDs : {len(selected)}")
    for decision, count in sorted(selected_counts.items()):
        print(f"  {decision:<17}: {count}")
    print(f"Matched record IDs  : {len({item.record_id for item in matches})}")
    print(f"Matched MCAP files  : {len(matches)}")
    print(f"Missing record IDs  : {len(missing)}")
    print(f"Transfer bytes      : {sum(item.size_bytes for item in matches)}")
    print(f"Manifest            : {reports.manifest}")
    print(f"Missing list        : {reports.missing}")
    print(f"Summary             : {reports.summary}")

    if not matches:
        raise TransferSelectionError(
            "No selected MCAP files were found under the local root"
        )
    if missing and args.require_all:
        raise TransferSelectionError(
            f"{len(missing)} selected record IDs are missing locally; "
            f"see {reports.missing}"
        )
    if missing:
        print(
            "Warning             : the transfer will contain only locally "
            "matched record IDs; use --require-all for a complete rebuild",
            file=sys.stderr,
        )

    command = build_rsync_command(
        rsync_path=rsync_path,
        local_root=local_root,
        remote_destination=args.remote,
        files_from=reports.files_from,
        ssh_port=args.ssh_port,
        identity_file=identity_file,
        execute=args.execute,
        compress=not args.no_compress,
        checksum=args.checksum,
        bandwidth_limit_kbps=args.bwlimit,
    )
    print("Mode                :", "EXECUTE" if args.execute else "DRY-RUN")
    print("Data command        :", shlex.join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        return result.returncode

    if args.include_decisions_csv:
        csv_command = build_csv_rsync_command(
            rsync_path=rsync_path,
            csv_path=csv_path,
            remote_destination=args.remote,
            ssh_port=args.ssh_port,
            identity_file=identity_file,
            execute=args.execute,
            compress=not args.no_compress,
        )
        print("CSV command         :", shlex.join(csv_command))
        result = subprocess.run(csv_command, check=False)
        if result.returncode != 0:
            return result.returncode

    if not args.execute:
        print("Dry-run completed. Add --execute only after reviewing the output.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TransferSelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
