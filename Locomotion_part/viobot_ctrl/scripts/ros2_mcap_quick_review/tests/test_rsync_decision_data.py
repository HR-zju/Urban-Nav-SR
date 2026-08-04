from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from rsync_decision_data import (
    TransferSelectionError,
    build_rsync_command,
    load_selected_decisions,
    match_local_mcap_files,
)


class RsyncDecisionDataTests(unittest.TestCase):
    def _write_decisions(self, root: Path) -> Path:
        path = root / "decisions.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "name",
                    "auto_recommendation",
                    "manual_decision",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "name": "old_prefix_rec_review_id.mcap",
                        "auto_recommendation": "KEEP",
                        "manual_decision": "review",
                    },
                    {
                        "name": "old_prefix_rec_dropid.mcap",
                        "auto_recommendation": "REVIEW",
                        "manual_decision": "DROP",
                    },
                    {
                        "name": "old_prefix_rec_keepid.mcap",
                        "auto_recommendation": "DROP",
                        "manual_decision": "KEEP",
                    },
                    {
                        "name": "old_prefix_rec_auto_drop.mcap",
                        "auto_recommendation": "DROP",
                        "manual_decision": "",
                    },
                ]
            )
        return path

    def test_selects_manual_review_and_drop_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decisions = load_selected_decisions(
                self._write_decisions(root),
                "manual_decision",
            )

        self.assertEqual(set(decisions), {"review", "dropid"})
        self.assertEqual(decisions["review"].decision, "REVIEW")
        self.assertEqual(decisions["dropid"].decision, "DROP")

    def test_matches_by_record_id_when_prefixes_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decisions = load_selected_decisions(
                self._write_decisions(root),
                "manual_decision",
            )
            matched_path = (
                root / "new" / "different_prefix_rec_review_id_repaired.mcap"
            )
            matched_path.parent.mkdir()
            matched_path.write_bytes(b"review")
            other_path = root / "new" / "rec_keepid.mcap"
            other_path.write_bytes(b"keep")

            matches, missing = match_local_mcap_files(root, decisions)

        self.assertEqual(
            [match.relative_path.as_posix() for match in matches],
            ["new/different_prefix_rec_review_id_repaired.mcap"],
        )
        self.assertEqual([item.record_id for item in missing], ["dropid"])

    def test_rejects_ambiguous_local_matches_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decisions = load_selected_decisions(
                self._write_decisions(root),
                "manual_decision",
            )
            (root / "a_rec_dropid.mcap").write_bytes(b"a")
            (root / "b_rec_dropid_repaired.mcap").write_bytes(b"b")

            with self.assertRaisesRegex(
                TransferSelectionError,
                "Multiple local MCAP files",
            ):
                match_local_mcap_files(root, decisions)

    def test_rsync_command_is_incremental_and_dry_run_by_default(self):
        command = build_rsync_command(
            rsync_path="/usr/bin/rsync",
            local_root=Path("/data/local"),
            remote_destination="user@example:/data/rebuild/",
            files_from=Path("/tmp/files-from.bin"),
            ssh_port=10522,
            identity_file=None,
            execute=False,
            compress=True,
            checksum=False,
            bandwidth_limit_kbps=None,
        )

        self.assertIn("--dry-run", command)
        self.assertIn("--partial", command)
        self.assertIn("--partial-dir=.rsync-partial", command)
        self.assertIn("--from0", command)
        self.assertIn("-avRz", command)
        self.assertNotIn("--delete", command)
        self.assertIn("ssh -p 10522", command)


if __name__ == "__main__":
    unittest.main()
