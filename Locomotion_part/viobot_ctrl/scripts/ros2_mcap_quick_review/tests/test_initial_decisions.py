from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from review_tool.initial_decisions import (
    attach_initial_decisions,
    extract_record_id,
    is_selected_manual_decision,
    load_initial_decisions,
)


class InitialDecisionTests(unittest.TestCase):
    def test_extracts_id_from_original_and_repaired_names(self):
        self.assertEqual(
            extract_record_id("prefix__rec_0eQJIq1Qt537ohaz.mcap"),
            "0eQJIq1Qt537ohaz",
        )
        self.assertEqual(
            extract_record_id("prefix_rec_0eQJIq1Qt537ohaz_repaired.mcap"),
            "0eQJIq1Qt537ohaz",
        )
        self.assertEqual(extract_record_id("bag_without_record_id.mcap"), "")

    def test_loads_bom_csv_and_marks_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "decisions.csv"
            pd.DataFrame([
                {
                    "name": "a_rec_same.mcap",
                    "manual_decision": "KEEP",
                    "auto_recommendation": "REVIEW",
                    "issues": "odom_jump_review",
                },
                {
                    "name": "b_rec_same_repaired.mcap",
                    "manual_decision": "DROP",
                    "auto_recommendation": "KEEP",
                    "issues": "",
                },
            ]).to_csv(csv_path, index=False, encoding="utf-8-sig")
            decisions = load_initial_decisions(csv_path)
        self.assertEqual(
            set(decisions["initial_match_status"]), {"decision_id_ambiguous"}
        )

    def test_selects_manual_keep_and_review_but_not_drop(self):
        decisions = pd.DataFrame([
            {
                "initial_record_id": "keepid",
                "initial_match_status": "available",
                "initial_manual_decision": "KEEP",
                "initial_auto_recommendation": "REVIEW",
                "initial_issues": "odom_jump_review",
                "initial_source_name": "old_rec_keepid.mcap",
            },
            {
                "initial_record_id": "dropid",
                "initial_match_status": "available",
                "initial_manual_decision": "DROP",
                "initial_auto_recommendation": "KEEP",
                "initial_issues": "",
                "initial_source_name": "old_rec_dropid.mcap",
            },
            {
                "initial_record_id": "reviewid",
                "initial_match_status": "available",
                "initial_manual_decision": "REVIEW",
                "initial_auto_recommendation": "REVIEW",
                "initial_issues": "drift_unobservable",
                "initial_source_name": "old_rec_reviewid.mcap",
            },
        ])
        dataset = pd.DataFrame([
            {"bag_id": "1", "bag_path": "/new/prefix_rec_keepid_repaired.mcap"},
            {"bag_id": "2", "bag_path": "/new/prefix_rec_dropid.mcap"},
            {"bag_id": "3", "bag_path": "/new/prefix_rec_reviewid.mcap"},
            {"bag_id": "4", "bag_path": "/new/no_rec_marker.mcap"},
        ])

        attached = attach_initial_decisions(dataset, decisions)
        selected = is_selected_manual_decision(attached, ["KEEP", "REVIEW"])

        self.assertEqual(attached.loc[0, "initial_match_status"], "matched")
        self.assertEqual(attached.loc[1, "initial_match_status"], "matched")
        self.assertEqual(attached.loc[2, "initial_match_status"], "matched")
        self.assertEqual(attached.loc[3, "initial_match_status"], "not_in_decisions")
        self.assertEqual(selected.tolist(), [True, False, True, False])

    def test_duplicate_dataset_ids_are_not_silently_matched(self):
        decisions = pd.DataFrame([{
            "initial_record_id": "same",
            "initial_match_status": "available",
            "initial_manual_decision": "KEEP",
            "initial_auto_recommendation": "KEEP",
            "initial_issues": "",
            "initial_source_name": "old_rec_same.mcap",
        }])
        dataset = pd.DataFrame([
            {"bag_id": "1", "bag_path": "/a_rec_same.mcap"},
            {"bag_id": "2", "bag_path": "/b_rec_same_repaired.mcap"},
        ])
        attached = attach_initial_decisions(dataset, decisions)
        self.assertEqual(
            set(attached["initial_match_status"]), {"dataset_id_ambiguous"}
        )


if __name__ == "__main__":
    unittest.main()
