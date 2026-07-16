from __future__ import annotations

import unittest
from unittest.mock import patch

from mcp_server.llm import GoalDecomposer, LLMSettings, _json_only_contract, _parse_json_object_from_text


class LLMJsonParsingTest(unittest.TestCase):
    """Tests for conservative strict-JSON response normalization."""

    def test_latex_escaped_underscores_are_normalized(self) -> None:
        parsed = _parse_json_object_from_text(
            r'{"destination\_kind":"generic\_poi","search\_keywords":"coffee\_shop"}'
        )

        self.assertEqual(parsed["destination_kind"], "generic_poi")
        self.assertEqual(parsed["search_keywords"], "coffee_shop")

    def test_legacy_dollar_prefixed_key_typo_is_normalized(self) -> None:
        parsed = _parse_json_object_from_text(r'{$search\_keywords":"coffee|cafe"}')

        self.assertEqual(parsed, {"search_keywords": "coffee|cafe"})

    def test_missing_container_closers_are_appended(self) -> None:
        parsed = _parse_json_object_from_text(
            '{"ranked_candidate_ids":["poi_1","poi_2"],"metadata":{"valid":true}'
        )

        self.assertEqual(parsed["ranked_candidate_ids"], ["poi_1", "poi_2"])
        self.assertEqual(parsed["metadata"], {"valid": True})

    def test_missing_string_content_is_not_synthesized(self) -> None:
        with self.assertRaisesRegex(ValueError, "No valid JSON object"):
            _parse_json_object_from_text('{"summary":"unfinished')

    def test_natural_language_is_not_promoted_to_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "No valid JSON object"):
            _parse_json_object_from_text("Choose the closest cafe first.")

    def test_json_contract_forbids_latex_underscore_escapes(self) -> None:
        contract = _json_only_contract({"destination_kind": "generic_poi"})

        self.assertIn("never escape them as \\_", contract)
        self.assertIn("never repeat tokens or fields", contract)

    def test_compact_itinerary_profile_omits_nonessential_fields(self) -> None:
        decomposer = GoalDecomposer(settings=_settings())

        with patch.dict("os.environ", {"LLM_ITINERARY_PROFILE": "compact"}, clear=False):
            messages = decomposer.build_messages("First visit a cafe, then a bookstore.")

        system_prompt = messages[0]["content"]
        self.assertIn("Preserve every requested stop and its order", system_prompt)
        self.assertNotIn('"hard_constraints"', system_prompt)
        self.assertNotIn('"rationale"', system_prompt)

    def test_compact_itinerary_output_receives_runtime_defaults(self) -> None:
        decomposer = GoalDecomposer(settings=_settings())
        parsed = {
            "goals": [
                {
                    "goal_index": 0,
                    "goal_type": "cafe_stop",
                    "request_text": "visit a cafe",
                    "destination_type": "cafe",
                    "destination_kind": "generic_poi",
                    "search_keywords": "coffee|cafe",
                    "mode": "foot",
                    "radius_m": 2000,
                    "optimize_for": "distance",
                }
            ],
            "global_mode": "foot",
            "notes": "ordered_goal_list",
        }

        normalized = decomposer._normalize_output("Visit a cafe.", parsed)

        self.assertEqual(normalized["goals"][0]["search_keywords"], "coffee|cafe")
        self.assertEqual(normalized["goals"][0]["hard_constraints"]["must_include_keywords"], [])
        self.assertEqual(normalized["goals"][0]["rationale"], "")

    def test_minimal_itinerary_profile_uses_runtime_compatible_schema(self) -> None:
        decomposer = GoalDecomposer(settings=_settings())

        with patch.dict("os.environ", {"LLM_ITINERARY_PROFILE": "minimal"}, clear=False):
            messages = decomposer.build_messages("First visit a cafe, then a bookstore.")

        system_prompt = messages[0]["content"]
        self.assertIn(
            "goal_index, request_text, destination_type, destination_kind, and search_keywords",
            system_prompt,
        )
        self.assertIn('"destination_kind"', system_prompt)
        self.assertNotIn('"optimize_for"', system_prompt)

    def test_minimal_itinerary_output_receives_runtime_defaults(self) -> None:
        decomposer = GoalDecomposer(settings=_settings())
        parsed = {
            "goals": [
                {
                    "goal_index": 0,
                    "request_text": "visit a cafe",
                    "destination_type": "cafe",
                    "destination_kind": "generic_poi",
                    "search_keywords": "coffee|cafe",
                }
            ],
            "global_mode": "foot",
        }

        normalized = decomposer._normalize_output("Visit a cafe.", parsed)

        self.assertEqual(normalized["goals"][0]["destination_kind"], "generic_poi")
        self.assertEqual(normalized["goals"][0]["mode"], "foot")
        self.assertEqual(normalized["goals"][0]["optimize_for"], "distance")


def _settings() -> LLMSettings:
    return LLMSettings(
        provider="openai",
        api_style="chat_completions",
        api_key="dummy",
        model="llava-next-video-32b-qwen",
        base_url="http://127.0.0.1:18244/v1",
    )


if __name__ == "__main__":
    unittest.main()
