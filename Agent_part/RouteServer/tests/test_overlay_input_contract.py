from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SERVER_FILE = ROOT / "src" / "mcp_server" / "server.py"
ZERO_FILE = ROOT / "examples" / "zero_shot_visual_prompt_agent_client.py"
YUQUAN_FILE = ROOT / "examples" / "yuquan_visual_prompt_agent_client.py"


def _function_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    names = [arg.arg for arg in node.args.args]
    names.extend(arg.arg for arg in node.args.kwonlyargs)
    return names


class OverlayInputContractTest(unittest.TestCase):
    def test_server_tool_accepts_overlay_and_alignment_args(self) -> None:
        tree = ast.parse(SERVER_FILE.read_text(encoding="utf-8"))
        tool_fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_zero_shot_visual_prompt_agent"
        )
        names = _function_arg_names(tool_fn)
        self.assertIn("fpv_segmentation_overlay_image_path", names)
        self.assertIn("fpv_segmentation_overlay_image_url", names)
        self.assertIn("enforce_alignment", names)

    def test_clients_expose_overlay_and_alignment_cli_flags(self) -> None:
        zero_text = ZERO_FILE.read_text(encoding="utf-8")
        yuquan_text = YUQUAN_FILE.read_text(encoding="utf-8")
        for text in [zero_text, yuquan_text]:
            self.assertIn("--fpv-segmentation-overlay-image-path", text)
            self.assertIn("--fpv-segmentation-overlay-image-url", text)
            self.assertIn("--enforce-alignment", text)
            self.assertIn("--no-enforce-alignment", text)


if __name__ == "__main__":
    unittest.main()
