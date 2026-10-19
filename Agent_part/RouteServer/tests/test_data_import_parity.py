"""Verify canonical modules remain importable after retiring compatibility paths."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MOVES = json.loads((ROOT / "docs/retired_module_paths.json").read_text())


@pytest.mark.parametrize("old,new", [(old, new) for old, new in MOVES.items()
                                     if old.startswith("mcp_server.data.")])
def test_canonical_owner_replaces_retired_module(old: str, new: str) -> None:
    """Old source files must be absent and the canonical owner must load directly."""
    old_path = ROOT / "src" / Path(*old.split("."))
    assert not old_path.with_suffix(".py").exists()
    assert not (old_path / "__init__.py").exists()
    module = importlib.import_module(new)
    assert module.__name__ == new
    assert Path(module.__file__).is_file()
