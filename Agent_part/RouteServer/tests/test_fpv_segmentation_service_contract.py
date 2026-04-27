from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "services" / "fpv_segmentation_service.py"
_SPEC = importlib.util.spec_from_file_location("fpv_segmentation_service", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

segment_fpv_frame = _MODULE.segment_fpv_frame


class FpvSegmentationServiceContractTest(unittest.TestCase):
    def test_contract_success_with_existing_artifacts(self) -> None:
        async def _run() -> dict:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:
                base = Path(tmp)
                frame_dir = base / "ride_foo" / "camera_front"
                frame_dir.mkdir(parents=True, exist_ok=True)
                np.save(frame_dir / "000005_segmentation.npy", np.zeros((6, 8), dtype=np.int32))
                (frame_dir / "class_mapping.json").write_text("{}", encoding="utf-8")
                (frame_dir / "000005_visualization.jpg").write_bytes(b"jpg")

                return await segment_fpv_frame(
                    sequence_name="ride_foo",
                    sam3_output_base_dir=str(base),
                    frame_name="000005",
                    camera_view="camera_front",
                    run_sam3=False,
                    request_id="req-contract-ok",
                )

        payload = asyncio.run(_run())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["provider"], "sam3")
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["request_id"], "req-contract-ok")
        self.assertIsNone(payload["error"])
        self.assertIn("timing_ms", payload)
        self.assertIn("total_ms", payload["timing_ms"])
        self.assertIn("run_ms", payload["timing_ms"])
        self.assertIn("artifact_resolve_ms", payload["timing_ms"])
        self.assertIsInstance(payload.get("artifacts"), dict)
        self.assertIsInstance(payload.get("fpv_segmentation"), dict)
        contract = (payload.get("artifacts") or {}).get("alignment_contract") or {}
        self.assertIsInstance(contract, dict)
        self.assertEqual(contract.get("schema_version"), 1)
        self.assertEqual((contract.get("frame_identity") or {}).get("frame_name"), "000005")
        self.assertEqual(contract.get("mask_shape_hw"), [6, 8])
        self.assertIsNone(contract.get("image_shape_hw"))
        self.assertIsNone(contract.get("shape_aligned_with_image"))
        self.assertEqual((payload.get("fpv_segmentation") or {}).get("alignment_contract"), contract)

    def test_contract_error_when_artifact_missing(self) -> None:
        async def _run() -> dict:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:
                return await segment_fpv_frame(
                    sequence_name="ride_missing",
                    sam3_output_base_dir=str(tmp),
                    frame_name="000005",
                    run_sam3=False,
                )

        payload = asyncio.run(_run())
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "SEGMENTATION_ARTIFACT_MISSING")
        self.assertFalse(bool(payload["error"].get("retryable")))

    def test_contract_error_when_provider_unsupported(self) -> None:
        async def _run() -> dict:
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:
                return await segment_fpv_frame(
                    sequence_name="ride_foo",
                    sam3_output_base_dir=str(tmp),
                    frame_name="000005",
                    provider="mock-provider",
                )

        payload = asyncio.run(_run())
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"]["code"], "SEGMENTATION_PROVIDER_UNSUPPORTED")
        self.assertFalse(bool(payload["error"].get("retryable")))


if __name__ == "__main__":
    unittest.main()
