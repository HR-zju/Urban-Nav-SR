"""CPU-only tests for DF-OPD coarse-motion cue rendering."""

from __future__ import annotations

from PIL import Image

from mcp_server.data.df_opd.cue_renderer import (
    DF_OPD_CUE_RENDER_POLICY_DIGEST_V1,
    render_df_opd_coarse_motion_cue,
)


_PROFILE = "fpv_degree10_terminal_centroid_v2"


def test_renderer_reduces_fine_direction_to_stable_coarse_private_icon(tmp_path):
    first = render_df_opd_coarse_motion_cue(
        tmp_path / "first.png",
        direction_profile_id=_PROFILE,
        direction_label_name="left_50",
    )
    second = render_df_opd_coarse_motion_cue(
        tmp_path / "second.png",
        direction_profile_id=_PROFILE,
        direction_label_name="left_50",
    )

    assert first.cue_class == "turn_left"
    assert first.render_policy_digest == DF_OPD_CUE_RENDER_POLICY_DIGEST_V1
    assert first.file_sha256 == second.file_sha256
    with Image.open(first.path) as image:
        assert image.mode == "RGB"
        assert image.size == (128, 128)


def test_stop_renderer_uses_only_coarse_stop_and_requires_png(tmp_path):
    cue = render_df_opd_coarse_motion_cue(
        tmp_path / "stop.png",
        direction_profile_id=_PROFILE,
        direction_label_name="stop",
    )

    assert cue.cue_class == "stop"
    assert "trajectory" not in cue.to_dict()
