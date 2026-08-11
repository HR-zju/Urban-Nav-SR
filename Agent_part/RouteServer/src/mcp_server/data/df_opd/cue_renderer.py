"""Render auditable coarse-motion cue images for private DF-OPD Teacher views.

The renderer has a deliberately narrow input space.  It maps a profile-owned
direction label to one of four coarse classes and draws a fixed icon.  It does
not accept trajectory points, speeds, timestamps, map coordinates, or any
other value that could encode an exact-future oracle in the media pixels.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from mcp_server.contracts.direction_profile import coarse_motion_for_direction_label
from mcp_server.data.io.jsonl import file_sha256


DF_OPD_CUE_RENDER_POLICY_V1 = {
    "schema_version": "urbannav_df_opd_coarse_cue_policy_v1",
    "canvas_px": 128,
    "background_rgb": [17, 24, 39],
    "arrow_rgb": [248, 250, 252],
    "stop_rgb": [239, 68, 68],
    "line_width_px": 12,
    "allowed_cue_classes": ["go_straight", "turn_left", "turn_right", "stop"],
    "forbidden_source_fields": [
        "trajectory_ego_t",
        "action_ego_t",
        "exact_future_trajectory",
        "timestamp",
        "speed",
        "map_coordinate",
    ],
}
DF_OPD_CUE_RENDER_POLICY_DIGEST_V1 = hashlib.sha256(
    json.dumps(DF_OPD_CUE_RENDER_POLICY_V1, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class RenderedDFOPDCueV1:
    """One persisted private Teacher cue with reproducibility stamps.

    Attributes:
        path: Relative or absolute path of the PNG artifact.
        cue_class: One of four coarse motion classes, never an exact label.
        render_policy_digest: Immutable renderer policy identity.
        file_sha256: Digest of the generated PNG bytes.
    """

    path: str
    cue_class: str
    render_policy_digest: str
    file_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe private-artifact manifest entry."""

        return asdict(self)


def render_df_opd_coarse_motion_cue(
    output_path: str | Path,
    *,
    direction_profile_id: str,
    direction_label_name: str,
) -> RenderedDFOPDCueV1:
    """Write a deterministic PNG cue for one profile-owned direction label.

    Args:
        output_path: Destination PNG path, normally below a Teacher-only cue
            artifact root.
        direction_profile_id: Direction profile that owns the input label.
        direction_label_name: Exact public decision label.  It is reduced to a
            coarse class before any pixels are drawn and is not stored in the
            returned manifest.

    Returns:
        A private-artifact record containing the coarse class and byte digest.

    Raises:
        ValueError: If the profile label cannot be reduced to an approved
            coarse class or the destination is not a PNG.
    """

    cue_class = coarse_motion_for_direction_label(direction_profile_id, direction_label_name)
    allowed = set(DF_OPD_CUE_RENDER_POLICY_V1["allowed_cue_classes"])
    if cue_class not in allowed:
        raise ValueError(f"unsupported DF-OPD coarse cue class: {cue_class!r}")
    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != ".png":
        raise ValueError("DF-OPD coarse cue output must use a .png extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = _render_cue_image(cue_class)
    _save_png_atomically(image, destination)
    return RenderedDFOPDCueV1(
        path=str(destination),
        cue_class=cue_class,
        render_policy_digest=DF_OPD_CUE_RENDER_POLICY_DIGEST_V1,
        file_sha256=file_sha256(destination),
    )


def _render_cue_image(cue_class: str) -> Image.Image:
    """Draw one fixed icon from a coarse class without semantic text labels."""

    policy = DF_OPD_CUE_RENDER_POLICY_V1
    size = int(policy["canvas_px"])
    center = size // 2
    image = Image.new("RGB", (size, size), tuple(policy["background_rgb"]))
    draw = ImageDraw.Draw(image)
    width = int(policy["line_width_px"])
    arrow_color = tuple(policy["arrow_rgb"])
    if cue_class == "stop":
        radius = 34
        points = [
            (center - radius // 2, center - radius),
            (center + radius // 2, center - radius),
            (center + radius, center - radius // 2),
            (center + radius, center + radius // 2),
            (center + radius // 2, center + radius),
            (center - radius // 2, center + radius),
            (center - radius, center + radius // 2),
            (center - radius, center - radius // 2),
        ]
        draw.polygon(points, fill=tuple(policy["stop_rgb"]))
        return image

    if cue_class == "go_straight":
        points = [(center, 100), (center, 30)]
        head = [(center, 18), (center - 18, 42), (center + 18, 42)]
    elif cue_class == "turn_left":
        points = [(96, 96), (64, 96), (36, 68), (36, 38)]
        head = [(24, 30), (48, 30), (36, 12)]
    elif cue_class == "turn_right":
        points = [(32, 96), (64, 96), (92, 68), (92, 38)]
        head = [(80, 30), (104, 30), (92, 12)]
    else:
        raise ValueError(f"unsupported DF-OPD coarse cue class: {cue_class!r}")
    draw.line(points, fill=arrow_color, width=width, joint="curve")
    draw.polygon(head, fill=arrow_color)
    return image


def _save_png_atomically(image: Image.Image, destination: Path) -> None:
    """Avoid partially written private cue artifacts on interrupted exports."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=".png",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", optimize=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "DF_OPD_CUE_RENDER_POLICY_DIGEST_V1",
    "DF_OPD_CUE_RENDER_POLICY_V1",
    "RenderedDFOPDCueV1",
    "render_df_opd_coarse_motion_cue",
]
