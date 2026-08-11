"""Render private audit-only exact-future oracle cues for DF-OPD.

Unlike the coarse TC/TS cue renderer, this module intentionally receives an
exact target trajectory.  Its output is therefore restricted to the TO
evaluation-only branch and must never enter a trainable or Student manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from mcp_server.data.io.jsonl import file_sha256

DF_OPD_ORACLE_CUE_RENDER_POLICY_V1 = {
    'schema_version': 'urbannav_df_opd_oracle_cue_policy_v1',
    'canvas_px': 256,
    'background_rgb': [17, 24, 39],
    'trajectory_rgb': [251, 191, 36],
    'origin_rgb': [248, 250, 252],
    'meters_per_half_canvas': 8.0,
    'line_width_px': 5,
    'evaluation_only': True,
}
DF_OPD_ORACLE_CUE_RENDER_POLICY_DIGEST_V1 = hashlib.sha256(
    json.dumps(DF_OPD_ORACLE_CUE_RENDER_POLICY_V1,
               sort_keys=True,
               separators=(',', ':')).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class RenderedDFOPDOracleCueV1:
    """One private, exact-future artifact bound to a canonical TO view."""

    path: str
    cue_class: str
    render_policy_digest: str
    file_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return the artifact record without copying target coordinates."""

        return asdict(self)


def render_df_opd_exact_future_oracle_cue(
    output_path: str | Path,
    *,
    trajectory_ego_t: Sequence[Sequence[float]],
    action_mask: Sequence[bool],
) -> RenderedDFOPDOracleCueV1:
    """Render one exact trajectory solely for the private TO audit condition.

    Args:
        output_path: Private PNG destination.
        trajectory_ego_t: Cumulative FLU ``[x, y, yaw]`` targets.
        action_mask: Valid-prefix target mask aligned to the trajectory.

    Returns:
        Persisted private oracle-cue metadata.

    Raises:
        ValueError: If target geometry is malformed or the output is not PNG.
    """

    destination = Path(output_path).expanduser()
    if destination.suffix.lower() != '.png':
        raise ValueError('DF-OPD oracle cue output must use a .png extension')
    points = _validated_valid_prefix(trajectory_ego_t, action_mask)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_png_atomically(_render_oracle_image(points), destination)
    return RenderedDFOPDOracleCueV1(
        path=str(destination),
        cue_class='exact_future_oracle',
        render_policy_digest=DF_OPD_ORACLE_CUE_RENDER_POLICY_DIGEST_V1,
        file_sha256=file_sha256(destination),
    )


def _validated_valid_prefix(
    trajectory_ego_t: Sequence[Sequence[float]],
    action_mask: Sequence[bool],
) -> list[tuple[float, float]]:
    """Validate FLU target shape and extract its contiguous valid prefix."""

    if len(trajectory_ego_t) != len(action_mask) or not trajectory_ego_t:
        raise ValueError(
            'oracle trajectory/action_mask must be aligned and non-empty')
    if any(not isinstance(value, bool) for value in action_mask):
        raise ValueError('oracle action_mask must contain booleans')
    if any(not action_mask[index] and action_mask[index + 1]
           for index in range(len(action_mask) - 1)):
        raise ValueError('oracle action_mask must be a valid-prefix mask')
    valid_count = sum(action_mask)
    if valid_count < 1:
        raise ValueError('oracle action_mask requires one valid target point')
    points: list[tuple[float, float]] = []
    for index, raw in enumerate(trajectory_ego_t[:valid_count]):
        if len(raw) != 3:
            raise ValueError(
                f'oracle trajectory point {index} must have three FLU values')
        x, y, _yaw = (float(value) for value in raw)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f'oracle trajectory point {index} contains a non-finite position'
            )
        points.append((x, y))
    return points


def _render_oracle_image(points: Sequence[tuple[float, float]]) -> Image.Image:
    """Draw one fixed-scale FLU trajectory plot without text metadata."""

    policy = DF_OPD_ORACLE_CUE_RENDER_POLICY_V1
    size = int(policy['canvas_px'])
    center = size // 2
    scale = center / float(policy['meters_per_half_canvas'])
    image = Image.new('RGB', (size, size), tuple(policy['background_rgb']))
    draw = ImageDraw.Draw(image)

    def project(point: tuple[float, float]) -> tuple[int, int]:
        # FLU uses x-forward and y-left: forward points towards the top edge.
        x, y = point
        return (
            int(round(center - y * scale)),
            int(round(center - x * scale)),
        )

    projected = [project(point) for point in points]
    draw.line(
        [(center, center), *projected],
        fill=tuple(policy['trajectory_rgb']),
        width=int(policy['line_width_px']),
        joint='curve',
    )
    radius = 6
    draw.ellipse(
        (center - radius, center - radius, center + radius, center + radius),
        fill=tuple(policy['origin_rgb']),
    )
    return image


def _save_png_atomically(image: Image.Image, destination: Path) -> None:
    """Avoid partial private oracle artifacts after interruption."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.stem}.',
        suffix='.png',
        dir=str(destination.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format='PNG', optimize=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    'DF_OPD_ORACLE_CUE_RENDER_POLICY_DIGEST_V1',
    'DF_OPD_ORACLE_CUE_RENDER_POLICY_V1',
    'RenderedDFOPDOracleCueV1',
    'render_df_opd_exact_future_oracle_cue',
]
