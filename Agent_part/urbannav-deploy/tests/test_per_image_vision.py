"""Validate the new numerical-policy adapter on unequal image shapes."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urbannav_deploy.per_image_vision import PerImageVision


class BatchSensitiveVision(torch.nn.Module):
    """Expose the observed dependency on batch size as a deterministic fixture."""
    dtype = torch.float32
    config = SimpleNamespace(spatial_merge_size=1)
    spatial_merge_size = 1

    def forward(self, pixels, grid_thw):
        return SimpleNamespace(pooler_output=pixels + len(grid_thw))


def test_output_independent_of_miss_grouping_with_unequal_images():
    adapter = PerImageVision(BatchSensitiveVision()).eval()
    pixels = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    grids = torch.tensor([[1, 1, 1], [1, 1, 2], [1, 2, 2]])
    combined = adapter(pixels, grids).pooler_output
    split = torch.cat([adapter(pixels[:1], grids[:1]).pooler_output,
                       adapter(pixels[1:], grids[1:]).pooler_output])
    assert torch.equal(combined, split)
    assert torch.equal(combined, pixels + 1)
    assert adapter.dtype == torch.float32
    assert adapter.spatial_merge_size == adapter.config.spatial_merge_size == 1
