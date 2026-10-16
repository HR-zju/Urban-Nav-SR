"""Fix the numerical image-encoding unit independently of cache miss batches."""
import torch


class PerImageVision(torch.nn.Module):
    """Encode each cache miss as one image using the frozen native vision tower.

    Args:
        visual: Loaded native Qwen vision module, already in evaluation mode.

    The image cache assumes E(image) is independent of request history. BF16
    batched kernels can violate bitwise independence when the miss batch changes.
    Fixing the encoder invocation to one image gives every cached entry the same
    numerical path. This adapter is installed inside CachedQwenVision; its output
    contract needs merged pooler_output only, matching that cache's consumer.
    """
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    @property
    def dtype(self):
        """Return the frozen native visual precision."""
        return self.visual.dtype

    @property
    def config(self):
        """Return the original vision configuration."""
        return self.visual.config

    @property
    def spatial_merge_size(self):
        """Return the original spatial patch merge factor."""
        return self.visual.spatial_merge_size

    def forward(self, hidden_states, grid_thw, **kwargs):
        """Concatenate image-local merged embeddings in their original order.

        Args:
            hidden_states: Concatenated image patches on the native device/dtype.
            grid_thw: Per-image patch dimensions [N,3].
            **kwargs: Native vision-forward options.
        Returns:
            BaseModelOutputWithPooling with merged embeddings in input order.
        """
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        sizes = [int(t * h * w) for t, h, w in grid_thw.tolist()]
        # 固定 E(I_i)=Vision(I_i, grid_i)，使缓存出生批次不再改变 BF16 数值路径。
        outputs = [self.visual(patches, grid_thw=grid_thw[i:i + 1], **kwargs).pooler_output
                   for i, patches in enumerate(hidden_states.split(sizes))]
        return BaseModelOutputWithPooling(pooler_output=torch.cat(outputs, dim=0))
