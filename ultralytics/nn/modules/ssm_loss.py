import torch
import torch.nn as nn


class SSMLoss(nn.Module):
    """Total Variation (TV) Loss for smoothing offset fields in BDFWarpUp."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, offsets: list[torch.Tensor]) -> float | torch.Tensor:
        if not offsets:
            return 0.0

        loss = 0.0
        for offset in offsets:
            # offset shape: [B, 2, H, W]
            diff_w = torch.abs(offset[..., :, 1:] - offset[..., :, :-1])
            diff_h = torch.abs(offset[..., 1:, :] - offset[..., :-1, :])
            loss += diff_h.mean() + diff_w.mean()

        return self.weight * loss
