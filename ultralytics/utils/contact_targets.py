"""Contact-zone target generation utilities for FA-CZS v1."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _morph_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply binary dilation with max pooling."""
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _morph_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply binary erosion with max pooling."""
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel, stride=1, padding=radius)


def _resize_mask(mask: torch.Tensor, proto_shape: tuple[int, int]) -> torch.Tensor:
    """Resize a single-instance mask to proto resolution."""
    if mask.shape[-2:] == proto_shape:
        return mask
    return F.interpolate(mask, size=proto_shape, mode="nearest")


def _pairwise_contact_core(
    instance_masks: list[torch.Tensor],
    min_area: float,
    dilation_radius: int,
    large_instance_thresh: float,
    large_instance_radius: int,
) -> torch.Tensor | None:
    """Build a narrow binary contact core from per-instance masks."""
    processed = []
    for instance_mask in instance_masks:
        if float(instance_mask.sum().item()) < min_area:
            continue
        radius = large_instance_radius if float(instance_mask.sum().item()) >= large_instance_thresh else dilation_radius
        binary_mask = (instance_mask > 0.5).float()
        processed.append((binary_mask, _morph_dilate(binary_mask, radius), _morph_erode(binary_mask, 1)))

    if len(processed) < 2:
        return None

    core = torch.zeros_like(processed[0][0], dtype=torch.bool)
    for i in range(len(processed)):
        _, dil_i, erode_i = processed[i]
        for j in range(i + 1, len(processed)):
            _, dil_j, erode_j = processed[j]
            overlap = (dil_i > 0.5) & (dil_j > 0.5)
            interiors = (erode_i > 0.5) | (erode_j > 0.5)
            core |= overlap & ~interiors
    return core.float()


def _soften_contact_core(core: torch.Tensor, blur_kernel: int, blur_passes: int) -> torch.Tensor:
    """Broaden the binary contact core into a narrow soft target."""
    blur_kernel = max(int(blur_kernel), 1)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    soft = core.float()
    for _ in range(max(int(blur_passes), 0)):
        soft = F.avg_pool2d(soft, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
    return torch.maximum(soft, core.float()).clamp_(0.0, 1.0)


def build_contact_targets(
    masks: torch.Tensor,
    proto_shape: tuple[int, int] | torch.Size,
    overlap: bool = True,
    batch_idx: torch.Tensor | None = None,
    batch_size: int | None = None,
    min_area: float = 4.0,
    dilation_radius: int = 1,
    large_instance_thresh: float = 256.0,
    large_instance_radius: int = 2,
    blur_kernel: int = 3,
    blur_passes: int = 1,
    return_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build soft contact-band targets at proto resolution from instance masks."""
    proto_shape = tuple(int(x) for x in proto_shape[-2:])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if overlap:
        if masks.ndim != 3:
            raise ValueError(f"Expected overlap masks with shape [B, H, W], but got {tuple(masks.shape)}.")
        inferred_batch = masks.shape[0]
    else:
        if masks.ndim != 3:
            raise ValueError(f"Expected per-instance masks with shape [N, H, W], but got {tuple(masks.shape)}.")
        if batch_idx is None:
            raise ValueError("batch_idx is required when overlap_mask=False.")
        batch_idx = batch_idx.view(-1)
        inferred_batch = int(batch_size or (int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0))

    targets = torch.zeros(inferred_batch, 1, *proto_shape, device=masks.device, dtype=torch.float32)
    contact_cores = torch.zeros_like(targets) if return_debug else None

    for image_idx in range(inferred_batch):
        if overlap:
            instance_index_mask = masks[image_idx]
            instance_masks = []
            instance_ids = torch.unique(instance_index_mask.long())
            instance_ids = instance_ids[instance_ids > 0]
            for instance_id in instance_ids:
                instance_mask = (instance_index_mask == instance_id).float().unsqueeze(0).unsqueeze(0)
                instance_masks.append(_resize_mask(instance_mask, proto_shape))
        else:
            instance_masks = []
            image_instances = masks[batch_idx == image_idx]
            for instance_mask in image_instances:
                instance_masks.append(_resize_mask(instance_mask.unsqueeze(0).unsqueeze(0).float(), proto_shape))

        core = _pairwise_contact_core(
            instance_masks=instance_masks,
            min_area=min_area,
            dilation_radius=dilation_radius,
            large_instance_thresh=large_instance_thresh,
            large_instance_radius=large_instance_radius,
        )
        if core is None:
            continue

        targets[image_idx] = _soften_contact_core(core, blur_kernel=blur_kernel, blur_passes=blur_passes)
        if return_debug:
            contact_cores[image_idx] = core

    if return_debug:
        return targets, {"contact_core": contact_cores, "contact_target": targets}
    return targets
