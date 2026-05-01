"""Target generation utilities for FA-OCD + APD-prior v1 (supervision-rewritten)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .contact_targets import _morph_dilate, _morph_erode, _resize_instance_stack


def _extract_instance_stack(
    masks: torch.Tensor,
    proto_shape: tuple[int, int],
    overlap: bool,
    image_idx: int,
    batch_idx: torch.Tensor | None,
) -> torch.Tensor:
    """Extract one image worth of instance masks at proto resolution."""
    if overlap:
        index_mask = masks[image_idx]
        instance_ids = torch.unique(index_mask.long())
        instance_ids = instance_ids[instance_ids > 0]
        if instance_ids.numel() == 0:
            return masks.new_zeros((0, 1, *proto_shape))
        instance_stack = (index_mask.unsqueeze(0) == instance_ids[:, None, None]).unsqueeze(1).float()
    else:
        assert batch_idx is not None
        instance_stack = masks[batch_idx == image_idx].unsqueeze(1).float()
    if instance_stack.numel() == 0:
        return instance_stack
    return _resize_instance_stack(instance_stack, proto_shape)


def _instance_stats(instance_stack: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centroid and area statistics for a stack of binary instance masks."""
    _, _, h, w = instance_stack.shape
    ys = torch.arange(h, device=instance_stack.device, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.arange(w, device=instance_stack.device, dtype=torch.float32).view(1, 1, 1, w)
    areas = instance_stack.sum(dim=(1, 2, 3)).clamp_min_(1.0)
    cy = (instance_stack * ys).sum(dim=(1, 2, 3)) / areas
    cx = (instance_stack * xs).sum(dim=(1, 2, 3)) / areas
    centroids = torch.stack((cx, cy), dim=1)
    return centroids, areas, instance_stack.new_zeros((0,))


def _canonical_axis_sign(axis: torch.Tensor) -> torch.Tensor:
    """Resolve PCA sign ambiguity deterministically."""
    if axis.abs()[0] > axis.abs()[1]:
        return axis if axis[0] >= 0 else -axis
    return axis if axis[1] >= 0 else -axis


def compute_apd_code(instance_mask: torch.Tensor) -> torch.Tensor:
    """Compute a compact 4D anisotropic pole descriptor from one binary instance mask."""
    device_type = instance_mask.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        instance_mask = instance_mask.to(dtype=torch.float32)
        coords = torch.nonzero(instance_mask > 0.5, as_tuple=False).to(dtype=torch.float32)
        if coords.shape[0] < 3:
            return instance_mask.new_tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        yx = coords[:, -2:]
        centered = yx - yx.mean(dim=0, keepdim=True)
        cov = centered.t().matmul(centered) / max(centered.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order]
        axis_yx = eigvecs[:, order[0]]
        axis_xy = _canonical_axis_sign(torch.stack((axis_yx[1], axis_yx[0])))
        axis_xy = axis_xy / axis_xy.norm().clamp_min(1e-6)

        major = eigvals[0].clamp_min(1e-6).sqrt()
        minor = eigvals[1].clamp_min(1e-6).sqrt()
        elongation = ((major - minor) / (major + minor + 1e-6)).clamp(0.0, 1.0)
        anisotropy = torch.tanh(torch.log((major + 1e-6) / (minor + 1e-6)))
        return torch.stack((axis_xy[0], axis_xy[1], elongation, anisotropy))


def _build_pair_fields(
    instance_stack: torch.Tensor,
    ambiguity_mask: torch.Tensor,
    membership: torch.Tensor,
    centroids: torch.Tensor,
    rho_temperature: float,
    build_xi: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build GT-local pair ids, anonymous-slot ownership, and pair-axis cue targets."""
    _, _, h, w = instance_stack.shape
    device = instance_stack.device
    pair_ids = torch.full((2, h, w), -1, device=device, dtype=torch.long)
    rho = torch.zeros((2, h, w), device=device, dtype=torch.float32)
    xi = torch.zeros((2, h, w), device=device, dtype=torch.float32) if build_xi else instance_stack.new_zeros((0,))

    if ambiguity_mask.sum() == 0:
        return pair_ids, rho, xi

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx.unsqueeze(0) - centroids[:, 0].view(-1, 1, 1)
    dy = yy.unsqueeze(0) - centroids[:, 1].view(-1, 1, 1)
    radial_score = membership / (dx.square() + dy.square() + 1.0)
    top2_score, top2_idx = radial_score.topk(k=min(2, radial_score.shape[0]), dim=0)
    if top2_idx.shape[0] < 2:
        return pair_ids, rho, xi

    idx0 = top2_idx[0]
    idx1 = top2_idx[1]
    cen0 = centroids[idx0]
    cen1 = centroids[idx1]
    swap = (cen0[..., 0] > cen1[..., 0]) | ((cen0[..., 0] == cen1[..., 0]) & (cen0[..., 1] > cen1[..., 1]))
    slot1 = torch.where(swap, idx1, idx0)
    slot2 = torch.where(swap, idx0, idx1)

    score1 = torch.where(swap, top2_score[1], top2_score[0])
    score2 = torch.where(swap, top2_score[0], top2_score[1])
    score = torch.stack((score1, score2), dim=0) / max(float(rho_temperature), 1e-6)
    rho_local = score.softmax(dim=0)

    xi_local = None
    if build_xi:
        c1 = centroids[slot1]
        c2 = centroids[slot2]
        axis = c2 - c1
        axis_norm = axis.norm(dim=-1, keepdim=True).clamp_min(1.0)
        axis = axis / axis_norm
        midpoint = 0.5 * (c1 + c2)
        proj = ((xx - midpoint[..., 0]) * axis[..., 0] + (yy - midpoint[..., 1]) * axis[..., 1]) / (
            0.5 * axis_norm[..., 0] + 1e-6
        )
        proj = proj.clamp(-1.0, 1.0)
        xi_local = torch.stack(((1.0 - proj) * 0.5, (1.0 + proj) * 0.5), dim=0)
        xi_local = xi_local / xi_local.sum(dim=0, keepdim=True).clamp_min(1e-6)

    pair_ids[0][ambiguity_mask] = slot1[ambiguity_mask]
    pair_ids[1][ambiguity_mask] = slot2[ambiguity_mask]
    rho[:, ambiguity_mask] = rho_local[:, ambiguity_mask]
    if build_xi and xi_local is not None:
        xi[:, ambiguity_mask] = xi_local[:, ambiguity_mask]
    return pair_ids, rho, xi


def build_ocd_targets(
    masks: torch.Tensor,
    proto_shape: tuple[int, int] | torch.Size,
    overlap: bool = True,
    batch_idx: torch.Tensor | None = None,
    batch_size: int | None = None,
    min_area: float = 4.0,
    dilation_radius: int = 1,
    large_instance_thresh: float = 256.0,
    large_instance_radius: int = 2,
    rho_temperature: float = 1.0,
    pair_axis_enabled: bool = False,
    apd_enabled: bool = False,
    return_debug: bool = False,
) -> dict[str, torch.Tensor]:
    """Build GT-local FA-OCD supervision targets at proto resolution."""
    proto_shape = tuple(int(x) for x in proto_shape[-2:])
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if overlap:
        if masks.ndim != 3:
            raise ValueError(f"Expected overlap masks with shape [B, H, W], got {tuple(masks.shape)}.")
        inferred_batch = masks.shape[0]
    else:
        if masks.ndim != 3:
            raise ValueError(f"Expected per-instance masks with shape [N, H, W], got {tuple(masks.shape)}.")
        if batch_idx is None:
            raise ValueError("batch_idx is required when overlap_mask=False.")
        batch_idx = batch_idx.view(-1)
        inferred_batch = int(batch_size or (int(batch_idx.max().item()) + 1 if batch_idx.numel() else 0))

    mult_tgt = torch.zeros((inferred_batch, *proto_shape), device=masks.device, dtype=torch.long)
    amb_mask = torch.zeros((inferred_batch, 1, *proto_shape), device=masks.device, dtype=torch.float32)
    rho_tgt = torch.zeros((inferred_batch, 2, *proto_shape), device=masks.device, dtype=torch.float32)
    xi_tgt = torch.zeros((inferred_batch, 2, *proto_shape), device=masks.device, dtype=torch.float32)
    pair_ids = torch.full((inferred_batch, 2, *proto_shape), -1, device=masks.device, dtype=torch.long)
    apd_codes: list[torch.Tensor] = [] if apd_enabled else []

    for image_idx in range(inferred_batch):
        instance_stack = _extract_instance_stack(
            masks=masks, proto_shape=proto_shape, overlap=overlap, image_idx=image_idx, batch_idx=batch_idx
        )
        if instance_stack.shape[0] == 0:
            if apd_enabled:
                apd_codes.append(torch.zeros((0, 4), device=masks.device, dtype=torch.float32))
            continue

        areas = instance_stack.sum(dim=(1, 2, 3))
        keep = areas >= float(min_area)
        instance_stack = instance_stack[keep]
        if instance_stack.shape[0] == 0:
            if apd_enabled:
                apd_codes.append(torch.zeros((0, 4), device=masks.device, dtype=torch.float32))
            continue

        centroids, _, _ = _instance_stats(instance_stack)
        binary = (instance_stack > 0.5).float()
        eroded = (_morph_erode(binary, 1) > 0.5).float()
        if int(dilation_radius) == int(large_instance_radius):
            dilated = (_morph_dilate(binary, int(dilation_radius)) > 0.5).float()
        else:
            large = areas[keep] >= float(large_instance_thresh)
            dilated = torch.zeros_like(binary)
            if (~large).any():
                dilated[~large] = (_morph_dilate(binary[~large], int(dilation_radius)) > 0.5).float()
            if large.any():
                dilated[large] = (_morph_dilate(binary[large], int(large_instance_radius)) > 0.5).float()

        active = (dilated * (1.0 - eroded)).float()
        interior = binary.sum(dim=0).squeeze(0)
        multiplicity = active.sum(dim=0).squeeze(0)
        mult = torch.zeros_like(multiplicity, dtype=torch.long)
        mult[interior > 0] = 1
        mult[multiplicity >= 2] = 2
        mult[multiplicity >= 3] = 3
        mult_tgt[image_idx] = mult

        ambiguity = multiplicity >= 2
        amb_mask[image_idx, 0] = ambiguity.float()
        membership = active.squeeze(1)
        pair_map, rho_map, xi_map = _build_pair_fields(
            instance_stack=instance_stack,
            ambiguity_mask=ambiguity,
            membership=membership,
            centroids=centroids,
            rho_temperature=rho_temperature,
            build_xi=pair_axis_enabled,
        )
        pair_ids[image_idx] = pair_map
        rho_tgt[image_idx] = rho_map
        if pair_axis_enabled:
            xi_tgt[image_idx] = xi_map

        if apd_enabled:
            apd_codes.append(torch.stack([compute_apd_code(mask[0]) for mask in instance_stack], dim=0))

    targets = {
        "mult_tgt": mult_tgt,
        "amb_mask": amb_mask,
        "rho_tgt": rho_tgt,
    }
    if pair_axis_enabled:
        targets["xi_tgt"] = xi_tgt
    if apd_enabled:
        targets["apd_tgt"] = apd_codes
    if return_debug:
        targets["pair_ids"] = pair_ids
    return targets
