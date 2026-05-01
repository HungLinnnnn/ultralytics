"""Shared realization utilities for FA-OCD + APD-prior v1 (supervision-rewritten)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .ops import crop_mask, scale_masks


def _candidate_order(boxes: torch.Tensor) -> torch.Tensor:
    """Return a deterministic lexicographic ordering rank for candidate boxes."""
    centers = 0.5 * (boxes[:, :2] + boxes[:, 2:4])
    order = torch.argsort(centers[:, 0] * 100000.0 + centers[:, 1])
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.shape[0], device=boxes.device)
    return rank


def _pair_apd_prior(box_i: torch.Tensor, box_j: torch.Tensor, apd_i: torch.Tensor, apd_j: torch.Tensor) -> tuple[float, float]:
    """Compute a small bounded pairwise APD prior from candidate geometry."""
    axis = 0.5 * (box_j[:2] + box_j[2:4]) - 0.5 * (box_i[:2] + box_i[2:4])
    axis = axis / axis.norm().clamp_min(1e-6)
    align_i = torch.abs(torch.dot(apd_i[:2], axis)).clamp(0.0, 1.0)
    align_j = torch.abs(torch.dot(apd_j[:2], axis)).clamp(0.0, 1.0)
    prior_i = torch.tanh(apd_i[3] * (0.5 + 0.5 * align_i) + apd_i[2] * 0.5)
    prior_j = torch.tanh(apd_j[3] * (0.5 + 0.5 * align_j) + apd_j[2] * 0.5)
    return float(prior_i.item()), float(prior_j.item())


def refine_mask_logits(
    mask_coeff: torch.Tensor,
    proto: torch.Tensor,
    boxes: torch.Tensor,
    mult_map: torch.Tensor | None = None,
    rho_map: torch.Tensor | None = None,
    xi_map: torch.Tensor | None = None,
    apd_code: torch.Tensor | None = None,
    ambiguity_threshold: float = 0.35,
    realization_alpha: float = 0.25,
    apd_gamma: float = 0.1,
    xi_bridge_weight: float = 0.15,
    xi_only: bool = False,
    stability_threshold: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Refine local ambiguous mask logits using anonymous-slot ownership and optional APD prior."""
    c, mh, mw = proto.shape
    mask_logits = (mask_coeff @ proto.float().view(c, -1)).view(-1, mh, mw)
    debug = {
        "amb_mask": torch.zeros((mh, mw), device=proto.device, dtype=torch.bool),
        "pair_top1": torch.full((mh, mw), -1, device=proto.device, dtype=torch.long),
        "pair_top2": torch.full((mh, mw), -1, device=proto.device, dtype=torch.long),
    }
    if mask_logits.shape[0] < 2 or mult_map is None:
        return mask_logits, debug

    mult_prob = mult_map.softmax(dim=0)
    amb_mask = mult_prob[2:].sum(dim=0) >= float(ambiguity_threshold)
    top2_vals, top2_idx = mask_logits.topk(k=2, dim=0)
    margin = (top2_vals[0] - top2_vals[1]).abs()
    amb_mask = amb_mask & (margin <= float(stability_threshold))
    if amb_mask.sum() == 0:
        return mask_logits, debug

    rank = _candidate_order(boxes)
    idx0 = top2_idx[0]
    idx1 = top2_idx[1]
    swap = rank[idx0] > rank[idx1]
    slot1 = torch.where(swap, idx1, idx0)
    slot2 = torch.where(swap, idx0, idx1)
    debug["amb_mask"] = amb_mask
    debug["pair_top1"][amb_mask] = slot1[amb_mask]
    debug["pair_top2"][amb_mask] = slot2[amb_mask]

    if rho_map is None and xi_map is None:
        return mask_logits, debug

    slot_logits = None
    if rho_map is not None and not xi_only:
        slot_logits = rho_map.clone()
    if xi_map is not None:
        slot_logits = xi_map.clone() if slot_logits is None else slot_logits + float(xi_bridge_weight) * xi_map
    slot_probs = slot_logits.softmax(dim=0)
    refined = mask_logits.clone()

    pair_token = slot1 * mask_logits.shape[0] + slot2
    for token in torch.unique(pair_token[amb_mask]):
        a = int((token // mask_logits.shape[0]).item())
        b = int((token % mask_logits.shape[0]).item())
        pair_mask = amb_mask & (slot1 == a) & (slot2 == b)
        if pair_mask.sum() == 0:
            continue

        prior_a = prior_b = 0.0
        if apd_code is not None and apd_code.shape[0] > max(a, b):
            prior_a, prior_b = _pair_apd_prior(boxes[a], boxes[b], apd_code[a], apd_code[b])

        base_prob_a = torch.sigmoid(mask_logits[a][pair_mask])
        base_prob_b = torch.sigmoid(mask_logits[b][pair_mask])
        budget = (base_prob_a + base_prob_b).clamp(0.0, 1.0)

        score_a = mask_logits[a][pair_mask] + float(realization_alpha) * torch.log(slot_probs[0][pair_mask].clamp_min(1e-6))
        score_b = mask_logits[b][pair_mask] + float(realization_alpha) * torch.log(slot_probs[1][pair_mask].clamp_min(1e-6))
        if apd_code is not None:
            score_a = score_a + float(apd_gamma) * prior_a
            score_b = score_b + float(apd_gamma) * prior_b

        refined_pair = torch.stack((score_a, score_b), dim=0).softmax(dim=0)
        refined_prob = (refined_pair * budget.unsqueeze(0)).clamp(1e-4, 1 - 1e-4)
        refined[a][pair_mask] = torch.logit(refined_prob[0])
        refined[b][pair_mask] = torch.logit(refined_prob[1])

    return refined, debug


def decode_masks_with_ocd(
    proto: torch.Tensor,
    mask_coeff: torch.Tensor,
    boxes: torch.Tensor,
    shape: tuple[int, int],
    mult_map: torch.Tensor | None = None,
    rho_map: torch.Tensor | None = None,
    xi_map: torch.Tensor | None = None,
    apd_code: torch.Tensor | None = None,
    native: bool = False,
    ambiguity_threshold: float = 0.35,
    realization_alpha: float = 0.25,
    apd_gamma: float = 0.1,
    xi_bridge_weight: float = 0.15,
    xi_only: bool = False,
    stability_threshold: float = 0.2,
    upsample: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decode masks and apply the shared FA-OCD realization bridge if enabled."""
    logits, debug = refine_mask_logits(
        mask_coeff=mask_coeff,
        proto=proto,
        boxes=boxes,
        mult_map=mult_map,
        rho_map=rho_map,
        xi_map=xi_map,
        apd_code=apd_code,
        ambiguity_threshold=ambiguity_threshold,
        realization_alpha=realization_alpha,
        apd_gamma=apd_gamma,
        xi_bridge_weight=xi_bridge_weight,
        xi_only=xi_only,
        stability_threshold=stability_threshold,
    )
    if native:
        masks = scale_masks(logits[None], shape)[0]
        masks = crop_mask(masks, boxes)
    else:
        width_ratio = logits.shape[-1] / shape[1]
        height_ratio = logits.shape[-2] / shape[0]
        ratios = torch.tensor([[width_ratio, height_ratio, width_ratio, height_ratio]], device=boxes.device)
        masks = crop_mask(logits, boxes=boxes * ratios)
        if upsample:
            masks = F.interpolate(masks[None], shape, mode="bilinear")[0]
    return masks.gt_(0.0).byte(), debug
