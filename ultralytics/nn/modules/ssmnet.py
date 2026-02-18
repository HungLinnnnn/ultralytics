import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaborLLDLow(nn.Module):
    """Pixel-wise softmax low-pass filter (depthwise) to obtain F_low."""

    def __init__(self, c: int, k: int = 5, r: int = 4):
        super().__init__()
        self.k = k
        self.pad = k // 2
        hidden = max(1, c // r)
        self.reduce = nn.Conv2d(c, hidden, 1)
        self.kernel_logits = nn.Conv2d(hidden, c * k * k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.k
        logits = self.kernel_logits(self.reduce(x))  # (b, c*k*k, h, w)
        logits = logits.view(b, c, k * k, h, w)
        weights = logits.softmax(dim=2)
        patches = F.unfold(x, k, padding=self.pad)  # (b, c*k*k, h*w)
        patches = patches.view(b, c, k * k, h, w)
        out = (weights * patches).sum(dim=2)
        return out


class GaborLLDHigh(nn.Module):
    """High-frequency guidance from shared low-pass: expects (x_raw, x_low) and processes residual with Gabor bank."""

    def __init__(
        self,
        c: int,
        k: int = 7,
        r: int = 4,
        cg: int = 256,
        n_orient: int = 4,
        n_scales: int = 2,
        magnitude: bool = True,
    ):
        super().__init__()
        mid = max(1, c // r)
        self.proj_r = nn.Conv2d(c, mid, 1)
        self.gabor_real, self.gabor_imag = self._build_gabor_bank(k, n_orient, n_scales)
        self.magnitude = magnitude
        self.fuse = nn.Sequential(
            nn.Conv2d(mid + mid * n_orient * n_scales, cg, 1, bias=False),
            nn.BatchNorm2d(cg),
            nn.SiLU(),
        )

    @staticmethod
    def _build_gabor_bank(k: int, n_orient: int, n_scales: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_base = k / 3.0
        filters_r = []
        filters_i = []
        for s in range(n_scales):
            sigma = sigma_base * (0.8 + 0.4 * s)
            lam = 0.8 * k / (s + 1)
            for o in range(n_orient):
                theta = math.pi * o / n_orient
                kernel_r = torch.zeros((k, k))
                kernel_i = torch.zeros((k, k))
                for y in range(k):
                    for x in range(k):
                        xp = (x - k // 2) * math.cos(theta) + (y - k // 2) * math.sin(theta)
                        yp = -(x - k // 2) * math.sin(theta) + (y - k // 2) * math.cos(theta)
                        gauss = math.exp(-(xp ** 2 + yp ** 2) / (2 * sigma * sigma))
                        carrier_r = math.cos(2 * math.pi * xp / lam)
                        carrier_i = math.sin(2 * math.pi * xp / lam)
                        kernel_r[y, x] = gauss * carrier_r
                        kernel_i[y, x] = gauss * carrier_i
                kernel_r -= kernel_r.mean()
                kernel_i -= kernel_i.mean()
                filters_r.append(kernel_r)
                filters_i.append(kernel_i)
        bank_r = torch.stack(filters_r).unsqueeze(1)  # (n,1,k,k)
        bank_i = torch.stack(filters_i).unsqueeze(1)
        return bank_r, bank_i

    def _apply_gabor(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, mid, h, w)
        b, mid, h, w = x.shape
        device = x.device
        dtype = x.dtype
        n = self.gabor_real.shape[0]
        weight_r = self.gabor_real.to(device=device, dtype=dtype).repeat(mid, 1, 1, 1)
        weight_i = self.gabor_imag.to(device=device, dtype=dtype).repeat(mid, 1, 1, 1)
        out_r = F.conv2d(x, weight_r, padding=self.gabor_real.shape[-1] // 2, groups=mid)
        out_i = F.conv2d(x, weight_i, padding=self.gabor_imag.shape[-1] // 2, groups=mid)
        if self.magnitude:
            out = torch.sqrt(out_r.pow(2) + out_i.pow(2) + 1e-6)
        else:
            out = torch.abs(out_r) + torch.abs(out_i)
        out = out.view(b, mid * n, h, w)
        return out

    def forward(self, x: torch.Tensor, x_low: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_low is None:
            if isinstance(x, (tuple, list)) and len(x) == 2:
                x, x_low = x
            else:
                raise ValueError("GaborLLDHigh expects inputs (x_raw, x_low).")
        residual = x - x_low
        r_proj = self.proj_r(residual)
        gabor_feat = self._apply_gabor(r_proj)
        fused = torch.cat([r_proj, gabor_feat], dim=1)
        return self.fuse(fused)


class MambaCM(nn.Module):
    """Lightweight 2D consistency module (approximate Mamba-style directional aggregation)."""

    def __init__(self, c: int, d_state: int = 16, expand: float = 2.0, directions: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(c * expand)
        self.norm = nn.LayerNorm(c)
        self.in_proj = nn.Conv2d(c, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.out_proj = nn.Conv2d(hidden, c, 1)
        self.directions = directions
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _cumulative_avg(x: torch.Tensor, dim: int) -> torch.Tensor:
        cumsum = torch.cumsum(x, dim=dim)
        counts = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
        shape = [1, 1, 1, 1]
        shape[dim] = -1
        counts = counts.view(shape)
        return cumsum / counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_flat).permute(0, 3, 1, 2).contiguous()
        y = self.in_proj(x_norm)
        y = self.dw(y)
        agg = 0
        if self.directions >= 1:
            agg = agg + self._cumulative_avg(y, dim=2)
        if self.directions >= 2:
            agg = agg + torch.flip(self._cumulative_avg(torch.flip(y, dims=[2]), dim=2), dims=[2])
        if self.directions >= 3:
            agg = agg + self._cumulative_avg(y, dim=3)
        if self.directions >= 4:
            agg = agg + torch.flip(self._cumulative_avg(torch.flip(y, dims=[3]), dim=3), dims=[3])
        agg = agg / float(min(self.directions, 4))
        out = self.out_proj(agg)
        out = self.dropout(out)
        return out


class BDFUpsample(nn.Module):
    """Boundary-aware deformable upsample guided by high- and low-frequency cues."""

    def __init__(self, c_deep: int, c_high: int, c_low: int, scale: int = 2, max_offset: float = 2.0):
        super().__init__()
        self.c = c_deep
        self.scale = scale
        self.max_offset = max_offset
        self.offset_net = nn.Sequential(
            nn.Conv2d(c_high + c_low, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(
        self, feat_deep: torch.Tensor, guide_high: torch.Tensor = None, guide_low: torch.Tensor = None
    ) -> torch.Tensor:
        if guide_high is None and isinstance(feat_deep, (list, tuple)) and len(feat_deep) == 3:
            feat_deep, guide_high, guide_low = feat_deep
        if guide_high is None or guide_low is None:
            raise ValueError("BDFUpsample expects inputs (feat_deep, guide_high, guide_low).")
        scale = self.scale
        feat_up = F.interpolate(feat_deep, scale_factor=scale, mode="bilinear", align_corners=False)
        guidance = torch.cat([guide_high, guide_low], dim=1)
        offset = torch.tanh(self.offset_net(guidance)) * self.max_offset
        b, _, h_up, w_up = offset.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h_up, device=offset.device, dtype=offset.dtype),
            torch.linspace(-1.0, 1.0, w_up, device=offset.device, dtype=offset.dtype),
            indexing="ij",
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.unsqueeze(0).expand(b, h_up, w_up, 2)
        offset_x = offset[:, 0] * (2.0 / max(w_up, 1))
        offset_y = offset[:, 1] * (2.0 / max(h_up, 1))
        flow = torch.stack((offset_x, offset_y), dim=-1)
        grid = base_grid + flow
        out = F.grid_sample(feat_up, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out


class SGF(nn.Module):
    """Spectral-Gated Fusion with residual gates for semantic and boundary cues."""

    def __init__(self, c_m: int, c_b: int, c_low: int, c_out: int = 256, residual_gate: bool = True):
        super().__init__()
        self.residual_gate = residual_gate
        self.sem_gate = nn.Sequential(nn.Conv2d(c_m + c_low, 1, kernel_size=1))
        self.edge_gate = nn.Sequential(nn.Conv2d(c_b, 1, kernel_size=1))
        self.proj = nn.Sequential(
            nn.Conv2d(c_m + c_b, c_out, kernel_size=1),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
        )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, m: torch.Tensor, b: torch.Tensor = None, g_low: torch.Tensor = None) -> torch.Tensor:
        if b is None and isinstance(m, (list, tuple)) and len(m) == 3:
            m, b, g_low = m
        if b is None or g_low is None:
            raise ValueError("SGF expects inputs (m, b, g_low).")
        mask = torch.sigmoid(self.sem_gate(torch.cat([m, g_low], dim=1)))
        b_mod = b + self.alpha * (b * mask) if self.residual_gate else b * mask
        w_edge = 1 + torch.tanh(self.edge_gate(b_mod))
        m_mod = m + self.beta * (m * w_edge) if self.residual_gate else m * w_edge
        out = self.proj(torch.cat([m_mod, b_mod], dim=1))
        return out


__all__ = [
    "GaborLLDLow",
    "GaborLLDHigh",
    "MambaCM",
    "BDFUpsample",
    "SGF",
]
