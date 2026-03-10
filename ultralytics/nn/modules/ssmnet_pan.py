# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""SSM-PAN modules for YOLOv8 segmentation neck experiments."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_scan_utils import HAS_SELECTIVE_SCAN_BACKEND, SelectiveScanCore, cross_selective_scan

__all__ = (
    "LLDLow",
    "LLDHigh",
    "LLDHigh_noGabor",
    "LowAggP5",
    "MambaCM",
    "BDFWarpUp",
    "SGF",
    "GateConcat",
)


def _has_selective_scan_backend() -> bool:
    """Return True when a selective-scan backend is available."""
    return HAS_SELECTIVE_SCAN_BACKEND


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Drop paths module."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class LSBlock(nn.Module):
    """Local perception branch used in Mamba-YOLO blocks."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return residual + self.drop(x)


class RGBlock(nn.Module):
    """Residual gated block used in Mamba-YOLO."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, stride=1, padding=0)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=hidden_features,
        )
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _flatten_by_direction(x: torch.Tensor, direction: str) -> torch.Tensor:
    """Flatten feature map to sequence in one of four directions."""
    if direction == "lr":
        return x.flatten(2)
    if direction == "rl":
        return torch.flip(x, dims=[-1]).flatten(2)
    if direction == "td":
        return x.transpose(2, 3).flatten(2)
    if direction == "bu":
        return torch.flip(x, dims=[-2]).transpose(2, 3).flatten(2)
    raise ValueError(f"Unsupported direction: {direction}")


def _unflatten_by_direction(seq: torch.Tensor, direction: str, h: int, w: int) -> torch.Tensor:
    """Map sequence back to feature map for one of four directions."""
    b, c, _ = seq.shape
    if direction == "lr":
        return seq.view(b, c, h, w)
    if direction == "rl":
        return torch.flip(seq.view(b, c, h, w), dims=[-1])
    if direction == "td":
        return seq.view(b, c, w, h).transpose(2, 3)
    if direction == "bu":
        return torch.flip(seq.view(b, c, w, h).transpose(2, 3), dims=[-2])
    raise ValueError(f"Unsupported direction: {direction}")


class SS2DFallback(nn.Module):
    """Dependency-free SS2D approximation using 4-direction cumulative aggregation."""

    def __init__(self, d_model: int, ssm_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        d_inner = max(int(d_model * ssm_ratio), d_model)
        self.in_proj = nn.Conv2d(d_model, d_inner, kernel_size=1, stride=1, padding=0, bias=False)
        self.dwconv = nn.Conv2d(d_inner, d_inner, kernel_size=3, stride=1, padding=1, groups=d_inner, bias=True)
        self.act = nn.SiLU()
        self.fuse = nn.Conv2d(d_inner * 4, d_inner, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_proj = nn.Conv2d(d_inner, d_model, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _scan(x: torch.Tensor, direction: str) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = _flatten_by_direction(x, direction)
        length = seq.shape[-1]
        denom = torch.arange(1, length + 1, device=seq.device, dtype=seq.dtype).view(1, 1, length)
        seq = torch.cumsum(seq, dim=-1) / denom
        return _unflatten_by_direction(seq, direction, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dwconv(self.in_proj(x)))
        y = torch.cat([self._scan(x, d) for d in ("lr", "rl", "td", "bu")], dim=1)
        y = self.fuse(y)
        y = self.out_proj(y)
        return self.drop(y)


class SS2DSelective(nn.Module):
    """Mamba-YOLO-style SS2D selective scan block."""

    backend_available = _has_selective_scan_backend()

    def __init__(self, d_model: int, d_state: int = 16, ssm_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.k_group = 4
        self.d_expand = max(int(ssm_ratio * d_model), d_model)
        self.d_state = int(d_state)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Conv2d(d_model, self.d_expand * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.dwconv = nn.Conv2d(
            self.d_expand,
            self.d_expand,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.d_expand,
            bias=True,
        )

        x_proj = [
            nn.Linear(self.d_expand, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        del x_proj

        self.dt_projs_weight = nn.Parameter(torch.randn(self.k_group, self.d_expand, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.randn(self.k_group, self.d_expand))
        self.A_logs = nn.Parameter(torch.zeros(self.k_group * self.d_expand, self.d_state))
        self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_expand))

        self.out_norm = nn.LayerNorm(self.d_expand)
        self.out_proj = nn.Conv2d(self.d_expand, d_model, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, z = self.in_proj(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x))

        y = cross_selective_scan(
            x=x,
            x_proj_weight=self.x_proj_weight,
            x_proj_bias=None,
            dt_projs_weight=self.dt_projs_weight,
            dt_projs_bias=self.dt_projs_bias,
            a_logs=self.A_logs,
            ds=self.Ds,
            out_norm=self.out_norm,
            out_norm_shape="v0",
            delta_softplus=True,
            force_fp32=self.training,
            selective_scan=SelectiveScanCore,
            ssoflex=self.training,
        )
        y = y.permute(0, 3, 1, 2).contiguous()
        y = y * self.act(z)
        y = self.out_proj(y)
        return self.drop(y)


class SS2DAuto(nn.Module):
    """Route SS2D implementation by runtime mode."""

    def __init__(self, d_model: int, d_state: int = 16, ssm_ratio: float = 2.0, dropout: float = 0.0, mode: str = "auto"):
        super().__init__()
        self.mode = mode
        self.has_backend = SS2DSelective.backend_available
        self.selective = SS2DSelective(d_model, d_state=d_state, ssm_ratio=ssm_ratio, dropout=dropout)
        self.fallback = SS2DFallback(d_model, ssm_ratio=ssm_ratio, dropout=dropout)
        self.last_route = "fallback"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "strict":
            if not self.has_backend:
                raise RuntimeError(
                    "SS2D strict mode requires selective-scan backend, but none was found. "
                    "Install selective-scan extensions or use mode='auto'/'fallback'."
                )
            if not x.is_cuda:
                self.last_route = "fallback_cpu_bootstrap"
                return self.fallback(x)
            self.last_route = "selective"
            return self.selective(x)

        if self.mode == "fallback":
            self.last_route = "fallback"
            return self.fallback(x)

        if self.mode != "auto":
            raise ValueError(f"Unsupported SS2D mode: {self.mode}")

        if self.has_backend and x.is_cuda:
            self.last_route = "selective"
            return self.selective(x)

        self.last_route = "fallback"
        return self.fallback(x)


class LLDLow(nn.Module):
    """
    Learnable Gaussian Low-Pass Filter (LLDLow).
    
    Mechanism:
        Instead of learning weights directly, this module learns the 'sigma' (standard deviation)
        of a Gaussian kernel per channel. This enforces a strict physical low-pass constraint
        while allowing the network to adapt the blur radius for different features.
    """

    def __init__(self, c1: int, k: int = 5):
        super().__init__()
        self.k = k
        self.c1 = c1
        self.padding = k // 2
        self.groups = c1
        
        # 1. Learnable Parameter: Sigma
        # Initialize sigma=1.0. A value of 1.0 is a balanced blur.
        self.sigma = nn.Parameter(torch.ones(c1, 1, 1, 1))
        
        # 2. Fixed Coordinate Grid (Buffer)
        self.register_buffer('dist_sq', self._build_dist_sq(k))

    @staticmethod
    def _build_dist_sq(k: int) -> torch.Tensor:
        """Generates the squared distance grid (x^2 + y^2) for a kxk kernel."""
        ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        return xx.pow(2) + yy.pow(2)

    def get_kernel(self) -> torch.Tensor:
        """Dynamically generates the Gaussian kernel based on current sigma."""
        sigma = F.softplus(self.sigma) + 0.1
        dist = self.dist_sq.to(sigma.device)
        gamma = 1.0 / (2 * sigma.pow(2))
        kernel = torch.exp(-dist * gamma)
        kernel_sum = kernel.sum(dim=(-2, -1), keepdim=True)
        return kernel / kernel_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.get_kernel()
        return F.conv2d(x, kernel, stride=1, padding=self.padding, groups=self.groups)


class LLDHigh(nn.Module):
    """Decomposes high frequency using Laplacian Residual + Fixed Gabor priors."""

    def __init__(self, c1: int | list[int]):
        super().__init__()
        c_in = c1[0] if isinstance(c1, (list, tuple)) else c1

        self.fusion = nn.Conv2d(c_in * 2, c_in, 1, 1, 0)

        # Gabor filters must be absolutely non-learnable.
        # We register them as a buffer so they are part of the state_dict but not parameters.
        gabor_weights = torch.zeros((c_in, 1, 3, 3))
        for i in range(c_in):
            theta = (i % 4) * (math.pi / 4)
            gabor_weights[i, 0] = self._gabor_kernel(3, theta)
        
        self.register_buffer('gabor_kernel', gabor_weights)

    @staticmethod
    def _gabor_kernel(
        k: int,
        theta: float,
        sigma: float = 1.0,
        lambd: float = 4.0,
        gamma: float = 0.5,
    ) -> torch.Tensor:
        d = k // 2
        coords = torch.arange(-d, d + 1, dtype=torch.float32)
        y, x = torch.meshgrid(coords, coords, indexing="ij")

        x_prime = x * math.cos(theta) + y * math.sin(theta)
        y_prime = -x * math.sin(theta) + y * math.cos(theta)

        g = torch.exp(-(x_prime**2 + (gamma**2) * (y_prime**2)) / (2 * (sigma**2)))
        g = g * torch.cos(2 * math.pi * x_prime / lambd)
        g = g - g.mean()
        g = g / (g.std() + 1e-6)
        return g

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)) or len(xs) != 2:
            raise ValueError("LLDHigh expects [F, L] as input.")

        f_raw, f_low = xs
        if f_low.shape[-2:] != f_raw.shape[-2:]:
            f_low = F.interpolate(f_low, size=f_raw.shape[-2:], mode="bilinear", align_corners=False)

        high_res = f_raw - f_low
        
        # Use functional conv2d with the fixed buffer
        high_edge = F.conv2d(f_raw, self.gabor_kernel, padding=1, groups=f_raw.shape[1])
        return self.fusion(torch.cat([high_res, high_edge], dim=1))


class LLDHigh_noGabor(nn.Module):
    """Ablation high-frequency branch using only Laplacian residual H = F - L."""

    def __init__(self, c1: int | list[int]):
        super().__init__()
        _ = c1  # keep constructor compatible with parse_model convention

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)) or len(xs) != 2:
            raise ValueError("LLDHigh_noGabor expects [F, L] as input.")

        f_raw, f_low = xs
        if f_low.shape[-2:] != f_raw.shape[-2:]:
            f_low = F.interpolate(f_low, size=f_raw.shape[-2:], mode="bilinear", align_corners=False)
        return f_raw - f_low


class LowAggP5(nn.Module):
    """Cross-scale low-frequency aggregation on P5 grid."""

    def __init__(self, in_channels: list[int] | tuple[int, int, int], d: int = 256, use_pool: bool = True):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("LowAggP5 expects three input channel sizes: [c3, c4, c5].")
        c3, c4, c5 = in_channels
        self.use_pool = use_pool
        self.proj3 = nn.Conv2d(c3, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj4 = nn.Conv2d(c4, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj5 = nn.Conv2d(c5, d, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)) or len(xs) != 3:
            raise ValueError("LowAggP5 expects [L3, L4, L5] as input.")
        l3, l4, l5 = xs
        p5_size = l5.shape[-2:]
        if self.use_pool:
            l3_5 = F.adaptive_avg_pool2d(l3, output_size=p5_size)
            l4_5 = F.adaptive_avg_pool2d(l4, output_size=p5_size)
        else:
            l3_5 = F.interpolate(l3, size=p5_size, mode="bilinear", align_corners=False)
            l4_5 = F.interpolate(l4, size=p5_size, mode="bilinear", align_corners=False)

        return torch.cat((self.proj3(l3_5), self.proj4(l4_5), self.proj5(l5)), dim=1)


class MambaCM(nn.Module):
    """Global topology mixer at P5 using Mamba-YOLO-style ODSS block."""

    def __init__(
        self,
        c1: int,
        d: int = 256,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        ss2d_mode: str = "auto",
    ):
        super().__init__()
        self.proj_conv = nn.Sequential(
            nn.Conv2d(c1, d, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(d),
            nn.SiLU(),
        )
        self.lsblock = LSBlock(d, d)
        self.norm1 = LayerNorm2d(d)
        self.ss2d = SS2DAuto(d_model=d, d_state=ssm_d_state, ssm_ratio=ssm_ratio, dropout=0.0, mode=ss2d_mode)
        self.drop_path = DropPath(drop_path)

        self.norm2 = LayerNorm2d(d)
        self.rg = RGBlock(in_features=d, hidden_features=int(d * mlp_ratio), out_features=d, act_layer=nn.GELU, drop=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_conv(x)
        x_local = self.lsblock(x)
        x = x + self.drop_path(self.ss2d(self.norm1(x_local)))
        x = x + self.drop_path(self.rg(self.norm2(x)))
        return x


class BDFWarpUp(nn.Module):
    """High-guided warping upsample with identity-safe offset initialization."""

    def __init__(self, c_src: int, c_high: int, c_low: int = 0, max_offset: float = 2.0, align_corners: bool = False):
        super().__init__()
        self.max_offset = float(max_offset)
        self.src_proj = nn.Conv2d(c_src, c_high, kernel_size=1, stride=1, padding=0, bias=False)
        self.low_proj = nn.Conv2d(c_low, c_high, kernel_size=1, stride=1, padding=0, bias=False) if c_low > 0 else None

        c_in = c_high * 2 + (c_high if c_low > 0 else 0)
        hidden = max(c_high, 16)
        self.offset_net = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, kernel_size=3, stride=1, padding=1, bias=True),
        )

        last = self.offset_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        self.last_offset: torch.Tensor | None = None
        self.align_corners = align_corners

    @staticmethod
    def _build_base_grid(batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((gx, gy), dim=-1)
        return grid.unsqueeze(0).expand(batch, h, w, 2)

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)) or len(xs) < 2:
            raise ValueError("BDFWarpUp expects [src, high_t, low_t(optional)] as input.")

        src = xs[0]
        high_t = xs[1]
        low_t = xs[2] if len(xs) > 2 else None

        target_size = high_t.shape[-2:]
        src_up = F.interpolate(src, size=target_size, mode="bilinear", align_corners=self.align_corners)

        feat = [high_t, self.src_proj(src_up)]
        if self.low_proj is not None and low_t is not None:
            if low_t.shape[-2:] != target_size:
                low_t = F.interpolate(low_t, size=target_size, mode="bilinear", align_corners=self.align_corners)
            feat.append(self.low_proj(low_t))

        offset_raw = self.offset_net(torch.cat(feat, dim=1))
        offset = self.max_offset * torch.tanh(offset_raw)

        b, _, h, w = offset.shape
        base_grid = self._build_base_grid(b, h, w, offset.device, offset.dtype)

        if w > 1:
            offset_x = offset[:, 0] * (2.0 / (w - 1))
        else:
            offset_x = torch.zeros_like(offset[:, 0])
        if h > 1:
            offset_y = offset[:, 1] * (2.0 / (h - 1))
        else:
            offset_y = torch.zeros_like(offset[:, 1])

        norm_offset = torch.stack((offset_x, offset_y), dim=-1)
        grid = base_grid + norm_offset
        aligned = F.grid_sample(src_up, grid, mode="bilinear", padding_mode="border", align_corners=self.align_corners)

        # Always store offset for visualization/debugging purposes
        self.last_offset = offset

        if self.training:
            return aligned, offset
        return aligned


class SGF(nn.Module):
    """Spectral Gated Fusion with residual safety path."""

    def __init__(self, c_m: int, c_b: int, c_e: int, c_out: int, residual: bool = True, alpha_init: float = 0.2):
        super().__init__()
        self.residual = bool(residual)

        self.m_gate = nn.Conv2d(c_m, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.e_gate = nn.Conv2d(c_e, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.fuse = nn.Conv2d(c_m + c_b, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.skip = nn.Conv2d(c_m, c_out, kernel_size=1, stride=1, padding=0, bias=False) if c_m != c_out else nn.Identity()
        
        # Channel-wise alpha to prevent gradient starvation
        self.alpha = nn.Parameter(torch.ones(1, c_out, 1, 1) * alpha_init)

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)) or len(xs) != 3:
            raise ValueError("SGF expects [M, B, E] as input.")

        m, b, e = xs
        # Handle tuple input from BDFWarpUp (training mode)
        if isinstance(m, (list, tuple)):
            m = m[0]
        target_size = m.shape[-2:]
        if b.shape[-2:] != target_size:
            b = F.interpolate(b, size=target_size, mode="bilinear", align_corners=False)
        if e.shape[-2:] != target_size:
            e = F.interpolate(e, size=target_size, mode="bilinear", align_corners=False)

        w_m = torch.sigmoid(self.m_gate(m))
        w_e = 1.0 + torch.tanh(self.e_gate(e))

        b_prime = b * w_m
        m_prime = m * w_e
        fused = self.fuse(torch.cat((m_prime, b_prime), dim=1))

        if self.residual:
            return self.skip(m) + self.alpha * fused
        return fused


class GateConcat(nn.Module):
    """Bottom-up selective routing gate for concat fusion."""

    def __init__(self, c_x: int, c_y: int, c_guide: int, init_bias: float = 2.0):
        super().__init__()
        _ = c_y
        self.gate = nn.Conv2d(c_guide, 1, kernel_size=1, stride=1, padding=0, bias=True)
        nn.init.constant_(self.gate.bias, float(init_bias))
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if not isinstance(xs, (list, tuple)) or len(xs) != 3:
            raise ValueError("GateConcat expects [x, y, guide] as input.")

        x, y, guide = xs
        # Handle tuple input from BDFWarpUp (if used as input)
        if isinstance(x, (list, tuple)):
            x = x[0]
        target_size = x.shape[-2:]
        if y.shape[-2:] != target_size:
            y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
        if guide.shape[-2:] != target_size:
            guide = F.interpolate(guide, size=target_size, mode="bilinear", align_corners=False)

        w = torch.sigmoid(self.gate(guide))
        x_gated = x * w
        return torch.cat((x_gated, y), dim=1)
