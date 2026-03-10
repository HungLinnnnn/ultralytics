# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Minimal Mamba-YOLO selective-scan bridge utilities."""

from __future__ import annotations

import importlib

import torch


def _load_scan_backend():
    """Load selective-scan backend following Mamba-YOLO priority."""
    for module_name in ("selective_scan_cuda_core", "selective_scan_cuda"):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "fwd") and hasattr(module, "bwd"):
                return module
        except Exception:
            continue
    return None


_SCAN_BACKEND = _load_scan_backend()
HAS_SELECTIVE_SCAN_BACKEND = _SCAN_BACKEND is not None


class CrossScan(torch.autograd.Function):
    """Convert BCHW map to 4 directional scan sequences."""

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        b, c, h, w = x.shape
        ctx.shape = (b, c, h, w)
        xs = x.new_empty((b, 4, c, h * w))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        b, c, h, w = ctx.shape
        length = h * w
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(b, 2, -1, length)
        y = ys[:, 0] + ys[:, 1].view(b, -1, w, h).transpose(dim0=2, dim1=3).contiguous().view(b, -1, length)
        return y.view(b, -1, h, w)


class CrossMerge(torch.autograd.Function):
    """Merge 4 directional outputs back to sequence map."""

    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        b, k, d, h, w = ys.shape
        ctx.shape = (h, w)
        ys = ys.view(b, k, d, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(b, 2, d, -1)
        y = ys[:, 0] + ys[:, 1].view(b, -1, w, h).transpose(dim0=2, dim1=3).contiguous().view(b, d, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        h, w = ctx.shape
        b, c, length = x.shape
        xs = x.new_empty((b, 4, c, length))
        xs[:, 0] = x
        xs[:, 1] = x.view(b, c, h, w).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs.view(b, 4, c, h, w)


class SelectiveScanCore(torch.autograd.Function):
    """CUDA selective-scan wrapper compatible with Mamba-YOLO calls."""

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        u,
        delta,
        a,
        b,
        c,
        d=None,
        delta_bias=None,
        delta_softplus=False,
        nrows=1,
        backnrows=1,
        oflex=True,
    ):
        del nrows, backnrows, oflex
        if _SCAN_BACKEND is None:
            raise RuntimeError("Selective-scan backend is not available.")

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if d is not None and d.stride(-1) != 1:
            d = d.contiguous()
        if b.stride(-1) != 1:
            b = b.contiguous()
        if c.stride(-1) != 1:
            c = c.contiguous()
        if b.dim() == 3:
            b = b.unsqueeze(dim=1)
            ctx.squeeze_b = True
        if c.dim() == 3:
            c = c.unsqueeze(dim=1)
            ctx.squeeze_c = True

        ctx.delta_softplus = delta_softplus
        out, x, *_ = _SCAN_BACKEND.fwd(u, delta, a, b, c, d, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, a, b, c, d, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        del args
        if _SCAN_BACKEND is None:
            raise RuntimeError("Selective-scan backend is not available.")

        u, delta, a, b, c, d, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, da, db, dc, dd, ddelta_bias, *_ = _SCAN_BACKEND.bwd(
            u, delta, a, b, c, d, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return du, ddelta, da, db, dc, dd, ddelta_bias, None, None, None, None


def cross_selective_scan(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    x_proj_bias: torch.Tensor | None,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    a_logs: torch.Tensor,
    ds: torch.Tensor,
    out_norm: torch.nn.Module | None,
    out_norm_shape: str = "v0",
    nrows: int = -1,
    backnrows: int = -1,
    delta_softplus: bool = True,
    to_dtype: bool = True,
    force_fp32: bool = False,
    ssoflex: bool = True,
    selective_scan=SelectiveScanCore,
) -> torch.Tensor:
    """Cross selective-scan entry used by SS2DSelective."""
    if _SCAN_BACKEND is None:
        raise RuntimeError("Selective-scan backend is not available.")

    b, _, h, w = x.shape
    _, n = a_logs.shape
    k, _, r = dt_projs_weight.shape
    length = h * w

    def selective_scan_fn(u, delta, a, b_term, c_term, d_term=None, delta_bias_term=None, softplus=True):
        return selective_scan.apply(
            u, delta, a, b_term, c_term, d_term, delta_bias_term, softplus, nrows, backnrows, ssoflex
        )

    xs = CrossScan.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, k, -1, 1)
    dts, bs, cs = torch.split(x_dbl, [r, n, n], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(b, -1, length)
    dts = dts.contiguous().view(b, -1, length)
    as_term = -torch.exp(a_logs.to(torch.float32))
    bs = bs.contiguous()
    cs = cs.contiguous()
    ds = ds.to(torch.float32)
    delta_bias = dt_projs_bias.view(-1).to(torch.float32)

    if force_fp32:
        xs = xs.to(torch.float32)
        dts = dts.to(torch.float32)
        bs = bs.to(torch.float32)
        cs = cs.to(torch.float32)

    ys = selective_scan_fn(xs, dts, as_term, bs, cs, ds, delta_bias, delta_softplus).view(b, k, -1, h, w)
    y = CrossMerge.apply(ys)

    if out_norm_shape == "v1":
        y = out_norm(y.view(b, -1, h, w)).permute(0, 2, 3, 1)
    else:
        y = y.transpose(dim0=1, dim1=2).contiguous()
        if out_norm is not None:
            y = out_norm(y)
        y = y.view(b, h, w, -1)

    return y.to(x.dtype) if to_dtype else y
