# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
FreqFusion integration for YOLOv8.

Pure-PyTorch CARAFE fallback is provided so mmcv is optional. `FreqFusionCat`
wraps FreqFusion to output the concatenation of enhanced high-resolution
features and upsampled low-resolution features.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional mmcv acceleration
    from mmcv.ops.carafe import normal_init, xavier_init, carafe  # type: ignore
except Exception:  # pragma: no cover - fallback

    def xavier_init(module: nn.Module, gain: float = 1.0, bias: float = 0.0, distribution: str = "normal") -> None:
        assert distribution in ["uniform", "normal"]
        if hasattr(module, "weight") and module.weight is not None:
            if distribution == "uniform":
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0.0) -> None:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def carafe(x: torch.Tensor, normed_mask: torch.Tensor, kernel_size: int, group: int = 1, up: int = 1) -> torch.Tensor:
        """Pure PyTorch CARAFE fallback (no CUDA extension)."""
        b, c, h, w = x.shape
        _, _, m_h, m_w = normed_mask.shape
        pad = kernel_size // 2
        pad_x = F.pad(x, pad=[pad] * 4, mode="reflect")
        unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1)
        unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
        if up != 1:
            unfold_x = F.interpolate(unfold_x, scale_factor=up, mode="nearest")
        unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
        normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
        out = (unfold_x * normed_mask).sum(dim=2)
        return out


def constant_init(module: nn.Module, val: float, bias: float = 0.0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def resize(input: torch.Tensor, size=None, scale_factor=None, mode: str = "nearest", align_corners=None, warning: bool = True):
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if (output_h > input_h or output_w > input_w) and ((output_h - 1) % (input_h - 1)) and ((output_w - 1) % (input_w - 1)):
            warnings.warn(
                f"When align_corners={align_corners}, output is better aligned if input {(input_h, input_w)} "
                f"and output {(output_h, output_w)} satisfy (out-1) % (in-1) == 0"
            )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def hamming2D(M: int, N: int) -> np.ndarray:
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    return np.outer(hamming_x, hamming_y)


class FreqFusion(nn.Module):
    def __init__(
        self,
        hr_channels: int,
        lr_channels: int,
        scale_factor: int = 1,
        lowpass_kernel: int = 5,
        highpass_kernel: int = 3,
        up_group: int = 1,
        encoder_kernel: int = 3,
        encoder_dilation: int = 1,
        compressed_channels: int = 64,
        align_corners: bool = False,
        upsample_mode: str = "nearest",
        feature_resample: bool = False,
        feature_resample_group: int = 4,
        comp_feat_upsample: bool = True,
        use_high_pass: bool = True,
        use_low_pass: bool = True,
        hr_residual: bool = True,
        semi_conv: bool = True,
        hamming_window: bool = True,
        feature_resample_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels

        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)

        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            lowpass_kernel**2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1,
        )

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample

        if self.feature_resample:
            self.dysampler = LocalSimGuidedSampler(
                in_channels=compressed_channels,
                scale=2,
                style="lp",
                groups=feature_resample_group,
                use_direct_scale=True,
                kernel_size=encoder_kernel,
                norm=feature_resample_norm,
            )

        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                highpass_kernel**2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1,
            )

        self.hamming_window = hamming_window
        if self.hamming_window:
            self.register_buffer("hamming_lowpass", torch.FloatTensor(hamming2D(lowpass_kernel, lowpass_kernel))[None, None,])
            self.register_buffer("hamming_highpass", torch.FloatTensor(hamming2D(highpass_kernel, highpass_kernel))[None, None,])
        else:
            self.register_buffer("hamming_lowpass", torch.FloatTensor([1.0]))
            self.register_buffer("hamming_highpass", torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask: torch.Tensor, kernel: int, hamming: torch.Tensor, scale_factor: int | None = None) -> torch.Tensor:
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask = mask / mask.sum(dim=(-1, -2), keepdim=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor, use_checkpoint: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, hr_feat, lr_feat)
        return self._forward(hr_feat, lr_feat)

    def _forward(self, hr_feat: torch.Tensor, lr_feat: torch.Tensor):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        if self.semi_conv:
            if self.comp_feat_upsample:
                if not self.use_high_pass:
                    raise NotImplementedError("High-pass path disabled is not implemented in this variant")

                mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass)
                compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(
                    compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1
                )

                mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)

                mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                mask_lr_lr_feat = F.interpolate(
                    carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
                    size=compressed_hr_feat.shape[-2:],
                    mode="nearest",
                )
                mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
                mask_hr_lr_feat = F.interpolate(
                    carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 2),
                    size=compressed_hr_feat.shape[-2:],
                    mode="nearest",
                )
                mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(
                    self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode="nearest"
                )
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(
                        self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode="nearest"
                    )
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode="nearest") + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass:
                mask_hr = self.content_encoder2(compressed_x)

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        if self.semi_conv:
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = resize(
                input=lr_feat,
                size=hr_feat.shape[2:],
                mode=self.upsample_mode,
                align_corners=None if self.upsample_mode == "nearest" else self.align_corners,
            )
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            hr_feat = hr_feat_hf + hr_feat if self.hr_residual else hr_feat_hf

        if self.feature_resample:
            lr_feat = self.dysampler(hr_x=compressed_hr_feat, lr_x=compressed_lr_feat, feat2sample=lr_feat)

        return mask_lr, hr_feat, lr_feat


class LocalSimGuidedSampler(nn.Module):
    """Offset generator used inside FreqFusion."""

    def __init__(
        self,
        in_channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        use_direct_scale: bool = True,
        kernel_size: int = 1,
        local_window: int = 3,
        sim_type: str = "cos",
        norm: bool = True,
        direction_feat: str = "sim_concat",
    ) -> None:
        super().__init__()
        assert scale == 2
        assert style == "lp"
        assert in_channels % groups == 0

        self.scale = scale
        self.style = style
        self.groups = groups
        self.local_window = local_window
        self.sim_type = sim_type
        self.direction_feat = direction_feat

        out_channels = 2 * groups * scale * scale
        if self.direction_feat == "sim":
            self.offset = nn.Conv2d(local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == "sim_concat":
            self.offset = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == "sim":
                self.direct_scale = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            elif self.direction_feat == "sim_concat":
                self.direct_scale = nn.Conv2d(in_channels + local_window**2 - 1, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            else:
                raise NotImplementedError
            constant_init(self.direct_scale, val=0.0)

        hr_out_channels = 2 * groups
        if self.direction_feat == "sim":
            self.hr_offset = nn.Conv2d(local_window**2 - 1, hr_out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        elif self.direction_feat == "sim_concat":
            self.hr_offset = nn.Conv2d(in_channels + local_window**2 - 1, hr_out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        else:
            raise NotImplementedError
        normal_init(self.hr_offset, std=0.001)

        if use_direct_scale:
            if self.direction_feat == "sim":
                self.hr_direct_scale = nn.Conv2d(in_channels, hr_out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            elif self.direction_feat == "sim_concat":
                self.hr_direct_scale = nn.Conv2d(in_channels + local_window**2 - 1, hr_out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            else:
                raise NotImplementedError
            constant_init(self.hr_direct_scale, val=0.0)

        self.norm = norm
        if self.norm:
            self.norm_hr = nn.GroupNorm(in_channels // 8, in_channels)
            self.norm_lr = nn.GroupNorm(in_channels // 8, in_channels)
        else:
            self.norm_hr = nn.Identity()
            self.norm_lr = nn.Identity()
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self) -> torch.Tensor:
        h = torch.arange((-(self.scale - 1)) / 2, ((self.scale - 1) / 2) + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x: torch.Tensor, offset: torch.Tensor, scale: int | None = None) -> torch.Tensor:
        if scale is None:
            scale = self.scale
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).to(x.device)
        coords = coords.type(x.dtype)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), scale).view(B, 2, -1, scale * H, scale * W)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(B * self.groups, -1, x.size(-2), x.size(-1)),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, scale * H, scale * W)

    def get_offset_lp(self, hr_x: torch.Tensor, lr_x: torch.Tensor, hr_sim: torch.Tensor, lr_sim: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "direct_scale"):
            offset = (self.offset(lr_sim) + F.pixel_unshuffle(self.hr_offset(hr_sim), self.scale)) * (
                self.direct_scale(lr_x) + F.pixel_unshuffle(self.hr_direct_scale(hr_x), self.scale)
            ).sigmoid() + self.init_pos
        else:
            offset = (self.offset(lr_x) + F.pixel_unshuffle(self.hr_offset(hr_x), self.scale)) * 0.25 + self.init_pos
        return offset

    def forward(self, hr_x: torch.Tensor, lr_x: torch.Tensor, feat2sample: torch.Tensor) -> torch.Tensor:
        hr_x = self.norm_hr(hr_x)
        lr_x = self.norm_lr(lr_x)

        if self.direction_feat == "sim":
            hr_sim = compute_similarity(hr_x, self.local_window, dilation=2, sim="cos")
            lr_sim = compute_similarity(lr_x, self.local_window, dilation=2, sim="cos")
        elif self.direction_feat == "sim_concat":
            hr_sim = torch.cat([hr_x, compute_similarity(hr_x, self.local_window, dilation=2, sim="cos")], dim=1)
            lr_sim = torch.cat([lr_x, compute_similarity(lr_x, self.local_window, dilation=2, sim="cos")], dim=1)
            hr_x, lr_x = hr_sim, lr_sim
        else:
            raise NotImplementedError

        offset = self.get_offset_lp(hr_x, lr_x, hr_sim, lr_sim)
        return self.sample(feat2sample, offset)


def compute_similarity(input_tensor: torch.Tensor, k: int = 3, dilation: int = 1, sim: str = "cos") -> torch.Tensor:
    B, C, H, W = input_tensor.shape
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)
    unfold_tensor = unfold_tensor.reshape(B, C, k * k, H, W)

    if sim == "cos":
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2 : k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == "dot":
        similarity = unfold_tensor[:, :, k * k // 2 : k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    similarity = torch.cat((similarity[:, : k * k // 2], similarity[:, k * k // 2 + 1 :]), dim=1)
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity


class FreqFusionCat(nn.Module):
    """Wrapper that concatenates enhanced hr_feat and upsampled lr_feat."""

    def __init__(
        self,
        hr_channels: int,
        lr_channels: int,
        compress_ratio: int = 8,
        feature_resample: bool = True,
        feature_resample_group: int = 4,
        feature_resample_norm: bool = True,
        lowpass_kernel: int = 5,
        highpass_kernel: int = 3,
        use_high_pass: bool = True,
        use_low_pass: bool = True,
        hr_residual: bool = True,
        semi_conv: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        compressed_channels = (hr_channels + lr_channels) // compress_ratio
        assert compressed_channels % feature_resample_group == 0, "compressed_channels must be divisible by feature_resample_group"

        self.freqfusion = FreqFusion(
            hr_channels=hr_channels,
            lr_channels=lr_channels,
            compressed_channels=compressed_channels,
            feature_resample=feature_resample,
            feature_resample_group=feature_resample_group,
            feature_resample_norm=feature_resample_norm,
            lowpass_kernel=lowpass_kernel,
            highpass_kernel=highpass_kernel,
            use_high_pass=use_high_pass,
            use_low_pass=use_low_pass,
            hr_residual=hr_residual,
            semi_conv=semi_conv,
            **kwargs,
        )

    def forward(self, x: list[torch.Tensor] | Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            lr_feat, hr_feat = x
        else:
            raise TypeError("FreqFusionCat expects a list/tuple [lr, hr]")
        _, hr_feat_enh, lr_feat_up = self.freqfusion(hr_feat=hr_feat, lr_feat=lr_feat)
        return torch.cat([hr_feat_enh, lr_feat_up], dim=1)


__all__ = ["FreqFusion", "FreqFusionCat"]

