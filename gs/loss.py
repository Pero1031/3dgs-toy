# -*- coding: utf-8 -*-
"""損失関数: L = (1-λ) L1 + λ (1 - SSIM)、λ=0.2（論文 式(7)）"""

import functools

import torch
import torch.nn.functional as F


@functools.lru_cache(maxsize=None)
def _window(size, sigma, device):
    g = torch.exp(-((torch.arange(size, dtype=torch.float32) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous().to(device)


def ssim(a, b, size=11, sigma=1.5):
    """(H,W,3) 同士の SSIM（11x11 ガウス窓、σ=1.5）"""
    return ssim_map(a, b, size, sigma).mean()


def ssim_map(a, b, size=11, sigma=1.5):
    """画素ごとの SSIM (1,3,H,W)。評価でマスク内だけ平均したいとき用"""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    w = _window(size, sigma, a.device)
    conv = functools.partial(F.conv2d, weight=w, padding=size // 2, groups=3)
    mx, my = conv(x), conv(y)
    sxx = conv(x * x) - mx * mx
    syy = conv(y * y) - my * my
    sxy = conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def psnr(a, b):
    return -10.0 * torch.log10(((a - b) ** 2).mean())
