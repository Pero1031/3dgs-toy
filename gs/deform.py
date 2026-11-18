# -*- coding: utf-8 -*-
"""
時刻に応じてガウシアンを動かす変形場（Deformable 3D Gaussians, Yang et al. CVPR 2024 の簡略版）。

ガウシアンは「基準の形」（canonical）を 1 組だけ持ち、時刻 t の形は小さな MLP がずらして作る:

    (δx, δq, δs) = MLP(γ(x), γ(t))     x は基準の位置、γ は NeRF と同じ sin/cos の位置符号化
    位置 x + δx、回転 normalize(q + δq)、大きさ exp(log s + δs)

色と不透明度は時刻によらない（動いても見た目は変わらない、という仮定）。

論文との違い:
    - 時刻の符号化のあとに置く小さな MLP（合成データ用）は省いた
    - δs は log スケールに足す（論文は活性化後の大きさに足す。負にならないように）
    - 出力層を 0 で初期化し、学習の始めは変形なし（静止したシーン）から動き出すようにした
"""

import torch
import torch.nn as nn


def positional_encoding(x, n_freqs):
    """γ(x) = [x, sin(2^k x), cos(2^k x)]_{k=0..L-1}（NeRF の実装と同じく π は掛けない）"""
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
    xf = x[..., None] * freqs                        # (..., D, L)
    enc = torch.cat([xf.sin(), xf.cos()], -1).flatten(-2)
    return torch.cat([x, enc], -1)


class DeformNet(nn.Module):
    def __init__(self, depth=8, width=256, x_freqs=10, t_freqs=6):
        super().__init__()
        self.x_freqs, self.t_freqs = x_freqs, t_freqs
        d_in = 3 * (1 + 2 * x_freqs) + (1 + 2 * t_freqs)
        self.skip = depth // 2                       # この層の入力に γ をもう一度つなぐ（NeRF と同じ）
        layers = []
        for i in range(depth):
            d = d_in if i == 0 else width + (d_in if i == self.skip else 0)
            layers.append(nn.Linear(d, width))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(width, 3 + 4 + 3)       # δx, δq, δs
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, xyz, t):
        """xyz (N,3) 基準の位置、t: 時刻（スカラー）-> {"means", "quats", "log_scales"} のずれ"""
        tt = torch.full((len(xyz), 1), float(t), device=xyz.device)
        g = torch.cat([positional_encoding(xyz, self.x_freqs), positional_encoding(tt, self.t_freqs)], -1)
        h = g
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, g], -1)
            h = torch.relu(layer(h))
        out = self.head(h)
        return {"means": out[:, :3], "quats": out[:, 3:7], "log_scales": out[:, 7:]}
