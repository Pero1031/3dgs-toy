# -*- coding: utf-8 -*-
"""
Deformation field that moves the Gaussians according to time
(a simplified version of Deformable 3D Gaussians, Yang et al. CVPR 2024).

The Gaussians have only one "canonical shape" (canonical), and a small MLP offsets it to make the shape at time t:

    (δx, δq, δs) = MLP(γ(x), γ(t))     x is the canonical position,
                                       γ is the same sin/cos positional encoding as NeRF
    position x + δx, rotation normalize(q + δq), scale exp(log s + δs)

Color and opacity do not depend on time (the assumption is that appearance does not change when things move).

Differences from the paper:
    - The small MLP placed after the time encoding (used for synthetic data) is omitted
    - δs is added to the log scale (the paper adds it to the activated scale). This keeps the scale from going negative
    - The output layer is initialized to 0, so training starts with no deformation (a static scene) and motion
      grows from there
"""

import torch
import torch.nn as nn


def positional_encoding(x, n_freqs):
    """γ(x) = [x, sin(2^k x), cos(2^k x)]_{k=0..L-1} (not multiplied by π, same as the NeRF implementation)"""
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
    xf = x[..., None] * freqs                        # (..., D, L)
    enc = torch.cat([xf.sin(), xf.cos()], -1).flatten(-2)
    return torch.cat([x, enc], -1)


class DeformNet(nn.Module):
    def __init__(self, depth=8, width=256, x_freqs=10, t_freqs=6):
        super().__init__()
        self.x_freqs, self.t_freqs = x_freqs, t_freqs
        d_in = 3 * (1 + 2 * x_freqs) + (1 + 2 * t_freqs)
        self.skip = depth // 2                       # γ is concatenated again to this layer's input (same as NeRF)
        layers = []
        for i in range(depth):
            d = d_in if i == 0 else width + (d_in if i == self.skip else 0)
            layers.append(nn.Linear(d, width))
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(width, 3 + 4 + 3)       # δx, δq, δs
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, xyz, t):
        """xyz (N,3) canonical positions, t: time (scalar) -> offsets {"means", "quats", "log_scales"}"""
        tt = torch.full((len(xyz), 1), float(t), device=xyz.device)
        g = torch.cat([positional_encoding(xyz, self.x_freqs), positional_encoding(tt, self.t_freqs)], -1)
        h = g
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, g], -1)
            h = torch.relu(layer(h))
        out = self.head(h)
        return {"means": out[:, :3], "quats": out[:, 3:7], "log_scales": out[:, 7:]}
