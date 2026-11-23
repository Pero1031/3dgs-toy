# -*- coding: utf-8 -*-
"""Render the set of Gaussians from a camera (activate parameters → project → color from SH → rasterize)."""

import torch
import torch.nn.functional as F

from .rasterize import conic_and_radius, covariance_3d, project_gaussians, rasterize
from .sh import eval_sh


def render(model, cam, bg, sh_degree, tile=8, max_entries=2 ** 25, near=0.2, delta=None):
    """
    delta: for dynamic scenes. Time-dependent offsets (produced by gs/deform.py or gs/spacetime.py).
           Any key may be omitted.
           "means" / "log_scales" (N,3)  added to the position and the log scale
           "quats" (N,4)                 added to the normalized quaternion, which is then normalized again
           "opacity" (N,)                multiplied into the opacity (how much it exists at that time)

    Returns dict:
        image (H,W,3), alpha (H,W)
        means2d (Nv,2)  screen-space centers. During training .grad holds ∂L/∂μ'
                        (used for the densification decision)
        ids (Nv,)       which original Gaussian each entry of means2d etc. belongs to
        on_screen (Nv,) whether it appeared on screen
        radius (Nv,)    screen-space radius 3σ [px] (used for the prune decision on ones that are too large)
    """
    p = model.params
    means_all, log_scales, quats = p["means"], p["log_scales"], p["quats"]
    opacity_all = torch.sigmoid(p["opacity_logits"]).squeeze(-1)
    delta = delta or {}
    if "means" in delta:
        means_all = means_all + delta["means"]
    if "log_scales" in delta:
        log_scales = log_scales + delta["log_scales"]
    if "quats" in delta:
        quats = F.normalize(quats, dim=-1) + delta["quats"]
    if "opacity" in delta:
        opacity_all = opacity_all * delta["opacity"]

    # First remove those behind or too close to the camera
    # (dividing by z<=0 mixes NaN gradients in, so they are dropped before the computation)
    with torch.no_grad():
        z = means_all @ cam.R[2] + cam.t[2]
        keep = z > near
        if "opacity" in delta:   # nearly absent at this time. Discarded with α < 1/255 at every pixel, so the result is the same
            keep &= opacity_all > 1.0 / 255.0
        ids = torch.nonzero(keep).squeeze(1)

    means = means_all[ids]
    scales = torch.exp(log_scales[ids])
    quats = F.normalize(quats[ids], dim=-1)
    opacity = opacity_all[ids]

    means2d, cov2d, depth = project_gaussians(means, covariance_3d(scales, quats), cam)
    conic, radius, tile_radius = conic_and_radius(cov2d, opacity)
    if means2d.requires_grad:
        means2d.retain_grad()

    # Color depending on the view direction (only SH up to the current degree is used)
    n_coef = (sh_degree + 1) ** 2
    sh = torch.cat([p["sh0"][ids], p["shN"][ids, : n_coef - 1]], 1)
    dirs = F.normalize(means - cam.center, dim=-1)
    color = (eval_sh(sh_degree, sh, dirs) + 0.5).clamp(min=0.0)

    image, alpha, on_screen = rasterize(means2d, conic, opacity, color, tile_radius, depth,
                                        cam.W, cam.H, bg, tile=tile, max_entries=max_entries)
    return {"image": image, "alpha": alpha, "means2d": means2d, "ids": ids,
            "on_screen": on_screen, "radius": radius}
