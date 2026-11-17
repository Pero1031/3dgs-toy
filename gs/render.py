# -*- coding: utf-8 -*-
"""ガウシアン群をカメラから描画する（パラメータの活性化 → 投影 → SH で色 → ラスタライズ）。"""

import torch
import torch.nn.functional as F

from .rasterize import conic_and_radius, covariance_3d, project_gaussians, rasterize
from .sh import eval_sh


def render(model, cam, bg, sh_degree, tile=8, max_entries=2 ** 25, near=0.2):
    """
    Returns dict:
        image (H,W,3), alpha (H,W)
        means2d (Nv,2)  画面上の中心。学習時は .grad に ∂L/∂μ' が入る（densification の判定に使う）
        ids (Nv,)       means2d などが元の何番目のガウシアンか
        on_screen (Nv,) 画面内に映ったか
        radius (Nv,)    画面上の半径 3σ [px]（大きすぎるものの prune 判定に使う）
    """
    p = model.params

    # カメラの後ろ・近すぎるものを先に取り除く（z<=0 で割ると NaN の勾配が混ざるため、計算前に抜く）
    with torch.no_grad():
        z = p["means"] @ cam.R[2] + cam.t[2]
        ids = torch.nonzero(z > near).squeeze(1)

    means = p["means"][ids]
    scales = torch.exp(p["log_scales"][ids])
    quats = F.normalize(p["quats"][ids], dim=-1)
    opacity = torch.sigmoid(p["opacity_logits"][ids]).squeeze(-1)

    means2d, cov2d, depth = project_gaussians(means, covariance_3d(scales, quats), cam)
    conic, radius, tile_radius = conic_and_radius(cov2d, opacity)
    if means2d.requires_grad:
        means2d.retain_grad()

    # 視線方向に応じた色（現在の次数までの SH だけ使う）
    n_coef = (sh_degree + 1) ** 2
    sh = torch.cat([p["sh0"][ids], p["shN"][ids, : n_coef - 1]], 1)
    dirs = F.normalize(means - cam.center, dim=-1)
    color = (eval_sh(sh_degree, sh, dirs) + 0.5).clamp(min=0.0)

    image, alpha, on_screen = rasterize(means2d, conic, opacity, color, tile_radius, depth,
                                        cam.W, cam.H, bg, tile=tile, max_entries=max_entries)
    return {"image": image, "alpha": alpha, "means2d": means2d, "ids": ids,
            "on_screen": on_screen, "radius": radius}
