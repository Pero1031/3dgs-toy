# -*- coding: utf-8 -*-
"""ガウシアン群をカメラから描画する（パラメータの活性化 → 投影 → SH で色 → ラスタライズ）。"""

import torch
import torch.nn.functional as F

from .rasterize import conic_and_radius, covariance_3d, project_gaussians, rasterize
from .sh import eval_sh


def render(model, cam, bg, sh_degree, tile=8, max_entries=2 ** 25, near=0.2, delta=None):
    """
    delta: 動くシーン用。時刻に応じたずれ（gs/deform.py や gs/spacetime.py が出す）。どのキーも省ける。
           "means" / "log_scales" (N,3)  位置と log スケールに足す
           "quats" (N,4)                 正規化した四元数に足してから正規化し直す
           "opacity" (N,)                不透明度に掛ける（その時刻にどれだけ存在するか）

    Returns dict:
        image (H,W,3), alpha (H,W)
        means2d (Nv,2)  画面上の中心。学習時は .grad に ∂L/∂μ' が入る（densification の判定に使う）
        ids (Nv,)       means2d などが元の何番目のガウシアンか
        on_screen (Nv,) 画面内に映ったか
        radius (Nv,)    画面上の半径 3σ [px]（大きすぎるものの prune 判定に使う）
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

    # カメラの後ろ・近すぎるものを先に取り除く（z<=0 で割ると NaN の勾配が混ざるため、計算前に抜く）
    with torch.no_grad():
        z = means_all @ cam.R[2] + cam.t[2]
        keep = z > near
        if "opacity" in delta:   # この時刻にほぼ存在しないもの。どの画素でも α < 1/255 で捨てられるので結果は同じ
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

    # 視線方向に応じた色（現在の次数までの SH だけ使う）
    n_coef = (sh_degree + 1) ** 2
    sh = torch.cat([p["sh0"][ids], p["shN"][ids, : n_coef - 1]], 1)
    dirs = F.normalize(means - cam.center, dim=-1)
    color = (eval_sh(sh_degree, sh, dirs) + 0.5).clamp(min=0.0)

    image, alpha, on_screen = rasterize(means2d, conic, opacity, color, tile_radius, depth,
                                        cam.W, cam.H, bg, tile=tile, max_entries=max_entries)
    return {"image": image, "alpha": alpha, "means2d": means2d, "ids": ids,
            "on_screen": on_screen, "radius": radius}
