# -*- coding: utf-8 -*-
"""
時刻を持つガウシアン（Spacetime Gaussians, Li et al. CVPR 2024 / 4D ガウシアン, Yang et al. ICLR 2024 の簡略版）。

gs/deform.py は「基準の形 1 組を MLP でずらす」。こちらは各ガウシアンが自分の時刻を持ち、
その前後でだけ現れて、まっすぐ動く:

    t_center    (N,1)  いちばん濃く見える時刻 μ_t
    log_t_scale (N,1)  時間方向の幅 σ_t の log
    velocity    (N,3)  速度 v

    時刻 t の位置      x + v (t - μ_t)
    時刻 t の不透明度  o · exp(-½ ((t - μ_t) / σ_t)²)

曲がった動きは、時間幅の短いガウシアンを時刻順に並べた折れ線で表す。基準の形が要らないので、
物体が大きく動いても、途中で現れたり消えたりしてもよい。かわりに、1 つのガウシアンを見られる
フレームは時間幅の中にあるものだけになる（1 時刻 1 視点のデータでは、見える視点の数が減る）。

論文との違い:
    - 動きは 1 次式（STG は 3 次の多項式）。回転と大きさは時刻によらない
    - 色は時刻によらない SH のまま（STG は特徴量を小さな MLP で色にする）
    - 時間方向の分割は、時間幅が t_split_thresh より長いものだけ（空間の split と同じく 1/1.6 に縮める）
    - 時間幅に下限を付けられる: σ_t = min_t_scale + exp(log_t_scale)。1 時刻 1 視点のデータでは、下限が無いと
      時間幅がフレーム数枚ぶんまで縮み、動きを学ばずに学習視点へ過適合する（experiments/dynamic）
"""

import math

import torch

from .gaussians import NAMES, GaussianModel


class SpacetimeModel(GaussianModel):
    names = NAMES + ("t_center", "log_t_scale", "velocity")

    def __init__(self, sh_degree=3, init_t_scale=0.2, t_split_thresh=0.05, min_t_scale=0.0):
        super().__init__(sh_degree)
        self.init_t_scale = init_t_scale
        self.min_t_scale = min_t_scale
        # 下限のすぐ上では分割しない（下限より先には縮められないので、分割しても時間幅がほとんど変わらない）
        self.t_split_thresh = max(t_split_thresh, 1.6 * min_t_scale)

    @property
    def t_scales(self):
        """時間幅 σ_t (N,1)"""
        return self.min_t_scale + torch.exp(self.params["log_t_scale"])

    def _set(self, tensors):
        if "t_center" not in tensors:   # 初期化: 時刻は [0,1] に一様、速度は 0
            n, dev = len(tensors["means"]), tensors["means"].device
            tensors = dict(tensors, t_center=torch.rand(n, 1, device=dev),
                           log_t_scale=torch.full((n, 1), math.log(max(self.init_t_scale - self.min_t_scale, 1e-3)),
                                                  device=dev),
                           velocity=torch.zeros(n, 3, device=dev))
        super()._set(tensors)

    def setup_optimizer(self, spatial_scale, lr_t=5e-4, lr_t_scale=5e-3, lr_velocity=1.6e-3, **kw):
        """速度の学習率は位置と同じくシーンの大きさに比例させる（減衰はさせない）"""
        super().setup_optimizer(spatial_scale, extra_lrs={
            "t_center": lr_t, "log_t_scale": lr_t_scale, "velocity": lr_velocity * spatial_scale}, **kw)

    def delta_at(self, t):
        """時刻 t のずれ（gs/render.py の delta）"""
        dt = float(t) - self.params["t_center"]                          # (N,1)
        w = torch.exp(-0.5 * (dt / self.t_scales) ** 2)
        return {"means": self.params["velocity"] * dt, "opacity": w.squeeze(-1)}

    # ---------------------------------------------------------------- densification
    def _long(self):
        return self.t_scales.squeeze(-1) > self.t_split_thresh

    def _is_big(self, size_thresh):
        return super()._is_big(size_thresh) | self._long()

    def _split_children(self, split_mask, size_thresh, ns=2):
        """空間で大きいものは空間で、時間幅が長いものは時間で分ける（両方に当てはまれば両方）"""
        split = super()._split_children(split_mask, size_thresh, ns)
        rep = lambda v: v[split_mask].repeat(ns, *[1] * (v.dim() - 1))
        big = rep(GaussianModel._is_big(self, size_thresh))
        for k in ("means", "log_scales"):          # 空間で大きくないものは、位置と大きさを元のままにする
            split[k] = torch.where(big[:, None], split[k], rep(self.params[k]))
        long = rep(self._long())[:, None]
        sigma_t = torch.exp(split["log_t_scale"])                    # 下限より上の部分。分割ではここを縮める
        dt = torch.where(long, torch.randn_like(sigma_t) * (self.min_t_scale + sigma_t), torch.zeros_like(sigma_t))
        split["t_center"] = split["t_center"] + dt
        split["means"] = split["means"] + split["velocity"] * dt     # 新しい時刻で、元と同じ軌道の上に置く
        split["log_t_scale"] = torch.where(long, torch.log(sigma_t / (0.8 * ns)), split["log_t_scale"])
        return split

    def state_dict(self):
        return dict(super().state_dict(), init_t_scale=self.init_t_scale, t_split_thresh=self.t_split_thresh,
                    min_t_scale=self.min_t_scale)

    @classmethod
    def from_state_dict(cls, sd, device="cuda"):
        m = cls(sd["max_sh_degree"], sd["init_t_scale"], sd["t_split_thresh"], sd.get("min_t_scale", 0.0))
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m
