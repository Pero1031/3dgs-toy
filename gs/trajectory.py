# -*- coding: utf-8 -*-
"""
時刻ごとの位置を直接持つガウシアン（Dynamic 3D Gaussians, Luiten et al. 3DV 2024 の簡略版）。

ガウシアンの数・色・不透明度・大きさ・向きは時刻によらず、位置だけが時刻で変わる。各ガウシアンが
時刻の節点（[0,1] に等間隔に n_knots 個）ごとのずれを持ち、節点の間は線形に補間する:

    offsets (N, n_knots, 3)      時刻 t の位置 = means + 補間(offsets, t)

gs/deform.py（基準の形を MLP でずらす）と違い、動きはガウシアンごとに自由で、基準の形で
すべての時刻をおおまかに表せる必要がない。gs/spacetime.py（現れて消える）と違い、同じガウシアンが
最初から最後まで存在するので、物体の同じ点を追い続けるしかない。

学習は時間の順に進める（train_dynamic.py --trajectory）: はじめは最初の数節点までのフレームだけで学習し、
一定回数ごとに次の節点を使い始める。新しい節点は、直前の 2 節点から外挿した位置で初期化する
（＝前の時刻の位置から追跡を始める）。全部の時刻を最初から同時に学習すると、まだ学習していない時刻では
ガウシアンが物体の無い場所に取り残され、そこへ引き寄せる勾配が無い。

論文との違い:
    - 論文は多視点（約 30 台）で、最初の時刻で形を作ってから 1 時刻ずつ最適化する。ここは 1 時刻 1 視点
      なので、最初の数節点ぶんのフレームをまとめて使い、節点を増やしたあとも前の時刻を学習し続ける
    - 回転は時刻によらない（論文は時刻ごと）。局所剛体などの損失は入れていない
      （train_dynamic.py の --lambda-smooth で、近くのガウシアンの速度をそろえる損失は足せる）
"""

import torch

from .gaussians import NAMES, GaussianModel


class TrajectoryModel(GaussianModel):
    names = NAMES + ("offsets",)

    def __init__(self, sh_degree=3, n_knots=21, n_active=3):
        super().__init__(sh_degree)
        self.n_knots = n_knots
        self.n_active = min(n_active, n_knots)   # 使い始めた節点の数

    @property
    def t_active(self):
        """学習に使ってよい時刻の上限（最後に使い始めた節点の時刻）"""
        return (self.n_active - 1) / (self.n_knots - 1)

    def _set(self, tensors):
        if "offsets" not in tensors:
            n, dev = len(tensors["means"]), tensors["means"].device
            tensors = dict(tensors, offsets=torch.zeros(n, self.n_knots, 3, device=dev))
        super()._set(tensors)

    def setup_optimizer(self, spatial_scale, lr_offsets=3.2e-4, **kw):
        """節点のずれの学習率はシーンの大きさに比例させる（後から使い始める節点があるので、減衰はさせない）"""
        super().setup_optimizer(spatial_scale, extra_lrs={"offsets": lr_offsets * spatial_scale}, **kw)

    def _segment(self, t):
        u = min(max(float(t), 0.0), 1.0) * (self.n_knots - 1)
        i = min(int(u), self.n_knots - 2)
        return i, u - i

    def delta_at(self, t):
        """時刻 t のずれ（gs/render.py の delta）"""
        i, w = self._segment(t)
        o = self.params["offsets"]
        return {"means": (1 - w) * o[:, i] + w * o[:, i + 1]}

    def velocity_at(self, t):
        """時刻 t の速度 (N,3)（節点の間で一定）"""
        i, _ = self._segment(t)
        o = self.params["offsets"]
        return (o[:, i + 1] - o[:, i]) * (self.n_knots - 1)

    @torch.no_grad()
    def activate_next(self):
        """次の節点を使い始める。直前の 2 節点から外挿した位置で初期化し、それより後の節点も同じ位置にしておく"""
        k = self.n_active
        if k >= self.n_knots:
            return False
        o = self.params["offsets"]
        o[:, k:] = (2 * o[:, k - 1] - o[:, k - 2])[:, None]
        self.n_active += 1
        return True

    def state_dict(self):
        return dict(super().state_dict(), n_knots=self.n_knots, n_active=self.n_active)

    @classmethod
    def from_state_dict(cls, sd, device="cuda"):
        m = cls(sd["max_sh_degree"], sd["n_knots"], sd["n_active"])
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m
