# -*- coding: utf-8 -*-
"""
ガウシアン群のパラメータと、その「数」を変える処理（densification）。

各ガウシアンが持つ学習パラメータ（括弧内は描画時の活性化）:
    means          (N,3)    中心位置
    log_scales     (N,3)    軸ごとの標準偏差の log          (exp)
    quats          (N,4)    回転クォータニオン [w,x,y,z]     (正規化)
    opacity_logits (N,1)    不透明度の logit                 (sigmoid)
    sh0            (N,1,3)  SH 0 次係数（拡散色）
    shN            (N,K,3)  SH 1 次以上の係数（視点依存の色）K = (deg+1)^2 - 1

densification（論文 Sec.5.2）:
    画面上の位置勾配が大きい = その付近の再構成が足りていない、とみなして
      - 小さいガウシアンは複製 (clone)：形状を覆い切れていない領域を埋める
      - 大きいガウシアンは 2 つに分割 (split)：細部を表現できていない領域を細かくする
    さらに不透明度がほぼ 0 のもの・大きすぎるものを削除 (prune) する。
    パラメータの数が変わるので、Adam のモーメントも同じ形に作り直す必要がある。
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rasterize import quat_to_rotmat
from .sh import rgb_to_sh0

NAMES = ("means", "log_scales", "quats", "opacity_logits", "sh0", "shN")


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@torch.no_grad()
def knn_mean_sq_dist(x, k=3, chunk=2048):
    """各点から最近傍 k 点までの距離の二乗の平均（初期スケールの決定用）"""
    out = []
    for i in range(0, len(x), chunk):
        d2 = torch.cdist(x[i:i + chunk], x).pow(2)
        out.append(d2.topk(k + 1, dim=1, largest=False).values[:, 1:].mean(1))  # 先頭は自分自身
    return torch.cat(out)


class GaussianModel:
    def __init__(self, sh_degree=3):
        self.max_sh_degree = sh_degree
        self.params = {}
        self.optimizer = None

    def __len__(self):
        return self.params["means"].shape[0]

    # ---------------------------------------------------------------- 活性化後の値
    @property
    def means(self):
        return self.params["means"]

    @property
    def scales(self):
        return torch.exp(self.params["log_scales"])

    @property
    def quats(self):
        return F.normalize(self.params["quats"], dim=-1)

    @property
    def opacities(self):
        return torch.sigmoid(self.params["opacity_logits"]).squeeze(-1)

    # ---------------------------------------------------------------- 初期化
    def init_random(self, n, half_extent, device="cuda"):
        """一辺 2*half_extent の立方体内にランダムに点を置く（公式実装の合成データ用の初期化と同じ方針）"""
        xyz = (torch.rand(n, 3, device=device) * 2 - 1) * half_extent
        rgb = torch.rand(n, 3, device=device)
        self.init_from_points(xyz, rgb)

    def init_from_points(self, xyz, rgb):
        n, dev = len(xyz), xyz.device
        dist = torch.sqrt(knn_mean_sq_dist(xyz).clamp(min=1e-7))  # 近傍点との間隔くらいの大きさの球にする
        K = (self.max_sh_degree + 1) ** 2 - 1
        self._set({
            "means": xyz,
            "log_scales": torch.log(dist)[:, None].repeat(1, 3),
            "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1),
            "opacity_logits": inverse_sigmoid(torch.full((n, 1), 0.1, device=dev)),
            "sh0": rgb_to_sh0(rgb)[:, None, :],
            "shN": torch.zeros(n, K, 3, device=dev),
        })

    def _set(self, tensors):
        self.params = {k: nn.Parameter(tensors[k].contiguous()) for k in NAMES}
        self._reset_stats()

    def _reset_stats(self):
        n, dev = len(self), self.means.device
        self.grad_accum = torch.zeros(n, device=dev)   # 画面上の位置勾配ノルムの累積
        self.denom = torch.zeros(n, device=dev)        # 画面に映った回数
        self.max_radii2d = torch.zeros(n, device=dev)  # 画面上の最大半径 [px]

    # ---------------------------------------------------------------- 最適化
    def setup_optimizer(self, spatial_scale, lr_means=1.6e-4, lr_means_final=1.6e-6,
                        lr_sh=2.5e-3, lr_opacity=0.05, lr_scale=5e-3, lr_rot=1e-3):
        """学習率は公開当初の公式実装の既定値。位置の学習率だけシーンの大きさに比例させ、指数減衰させる。"""
        self.lr_means_init = lr_means * spatial_scale
        self.lr_means_final = lr_means_final * spatial_scale
        lrs = {"means": self.lr_means_init, "sh0": lr_sh, "shN": lr_sh / 20,
               "opacity_logits": lr_opacity, "log_scales": lr_scale, "quats": lr_rot}
        groups = [{"params": [self.params[k]], "lr": lrs[k], "name": k} for k in NAMES]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def update_lr(self, step, max_steps):
        t = min(step / max_steps, 1.0)
        lr = math.exp((1 - t) * math.log(self.lr_means_init) + t * math.log(self.lr_means_final))
        for g in self.optimizer.param_groups:
            if g["name"] == "means":
                g["lr"] = lr

    def _replace(self, name, tensor, state_fn):
        """パラメータ name を tensor に差し替え、Adam の 1 次・2 次モーメントも state_fn で同じ形にする。"""
        new = nn.Parameter(tensor.contiguous())
        if self.optimizer is not None:
            group = next(g for g in self.optimizer.param_groups if g["name"] == name)
            state = self.optimizer.state.pop(group["params"][0], None)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    state[key] = state_fn(state[key])
                self.optimizer.state[new] = state
            group["params"][0] = new
        self.params[name] = new

    def _append(self, extra):
        for k in NAMES:
            e = extra[k]
            self._replace(k, torch.cat([self.params[k].detach(), e]),
                          lambda s, e=e: torch.cat([s, torch.zeros_like(e)]))  # 新しい点のモーメントは 0

    def _keep(self, mask):
        for k in NAMES:
            self._replace(k, self.params[k].detach()[mask], lambda s: s[mask])

    # ---------------------------------------------------------------- densification
    @torch.no_grad()
    def add_densification_stats(self, grad2d, ids, radius, W, H):
        """1 回の描画ぶんの統計を足し込む。grad2d は画素座標での ∂L/∂μ' (len(ids), 2)。

        公式実装は NDC 座標（画面幅が 2）での勾配を見て閾値 0.0002 と比べるので、
        画素 -> NDC の換算 (W/2, H/2) を掛けて揃える。
        """
        scale = torch.tensor([0.5 * W, 0.5 * H], device=grad2d.device)
        self.grad_accum[ids] += (grad2d * scale).norm(dim=-1)
        self.denom[ids] += 1
        self.max_radii2d[ids] = torch.maximum(self.max_radii2d[ids], radius)

    @torch.no_grad()
    def densify_and_prune(self, grad_thresh, min_opacity, extent, max_screen_size=None,
                          percent_dense=0.01):
        n0 = len(self)
        grads = self.grad_accum / self.denom.clamp(min=1)
        scales = self.scales
        is_big = scales.max(dim=1).values > percent_dense * extent
        clone_mask = (grads >= grad_thresh) & ~is_big
        split_mask = (grads >= grad_thresh) & is_big
        big_screen = self.max_radii2d > max_screen_size if max_screen_size else None

        # clone: 同じ位置にそのまま複製（深度順で前後が決まるので、以後の勾配で別々に動いていく）
        new = {k: v[clone_mask] for k, v in self.params.items()}

        # split: 元のガウシアンの分布から位置を 2 つサンプルし、1/1.6 に縮めた複製を置く
        ns = 2
        split = {k: v[split_mask].repeat(ns, *[1] * (v.dim() - 1)) for k, v in self.params.items()}
        std = scales[split_mask].repeat(ns, 1)
        R = quat_to_rotmat(self.quats[split_mask]).repeat(ns, 1, 1)
        split["means"] = split["means"] + (R @ (torch.randn_like(std) * std)[..., None]).squeeze(-1)
        split["log_scales"] = torch.log(std / (0.8 * ns))

        self._append({k: torch.cat([new[k], split[k]]) for k in NAMES})

        # prune: split の元、ほぼ透明なもの、（opacity reset 後は）大きすぎるもの
        pad = torch.zeros(len(self) - n0, dtype=torch.bool, device=self.means.device)
        prune = torch.cat([split_mask, pad]) | (self.opacities < min_opacity)
        if big_screen is not None:
            prune |= torch.cat([big_screen, pad])
            prune |= self.scales.max(dim=1).values > 0.1 * extent
        self._keep(~prune)
        self._reset_stats()
        return int(clone_mask.sum()), int(split_mask.sum()), int(prune.sum())

    @torch.no_grad()
    def reset_opacity(self, value=0.01):
        """不透明度を一斉に小さくする（論文 Sec.5.2）。

        カメラ近くに浮いた「ゴミ」ガウシアンは、本当に必要なら最適化で不透明度が戻り、
        不要なら戻らずに prune される。
        """
        cap = inverse_sigmoid(torch.tensor(value, device=self.means.device))
        self._replace("opacity_logits", torch.minimum(self.params["opacity_logits"].detach(), cap),
                      torch.zeros_like)

    # ---------------------------------------------------------------- 保存・読み込み
    def state_dict(self):
        return {"max_sh_degree": self.max_sh_degree,
                "params": {k: v.detach().cpu() for k, v in self.params.items()}}

    @staticmethod
    def from_state_dict(sd, device="cuda"):
        m = GaussianModel(sd["max_sh_degree"])
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m

    @torch.no_grad()
    def save_ply(self, path):
        """公式実装と同じ形式の .ply（SuperSplat などの 3DGS ビューアで開ける）"""
        p = {k: v.detach().cpu() for k, v in self.params.items()}
        n = len(self)
        f_dc = p["sh0"].transpose(1, 2).reshape(n, -1)
        f_rest = p["shN"].transpose(1, 2).reshape(n, -1)  # チャンネルごとにまとめる (R...,G...,B...)
        cols = [p["means"], torch.zeros(n, 3), f_dc, f_rest, p["opacity_logits"], p["log_scales"], p["quats"]]
        data = torch.cat(cols, 1).numpy().astype("<f4")
        names = (["x", "y", "z", "nx", "ny", "nz"]
                 + [f"f_dc_{i}" for i in range(f_dc.shape[1])]
                 + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
                 + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
        header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
        header += "".join(f"property float {s}\n" for s in names) + "end_header\n"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(np.ascontiguousarray(data).tobytes())
