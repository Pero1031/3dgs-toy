# -*- coding: utf-8 -*-
"""
段階 3: 3dgs-toy に遅延シェーディングを足す（反射方向で共有の環境照明を引く。3DGS-DR や Ref-GS と同じ系統の簡略版）。

各ガウシアンは、色の SH の代わりに次を持つ:
    法線   一番短い軸（カメラ側に向ける）。パラメータは持たず、回転と大きさから決まる
    base   放射輝度の定数項 (3)（softplus）。オラクルの「点ごとの定数」に当たる
    rough  材質の粗さに当たる、反射方向でのぼけの幅 (1)（sigmoid × SIG_MAX）

描画は 2 段階:
    1. ラスタライザで、法線・base・rough を色と同じように α ブレンドする（gs/rasterize.py。チャンネル数は任意）
    2. 画素ごとに、合成した法線 Σwn の向きで視線を反射させ、共有の環境照明をぼけの幅 σ でぼかして引く:
           放射輝度 = Σw·base + A·L_env(r, σ) + (1 − A)·背景      （A は累積不透明度）
       σ の決め方（mode）:
           learned  σ = 合成した rough / A                          既存手法と同じく、粗さを学習する
           toksvig  σ = 2√(1 − |n̄|) · √((1 + cos²θ)/2)、|n̄| = |Σwn| / A   粗さは学習しない（段階 2 の換算と同じ）
           both     σ = √(toksvig² + learned²)                       形状のぼけは換算、材質の粗さだけ学習する
       表示値への変換（Reinhard → sRGB の曲線）も画像と同じにして、表示値で誤差をとる。

環境照明は SH（LE 次）の係数で持つ。ぼけた照明の引き方は、SIG_LEVELS の各幅でぼかした SH を正距円筒の
テクスチャに展開し（基底は前もって計算しておく）、反射方向と σ で 3 次元の線形補間をする（grid_sample）。
SH の高い次数を画素ごとに微分可能に評価すると重すぎるので、テクスチャを介して照明・方向・σ のすべてに勾配を流す。
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from gs.gaussians import GaussianModel, inverse_sigmoid, knn_mean_sq_dist  # noqa: E402
from gs.rasterize import conic_and_radius, covariance_3d, project_gaussians, quat_to_rotmat, rasterize  # noqa: E402
from gs.sh import sh_basis  # noqa: E402

LE = 48                                   # 環境照明の SH の次数（オラクルと同じ）
SIG_MAX = math.radians(40.0)              # ぼけの幅の上限
SIG_STEP = math.radians(2.0)              # ぼかした照明を用意する幅の間隔（その間は線形補間）
TEX_H, TEX_W = 128, 256                   # 照明を展開する正距円筒のテクスチャ（1 画素 1.4°）
MODES = ("learned", "toksvig", "both")


# ---- 表示値と放射輝度の変換（env_rough.py と同じ。画像は Reinhard → sRGB の曲線） ----

def lin_to_srgb(t):
    t = t.clamp(min=0)
    return torch.where(t <= 0.0031308, 12.92 * t, 1.055 * t.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def srgb_to_lin(y):
    return torch.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def radiance_to_display(x):
    x = x.clamp(min=0)
    return lin_to_srgb(x / (1 + x)).clamp(0, 1)


def display_to_radiance(y):
    t = srgb_to_lin(torch.as_tensor(y)).clamp(max=0.995)
    return t / (1 - t)


# ---- 環境照明 ----

def dir_to_uv(r):
    """方向 -> 正距円筒の (u, v) ∈ [0,1)×[0,1]。u は +z から +x への経度、v は +y（北）からの角度"""
    u = torch.atan2(r[..., 0], r[..., 2]) / (2 * math.pi) % 1.0
    v = torch.acos(r[..., 1].clamp(-1 + 1e-6, 1 - 1e-6)) / math.pi
    return u, v


class EnvSH:
    """SH の係数で持つ環境照明（放射輝度）。lookup(r, σ) で、反射方向 r を幅 σ のガウスでぼかした値を返す"""

    def __init__(self, Le=LE, init=0.3, device="cuda"):
        self.Le, self.Ke = Le, (Le + 1) ** 2
        coef = torch.zeros(self.Ke, 3, device=device)
        coef[0] = init / 0.28209479177387814  # 一様な明るさ init から始める
        self.coef = nn.Parameter(coef)
        v = (torch.arange(TEX_H, device=device) + 0.5) / TEX_H
        u = (torch.arange(TEX_W, device=device) + 0.5) / TEX_W
        vv, uu = torch.meshgrid(v * math.pi, u * 2 * math.pi, indexing="ij")
        d = torch.stack([vv.sin() * uu.sin(), vv.cos(), vv.sin() * uu.cos()], -1).reshape(-1, 3)
        self.B = sh_basis(Le, d).float()                                     # (texels, Ke)
        self.l = torch.tensor([math.isqrt(k) for k in range(self.Ke)], dtype=torch.float32, device=device)
        sig = torch.arange(0, SIG_MAX + 1e-6, SIG_STEP, device=device)
        self.n_levels = len(sig)
        self.att = torch.exp(-self.l * (self.l + 1) * sig[:, None] ** 2 / 2)  # (levels, Ke)

    def textures(self):
        """各幅でぼかした照明のテクスチャ (1, 3, levels, TEX_H, TEX_W + 2)。経度方向の継ぎ目のために左右を 1 列ずつ巻いて足す"""
        E = self.att[:, :, None] * self.coef[None]                            # (levels, Ke, 3)
        T = (self.B @ E.permute(1, 0, 2).reshape(self.Ke, -1))                # (texels, levels·3)
        T = T.reshape(TEX_H, TEX_W, self.n_levels, 3).permute(3, 2, 0, 1)    # (3, levels, H, W)
        T = torch.cat([T[..., -1:], T, T[..., :1]], -1)
        return T[None]

    def lookup(self, r, sig, T=None):
        """r (M,3) 単位ベクトル、sig (M,) ラジアン -> (M,3)。T は textures() の結果（同じ反復で使い回す）"""
        T = self.textures() if T is None else T
        u, v = dir_to_uv(r)
        x = -1 + 2 * (u * TEX_W + 0.5) / (TEX_W + 1)                        # 巻いて足した列を含めた座標（align_corners=True）
        y = (-1 + 2 * (v * TEX_H - 0.5) / (TEX_H - 1)).clamp(-1, 1)
        z = (-1 + 2 * (sig / SIG_STEP) / (self.n_levels - 1)).clamp(-1, 1)
        grid = torch.stack([x, y, z], -1).reshape(1, 1, 1, -1, 3)
        return F.grid_sample(T, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, 0].T

    def smoothness(self):
        """高い次数ほど重い罰 Σ l(l+1)|e|² / Ke（照明の雑音を抑える）"""
        return ((self.l * (self.l + 1))[:, None] * self.coef ** 2).mean()


# ---- ガウシアン ----

class DeferredModel(GaussianModel):
    names = ("means", "log_scales", "quats", "opacity_logits", "base", "rough")

    def __init__(self):
        super().__init__(sh_degree=0)

    def init_on_sphere(self, n, radius, center=(0.0, 0.0, 0.0), flat=0.1, rough_deg=2.0, device="cuda"):
        """半径 radius の球面に一様に n 個、面に沿った円盤（法線方向の大きさは flat 倍）として置く。
        形状の大まかな初期値（3DGS では SfM の点に当たる）は分かっているとする"""
        g = torch.Generator(device="cpu").manual_seed(0)
        nrm = F.normalize(torch.randn(n, 3, generator=g), dim=-1).to(device)
        xyz = radius * nrm + torch.tensor(center, device=device)
        dist = torch.sqrt(knn_mean_sq_dist(xyz).clamp(min=1e-10))
        # 局所の z 軸を法線に向ける回転（z × n の軸まわりに acos(n_z)）。[w, x, y, z]
        q = F.normalize(torch.stack([1 + nrm[:, 2], -nrm[:, 1], nrm[:, 0], torch.zeros_like(nrm[:, 0])], -1), dim=-1)
        ls = torch.log(dist)[:, None].repeat(1, 3)
        ls[:, 2] += math.log(flat)
        self._set({
            "means": xyz,
            "log_scales": ls,
            "quats": q,
            "opacity_logits": inverse_sigmoid(torch.full((n, 1), 0.5, device=device)),
            "base": torch.full((n, 3), math.log(math.expm1(0.01)), device=device),   # softplus で 0.01
            "rough": inverse_sigmoid(torch.full((n, 1), rough_deg / math.degrees(SIG_MAX), device=device)),
        })

    def state_dict(self):
        return {"params": {k: v.detach().cpu() for k, v in self.params.items()}}

    @staticmethod
    def from_state_dict(sd, device="cuda"):
        m = DeferredModel()
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m


def gaussian_normals(scales, quats, means, cam_center):
    """一番短い軸の向き（カメラ側に向ける）(N,3)"""
    R = quat_to_rotmat(quats)
    idx = scales.argmin(dim=1)
    n = R.gather(2, idx[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    flip = ((cam_center - means) * n).sum(-1, keepdim=True) < 0
    return torch.where(flip, -n, n)


def pixel_dirs(cam):
    """画素中心を通る視線（カメラ → 物体）のワールドでの向き (H, W, 3)"""
    dev = cam.R.device
    j, i = torch.meshgrid(torch.arange(cam.H, device=dev), torch.arange(cam.W, device=dev), indexing="ij")
    d = torch.stack([(i + 0.5 - cam.cx) / cam.fx, (j + 0.5 - cam.cy) / cam.fy, torch.ones_like(i, dtype=torch.float32)], -1)
    return F.normalize(d @ cam.R, dim=-1)   # cam.R は world -> camera なので、行ベクトルに右から掛けると camera -> world


def render_deferred(model, env, cam, mode, bg_rad, tile=8, max_entries=2 ** 24, near=0.2, T=None):
    """
    Returns dict:
        image (H,W,3) 表示値、alpha (H,W)、means2d / ids / on_screen / radius（densification 用。render.py と同じ）
        len_n (H,W) |n̄|、sig_geo (H,W) 換算のぼけ、sig_mat (H,W) 合成した rough、sigma (H,W) 実際に使ったぼけ
    """
    p = model.params
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

    nrm = gaussian_normals(scales, quats, means, cam.center)
    feat = torch.cat([nrm, F.softplus(p["base"][ids]), torch.sigmoid(p["rough"][ids]) * SIG_MAX], -1)
    buf, A, on_screen = rasterize(means2d, conic, opacity, feat, tile_radius, depth, cam.W, cam.H,
                                  torch.zeros(feat.shape[-1], device=feat.device), tile=tile, max_entries=max_entries)
    nsum, base, rsum = buf[..., :3], buf[..., 3:6], buf[..., 6]
    nlen = nsum.norm(dim=-1).clamp(min=1e-8)
    nbar = nsum / nlen[..., None]
    len_n = (nlen / A.clamp(min=1e-6)).clamp(max=1.0)
    d = pixel_dirs(cam)
    dn = (d * nbar).sum(-1, keepdim=True)
    r = d - 2 * dn * nbar
    cos2 = (dn[..., 0] ** 2).clamp(max=1.0)
    sig_geo = 2 * torch.sqrt((1 - len_n).clamp(min=0) + 1e-6) * torch.sqrt((1 + cos2) / 2)
    sig_mat = rsum / A.clamp(min=1e-6)
    sigma = {"learned": sig_mat, "toksvig": sig_geo,
             "both": torch.sqrt(sig_geo ** 2 + sig_mat ** 2)}[mode]
    L = env.lookup(r.reshape(-1, 3), sigma.reshape(-1), T).reshape(cam.H, cam.W, 3)
    rad = base + A[..., None] * L + (1 - A[..., None]) * bg_rad
    return {"image": radiance_to_display(rad), "alpha": A, "means2d": means2d, "ids": ids,
            "on_screen": on_screen, "radius": radius, "len_n": len_n, "sig_geo": sig_geo,
            "sig_mat": sig_mat, "sigma": sigma}
