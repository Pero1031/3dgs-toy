# -*- coding: utf-8 -*-
"""
段階 2 のオラクル: 条件を制御した球（make_sphere.py）で、点ごとに最適化したぼけ（粗さ）と、
画素の中の法線のばらつきから換算したぼけを比べる（3DGS を使わない検証。env_rough.py の球版）。

問い
    1. 凹凸が画素より細かいとき、点ごとに最適化したぼけは Toksvig の換算 σ_r = 2σ_n に合うか
    2. 画素の大きさ（512 / 256 / 128 px）を変えると、最適なぼけは「画素の中のばらつき」の変化に従うか
    3. 材質の粗さ（GGX α）によるぼけは、どの幅のガウスに当たるか（brdf の区画）

モデル（どれも共有の照明 SH 48 次 + 点ごとの定数。解き方は env_rough.py と同じ）
    material      σ = 0。法線は点ごとの足跡で平均した向き（以下のモデルも、measured_obs 以外は同じ法線）
    measured_pt   点ごとの足跡（接平面で等方。1 軸あたり、画素の箱型と、観測を拾う双線形補間の三角形を合わせた広がり、
                  幅はその点を見る学習視点の平均の奥行きでの 1 画素）で法線を平均し、σ = 2√(1 − |n̄|)。像の measured と同じ
    measured_obs  観測ごとに、その画素の足跡（画像面の箱型 ⊗ 三角形を球面に投影したもの。斜めから見ると伸びる）で
                  法線を平均し、その向きで反射させ、σ = 2√(1 − |n̄|) でぼかす。Toksvig 流の前処理の理想形
    measured_obs_aniso  measured_obs の σ を √((1 + cos²θ)/2) 倍する（θ は視線と平均法線の角）。法線を入射面の中で
                  δ 傾けると反射方向は 2δ 曲がるが、入射面に垂直に傾けると 2δ cosθ しか曲がらない。その 2 軸の
                  分散の平均に当たる等方のぼけ
    supersampled  ぼけの近似を使わない厳密なモデル。観測ごとに、足跡のサンプル（SUPER_SAMPLES 個）の法線で
                  それぞれ反射させた方向の SH を平均して、照明に掛ける。等方なガウスでぼかすモデルの上限
    fitted        点ごとに σ を格子から選ぶ（照明と交互に解く）。既存手法が学習する粗さの代わり
    sh6           点ごとの SH 6 次（参考）

法線は凹凸の解析的な式から求める（レンダラーは頂点法線を補間するので、傾きは約 3% 小さい。make_sphere.py mesh の検算）。
凹凸による位置のずれ（最大 1.3 mm）は投影では無視し、基準の球面で扱う。

区画の扱い: brdf の区画は形がなめらかで、measured は材質の粗さを知らないので、モデルの比較は smooth と bump の
5 区画の点だけで行う。brdf の区画の最適なぼけは、全区画の点で fitted を解き直して求める。

    python experiments/gloss/sphere_rough.py [--scales 512 256 128] [--no-denoised-check]

ほかに、measured_obs の σ を一律に k 倍したときの誤差（最適な倍率）と、supersampled の平均した SH が
Y(n̄)·exp(−l(l+1)σ²/2) に当たるとしたときの等価な σ（l = 4 と 12）も記録する。

出力: experiments/gloss/sphere_rough.json、data/gloss_sphere/sphere_rough_points.npz（点ごとの値。図を描き直す用）
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env_rough as er  # noqa: E402
import make_sphere as ms  # noqa: E402
import sh_oracle as so  # noqa: E402

DEV = er.DEV
N_PER_SECTOR = 800       # 区画ごとの点の数
LAT_MAX = 50.0           # 緯度がこれを超える点は使わない（凹凸を弱める範囲と極を避ける）
BORDER_MARGIN = 7.0      # 区画の境目からこれ以上離れた点だけ使う [度]（つなぎの幅 4° と足跡の広がりを避ける）
FP_SAMPLES = 512         # 足跡の中で法線を平均するサンプル数
SUPER_SAMPLES = 64       # supersampled で反射方向の SH を平均するサンプル数（FP_SAMPLES の先頭。Sobol 列なので偏らない）
K_SCAN = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # measured_obs の σ に掛ける倍率の候補
EQUIV_L = [4, 12]        # 等価な σ を読む SH の次数
LP, S = 0, 0.1           # 点ごとの項の次数、照明のなめらかさの罰（env_rough.py で粗さの比較に使った設定）
SH_DEG = 6
GEOM = [k for k, (_, _, mat) in enumerate(ms.SECTORS) if mat == "au"]  # smooth と bump の区画
SEED = 3
OUT_NPZ = ms.DATA / "sphere_rough_points.npz"


# ---- 画像とカメラ ----

def read_pfm(path):
    with open(path, "rb") as fh:
        kind = fh.readline().strip()
        w, h = map(int, fh.readline().split())
        sc = float(fh.readline())
        a = np.frombuffer(fh.read(), dtype="<f4" if sc < 0 else ">f4")
    return a.reshape(h, w, 3 if kind == b"PF" else 1)[::-1].squeeze().copy()


def load_split(split, scale, denoised):
    """表示値の画像 (V, S, S, 3) と、3x3 で収縮したマスク (V, S, S)。縮小は放射輝度で平均してから表示値に直す
    （画素が大きいカメラで撮ったのと同じ）。視点の並びは make_sphere.views() の順"""
    k = ms.SIZE // scale
    names = [(n, frm) for s, n, frm in ms.views() if s == split]
    img, msk = [], []
    for name, _ in names:
        base = ms.DATA / "render" / split / name
        x = torch.from_numpy(read_pfm(f"{base}{'_denoised' if denoised else ''}.pfm")).to(DEV)
        m = torch.from_numpy(read_pfm(f"{base}_mask.pfm")).to(DEV)
        x = F.avg_pool2d(x.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)
        m = F.avg_pool2d(m[None, None], k)[0, 0]
        img.append(er.radiance_to_display(x).float())
        msk.append(-F.max_pool2d(-m[None, None], 3, 1, 1)[0, 0])
    return [frm for _, frm in names], torch.stack(img), torch.stack(msk)


class Cam:
    def __init__(self, frm, scale):
        c2w = np.array(ms.c2w(frm))
        Rcv = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])   # OpenCV 流儀: x 右、y 下、z 前
        self.Rc2w = torch.tensor(Rcv, dtype=torch.float32, device=DEV)
        self.o = torch.tensor(frm, dtype=torch.float32, device=DEV)
        self.f = (scale / 2) / math.tan(math.radians(ms.FOV / 2))
        self.c = scale / 2
        self.S = scale

    def project(self, X):
        pc = (X - self.o) @ self.Rc2w
        return self.f * pc[..., 0] / pc[..., 2] + self.c, self.f * pc[..., 1] / pc[..., 2] + self.c, pc[..., 2]

    def hit_sphere(self, u, v):
        """画素の位置 (u, v) を通る光線と基準の球の、手前の交点の法線 n0"""
        d = torch.stack([(u - self.c) / self.f, (v - self.c) / self.f, torch.ones_like(u)], -1) @ self.Rc2w.T
        d = F.normalize(d, dim=-1)
        b = (d * self.o).sum(-1)
        disc = (b * b - (self.o @ self.o - ms.R0 ** 2)).clamp(min=0)
        X = self.o + (-b - disc.sqrt())[..., None] * d
        return F.normalize(X, dim=-1)


# ---- 凹凸の法線（make_sphere.surface の torch 版。区画の中ほどだけを使うので振幅は点ごとの定数） ----

FREQ = torch.tensor(ms.FREQ, dtype=torch.float32, device=DEV)
PHASE = torch.tensor(ms.PHASE, dtype=torch.float32, device=DEV)


def normals(n0, a, chunk=1 << 17):
    """基準の球の法線 n0 (m, 3) の所の、凹凸をつけた表面の法線。a (m,) は振幅 [m]"""
    out = torch.empty_like(n0)
    c = math.sqrt(2.0 / ms.K_WAVES)
    for i in range(0, len(n0), chunk):
        nn = n0[i:i + chunk]
        ph = 2 * math.pi * (ms.R0 * nn) @ FREQ.T + PHASE
        dg = -c * 2 * math.pi * (torch.sin(ph) @ FREQ)
        dt = dg - (dg * nn).sum(-1, keepdim=True) * nn
        out[i:i + chunk] = F.normalize(nn - a[i:i + chunk, None] * dt, dim=-1)
    return out


def footprint_offsets():
    """足跡の中のサンプル位置（画素単位、1 軸あたり一様分布 3 つの和 = 箱型 ⊗ 双線形補間の三角形）(M, 2)"""
    s = torch.quasirandom.SobolEngine(6, scramble=True, seed=SEED).draw(FP_SAMPLES).to(DEV)
    return torch.stack([s[:, :3].sum(1) - 1.5, s[:, 3:].sum(1) - 1.5], 1)


def mean_normal(n0s, a):
    """足跡のサンプルの基準の法線 n0s (m, M, 3) で凹凸の法線を求めて平均する -> 向き (m, 3)、σ = 2√(1 − |n̄|) (m,)"""
    m, M, _ = n0s.shape
    n = normals(n0s.reshape(-1, 3), a[:, None].expand(m, M).reshape(-1)).reshape(m, M, 3).mean(1)
    ln = n.norm(dim=-1).clamp(max=1)
    return n / ln[:, None], 2 * (1 - ln).sqrt()


# ---- 点と観測 ----

def sample_points():
    """区画ごとに N_PER_SECTOR 点（球面上で一様、境目と高緯度を除く）-> 基準の法線 (n, 3)、区画の番号 (n,)"""
    rng = np.random.default_rng(SEED)
    out, sec = [], []
    need = [N_PER_SECTOR] * len(ms.SECTORS)
    while any(need):
        u = rng.normal(size=(200000, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        lon, lat = ms.lon_lat(u)
        off = lon % ms.SECTOR_DEG
        k = (lon // ms.SECTOR_DEG).astype(int)
        ok = (np.abs(lat) <= LAT_MAX) & (off >= BORDER_MARGIN) & (off <= ms.SECTOR_DEG - BORDER_MARGIN)
        for kk in range(len(ms.SECTORS)):
            take = u[ok & (k == kk)][:need[kk]]
            out.append(take)
            sec += [kk] * len(take)
            need[kk] -= len(take)
    return np.concatenate(out), np.array(sec)


@torch.no_grad()
def observe(cams, img, msk, n0, a, offs):
    """各視点で、点が見えているか・観測（表示値、双線形補間）・視線方向・奥行き・観測ごとの足跡の平均法線とぼけ"""
    P = ms.R0 * n0
    V, n = len(cams), len(n0)
    vis = torch.zeros(V, n, dtype=torch.bool, device=DEV)
    col = torch.zeros(V, n, 3, dtype=torch.float64, device=DEV)
    dirs = torch.zeros(V, n, 3, device=DEV)
    depth = torch.zeros(V, n, device=DEV)
    n_obs = torch.zeros(V, n, 3, device=DEV)
    s_obs = torch.zeros(V, n, device=DEV)
    for i, cam in enumerate(cams):
        u, v, z = cam.project(P)
        to_cam = F.normalize(cam.o - P, dim=-1)
        inb = (u >= 0) & (u < cam.S) & (v >= 0) & (v < cam.S)
        pix = v.clamp(0, cam.S - 1).long() * cam.S + u.clamp(0, cam.S - 1).long()
        ok = inb & ((n0 * to_cam).sum(-1) > so.COS_MIN) & (msk[i].reshape(-1)[pix] > 0.999)
        grid = torch.stack([2 * u / cam.S - 1, 2 * v / cam.S - 1], -1)[None, None]
        col[i] = F.grid_sample(img[i].permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T.double()
        vis[i], dirs[i], depth[i] = ok, -to_cam, z
        idx = ok.nonzero()[:, 0]
        for j in range(0, len(idx), 1024):
            jj = idx[j:j + 1024]
            n0s = cam.hit_sphere(u[jj, None] + offs[None, :, 0], v[jj, None] + offs[None, :, 1])
            n_obs[i, jj], s_obs[i, jj] = mean_normal(n0s, a[jj])
    return vis, col, dirs, depth, n_obs, s_obs


@torch.no_grad()
def point_footprint(n0, a, width, offs):
    """点ごとの等方な足跡（接平面、1 軸あたり画素幅 width [m] の箱型 ⊗ 三角形）で法線を平均する"""
    t1, t2 = (torch.tensor(t, dtype=torch.float32, device=DEV) for t in ms.tangent_frame(n0.cpu().numpy().astype(np.float64)))
    out_n, out_s = torch.empty_like(n0), torch.empty(len(n0), device=DEV)
    for j in range(0, len(n0), 1024):
        sl = slice(j, j + 1024)
        X = ms.R0 * n0[sl, None] + width[sl, None, None] * (offs[None, :, 0, None] * t1[sl, None]
                                                             + offs[None, :, 1, None] * t2[sl, None])
        out_n[sl], out_s[sl] = mean_normal(F.normalize(X, dim=-1), a[sl])
    return out_n, out_s


@torch.no_grad()
def supersampled_Y(cams, vis, dirs, n0, a, offs):
    """観測ごと（vis.nonzero の順）に、足跡のサンプルの法線でそれぞれ反射させた方向の SH を平均する (n_obs, Ke)"""
    P = ms.R0 * n0
    M = SUPER_SAMPLES
    out = []
    for i, cam in enumerate(cams):
        jj_all = vis[i].nonzero()[:, 0]
        u, v, _ = cam.project(P)
        for j in range(0, len(jj_all), 128):
            jj = jj_all[j:j + 128]
            n0s = cam.hit_sphere(u[jj, None] + offs[None, :M, 0], v[jj, None] + offs[None, :M, 1])
            m = len(jj)
            ns = normals(n0s.reshape(-1, 3), a[jj, None].expand(m, M).reshape(-1)).reshape(m, M, 3)
            dd = dirs[i, jj][:, None, :]
            r = dd - 2 * (dd * ns).sum(-1, keepdim=True) * ns
            out.append(er.sh_basis(er.LE, r.reshape(-1, 3)).float().reshape(m, M, -1).mean(1))
    return torch.cat(out)


def equivalent_sigma(Y_sup, Y_mean, l):
    """平均した SH の次数 l の成分が、平均法線の向きの SH の a_l 倍だとして、a_l = exp(−l(l+1)σ²/2) から σ を読む (n_obs,)"""
    m = er.L_OF == l
    a_l = (Y_sup[:, m] * Y_mean[:, m]).sum(1) / (Y_mean[:, m] ** 2).sum(1).clamp(min=1e-12)
    return (-2 * a_l.clamp(1e-4, 1).log() / (l * (l + 1))).clamp(min=0).sqrt()


def per_point_rms(x, pid, n_pts):
    s2 = torch.zeros(n_pts, device=DEV).index_add_(0, pid, x.float() ** 2)
    cnt = torch.zeros(n_pts, device=DEV).index_add_(0, pid, torch.ones_like(x, dtype=torch.float32))
    return (s2 / cnt.clamp(min=1)).sqrt()


# ---- 解析 ----

def identifiable(loss):
    rmse = loss.sqrt()
    return ((rmse.max(1).values - rmse.min(1).values) > er.ID_CONTRAST).cpu().numpy()


def sector_psnr(o, pred, sec_pts):
    """区画ごとの、表示値の PSNR"""
    err = ((o.display(pred) - o.ydisp) ** 2).mean(1)
    sec = torch.tensor(sec_pts, device=DEV)[o.pid]
    out = {}
    for k in np.unique(sec_pts):
        m = sec == int(k)
        out[ms.SECTORS[k][0]] = -10 * math.log10(err[m].mean().item())
    return out


def run_scale(scale, n0_all, sec_all, a_all, offs, grid, t0, denoised=False, quick=False):
    data, cams = {}, {}
    for split in ("train", "test"):
        frms, img, msk = load_split(split, scale, denoised)
        cams[split] = [Cam(f, scale) for f in frms]
        data[split] = observe(cams[split], img, msk, n0_all, a_all, offs)
        del img, msk
    torch.cuda.empty_cache()
    keep = (data["train"][0].sum(0) >= so.MIN_TRAIN_OBS).cpu().numpy()
    vis_tr, _, _, depth_tr, _, s_obs_tr = data["train"]
    z_mean = (depth_tr * vis_tr).sum(0) / vis_tr.sum(0).clamp(min=1)
    width = z_mean * 2 * math.tan(math.radians(ms.FOV / 2)) / scale            # 点の所での 1 画素の幅 [m]
    n_pt, s_pt = point_footprint(n0_all, a_all, width, offs)
    s_obs_rms = ((s_obs_tr ** 2 * vis_tr).sum(0) / vis_tr.sum(0).clamp(min=1)).sqrt()
    print(f"[{scale} px{'・デノイズ後' if denoised else ''}] 観測と足跡: 点 {keep.sum()} / {len(keep)}、"
          f"学習の観測 {int(vis_tr[:, keep].sum())}（{time.time() - t0:.0f} 秒）", flush=True)

    res = {"pixel_width_mm_median": float(width.median() * 1000)}
    pts = {"keep": keep, "width_mm": width.cpu().numpy() * 1000,
           "measured_pt_deg": np.degrees(s_pt.cpu().numpy()), "measured_obs_rms_deg": np.degrees(s_obs_rms.cpu().numpy())}

    def subset(sel):
        idx = torch.tensor(np.nonzero(sel)[0], device=DEV)
        return {sp: [x[:, idx] for x in d] for sp, d in data.items()}, idx

    def fitted_run(sel, tag):
        d, idx = subset(sel)
        o_tr, o_te = er.Obs(*d["train"][:3], n_pt[idx], "radiance"), er.Obs(*d["test"][:3], n_pt[idx], "radiance")
        sig, _, ptr, pte, loss, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
        out = (np.degrees(sig.cpu().numpy()), identifiable(loss), er.psnr(o_tr, ptr), er.psnr(o_te, pte),
               sector_psnr(o_te, pte, sec_all[sel]))
        print(f"  fitted（{tag}）: 学習 {out[2]:.2f} / テスト {out[3]:.2f} dB（{time.time() - t0:.0f} 秒）", flush=True)
        del o_tr, o_te
        torch.cuda.empty_cache()
        return out

    geom = keep & np.isin(sec_all, GEOM)
    if quick:  # デノイズ後の確認: fitted だけ
        f_deg, f_id, *_ = fitted_run(geom, "smooth と bump の区画")
        pts["fitted_geom_deg"], pts["fitted_geom_ident"] = np.full(len(keep), np.nan), np.zeros(len(keep), bool)
        pts["fitted_geom_deg"][geom], pts["fitted_geom_ident"][geom] = f_deg, f_id
        return res, pts

    # モデルの比較（smooth と bump の区画）
    d, idx = subset(geom)
    o_tr, o_te = er.Obs(*d["train"][:3], n_pt[idx], "radiance"), er.Obs(*d["test"][:3], n_pt[idx], "radiance")
    res["n_points_geom"], res["n_obs_geom"] = int(geom.sum()), {"train": len(o_tr.pid), "test": len(o_te.pid)}
    res["models"] = {}

    def record(key, ptr, pte, o_tr, o_te):
        res["models"][key] = {"train": er.psnr(o_tr, ptr), "test": er.psnr(o_te, pte),
                              "test_by_sector": sector_psnr(o_te, pte, sec_all[geom])}
        print(f"  {key:<13} 学習 {res['models'][key]['train']:6.2f}  テスト {res['models'][key]['test']:6.2f} dB"
              f"（{time.time() - t0:.0f} 秒）", flush=True)

    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, None, SH_DEG, None)
    record("sh6", ptr, pte, o_tr, o_te)
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, torch.zeros(o_tr.n_pts, device=DEV), LP, S)
    record("material", ptr, pte, o_tr, o_te)
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, s_pt[idx], LP, S)
    record("measured_pt", ptr, pte, o_tr, o_te)
    sig, _, ptr, pte, loss, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
    record("fitted", ptr, pte, o_tr, o_te)
    f_deg, f_id = np.degrees(sig.cpu().numpy()), identifiable(loss)
    del o_tr, o_te, ptr, pte
    torch.cuda.empty_cache()

    o_tr = er.Obs(*d["train"][:3], d["train"][4], "radiance")
    o_te = er.Obs(*d["test"][:3], d["test"][4], "radiance")
    o_tr.sig_obs, o_te.sig_obs = d["train"][5][d["train"][0]], d["test"][5][d["test"][0]]
    s_tr, s_te = o_tr.sig_obs, o_te.sig_obs
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
    record("measured_obs", ptr, pte, o_tr, o_te)

    # 反射の異方性を補正した等方のぼけ
    def cos_obs(sp):
        vis, dirs, n_obs = d[sp][0], d[sp][2], d[sp][4]
        return (-(dirs * n_obs).sum(-1))[vis]
    f_tr, f_te = ((1 + cos_obs("train") ** 2) / 2).sqrt(), ((1 + cos_obs("test") ** 2) / 2).sqrt()
    o_tr.sig_obs, o_te.sig_obs = s_tr * f_tr, s_te * f_te
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
    record("measured_obs_aniso", ptr, pte, o_tr, o_te)
    aniso_rms = per_point_rms(s_tr * f_tr, o_tr.pid, o_tr.n_pts)

    # σ を一律に k 倍したとき
    res["scan_measured_obs"] = {}
    for k in K_SCAN:
        o_tr.sig_obs, o_te.sig_obs = k * s_tr, k * s_te
        _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
        res["scan_measured_obs"][str(k)] = {"train": er.psnr(o_tr, ptr), "test": er.psnr(o_te, pte)}
    best = max(res["scan_measured_obs"].items(), key=lambda kv: kv[1]["train"])
    print(f"  measured_obs の倍率: 学習で最良の k = {best[0]}（学習 {best[1]['train']:.2f} / テスト "
          f"{best[1]['test']:.2f} dB）（{time.time() - t0:.0f} 秒）", flush=True)

    # 足跡の中で反射方向の SH を平均する厳密なモデルと、等価な σ
    Ys_tr = supersampled_Y(cams["train"], d["train"][0], d["train"][2], n0_all[idx], a_all[idx], offs)
    Ys_te = supersampled_Y(cams["test"], d["test"][0], d["test"][2], n0_all[idx], a_all[idx], offs)
    equiv = {l: per_point_rms(equivalent_sigma(Ys_tr, o_tr.Y, l), o_tr.pid, o_tr.n_pts) for l in EQUIV_L}
    o_tr.Y, o_te.Y = Ys_tr, Ys_te
    o_tr.sig_obs, o_te.sig_obs = torch.zeros_like(s_tr), torch.zeros_like(s_te)
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
    record("supersampled", ptr, pte, o_tr, o_te)
    del o_tr, o_te, ptr, pte, d, Ys_tr, Ys_te
    torch.cuda.empty_cache()

    def scatter_geom(x):
        out = np.full(len(keep), np.nan)
        out[geom] = np.degrees(x.cpu().numpy())
        return out
    pts["measured_obs_aniso_rms_deg"] = scatter_geom(aniso_rms)
    for l in EQUIV_L:
        pts[f"equiv_l{l}_deg"] = scatter_geom(equiv[l])

    pts["fitted_geom_deg"], pts["fitted_geom_ident"] = np.full(len(keep), np.nan), np.zeros(len(keep), bool)
    pts["fitted_geom_deg"][geom], pts["fitted_geom_ident"][geom] = f_deg, f_id

    # 全区画の点で fitted を解き直す（brdf の区画の最適なぼけ）
    a_deg, a_id, tr, te, by_sec = fitted_run(keep, "全区画")
    res["fitted_all"] = {"train": tr, "test": te, "test_by_sector": by_sec}
    pts["fitted_all_deg"], pts["fitted_all_ident"] = np.full(len(keep), np.nan), np.zeros(len(keep), bool)
    pts["fitted_all_deg"][keep], pts["fitted_all_ident"][keep] = a_deg, a_id
    return res, pts


def summarize(res, pts, sec_all):
    """区画ごとの中央値と平均（mean。σ は格子の値しか取らず中央値が飛ぶので、図と表は平均を使う）と、順位相関"""
    keep = pts["keep"]
    rows = {}
    for k, (name, sig, mat) in enumerate(ms.SECTORS):
        m = keep & (sec_all == k)
        r = {"n_points": int(m.sum()), "design_blur_deg": 2 * (sig if mat == "au" else ms.BRDF_SIGMA[mat]),
             "measured_pt_deg": float(np.median(pts["measured_pt_deg"][m])),
             "measured_obs_rms_deg": float(np.median(pts["measured_obs_rms_deg"][m]))}
        r["mean"] = {"measured_pt_deg": float(np.mean(pts["measured_pt_deg"][m])),
                     "measured_obs_rms_deg": float(np.mean(pts["measured_obs_rms_deg"][m]))}
        for key in ["measured_obs_aniso_rms_deg"] + [f"equiv_l{l}_deg" for l in EQUIV_L]:
            if key in pts and np.isfinite(pts[key][m]).any():
                r[key] = float(np.nanmedian(pts[key][m]))
                r["mean"][key] = float(np.nanmean(pts[key][m]))
        for key in ("fitted_geom", "fitted_all"):
            if f"{key}_deg" not in pts:
                continue
            mi = m & pts[f"{key}_ident"]
            if mi.sum() == 0:
                continue
            q = np.percentile(pts[f"{key}_deg"][mi], [25, 50, 75])
            r[key] = {"n_identifiable": int(mi.sum()), "median_deg": float(q[1]), "q25_deg": float(q[0]),
                      "q75_deg": float(q[2]), "mean_deg": float(np.mean(pts[f"{key}_deg"][mi]))}
        rows[name] = r
    res["sectors"] = rows
    if "fitted_geom_deg" in pts:
        m = keep & np.isin(sec_all, GEOM) & pts["fitted_geom_ident"]
        bump = m & np.isin(sec_all, [k for k, (_, s, _) in enumerate(ms.SECTORS) if s > 0])
        res["spearman"] = {
            "geom_measured_pt_vs_fitted": float(spearmanr(pts["measured_pt_deg"][m], pts["fitted_geom_deg"][m])[0]),
            "geom_measured_obs_rms_vs_fitted": float(spearmanr(pts["measured_obs_rms_deg"][m],
                                                               pts["fitted_geom_deg"][m])[0])}
        for k in GEOM:  # 区画の中だけでの順位相関（区画の違いを除いた、点ごとのばらつきの効き）
            mk = m & (sec_all == k)
            if mk.sum() > 30 and ms.SECTORS[k][1] > 0:
                res["spearman"][f"within_{ms.SECTORS[k][0]}"] = float(
                    spearmanr(pts["measured_pt_deg"][mk], pts["fitted_geom_deg"][mk])[0])
        res["spearman"]["n_bump_identifiable"] = int(bump.sum())


# ---- 図 ----

SECTOR_LABEL = {"smooth_a": "なめらか A", "bump3": "凹凸 3°", "bump6": "凹凸 6°", "bump12": "凹凸 12°",
                "smooth_b": "なめらか B", "brdf3": "GGX 3°", "brdf6": "GGX 6°", "brdf12": "GGX 12°"}
C_CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # 図のカテゴリ色（この順で割り当てる）
SCALE_STYLE = {"512": ("o", "#a3a29e"), "256": ("^", "#6f6e6a"), "128": ("s", "#0b0b0b")}  # 画素の大きさ（形と濃さ）
VIEW_IDS = ["view_1_003", "view_1_009", "view_1_015", "view_1_021"]  # 赤道の視点（経度 49, 139, 229, 319°）


def _axes_style(ax):
    ax.set_facecolor(so.C_SURF)
    ax.grid(color=so.C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(out, path):
    """(a)〜(c) 凹凸の区画ごとの、ぼけの幅と画素の大きさ、(d) テスト視点の PSNR、(e) GGX の区画の最適なぼけ"""
    plt = er.setup_plt()
    sc = [k for k in ("512", "256", "128") if k in out["scales"]]
    R = {k: out["scales"][k] for k in sc}
    width = np.array([R[k]["pixel_width_mm_median"] for k in sc])
    fig = plt.figure(figsize=(11.6, 8.4), dpi=200, facecolor=so.C_SURF)
    gs_ = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.08], hspace=0.42, wspace=1.1)
    series = [("fitted", "点ごとに最適化（帯は 25〜75%）", C_CAT[0], "o", 2.2),
              ("measured_pt_deg", "換算: 点の足跡（像の解析と同じ）", C_CAT[1], "s", 1.4),
              ("measured_obs_rms_deg", "換算: 観測ごとの足跡（斜めから見ると伸びる）", C_CAT[2], "^", 1.4),
              ("measured_obs_aniso_rms_deg", "換算: 観測ごと × 反射の異方性の補正", C_CAT[3], "D", 1.4)]
    for j, name in enumerate(["bump3", "bump6", "bump12"]):
        ax = fig.add_subplot(gs_[0, 2 * j:2 * j + 2])
        _axes_style(ax)
        rows = [R[k]["sectors"][name] for k in sc]
        design = rows[0]["design_blur_deg"]
        ax.axhline(design, color=so.C_TEXT2, lw=1.0, ls=(0, (4, 3)),
                   label="換算: 十分大きい足跡（2 × 設計の傾き）" if j == 0 else None)
        for key, lab, col, mk, lw in series:
            if key == "fitted":
                f = [r.get("fitted_geom") for r in rows]
                y = np.array([x["mean_deg"] for x in f])
                ax.fill_between(width, [x["q25_deg"] for x in f], [x["q75_deg"] for x in f], color=col, alpha=0.15, lw=0)
            else:
                y = np.array([r["mean"].get(key, np.nan) for r in rows])
            ax.plot(width, y, color=col, lw=lw, marker=mk, ms=5, mec=so.C_SURF, mew=1.0, label=lab if j == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(width)
        ax.set_xticklabels([f"{w:.2f}\n({k} px)" for w, k in zip(width, sc)], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlim(width.min() / 1.25, width.max() * 1.25)
        ax.set_ylim(0, design * 1.12)
        ax.set_xlabel("1 画素の幅 [mm]（点の所、中央値）", fontsize=8)
        if j == 0:
            ax.set_ylabel("反射方向でのぼけの幅 σ [度]（区画の平均）")
        ax.set_title(f"({'abc'[j]}) {SECTOR_LABEL[name]}（1 軸あたりの傾きの標準偏差）", fontsize=9, loc="left")
    fig.legend(loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=7.5, frameon=False)

    # (d) テスト視点の PSNR
    ax = fig.add_subplot(gs_[1, :4])
    _axes_style(ax)
    models = [("material", "ぼけなし（σ = 0）"), ("measured_pt", "換算: 点の足跡"),
              ("measured_obs", "換算: 観測ごとの足跡"), ("measured_obs_aniso", "換算: 観測ごと × 異方性の補正"),
              ("scan", "換算: 観測ごと × 最良の倍率 k"), ("fitted", "点ごとに最適化"),
              ("supersampled", "足跡の中で反射を平均（ぼけの近似なし）")]
    ys = np.arange(len(models))
    for k in sc:
        mk, col = SCALE_STYLE[k]
        xs, ks = [], None
        for key, _ in models:
            if key == "scan":
                best = max(R[k]["scan_measured_obs"].items(), key=lambda kv: kv[1]["train"])
                xs.append(best[1]["test"])
                ks = best[0]
            else:
                xs.append(R[k]["models"][key]["test"])
        ax.scatter(xs, ys, marker=mk, s=34, color=col, edgecolor=so.C_SURF, lw=0.8, zorder=3,
                   label=f"{k} px（最良の k = {ks}）")
    ax.set_yticks(ys)
    ax.set_yticklabels([m[1] for m in models], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("テスト視点の観測色の PSNR [dB]（なめらかと凹凸の 5 区画の点）", fontsize=8)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, title="画素の大きさ", title_fontsize=7.5)
    ax.set_title("(d) 照明を共有するモデルの再現度（どれも共有の照明 SH 48 次 + 点ごとの定数）", fontsize=9, loc="left")

    # (e) GGX の区画
    ax = fig.add_subplot(gs_[1, 4:])
    _axes_style(ax)
    top = 34
    ax.plot([0, top], [0, top], color=so.C_TEXT2, lw=1.0, ls=(0, (4, 3)), label="y = x")
    groups = [(["brdf3", "brdf6", "brdf12"], "fitted_all", C_CAT[4], "GGX の区画（形はなめらか）"),
              (["bump3", "bump6", "bump12"], "fitted_geom", C_CAT[0], "凹凸の区画")]
    for names, key, col, lab in groups:
        for i, k in enumerate(sc):
            mk, _ = SCALE_STYLE[k]
            x = np.array([R[k]["sectors"][n]["design_blur_deg"] for n in names]) + (i - 1) * 0.7
            f = [R[k]["sectors"][n][key] for n in names]
            y = np.array([v["mean_deg"] for v in f])
            err = np.array([[max(v["mean_deg"] - v["q25_deg"], 0) for v in f], [max(v["q75_deg"] - v["mean_deg"], 0) for v in f]])
            ax.errorbar(x, y, yerr=err, fmt=mk, ms=5, color=col, mec=so.C_SURF, mew=0.8, elinewidth=1.0, capsize=0,
                        label=f"{lab}" if i == 0 else None)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xticks([0, 6, 12, 24])
    ax.set_xlabel("設計のぼけ [度]（GGX は半値幅を合わせたガウス）", fontsize=8)
    ax.set_ylabel("点ごとに最適なぼけ σ [度]（平均、線は 25〜75%）", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)
    ax.text(0.98, 0.03, "形は画素の大きさ（○ 512 / △ 256 / □ 128 px）", transform=ax.transAxes, ha="right",
            fontsize=7, color=so.C_TEXT2)
    ax.set_title("(e) 材質の粗さに当たるぼけ", fontsize=9, loc="left")
    fig.suptitle("凹凸の球: 点ごとに最適化したぼけと、画素の中の法線のばらつきからの換算", fontsize=10.5, x=0.01,
                 y=0.995, ha="left")
    fig.savefig(path, facecolor=so.C_SURF, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _meridian(cam, frm, lon, lats_deg):
    """経度 lon の子午線上の点を投影 -> u, v, カメラに向いているか（向きの余弦 > 0.3）"""
    lat = np.radians(np.asarray(lats_deg, dtype=float))
    q = np.stack([np.cos(lat) * np.sin(np.radians(lon)), np.sin(lat), np.cos(lat) * np.cos(np.radians(lon))], -1)
    X = torch.tensor(ms.R0 * q, dtype=torch.float32, device=DEV)
    o = torch.tensor(frm, dtype=torch.float32, device=DEV)
    cosv = (F.normalize(o - X, dim=-1) * F.normalize(X, dim=-1)).sum(-1)
    u, v, _ = cam.project(X)
    return u.cpu().numpy(), v.cpu().numpy(), cosv.cpu().numpy()


def plot_views(path):
    """赤道の 4 視点（512 px の中央、区画の名前と境目つき）と、1 視点の中央を 3 つの画素の大きさで"""
    plt = er.setup_plt()
    frm = {n: f for s, n, f in ms.views() if s == "train"}
    lo, hi = 96, 416                       # 上段の切り出し（球が収まる範囲）
    name2, c, h = VIEW_IDS[1], 256, 96     # 下段: この視点の中央 2h 画素四方
    fig = plt.figure(figsize=(11.6, 7.7), dpi=200, facecolor=so.C_SURF)
    gs_ = fig.add_gridspec(2, 12, height_ratios=[2.9, 3.87], hspace=0.1, wspace=0.12, left=0.01, right=0.99,
                           top=0.95, bottom=0.01)
    rad = {}
    for j, name in enumerate(VIEW_IDS):
        rad[name] = torch.from_numpy(read_pfm(ms.DATA / "render" / "train" / f"{name}.pfm"))
        ax = fig.add_subplot(gs_[0, 3 * j:3 * j + 3])
        ax.imshow(er.radiance_to_display(rad[name][lo:hi, lo:hi]).numpy().clip(0, 1),
                  extent=(lo, hi, hi, lo))
        cam = Cam(frm[name], ms.SIZE)
        for k, (sname, _, _) in enumerate(ms.SECTORS):
            u, v, cv = _meridian(cam, frm[name], ms.SECTOR_DEG * k, np.linspace(-62, 62, 60))
            ax.plot(np.where(cv > 0.3, u, np.nan), np.where(cv > 0.3, v, np.nan), color="#fcfcfb", lw=0.8,
                    ls=(0, (3, 3)), alpha=0.85)
            u, v, cv = _meridian(cam, frm[name], ms.SECTOR_DEG * (k + 0.5), [0.0])
            if cv[0] > 0.6:  # 区画の中央が正面に近いものだけ名前を出す（1 視点に 2 つ）
                ax.text(u[0], lo + 6, SECTOR_LABEL[sname], ha="center", va="top", fontsize=8, color=so.C_TEXT,
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9))
        ax.set_xlim(lo, hi)
        ax.set_ylim(hi, lo)
        ax.set_axis_off()
    for j, k in enumerate(ms.SCALES):
        f = ms.SIZE // k
        x = F.avg_pool2d(rad[name2].permute(2, 0, 1)[None], f)[0].permute(1, 2, 0)
        a0, a1 = (c - h) // f, (c + h) // f
        ax = fig.add_subplot(gs_[1, 4 * j:4 * j + 4])
        ax.imshow(er.radiance_to_display(x[a0:a1, a0:a1]).numpy().clip(0, 1), interpolation="nearest",
                  extent=(c - h, c + h, c + h, c - h))  # 512 px の画素座標で描く（境目の線と合わせるため）
        cam = Cam(frm[name2], ms.SIZE)
        u, v, cv = _meridian(cam, frm[name2], 135.0, np.linspace(-62, 62, 200))
        ax.plot(u, v, color="#fcfcfb", lw=1.0, ls=(0, (3, 3)), alpha=0.9)
        ax.set_xlim(c - h, c + h)
        ax.set_ylim(c + h, c - h)
        ax.set_axis_off()
        ax.set_title(f"{k} px（1 画素 ≈ {1.36 * f:.2f} mm）", fontsize=8, loc="left")
        if j == 0:
            for dx, lab in ((-50, "凹凸 6°"), (50, "凹凸 12°")):
                ax.text(c + dx, c + h - 6, lab, ha="center", va="bottom", fontsize=8, color=so.C_TEXT,
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9))
    lon2 = ms.lon_lat(np.array(frm[name2]))[0]
    fig.suptitle(f"凹凸の球（半径 {ms.R0 * 100:.0f} cm、凹凸の波長 {ms.LAMBDA * 1000:.0f} mm、点線は区画の境目）: "
                 f"上は赤道の 4 視点、下は経度 {lon2:.0f}° の視点の中央を画素の大きさを変えて",
                 fontsize=9.5, x=0.01, y=0.995, ha="left")
    fig.savefig(path, facecolor=so.C_SURF, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=ms.SCALES)
    ap.add_argument("--no-denoised-check", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="sphere_rough.json から図だけ描き直す")
    a = ap.parse_args()
    fig_dir = HERE / "figures"
    if a.plot_only:  # 点ごとの値（npz）から区画の集計を取り直して、図を描く
        out = json.loads((HERE / "sphere_rough.json").read_text(encoding="utf-8"))
        z = np.load(OUT_NPZ)
        sec_all = z["sector"]
        for prefix, res in [(f"s{k}_", v) for k, v in out["scales"].items()] + [("dn256_", {})]:
            pts = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
            if not pts:
                continue
            summarize(res, pts, sec_all)
            if prefix == "dn256_":
                out["denoised_256"] = {k: {"fitted_geom": v.get("fitted_geom")} for k, v in res["sectors"].items()
                                       if ms.SECTORS[[x[0] for x in ms.SECTORS].index(k)][2] == "au"}
        (HERE / "sphere_rough.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        plot_results(out, fig_dir / "sphere_rough.png")
        plot_views(fig_dir / "sphere_views.png")
        return
    t0 = time.time()
    grid = torch.tensor(np.radians(er.SIGMA_GRID_DEG), dtype=torch.float32, device=DEV)
    n0_np, sec_all = sample_points()
    amp = ms.amplitude(ms.R0 * n0_np)
    n0 = torch.tensor(n0_np, dtype=torch.float32, device=DEV)
    a_t = torch.tensor(amp, dtype=torch.float32, device=DEV)
    offs = footprint_offsets()
    out = {"settings": {"n_per_sector": N_PER_SECTOR, "lat_max_deg": LAT_MAX, "border_margin_deg": BORDER_MARGIN,
                        "footprint_samples": FP_SAMPLES, "super_samples": SUPER_SAMPLES, "local_degree": LP, "env_smooth": S, "env_degree": er.LE,
                        "identifiable_contrast": er.ID_CONTRAST, "sectors": ms.SECTORS,
                        "materials": ms.MATERIALS}, "scales": {}}
    arrays = {"n0": n0_np, "sector": sec_all}
    for scale in a.scales:
        res, pts = run_scale(scale, n0, sec_all, a_t, offs, grid, t0)
        summarize(res, pts, sec_all)
        out["scales"][str(scale)] = res
        arrays.update({f"s{scale}_{k}": v for k, v in pts.items()})
        for name, r in res["sectors"].items():
            fg = r.get("fitted_geom", {}).get("median_deg", float("nan"))
            fa = r.get("fitted_all", {}).get("median_deg", float("nan"))
            ex = "".join(f"  {lab} {r[k]:5.1f}°" for lab, k in
                         [("異方性補正", "measured_obs_aniso_rms_deg"), ("等価 l4", "equiv_l4_deg"),
                          ("等価 l12", "equiv_l12_deg")] if k in r)
            print(f"    {name:<9} 設計 {r['design_blur_deg']:4.0f}°  測った（点）{r['measured_pt_deg']:5.1f}°  "
                  f"測った（観測）{r['measured_obs_rms_deg']:5.1f}°{ex}  最適化 {fg:5.1f}° / 全区画で {fa:5.1f}°")
        print(f"    順位相関: {res.get('spearman')}", flush=True)
        (HERE / "sphere_rough.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(OUT_NPZ, **arrays)
    if not a.no_denoised_check and 256 in a.scales:  # 像と同じくデノイズ後の画像を使うと、最適なぼけは変わるか
        res, pts = run_scale(256, n0, sec_all, a_t, offs, grid, t0, denoised=True, quick=True)
        summarize(res, pts, sec_all)
        out["denoised_256"] = {k: {"fitted_geom": v.get("fitted_geom")} for k, v in res["sectors"].items()
                               if ms.SECTORS[[s[0] for s in ms.SECTORS].index(k)][2] == "au"}
        arrays.update({f"dn256_{k}": v for k, v in pts.items()})
        for name, r in out["denoised_256"].items():
            print(f"    デノイズ後 {name:<9} 最適化 {r['fitted_geom']['median_deg'] if r['fitted_geom'] else float('nan'):5.1f}°")
    (HERE / "sphere_rough.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT_NPZ, **arrays)
    if len(out["scales"]) == len(ms.SCALES):
        plot_results(out, fig_dir / "sphere_rough.png")
        plot_views(fig_dir / "sphere_views.png")
    print(f"-> {HERE / 'sphere_rough.json'}, {OUT_NPZ}、figures/sphere_*.png（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
