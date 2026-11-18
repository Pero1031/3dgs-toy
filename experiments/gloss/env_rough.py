# -*- coding: utf-8 -*-
"""
画素の中の法線のばらつきから決めた「実効的な粗さ」は、点ごとに最適化した粗さと合うか（3DGS を使わない検証）。

env_oracle.py の共有の照明に、点ごとのぼかしを足す:
    y(p, v) ≈ Σ_k c_{p,k} Y_k(d)  +  Σ_j a_{l(j)}(σ_p) e_j Y_j(r)
    a_l(σ) = exp(−l(l+1) σ² / 2)
球面上を幅 σ（1 軸あたりの標準偏差、ラジアン）のガウスでぼかすことは、SH では次数 l の係数に a_l を掛けることに当たる
（球面上の熱拡散。Ref-NeRF の IDE の減衰と同じ形）。σ_p を決めれば予測は e について線形のままなので、
env_oracle.py と同じく点ごとの項を消去して e を解ける。

σ_p の決め方を 3 通り比べる:
    material  σ_p = 0。材質の粗さ（GGX α = 0.01、反射方向で約 1°）は照明の SH の分解能（約 4°）より細かいので 0 とみなす。
              全点に共通のぼけは照明 e 自体がぼけることで吸収できるので、「全点で同じ粗さ」のモデルでもある
    measured  真のメッシュで、点のまわり半径 R にある表面の法線（レンダラーと同じ、頂点法線を補間したもの）の
              平均ベクトル n̄ を測り、σ_p = 2√(1 − |n̄|) とする。法線が 1 軸あたり標準偏差 σ_n でばらつくと
              |n̄| ≈ 1 − σ_n²（Toksvig の手がかり）、反射方向は法線の 2 倍ずれる。自由なパラメータはない
    fitted    点ごとに σ_p を格子から選ぶ（学習視点の誤差が最小のもの）。照明 e と交互に解く。既存手法が学習する粗さの代わり

値の空間: レンダラーの PNG は「放射輝度 x → Reinhard x/(1+x) → sRGB の曲線」。画素の中での平均（ぼけ）は放射輝度で
起きるので、モデルは放射輝度で組み、誤差は表示値で測る（評価は前回までと同じ表示値の PSNR）。表示値への変換は非線形
なので、予測のまわりで一次近似して重み付き最小二乗を解き、近似を取り直すことを繰り返す（ガウス・ニュートン法）。
重みを観測値の所で固定すると、明るい映り込みの重みが背景の数千分の一になり、外しても無視されてしまう。

相互反射は扱わないので、env_diagnose.py の判定で像自身に遮られる観測は除く。

    python experiments/gloss/env_rough.py

出力: experiments/gloss/env_rough.json, figures/env_rough.png
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env_diagnose as ed  # noqa: E402
import env_oracle as eo  # noqa: E402
import sh_oracle as so  # noqa: E402
from gs.sh import sh_basis  # noqa: E402

DEV = so.DEV
F64 = torch.float64
LE = 48                                  # 共有の照明の SH の次数（分解能はおよそ 180°/48 ≈ 4°）
ENV_SMOOTH = [0.03, 0.1, 0.3]            # 共有の照明のなめらかさの罰（最初の実行で 0.1 が最良、10 は常に悪かった）
LOCAL_DEGREES = [0, 2]                   # 点ごとの項の次数（0 = 点ごとの定数）
SIGMA_GRID_DEG = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40]  # 反射方向でのぼけの幅の候補
RADII_MM = [0.68, 1.36, 2.72, 4.0]       # 法線のばらつきを測る半径
R_NOMINAL_MM = 2.72                      # 1 画素の幅（距離 1.3 m）。画素の箱型と、観測を拾う双線形補間（±1 画素の三角形）を
                                         # 合わせた広がりは 1 軸あたり標準偏差 1.36 mm で、半径 2.72 mm の一様な円板と同じ
SH_REF_DEGREES = [2, 3, 4, 6]            # 参考: 点ごとの SH（sh_oracle.py）
N_ITER = 10                              # fitted の交互の更新の回数
GN_ITERS = 6                             # ガウス・ニュートン法の反復の回数（σ を固定したモデル）
GN_FINAL = 3                             # fitted で σ を決めたあとの、ガウス・ニュートン法の反復の回数
X_MAX = 50.0                             # 一次近似する位置の上限（放射輝度。表示値で約 0.99）
ID_CONTRAST = 0.02                       # σ を変えると学習視点の誤差（表示値の RMSE）がこれ以上変わる点を「粗さが決まる点」とする
OBS_CHUNK = 4096
KE = (LE + 1) ** 2
L_OF = torch.tensor([math.isqrt(k) for k in range(KE)], dtype=torch.float32, device=DEV)


# ---- 表示値と放射輝度の変換（GoLD_rayt の PNG: Reinhard → sRGB の曲線、露出 0） ----

def srgb_to_lin(y):
    return torch.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(t):
    t = t.clamp(min=0)
    return torch.where(t <= 0.0031308, 12.92 * t, 1.055 * t.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def to_radiance(y):
    t = srgb_to_lin(y).clamp(max=0.995)
    return t / (1 - t)


def radiance_to_display(x):
    x = x.clamp(min=0)
    return lin_to_srgb(x / (1 + x)).clamp(0, 1)


def display_slope(x):
    """表示値 D(x) = srgb(x / (1 + x)) を放射輝度 x で微分した値"""
    t = (x / (1 + x)).clamp(min=1e-6)
    d = torch.where(t <= 0.0031308, torch.full_like(t, 12.92), 1.055 / 2.4 * t ** (1 / 2.4 - 1))
    return d * (1 - t) ** 2  # dt/dx = 1 / (1 + x)² = (1 − t)²


# ---- 観測（見えている (点, 視点) の組を 1 列に並べたもの） ----

class Obs:
    def __init__(self, vis, col, dirs, Ns, space):
        v_idx, p_idx = vis.nonzero(as_tuple=True)
        self.pid, self.n_pts, self.space = p_idx, vis.shape[1], space
        self.ydisp = col[v_idx, p_idx].double()
        d = dirs[v_idx, p_idx]
        n = Ns[v_idx, p_idx] if Ns.dim() == 3 else Ns[p_idx]  # 法線は点ごと (n_pts, 3) か観測ごと (V, n_pts, 3)
        self.d = d
        r = d - 2 * (d * n).sum(-1, keepdim=True) * n  # 視線を法線で折り返した方向（env_oracle.reflect と同じ）
        self.Y = torch.cat([sh_basis(LE, r[i:i + OBS_CHUNK]).float() for i in range(0, len(r), OBS_CHUNK)])
        self.Bl = {}
        self.linearize(None)

    def linearize(self, pred):
        """表示値の誤差を、予測 pred（放射輝度）のまわりで一次近似する（ガウス・ニュートン法）。
        D(x) ≈ D(p) + D'(p)(x − p) なので |D(x) − y|² ≈ D'(p)² |x − z|²、z = p + (y − D(p)) / D'(p)。
        この重み D'(p)² と目標 z で重み付き最小二乗を解き、予測を更新して近似し直す。pred=None は観測値の所で近似する。
        重みはチャンネルごと (m, 3)"""
        if self.space == "display":
            self.y, self.w = self.ydisp, torch.ones_like(self.ydisp)
            return
        p = (to_radiance(self.ydisp) if pred is None else pred).clamp(0, X_MAX)
        g = display_slope(p)
        self.w = g ** 2
        self.y = p + (self.ydisp - radiance_to_display(p)) / g

    def local(self, L):
        if L not in self.Bl:
            self.Bl[L] = sh_basis(L, self.d).double()
        return self.Bl[L]

    def display(self, pred):
        return radiance_to_display(pred) if self.space == "radiance" else pred.clamp(0, 1)


def attenuation(sig, Le=LE):
    """反射方向で幅 sig（ラジアン）のガウスでぼかすときの、SH の次数ごとの減衰 (…, Ke)"""
    l = L_OF[:(Le + 1) ** 2]
    return torch.exp(-l * (l + 1) * sig.float()[..., None] ** 2 / 2)


def att_rows(o, sig, Le=LE):
    """観測の行の範囲 sl を受けて、その行のぼけの減衰 (len(sl), Ke) を返す関数。
    sig は点ごとの σ (n_pts,)、または文字列 "obs"（観測ごとの σ を o.sig_obs (n_obs,) に置いたもの）"""
    if isinstance(sig, str):
        assert sig == "obs"
        s_obs = o.sig_obs
        return lambda sl: attenuation(s_obs[sl], Le)
    a_pt = attenuation(sig, Le)
    return lambda sl: a_pt[o.pid[sl]]


def local_chol(o, L):
    """点ごとの項の正規方程式 A_p = Σ w Bl Blᵀ + R のコレスキー因子（チャンネルごと） (3, n_pts, K, K)"""
    Bl = o.local(L)
    K = Bl.shape[1]
    A = torch.zeros(3, o.n_pts, K, K, dtype=F64, device=DEV)
    for c in range(3):
        A[c].index_add_(0, o.pid, o.w[:, c, None, None] * Bl[:, :, None] * Bl[:, None, :])
    return torch.linalg.cholesky(A + eo.penalty(L, eo.S_LOC))


def fit_env(o, sig_pt, Lp, smooths, Le=LE):
    """点ごとのぼけ sig_pt (n_pts,)（または "obs"、att_rows を参照）を固定して、共有の照明 e を解く
    （点ごとの項は Schur 補元で消去） -> {s: e (Ke, 3)}"""
    Ke = (Le + 1) ** 2
    att = att_rows(o, sig_pt, Le)
    Bl = o.local(Lp)
    K = Bl.shape[1]
    Lc = local_chol(o, Lp)
    out = {s: torch.zeros(Ke, 3, dtype=F64, device=DEV) for s in smooths}
    for c in range(3):  # 重みがチャンネルごとに違うので、チャンネルごとに解く
        w, y = o.w[:, c], o.y[:, c]
        S = torch.zeros(Ke, Ke, dtype=F64, device=DEV)
        rhs = torch.zeros(Ke, dtype=F64, device=DEV)
        M = torch.zeros(o.n_pts, K, Ke, device=DEV)  # Σ w Bl Beᵀ（点ごと）
        for i in range(0, len(o.pid), OBS_CHUNK):
            sl = slice(i, i + OBS_CHUNK)
            Be = o.Y[sl, :Ke] * att(sl)
            Bw = Be * w[sl, None].float()
            S += (Bw.T @ Be).double()
            rhs += Bw.T.double() @ y[sl]
            M.index_add_(0, o.pid[sl], (w[sl, None] * Bl[sl]).float()[:, :, None] * Be[:, None, :])
        b = torch.zeros(o.n_pts, K, dtype=F64, device=DEV).index_add_(0, o.pid, w[:, None] * Bl * y[:, None])
        for i in range(0, o.n_pts, 512):
            sl = slice(i, i + 512)
            G = torch.linalg.solve_triangular(Lc[c, sl], M[sl].double(), upper=False).reshape(-1, Ke)
            g = torch.linalg.solve_triangular(Lc[c, sl], b[sl, :, None], upper=False).reshape(-1)
            S -= (G.T.float() @ G.float()).double()
            rhs -= G.T @ g
        for s in smooths:
            out[s][:, c] = torch.linalg.solve(S + eo.penalty(Le, s), rhs)
    return out


def env_pred(o, e, sig_pt, Le=LE):
    Ke = (Le + 1) ** 2
    att = att_rows(o, sig_pt, Le)
    out = torch.empty(len(o.pid), 3, dtype=F64, device=DEV)
    for i in range(0, len(o.pid), OBS_CHUNK):
        sl = slice(i, i + OBS_CHUNK)
        out[sl] = ((o.Y[sl, :Ke] * att(sl)) @ e.float()).double()
    return out


def predict(o_tr, o_te, e, sig_pt, Lp, Le=LE):
    """照明 e とぼけ σ を固定し、点ごとの係数を学習視点で解いて、学習・テストの予測（当てはめの空間）を返す。
    e=None なら点ごとの項だけ（点ごとの SH）"""
    if e is None:
        ptr = torch.zeros(len(o_tr.pid), 3, dtype=F64, device=DEV)
        pte = torch.zeros(len(o_te.pid), 3, dtype=F64, device=DEV)
    else:
        ptr, pte = env_pred(o_tr, e, sig_pt, Le), env_pred(o_te, e, sig_pt, Le)
    Bl = o_tr.local(Lp)
    Lc = local_chol(o_tr, Lp)
    coef = torch.zeros(o_tr.n_pts, Bl.shape[1], 3, dtype=F64, device=DEV)
    for c in range(3):
        b = torch.zeros(o_tr.n_pts, Bl.shape[1], dtype=F64, device=DEV).index_add_(
            0, o_tr.pid, o_tr.w[:, c, None] * Bl * (o_tr.y[:, c] - ptr[:, c])[:, None])
        coef[:, :, c] = torch.cholesky_solve(b[..., None], Lc[c])[..., 0]
    ptr = ptr + torch.einsum("mk,mkc->mc", Bl, coef[o_tr.pid])
    pte = pte + torch.einsum("mk,mkc->mc", o_te.local(Lp), coef[o_te.pid])
    return ptr, pte


def mse(o, pred):
    return ((o.display(pred) - o.ydisp) ** 2).mean().item()


def psnr(o, pred):
    return -10 * math.log10(mse(o, pred))


def search_sigma(o, e, Lp, grid):
    """照明 e を固定し、点ごとに格子の各 σ で点ごとの項を解き直したときの、学習視点の表示値の平均二乗誤差 (n_pts, S)"""
    nS = len(grid)
    A = attenuation(grid)                                                        # (S, Ke)
    E_all = (A[:, :, None] * e.float()[None]).permute(1, 0, 2).reshape(KE, -1)  # (Ke, S·3)
    pred = torch.empty(len(o.pid), nS, 3, dtype=F64, device=DEV)
    for i in range(0, len(o.pid), OBS_CHUNK):
        sl = slice(i, i + OBS_CHUNK)
        pred[sl] = (o.Y[sl] @ E_all).double().reshape(-1, nS, 3)
    Bl = o.local(Lp)
    K = Bl.shape[1]
    Lc = local_chol(o, Lp)
    err = torch.zeros(len(o.pid), nS, dtype=F64, device=DEV)
    for c in range(3):
        t = o.y[:, c, None] - pred[:, :, c]                                       # (m, S)
        b = torch.zeros(o.n_pts, K, nS, dtype=F64, device=DEV).index_add_(
            0, o.pid, (o.w[:, c, None] * Bl)[:, :, None] * t[:, None, :])
        coef = torch.cholesky_solve(b, Lc[c])                                    # (n_pts, K, S)
        full = pred[:, :, c] + torch.einsum("mk,mks->ms", Bl, coef[o.pid])
        err += (o.display(full) - o.ydisp[:, c, None]) ** 2 / 3
    loss = torch.zeros(o.n_pts, nS, dtype=F64, device=DEV).index_add_(0, o.pid, err)
    cnt = torch.zeros(o.n_pts, dtype=F64, device=DEV).index_add_(0, o.pid, torch.ones_like(o.pid, dtype=F64))
    return loss / cnt[:, None]


def run_fixed(o_tr, o_te, sig, Lp, s, iters=GN_ITERS):
    """σ を固定して、ガウス・ニュートン法で表示値の誤差を最小にする。sig=None は点ごとの項だけ（照明なし）"""
    o_tr.linearize(None)
    hist = []
    e = None
    for _ in range(iters if o_tr.space == "radiance" else 1):
        e = fit_env(o_tr, sig, Lp, [s])[s] if sig is not None else None
        ptr, pte = predict(o_tr, o_te, e, sig, Lp)
        o_tr.linearize(ptr)
        hist.append(mse(o_tr, ptr))
    return e, ptr, pte, hist


def run_fitted(o_tr, o_te, Lp, s, grid):
    """点ごとの σ と照明 e を交互に解く（σ = 0 から始める）。各回でガウス・ニュートン法の近似も取り直す"""
    o_tr.linearize(None)
    sig = torch.zeros(o_tr.n_pts, device=DEV)
    hist = []
    for _ in range(N_ITER):
        e = fit_env(o_tr, sig, Lp, [s])[s]
        new = grid[search_sigma(o_tr, e, Lp, grid).argmin(1)]
        changed = float((new != sig).float().mean())
        sig = new
        ptr, pte = predict(o_tr, o_te, e, sig, Lp)
        o_tr.linearize(ptr)
        hist.append({"changed": changed, "train_mse": mse(o_tr, ptr)})
    for _ in range(GN_FINAL):
        e = fit_env(o_tr, sig, Lp, [s])[s]
        ptr, pte = predict(o_tr, o_te, e, sig, Lp)
        o_tr.linearize(ptr)
    return sig, e, ptr, pte, search_sigma(o_tr, e, Lp, grid), hist


# ---- 法線のばらつき ----

def mean_normal_length(P, Nc, D, ND, radii_m):
    """各点のまわり半径 R にある表面の法線の平均ベクトルの長さ |n̄|（R ごと）。点の法線と反対を向く面（薄い部分の裏側）は除く"""
    tree = cKDTree(D)
    out = {}
    for R in radii_m:
        ln = np.ones(len(P))
        for i, nb in enumerate(tree.query_ball_point(P, R)):
            nb = np.asarray(nb, dtype=int)
            if len(nb):
                nb = nb[ND[nb] @ Nc[i] > 0]
            if len(nb):
                ln[i] = np.linalg.norm(ND[nb].mean(0))
        out[R] = ln
    return out


def sigma_from_length(ln):
    """|n̄| から反射方向のぼけの幅 σ_r = 2 σ_n、σ_n = √(1 − |n̄|)（ラジアン、格子の最大値で頭打ち）"""
    return np.minimum(2 * np.sqrt(np.clip(1 - ln, 0, None)), math.radians(SIGMA_GRID_DEG[-1]))


# ---- 図 ----

def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9,
                         "axes.edgecolor": so.C_TEXT2, "axes.labelcolor": so.C_TEXT, "xtick.color": so.C_TEXT2,
                         "ytick.color": so.C_TEXT2, "text.color": so.C_TEXT})
    return plt


def scatter_panel(ax, meas_deg, fit_deg, title, ylabel="点ごとに最適なぼけの幅 [度]"):
    """横軸 = 測ったばらつきから換算したぼけ、縦軸 = 当てはめたぼけ。点と、横軸の 8 分位ごとの中央値（帯は 25〜75%）"""
    top = SIGMA_GRID_DEG[-1] + 2
    ax.set_facecolor(so.C_SURF)
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.35, 0.35, len(fit_deg))  # 縦は格子の値しか取らないので、重なりが見えるように少しずらす
    ax.scatter(meas_deg, fit_deg + jit, s=5, color=so.C_SERIES["gold"], alpha=0.25, lw=0, label="点（粗さが観測から決まる点）")
    ax.plot([0, top], [0, top], color=so.C_TEXT2, lw=1.0, ls=(0, (4, 3)), label="y = x（換算どおり）")
    edges = np.quantile(meas_deg, np.linspace(0, 1, 9))
    xc, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (meas_deg >= a) & (meas_deg <= b)
        if m.sum() < 10:
            continue
        xc.append(np.median(meas_deg[m]))
        q = np.percentile(fit_deg[m], [25, 50, 75])
        q1.append(q[0]), med.append(q[1]), q3.append(q[2])
    xc, med = np.array(xc), np.array(med)
    ax.fill_between(xc, q1, q3, color=so.C_SERIES["matte"], alpha=0.18, lw=0)
    ax.plot(xc, med, color=so.C_SERIES["matte"], lw=2, marker="o", ms=4.5, mec=so.C_SURF, mew=1.0,
            label="測ったばらつきの 8 分位ごとの中央値（帯は 25〜75%）")
    ax.set_xlim(0, top)
    ax.set_ylim(-1, top)
    ax.set_xlabel(f"測った法線のばらつきから換算したぼけの幅 [度]（半径 {R_NOMINAL_MM:g} mm）")
    ax.set_ylabel(ylabel)
    ax.grid(color=so.C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_title(title, fontsize=9, loc="left")


def plot(meas_deg, fit_deg, rho, n_id, rows, path):
    plt = setup_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=200, facecolor=so.C_SURF,
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    scatter_panel(ax1, meas_deg, fit_deg, f"(a) 粗さの比較（反射方向で、{n_id} 点、Spearman ρ = {rho:.2f}）")

    ax2.set_facecolor(so.C_SURF)
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ys = np.arange(len(rows))
    ax2.barh(ys, vals, height=0.6, color=[so.C_SERIES["matte"] if r[2] else so.C_SERIES["gold"] for r in rows])
    for y, v in zip(ys, vals):
        ax2.text(v + 0.08, y, f"{v:.2f}", va="center", fontsize=7.5, color=so.C_TEXT)
    lo = math.floor(min(vals) - 1)
    ax2.set_xlim(lo, max(vals) + 1.0)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(names, fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("テスト視点の観測色の PSNR [dB]（遮られない観測だけ）")
    ax2.grid(axis="x", color=so.C_GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title(f"(b) テスト視点での再現度（横軸は {lo} dB から）", fontsize=9, loc="left")
    fig.suptitle("金の像: 画素の中の法線のばらつきから決めた粗さと、点ごとに最適化した粗さ（共有の照明 + 点ごとの定数）",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=so.C_SURF)
    plt.close(fig)


def prepare(t0):
    """点・法線のばらつき・遮られない観測を用意する（env_rough_check.py からも使う）"""
    v, f = so.load_mesh()
    P, N = so.sample_surface(v, f, 10000, seed=1)       # 面の法線 N は可視判定に使う（前回までと同じ）
    Ns = eo.smooth_normals(10000, seed=1)                 # 反射方向はレンダラーと同じ、なめらかな法線で
    D, ND = so.sample_surface(v, f, 2_000_000, seed=2)
    NDs = eo.smooth_normals(2_000_000, seed=2)

    radii = [r / 1000 for r in RADII_MM]
    ln_smooth = mean_normal_length(P, Ns, D, NDs, radii)
    ln_face = mean_normal_length(P, N, D, ND, [R_NOMINAL_MM / 1000, 0.00135])
    print(f"法線の読み込みとばらつきの計測: {time.time() - t0:.0f} 秒", flush=True)

    lo, hi = v.min(0) - 0.01, v.max(0) + 0.01
    P_t, N_t, Ns_t, D_t = (torch.tensor(x, dtype=torch.float32, device=DEV) for x in (P, N, Ns, D))
    occ, lo_t, shape = ed.occupancy(D_t, lo, hi)
    masks = {s: so.load_masks(so.ROOT / "data" / "gloss_matte", s) for s in ("train", "test")}
    obs = {}
    for split in ("train", "test"):
        cams = so.load_cameras(so.ROOT / "data" / "gloss_gold", split, False, so.BG, DEV)
        obs[split] = so.observe(cams, masks[split], P_t, N_t, D_t)
    keep = obs["train"][0].sum(0) >= so.MIN_TRAIN_OBS
    obs = {s: [x[:, keep] for x in o] for s, o in obs.items()}
    Nk = Ns_t[keep]
    for split, (vis, col, dirs) in obs.items():  # 像自身に遮られる観測を除く（env_diagnose.py と同じ判定）
        r = eo.reflect(dirs, Nk)
        V, n = vis.shape
        Pb = P_t[keep][None].expand(V, n, 3)
        cos_rn = (r * Nk[None]).sum(-1)
        occl = ed.occluded(Pb.reshape(-1, 3), r.reshape(-1, 3), cos_rn.reshape(-1), occ, lo_t, shape).reshape(V, n)
        obs[split][0] = vis & ~occl
    ok = obs["train"][0].sum(0) >= so.MIN_TRAIN_OBS
    obs = {s: [x[:, ok] for x in o] for s, o in obs.items()}
    Np = Nk[ok]
    sel = keep.cpu().numpy().nonzero()[0][ok.cpu().numpy()]  # 元の 10000 点のうち使う点
    print(f"点 {len(sel)}、遮られない観測 学習 {int(obs['train'][0].sum())} / テスト {int(obs['test'][0].sum())}"
          f"（{time.time() - t0:.0f} 秒）", flush=True)
    sig_meas = {R: torch.tensor(sigma_from_length(ln_smooth[R][sel]), dtype=torch.float32, device=DEV) for R in radii}
    return obs, Np, sel, radii, ln_smooth, ln_face, sig_meas


def main():
    t0 = time.time()
    grid = torch.tensor(np.radians(SIGMA_GRID_DEG), dtype=torch.float32, device=DEV)
    obs, Np, sel, radii, ln_smooth, ln_face, sig_meas = prepare(t0)

    spread = {}
    for name, ln in (("smooth", ln_smooth), ("face", ln_face)):
        for R, x in ln.items():
            deg = np.degrees(np.arccos(np.clip(x[sel], -1, 1)))
            spread[f"{name}_R{R * 1000:.2f}mm"] = {"median": float(np.median(deg)), "p90": float(np.percentile(deg, 90))}
            print(f"法線のばらつき（{'なめらかな法線' if name == 'smooth' else '面の法線'}、半径 {R * 1000:.2f} mm）: "
                  f"中央値 {np.median(deg):.1f}°、90% 点 {np.percentile(deg, 90):.1f}°")

    # 確認: 表示値のまま・ぼけなし・照明 32 次・罰 1 で、env_diagnose.py の「すべての点・遮られない観測だけ」を再現する
    o_tr, o_te = Obs(*obs["train"], Np, "display"), Obs(*obs["test"], Np, "display")
    z = torch.zeros(o_tr.n_pts, device=DEV)
    e = fit_env(o_tr, z, 0, [1.0], Le=32)[1.0]  # 表示値のままなら重みは 1 で、前回と同じ問題になる
    check = psnr(o_te, predict(o_tr, o_te, e, z, 0, Le=32)[1])
    print(f"確認（env_diagnose.py の 24.24 dB と一致するはず）: {check:.2f} dB", flush=True)
    del o_tr, o_te
    torch.cuda.empty_cache()

    o_tr, o_te = Obs(*obs["train"], Np, "radiance"), Obs(*obs["test"], Np, "radiance")
    res = {"n_points": int(o_tr.n_pts), "n_obs": {"train": len(o_tr.pid), "test": len(o_te.pid)},
           "check_display_space_psnr": check, "spread_deg": spread, "models": {}}

    def record(key, ptr, pte, hist):
        res["models"][key] = {"train": psnr(o_tr, ptr), "test": psnr(o_te, pte), "train_mse_history": hist}
        print(f"  {key:<34} 学習 {res['models'][key]['train']:6.2f}  テスト {res['models'][key]['test']:6.2f} dB",
              flush=True)

    for L in SH_REF_DEGREES:
        _, ptr, pte, hist = run_fixed(o_tr, o_te, None, L, None)
        record(f"per_point_sh_L{L}", ptr, pte, hist)
    z = torch.zeros(o_tr.n_pts, device=DEV)
    fitted = {}
    for Lp in LOCAL_DEGREES:
        for s in ENV_SMOOTH:
            print(f"点ごとの項 L={Lp}、照明の罰 {s:g}（{time.time() - t0:.0f} 秒）", flush=True)
            _, ptr, pte, hist = run_fixed(o_tr, o_te, z, Lp, s)
            record(f"Lp{Lp}_s{s:g}_material", ptr, pte, hist)
            for R in radii:
                _, ptr, pte, hist = run_fixed(o_tr, o_te, sig_meas[R], Lp, s)
                record(f"Lp{Lp}_s{s:g}_measured_R{R * 1000:.2f}", ptr, pte, hist)
            sig, _, ptr, pte, loss, hist = run_fitted(o_tr, o_te, Lp, s, grid)
            record(f"Lp{Lp}_s{s:g}_fitted", ptr, pte, hist)
            fitted[(Lp, s)] = (sig, loss)

    def best(prefix, suffix):
        ks = [k for k in res["models"] if k.startswith(prefix) and k.endswith(suffix)]
        k = max(ks, key=lambda k: res["models"][k]["test"])
        return k, res["models"][k]["test"]

    # 粗さの比較は、点ごとの定数のモデルで最もテストの良い罰のもの
    k_fit, _ = best("Lp0_", "_fitted")
    s_fit = float(k_fit.split("_s")[1].split("_")[0])
    sig, loss = fitted[(0, s_fit)]
    rmse = loss.sqrt()
    ident = ((rmse.max(1).values - rmse.min(1).values) > ID_CONTRAST).cpu().numpy()
    fit_deg = np.degrees(sig.cpu().numpy())
    corr = {}
    for R in radii:
        m_deg = np.degrees(sig_meas[R].cpu().numpy())
        corr[f"R{R * 1000:.2f}mm"] = {"spearman_identifiable": float(spearmanr(m_deg[ident], fit_deg[ident])[0]),
                                      "spearman_all": float(spearmanr(m_deg, fit_deg)[0])}
    meas_deg = np.degrees(sig_meas[R_NOMINAL_MM / 1000].cpu().numpy())
    rho = corr[f"R{R_NOMINAL_MM:.2f}mm"]["spearman_identifiable"]
    res["roughness_comparison"] = {
        "local_degree": 0, "env_smooth": s_fit, "identifiable_contrast": ID_CONTRAST,
        "n_identifiable": int(ident.sum()), "frac_identifiable": float(ident.mean()), "spearman": corr,
        "fitted_deg": {"median_identifiable": float(np.median(fit_deg[ident])), "median_all": float(np.median(fit_deg)),
                       "frac_zero_identifiable": float((fit_deg[ident] == 0).mean())},
        "measured_deg_nominal": {"median_identifiable": float(np.median(meas_deg[ident])),
                                 "median_all": float(np.median(meas_deg))}}
    print(f"粗さが決まる点: {ident.sum()} / {len(ident)}、Spearman ρ（半径 {R_NOMINAL_MM} mm）= {rho:.3f}")
    for k, c in corr.items():
        print(f"  {k}: ρ（決まる点）{c['spearman_identifiable']:.3f}、ρ（全点）{c['spearman_all']:.3f}")

    rows = [("点ごとの SH（参考、最良の次数）", max(res["models"][f"per_point_sh_L{L}"]["test"] for L in SH_REF_DEGREES), False),
            ("材質の粗さだけ（全点で共通のぼけ）", best("Lp0_", "_material")[1], False)]
    for R in RADII_MM:
        rows.append((f"測ったばらつきから換算（半径 {R:g} mm）", best("Lp0_", f"_measured_R{R:.2f}")[1], R == R_NOMINAL_MM))
    rows.append(("点ごとに最適化した粗さ", best("Lp0_", "_fitted")[1], False))
    res["summary_test_psnr_Lp0"] = {name: v for name, v, _ in rows}
    res["summary_test_psnr_Lp2"] = {"material": best("Lp2_", "_material")[1],
                                    "measured_nominal": best("Lp2_", f"_measured_R{R_NOMINAL_MM:.2f}")[1],
                                    "fitted": best("Lp2_", "_fitted")[1]}
    (HERE / "env_rough.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    plot(meas_deg[ident], fit_deg[ident], rho, int(ident.sum()), rows, HERE / "figures" / "env_rough.png")
    for name, v, _ in rows:
        print(f"{name:<32} {v:6.2f} dB")
    print(f"-> {HERE / 'env_rough.json'}, figures/env_rough.png（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
