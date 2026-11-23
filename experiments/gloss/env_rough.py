# -*- coding: utf-8 -*-
"""
Does the "effective roughness" determined from the normal variation within a pixel agree with the roughness optimized
per point? (a check that does not use 3DGS)

Add a per-point blur to the shared lighting of env_oracle.py:
    y(p, v) ≈ Σ_k c_{p,k} Y_k(d)  +  Σ_j a_{l(j)}(σ_p) e_j Y_j(r)
    a_l(σ) = exp(−l(l+1) σ² / 2)
Blurring on the sphere with a Gaussian of width σ (standard deviation per axis, radians) corresponds, in SH, to
multiplying the coefficients of degree l by a_l (heat diffusion on the sphere. The same form as the attenuation in the
IDE of Ref-NeRF). Once σ_p is fixed, the prediction remains linear in e, so e can be solved by eliminating the
per-point term, as in env_oracle.py.

Three ways of setting σ_p are compared:
    material  σ_p = 0. The material roughness (GGX α = 0.01, about 1° in reflection direction) is finer than the
              resolution of the lighting SH (about 4°), so it is treated as 0. A blur common to all points can be
              absorbed by the lighting e itself becoming blurred, so this is also the "same roughness at all points"
              model
    measured  On the true mesh, measure the mean vector n̄ of the surface normals (the same as the renderer:
              interpolated from vertex normals) within radius R around the point, and set σ_p = 2√(1 − |n̄|). When
              normals vary with a standard deviation of σ_n per axis, |n̄| ≈ 1 − σ_n² (Toksvig's cue), and the
              reflection direction shifts by twice the shift of the normal. No free parameters
    fitted    Choose σ_p per point from a grid (the one with the smallest error on the training views). Solved
              alternately with the lighting e. A stand-in for the roughness that existing methods learn

Value space: the renderer's PNG is "radiance x → Reinhard x/(1+x) → sRGB curve". Averaging within a pixel (the blur)
happens in radiance, so the model is built in radiance and the error is measured in display values (the evaluation is
PSNR of display values, as before). The conversion to display values is nonlinear, so it is linearized around the
prediction, a weighted least squares is solved, and the linearization is redone, repeatedly (Gauss-Newton).
If the weights are fixed at the observed values, the weight of bright reflections becomes a few thousandths of that of
the background, and missing them is ignored.

Interreflection is not handled, so observations judged by env_diagnose.py to be occluded by the statue itself are
removed.

    python experiments/gloss/env_rough.py

Output: experiments/gloss/env_rough.json, figures/env_rough.png
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
LE = 48                                  # SH degree of the shared lighting (resolution is about 180°/48 ≈ 4°)
ENV_SMOOTH = [0.03, 0.1, 0.3]            # smoothness penalty of the shared lighting (0.1 was best in the first run, 10 was always bad)
LOCAL_DEGREES = [0, 2]                   # degree of the per-point term (0 = per-point constant)
SIGMA_GRID_DEG = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40]  # candidate blur widths in reflection direction
RADII_MM = [0.68, 1.36, 2.72, 4.0]       # radii for measuring the normal variation
R_NOMINAL_MM = 2.72                      # width of one pixel (at distance 1.3 m). Combining the pixel box and the
                                         # bilinear interpolation used when picking up observations (a triangle of
                                         # ±1 pixel), the spread is a standard deviation of 1.36 mm per axis, the
                                         # same as a uniform disc of radius 2.72 mm
SH_REF_DEGREES = [2, 3, 4, 6]            # reference: per-point SH (sh_oracle.py)
N_ITER = 10                              # number of alternating updates in fitted
GN_ITERS = 6                             # number of Gauss-Newton iterations (models with σ fixed)
GN_FINAL = 3                             # number of Gauss-Newton iterations after σ is chosen in fitted
X_MAX = 50.0                             # upper limit of the linearization point (radiance; about 0.99 in display values)
ID_CONTRAST = 0.02                       # points where changing σ changes the training-view error (RMSE of display
                                         # values) by more than this count as "points where the roughness is
                                         # determined"
OBS_CHUNK = 4096
KE = (LE + 1) ** 2
L_OF = torch.tensor([math.isqrt(k) for k in range(KE)], dtype=torch.float32, device=DEV)


# ---- Conversion between display values and radiance (GoLD_rayt PNG: Reinhard → sRGB curve, exposure 0) ----

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
    """Derivative of the display value D(x) = srgb(x / (1 + x)) with respect to the radiance x"""
    t = (x / (1 + x)).clamp(min=1e-6)
    d = torch.where(t <= 0.0031308, torch.full_like(t, 12.92), 1.055 / 2.4 * t ** (1 / 2.4 - 1))
    return d * (1 - t) ** 2  # dt/dx = 1 / (1 + x)² = (1 − t)²


# ---- Observations (the visible (point, view) pairs laid out in one list) ----

class Obs:
    def __init__(self, vis, col, dirs, Ns, space):
        v_idx, p_idx = vis.nonzero(as_tuple=True)
        self.pid, self.n_pts, self.space = p_idx, vis.shape[1], space
        self.ydisp = col[v_idx, p_idx].double()
        d = dirs[v_idx, p_idx]
        n = Ns[v_idx, p_idx] if Ns.dim() == 3 else Ns[p_idx]  # normals are per point (n_pts, 3) or per observation (V, n_pts, 3)
        self.d = d
        r = d - 2 * (d * n).sum(-1, keepdim=True) * n  # the view direction mirrored about the normal (same as env_oracle.reflect)
        self.Y = torch.cat([sh_basis(LE, r[i:i + OBS_CHUNK]).float() for i in range(0, len(r), OBS_CHUNK)])
        self.Bl = {}
        self.linearize(None)

    def linearize(self, pred):
        """Linearize the display-value error around the prediction pred (radiance) (Gauss-Newton).
        D(x) ≈ D(p) + D'(p)(x − p), so |D(x) − y|² ≈ D'(p)² |x − z|², z = p + (y − D(p)) / D'(p).
        Solve a weighted least squares with this weight D'(p)² and target z, update the prediction, and linearize
        again. pred=None linearizes at the observed values.
        The weights are per channel (m, 3)"""
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
    """Per-degree SH attenuation for blurring with a Gaussian of width sig (radians) in reflection direction (…, Ke)"""
    l = L_OF[:(Le + 1) ** 2]
    return torch.exp(-l * (l + 1) * sig.float()[..., None] ** 2 / 2)


def att_rows(o, sig, Le=LE):
    """Function that takes a range sl of observation rows and returns the blur attenuation of those rows (len(sl), Ke).
    sig is the per-point σ (n_pts,), or the string "obs" (the per-observation σ is placed in o.sig_obs (n_obs,))"""
    if isinstance(sig, str):
        assert sig == "obs"
        s_obs = o.sig_obs
        return lambda sl: attenuation(s_obs[sl], Le)
    a_pt = attenuation(sig, Le)
    return lambda sl: a_pt[o.pid[sl]]


def local_chol(o, L):
    """Cholesky factor of the normal equations of the per-point term, A_p = Σ w Bl Blᵀ + R (per channel)
    (3, n_pts, K, K)"""
    Bl = o.local(L)
    K = Bl.shape[1]
    A = torch.zeros(3, o.n_pts, K, K, dtype=F64, device=DEV)
    for c in range(3):
        A[c].index_add_(0, o.pid, o.w[:, c, None, None] * Bl[:, :, None] * Bl[:, None, :])
    return torch.linalg.cholesky(A + eo.penalty(L, eo.S_LOC))


def fit_env(o, sig_pt, Lp, smooths, Le=LE):
    """Solve the shared lighting e with the per-point blur sig_pt (n_pts,) (or "obs", see att_rows) fixed
    (the per-point term is eliminated by the Schur complement) -> {s: e (Ke, 3)}"""
    Ke = (Le + 1) ** 2
    att = att_rows(o, sig_pt, Le)
    Bl = o.local(Lp)
    K = Bl.shape[1]
    Lc = local_chol(o, Lp)
    out = {s: torch.zeros(Ke, 3, dtype=F64, device=DEV) for s in smooths}
    for c in range(3):  # the weights differ per channel, so solve per channel
        w, y = o.w[:, c], o.y[:, c]
        S = torch.zeros(Ke, Ke, dtype=F64, device=DEV)
        rhs = torch.zeros(Ke, dtype=F64, device=DEV)
        M = torch.zeros(o.n_pts, K, Ke, device=DEV)  # Σ w Bl Beᵀ (per point)
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
    """With the lighting e and the blur σ fixed, solve the per-point coefficients on the training views and return
    the training and test predictions (in the space of the fit).
    e=None means the per-point term only (per-point SH)"""
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
    """With the lighting e fixed, the mean squared error of display values on the training views when the per-point
    term is solved again for each σ of the grid, per point (n_pts, S)"""
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
    """Minimize the display-value error by Gauss-Newton with σ fixed.
    sig=None means the per-point term only (no lighting)"""
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
    """Solve the per-point σ and the lighting e alternately (starting from σ = 0).
    The Gauss-Newton linearization is also redone each time"""
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


# ---- Normal variation ----

def mean_normal_length(P, Nc, D, ND, radii_m):
    """Length |n̄| of the mean vector of the surface normals within radius R around each point (for each R).
    Faces pointing away from the normal of the point (the back side of thin parts) are excluded"""
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
    """From |n̄|, the blur width in reflection direction σ_r = 2 σ_n, σ_n = √(1 − |n̄|)
    (radians, capped at the maximum of the grid)"""
    return np.minimum(2 * np.sqrt(np.clip(1 - ln, 0, None)), math.radians(SIGMA_GRID_DEG[-1]))


# ---- Figures ----

def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9,
                         "axes.edgecolor": so.C_TEXT2, "axes.labelcolor": so.C_TEXT, "xtick.color": so.C_TEXT2,
                         "ytick.color": so.C_TEXT2, "text.color": so.C_TEXT})
    return plt


def scatter_panel(ax, meas_deg, fit_deg, title, ylabel="Optimal per-point blur width [degrees]"):
    """Horizontal axis = blur converted from the measured variation, vertical axis = fitted blur.
    Points, and the median for each octile of the horizontal axis (the band is 25–75%)"""
    top = SIGMA_GRID_DEG[-1] + 2
    ax.set_facecolor(so.C_SURF)
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.35, 0.35, len(fit_deg))  # the vertical axis takes only grid values, so jitter a little to show overlaps
    ax.scatter(meas_deg, fit_deg + jit, s=5, color=so.C_SERIES["gold"], alpha=0.25, lw=0, label="Points (points where the roughness is determined by the observations)")
    ax.plot([0, top], [0, top], color=so.C_TEXT2, lw=1.0, ls=(0, (4, 3)), label="y = x (as converted)")
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
            label="Median for each octile of the measured variation (the band is 25–75%)")
    ax.set_xlim(0, top)
    ax.set_ylim(-1, top)
    ax.set_xlabel(f"Blur width converted from the measured normal variation [degrees] (radius {R_NOMINAL_MM:g} mm)")
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
    scatter_panel(ax1, meas_deg, fit_deg, f"(a) Roughness comparison (in reflection direction, {n_id} points, Spearman ρ = {rho:.2f})")

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
    ax2.set_xlabel("Test-view PSNR of the observed colour [dB] (unoccluded observations only)")
    ax2.grid(axis="x", color=so.C_GRID, lw=0.6)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title(f"(b) Reconstruction quality on the test views (horizontal axis starts at {lo} dB)", fontsize=9, loc="left")
    fig.suptitle("Gold statue: roughness determined from the normal variation within a pixel, and roughness optimized per point (shared lighting + per-point constant)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=so.C_SURF)
    plt.close(fig)


def prepare(t0):
    """Prepare the points, the normal variation and the unoccluded observations (also used from env_rough_check.py)"""
    v, f = so.load_mesh()
    P, N = so.sample_surface(v, f, 10000, seed=1)       # the face normal N is used for the visibility test (as before)
    Ns = eo.smooth_normals(10000, seed=1)                 # for the reflection direction, the same smooth normals as the renderer
    D, ND = so.sample_surface(v, f, 2_000_000, seed=2)
    NDs = eo.smooth_normals(2_000_000, seed=2)

    radii = [r / 1000 for r in RADII_MM]
    ln_smooth = mean_normal_length(P, Ns, D, NDs, radii)
    ln_face = mean_normal_length(P, N, D, ND, [R_NOMINAL_MM / 1000, 0.00135])
    print(f"Loading normals and measuring the variation: {time.time() - t0:.0f} s", flush=True)

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
    for split, (vis, col, dirs) in obs.items():  # remove observations occluded by the statue itself (same test as env_diagnose.py)
        r = eo.reflect(dirs, Nk)
        V, n = vis.shape
        Pb = P_t[keep][None].expand(V, n, 3)
        cos_rn = (r * Nk[None]).sum(-1)
        occl = ed.occluded(Pb.reshape(-1, 3), r.reshape(-1, 3), cos_rn.reshape(-1), occ, lo_t, shape).reshape(V, n)
        obs[split][0] = vis & ~occl
    ok = obs["train"][0].sum(0) >= so.MIN_TRAIN_OBS
    obs = {s: [x[:, ok] for x in o] for s, o in obs.items()}
    Np = Nk[ok]
    sel = keep.cpu().numpy().nonzero()[0][ok.cpu().numpy()]  # the points used, out of the original 10000
    print(f"Points {len(sel)}, unoccluded observations: train {int(obs['train'][0].sum())} / test {int(obs['test'][0].sum())}"
          f" ({time.time() - t0:.0f} s)", flush=True)
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
            print(f"Normal variation ({'smooth normals' if name == 'smooth' else 'face normals'}, radius {R * 1000:.2f} mm): "
                  f"median {np.median(deg):.1f}°, 90th percentile {np.percentile(deg, 90):.1f}°")

    # Check: in display values, with no blur, lighting of degree 32 and penalty 1, reproduce
    # "all points, unoccluded observations only" of env_diagnose.py
    o_tr, o_te = Obs(*obs["train"], Np, "display"), Obs(*obs["test"], Np, "display")
    z = torch.zeros(o_tr.n_pts, device=DEV)
    e = fit_env(o_tr, z, 0, [1.0], Le=32)[1.0]  # in display values the weights are 1, and the problem is the same as before
    check = psnr(o_te, predict(o_tr, o_te, e, z, 0, Le=32)[1])
    print(f"Check (should match the 24.24 dB of env_diagnose.py): {check:.2f} dB", flush=True)
    del o_tr, o_te
    torch.cuda.empty_cache()

    o_tr, o_te = Obs(*obs["train"], Np, "radiance"), Obs(*obs["test"], Np, "radiance")
    res = {"n_points": int(o_tr.n_pts), "n_obs": {"train": len(o_tr.pid), "test": len(o_te.pid)},
           "check_display_space_psnr": check, "spread_deg": spread, "models": {}}

    def record(key, ptr, pte, hist):
        res["models"][key] = {"train": psnr(o_tr, ptr), "test": psnr(o_te, pte), "train_mse_history": hist}
        print(f"  {key:<34} train {res['models'][key]['train']:6.2f}  test {res['models'][key]['test']:6.2f} dB",
              flush=True)

    for L in SH_REF_DEGREES:
        _, ptr, pte, hist = run_fixed(o_tr, o_te, None, L, None)
        record(f"per_point_sh_L{L}", ptr, pte, hist)
    z = torch.zeros(o_tr.n_pts, device=DEV)
    fitted = {}
    for Lp in LOCAL_DEGREES:
        for s in ENV_SMOOTH:
            print(f"Per-point term L={Lp}, lighting penalty {s:g} ({time.time() - t0:.0f} s)", flush=True)
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

    # The roughness comparison uses the per-point constant model, with the penalty that is best on the test views
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
    print(f"Points where the roughness is determined: {ident.sum()} / {len(ident)}, Spearman ρ (radius {R_NOMINAL_MM} mm) = {rho:.3f}")
    for k, c in corr.items():
        print(f"  {k}: ρ (determined points) {c['spearman_identifiable']:.3f}, ρ (all points) {c['spearman_all']:.3f}")

    rows = [("Per-point SH (reference, best degree)", max(res["models"][f"per_point_sh_L{L}"]["test"] for L in SH_REF_DEGREES), False),
            ("Material roughness only (same blur for all points)", best("Lp0_", "_material")[1], False)]
    for R in RADII_MM:
        rows.append((f"Converted from the measured variation (radius {R:g} mm)", best("Lp0_", f"_measured_R{R:.2f}")[1], R == R_NOMINAL_MM))
    rows.append(("Per-point optimized roughness", best("Lp0_", "_fitted")[1], False))
    res["summary_test_psnr_Lp0"] = {name: v for name, v, _ in rows}
    res["summary_test_psnr_Lp2"] = {"material": best("Lp2_", "_material")[1],
                                    "measured_nominal": best("Lp2_", f"_measured_R{R_NOMINAL_MM:.2f}")[1],
                                    "fitted": best("Lp2_", "_fitted")[1]}
    (HERE / "env_rough.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    plot(meas_deg[ident], fit_deg[ident], rho, int(ident.sum()), rows, HERE / "figures" / "env_rough.png")
    for name, v, _ in rows:
        print(f"{name:<32} {v:6.2f} dB")
    print(f"-> {HERE / 'env_rough.json'}, figures/env_rough.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
