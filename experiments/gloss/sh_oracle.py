# -*- coding: utf-8 -*-
"""
Measure the limit of the representation itself (SH of the view direction) with "per-point SH fits" that do not use
3DGS.

The true shape (the Erato mesh) is known, so for each point on the surface:
    - collect the observed colours (image values) and view directions from the training views that see the point,
    - fit real SH of degree L by least squares (with a very weak ridge),
    - on the test views, predict the colour with the fitted SH and compare it with the observation.
This removes the effects of 3DGS optimization, densification, and projection, and gives the upper bound of the
representation "each point holds its colour as SH of the view direction" (the ideal case where one Gaussian takes
care of one point on the surface).

    python experiments/gloss/sh_oracle.py

Output: experiments/gloss/sh_oracle.json, figures/sh_oracle.png, figures/sh_oracle_point.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from geom_eval import load_mesh, sample_surface  # noqa: E402
from gs.dataset import load_cameras  # noqa: E402
from gs.sh import sh_basis  # noqa: E402

DEV = "cuda"
BG = [0.396] * 3
COS_MIN = math.cos(math.radians(75))  # observations seen more obliquely than this are not used
LAMBDA = 1e-3                         # ridge (so that the solution is determined even with more coefficients than observations)
MIN_TRAIN_OBS = 10                    # points seen in fewer than this many training views are not used
SMOOTH_FIG = 0.03                     # strength of the smoothness penalty shown on the right of figure 1
EXAMPLES = [(3, 0.0), (12, 0.0), (12, SMOOTH_FIG)]  # (degree, smoothness penalty) shown in figure 2
LUMA = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float64)

# Figure colours (dataviz reference palette: categories 1, 2 and the text / surface colours)
C_SERIES = {"gold": "#2a78d6", "matte": "#eb6834"}
# Text, ticks and axis lines are black. Grid lines are a clearly visible gray (pale gray is unreadable).
# C_SHADE is only for shaded background regions.
C_TEXT, C_GRID, C_SHADE, C_SURF = "#0b0b0b", "#b0b0b0", "#e4e3df", "#fcfcfb"
LABEL = {"gold": "Gold", "matte": "Diffuse"}


def load_masks(root, split):
    """Statue mask eroded by 3x3 (edge pixels are mixed with the background, so they are not used)"""
    frames = json.loads((Path(root) / f"transforms_{split}.json").read_text(encoding="utf-8"))["frames"]
    out = []
    for f in frames:
        m = torch.from_numpy(imageio.imread(Path(root) / (f["file_path"].replace("_denoised", "") + "_mask.png"))
                             .astype(np.float32) / 255.0).to(DEV)
        out.append(-F.max_pool2d(-m[None, None], 3, 1, 1)[0, 0])
    return out


@torch.no_grad()
def observe(cams, masks, P, N, D):
    """For each view, return whether point P (n,3) (normal N) is visible, the observed colour, and the view direction
    (camera → point).

    Visibility test: build a per-pixel minimum depth (z-buffer) from points D that densely cover the surface. A point
    is visible if its depth is no more than that (with a tolerance for oblique surfaces), it is seen from near the
    front, and its pixel is inside the statue.
    """
    vis, col, dirs = [], [], []
    for cam, m in zip(cams, masks):
        def proj(X):
            pc = X @ cam.R.T + cam.t
            return cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy, pc[:, 2]

        ud, vd, zd = proj(D)
        ok = (ud >= 0) & (ud < cam.W) & (vd >= 0) & (vd < cam.H) & (zd > 0)
        zbuf = torch.full((cam.H * cam.W,), float("inf"), device=DEV).scatter_reduce(
            0, vd[ok].long() * cam.W + ud[ok].long(), zd[ok], "amin")

        u, v, z = proj(P)
        inb = (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        pix = v.clamp(0, cam.H - 1).long() * cam.W + u.clamp(0, cam.W - 1).long()
        to_cam = F.normalize(cam.center - P, dim=-1)
        cos = (N * to_cam).sum(-1)
        tan = torch.sqrt((1 - cos * cos).clamp(min=0)) / cos.clamp(min=COS_MIN)
        tol = 0.002 + 1.5 * (z / cam.fx) * tan  # even within one pixel, the depth of an oblique surface changes by about this much
        visible = inb & (cos > COS_MIN) & (z <= zbuf[pix] + tol) & (m.reshape(-1)[pix] > 0.999)

        grid = torch.stack([2 * u / cam.W - 1, 2 * v / cam.H - 1], -1)[None, None]
        rgb = F.grid_sample(cam.image.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T
        vis.append(visible)
        col.append(rgb)
        dirs.append(-to_cam)
    return torch.stack(vis), torch.stack(col).double(), torch.stack(dirs)


def chunk_size(K):
    return max(200, min(2000, int(4e8 / (K * K * 8))))


def fit(deg, vis, col, dirs, smooth=0.0):
    """Solve min Σ_v w_v |B_v c - y_v|² + Σ_k (λ + s·l_k(l_k+1)) c_k² per point -> coefficients (n, K, 3)

    λ = LAMBDA is a very weak ridge (so that the solution is determined even with more coefficients than
    observations).
    s is the smoothness penalty: the coefficients of degree l are penalized with weight l(l+1) (this corresponds to
    the sum of squares of the gradient magnitude on the sphere).
    The constant term (l=0) is not penalized, so the mean colour is not biased.
    """
    V, n = vis.shape
    K = (deg + 1) ** 2
    coef = torch.zeros(n, K, 3, dtype=torch.float64, device=DEV)
    l = torch.tensor([math.isqrt(k) for k in range(K)], dtype=torch.float64, device=DEV)
    eye = torch.diag(LAMBDA + smooth * l * (l + 1))
    for i in range(0, n, chunk_size(K)):
        sl = slice(i, i + chunk_size(K))
        B = sh_basis(deg, dirs[:, sl].reshape(-1, 3)).double().reshape(V, -1, K)
        Bw = B * vis[:, sl, None]
        A = torch.einsum("vnk,vnj->nkj", Bw, B) + eye
        coef[sl] = torch.linalg.solve(A, torch.einsum("vnk,vnc->nkc", Bw, col[:, sl]))
    return coef


def predict(deg, coef, dirs):
    V, n = dirs.shape[:2]
    K = coef.shape[1]
    out = torch.empty(V, n, 3, dtype=torch.float64, device=DEV)
    for i in range(0, n, chunk_size(K)):
        sl = slice(i, i + chunk_size(K))
        B = sh_basis(deg, dirs[:, sl].reshape(-1, 3)).double().reshape(V, -1, K)
        out[:, sl] = torch.einsum("vnk,nkc->vnc", B, coef[sl])
    return out.clamp(0, 1)


def psnr(pred, col, vis):
    se = ((pred - col) ** 2).mean(-1)
    return -10 * math.log10(((se * vis).sum() / vis.sum()).item())


def to_cam_grid(n_az=361, n_el=101, center=0.0):
    az = torch.linspace(-180, 180, n_az, dtype=torch.float64)
    el = torch.linspace(-10, 90, n_el, dtype=torch.float64)
    E, A = torch.meshgrid(torch.deg2rad(el), torch.deg2rad(az + center), indexing="ij")
    return az, el, torch.stack([torch.cos(E) * torch.sin(A), torch.sin(E), torch.cos(E) * torch.cos(A)], -1)


def az_el(to_cam):
    t = to_cam.cpu().double()
    return torch.rad2deg(torch.atan2(t[:, 0], t[:, 2])).numpy(), torch.rad2deg(torch.asin(t[:, 1].clamp(-1, 1))).numpy()


def plot_curves(res, n_obs_median, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9,
                         "axes.edgecolor": C_TEXT, "axes.labelcolor": C_TEXT, "xtick.color": C_TEXT,
                         "ytick.color": C_TEXT, "text.color": C_TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200, facecolor=C_SURF, sharey=True)
    degs = res["degrees"]
    L_over = math.sqrt(n_obs_median) - 1  # the degree at which the number of coefficients (L+1)^2 exceeds the median number of training observations per point
    panels = [("(a) Almost plain least squares (very weak ridge only)", lambda r: r),
              (f"(b) With smoothness penalty (s = {SMOOTH_FIG:g})", lambda r: r["smooth_check"][str(SMOOTH_FIG)])]
    for ax, (title, get) in zip(axes, panels):
        ax.set_facecolor(C_SURF)
        ax.axvspan(L_over, degs[-1] + 0.5, color=C_SHADE, alpha=0.55, lw=0)
        ax.text(L_over + 0.15, 0.03, f"number of coefficients (L+1)² exceeds the median number\nof training observations per point ({n_obs_median:.0f})",
                transform=ax.get_xaxis_transform(), fontsize=7, color=C_TEXT, va="bottom")
        ax.axvline(3, color=C_TEXT, lw=0.8, ls=":")
        ax.text(3.15, 0.97, "3DGS default (L=3)", transform=ax.get_xaxis_transform(), fontsize=7, color=C_TEXT,
                va="top")
        ends = []
        for cond in res["conditions"]:
            r = get(res["conditions"][cond])
            for split, ls in (("train", (0, (4, 2))), ("test", "-")):
                y = r[f"{split}_psnr"]
                ax.plot(degs, y, color=C_SERIES[cond], lw=2, ls=ls, marker="o" if split == "test" else "s", ms=4.5,
                        mec=C_SURF, mew=1.2, label=f"{LABEL[cond]}, {'training views' if split == 'train' else 'test views'}")
                ends.append([y[-1], f"{LABEL[cond]}, {'training' if split == 'train' else 'test'}"])
        # Keep the labels at the right end of the lines at least GAP dB apart vertically so that they do not overlap
        # (push them down in order from the top)
        GAP = 3.5
        ends.sort(key=lambda e: -e[0])
        pos = [ends[0][0]]
        for y, _ in ends[1:]:
            pos.append(min(y, pos[-1] - GAP))
        for (y, text), yl in zip(ends, pos):
            ax.annotate(text, (degs[-1], y), xytext=(degs[-1] + 0.45, yl), textcoords="data", va="center",
                        fontsize=7, color=C_TEXT)
        ax.set_xticks(degs)
        ax.set_xticklabels([f"{L}\n({(L + 1) ** 2})" for L in degs], fontsize=7)
        ax.set_xlabel("SH degree L (in parentheses: number of coefficients per channel)")
        ax.set_xlim(-0.5, degs[-1] + 3.2)
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_ylabel("PSNR of the observed colours [dB]")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=4, frameon=False, fontsize=7.5,
                   handlelength=2.6, columnspacing=1.2, borderaxespad=0.2)
    fig.suptitle("Reconstruction quality when SH of the view direction is fitted per surface point (upper bound without 3DGS)", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=C_SURF)
    plt.close(fig)


def plot_point(ex, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9})
    c = ex["az_center"]
    az, el, grid = to_cam_grid(center=c)  # azimuth is relative to the centre of this point's observations (avoids the wrap at -180°)
    rel = lambda a: (np.asarray(a) - c + 180) % 360 - 180  # noqa: E731
    fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(4.0 * len(EXAMPLES), 3.1), dpi=200, facecolor=C_SURF,
                             sharey=True, constrained_layout=True)
    for ax, (deg, s), coef in zip(axes, EXAMPLES, ex["coef"]):
        coef = torch.tensor(coef, dtype=torch.float64)
        B = sh_basis(deg, -grid.reshape(-1, 3))  # the SH is evaluated in the camera → point direction
        lum = ((B @ coef).clamp(0, 1) @ LUMA).reshape(grid.shape[:2]).numpy()
        im = ax.imshow(lum, extent=[-180, 180, el[0], el[-1]], origin="lower", cmap="gray", vmin=0, vmax=1,
                       aspect="auto", interpolation="bilinear")
        for split, mk, ec, lab in (("train", "o", C_SERIES["gold"], "training-view observations"),
                                   ("test", "^", C_SERIES["matte"], "test-view observations")):
            ax.scatter(rel(ex[split]["az"]), ex[split]["el"], c=ex[split]["lum"], cmap="gray", vmin=0, vmax=1,
                       marker=mk, s=60, edgecolors=ec, linewidths=1.6, label=lab, zorder=3)
        ax.set_title(f"L={deg} ({(deg + 1) ** 2} coefficients)" + (f", smoothness penalty s={s:g}" if s else ", no penalty"),
                     fontsize=9, loc="left")
        ax.set_xlabel("Camera azimuth (relative to the centre of the observations, degrees)")
        ax.set_xlim(-110, 110)
        ax.set_ylim(-10, 90)
    axes[0].set_ylabel("Camera elevation (degrees)")
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.85)
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("Luminance (image value)", fontsize=8)
    fig.suptitle("One point on the gold statue: appearance by view direction (markers = actual observations, background = SH interpolation)", fontsize=9, x=0.01,
                 ha="left")
    fig.savefig(path, facecolor=C_SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16])
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--check-smooth", type=float, nargs="*", default=[0.003, 0.01, SMOOTH_FIG, 0.1])
    args = ap.parse_args()

    v, f = load_mesh()
    P, N = sample_surface(v, f, args.n_points, seed=1)
    D, _ = sample_surface(v, f, 2_000_000, seed=2)
    P, N, D = (torch.tensor(x, dtype=torch.float32, device=DEV) for x in (P, N, D))
    masks = {s: load_masks(ROOT / "data" / "gloss_matte", s) for s in ("train", "test")}

    res = {"degrees": args.degrees, "lambda": LAMBDA, "n_points_sampled": args.n_points, "conditions": {}}
    example = None
    for cond in ("gold", "matte"):
        obs = {}
        for split in ("train", "test"):
            cams = load_cameras(ROOT / "data" / f"gloss_{cond}", split, False, BG, DEV)
            obs[split] = observe(cams, masks[split], P, N, D)
        keep = obs["train"][0].sum(0) >= MIN_TRAIN_OBS
        tr = [x[:, keep] for x in obs["train"]]
        te = [x[:, keep] for x in obs["test"]]
        n_tr = tr[0].sum(0).float()
        r = {"n_points": int(keep.sum()), "train_obs_per_point_median": float(n_tr.median()),
             "train_obs_per_point_mean": float(n_tr.mean()), "test_obs_total": int(te[0].sum()),
             "train_psnr": [], "test_psnr": []}
        for deg in args.degrees:
            coef = fit(deg, *tr)
            r["train_psnr"].append(psnr(predict(deg, coef, tr[2]), tr[1], tr[0]))
            r["test_psnr"].append(psnr(predict(deg, coef, te[2]), te[1], te[0]))
            print(f"{cond:<6} L={deg:2d} K={(deg + 1) ** 2:3d}  train {r['train_psnr'][-1]:6.2f} dB  "
                  f"test {r['test_psnr'][-1]:6.2f} dB", flush=True)
        res["conditions"][cond] = r
        print(f"{cond}: points {r['n_points']}, training observations per point: median {r['train_obs_per_point_median']:.0f}"
              f" (mean {r['train_obs_per_point_mean']:.1f}), test observations {r['test_obs_total']}")

        if cond == "gold":  # the point shown as an example: well seen in both training and test, with the largest change in appearance
            lum_tr = (tr[1] @ LUMA.to(DEV))
            w = tr[0].double()
            mean = (lum_tr * w).sum(0) / w.sum(0)
            var = (((lum_tr - mean) ** 2) * w).sum(0) / w.sum(0)
            var[(tr[0].sum(0) < 30) | (te[0].sum(0) < 3)] = -1
            j = int(var.argmax())
            one = [x[:, j:j + 1] for x in tr]
            ex = {"examples": EXAMPLES, "coef": [fit(d, *one, smooth=s)[0].cpu().tolist() for d, s in EXAMPLES]}
            for split, o in (("train", tr), ("test", te)):
                sel = o[0][:, j]
                a, e = az_el(-o[2][sel, j])
                ex[split] = {"az": a.tolist(), "el": e.tolist(), "lum": (o[1][sel, j] @ LUMA.to(DEV)).cpu().tolist()}
            allaz = np.radians(np.array(ex["train"]["az"] + ex["test"]["az"]))
            ex["az_center"] = math.degrees(math.atan2(np.sin(allaz).mean(), np.cos(allaz).mean()))
            example = ex

        # Dependence on regularization: with a smoothness penalty, do the test views get better even at high degrees?
        r["smooth_check"] = {}
        for s in args.check_smooth:
            fits = [(d, fit(d, *tr, smooth=s)) for d in args.degrees]
            tr_ps = [psnr(predict(d, c, tr[2]), tr[1], tr[0]) for d, c in fits]
            te_ps = [psnr(predict(d, c, te[2]), te[1], te[0]) for d, c in fits]
            k = int(np.argmax(te_ps))
            r["smooth_check"][str(s)] = {"train_psnr": tr_ps, "test_psnr": te_ps, "best_degree": args.degrees[k],
                                         "best_test_psnr": te_ps[k]}
            print(f"{cond:<6} s={s:g}: best on the test views {te_ps[k]:.2f} dB (L={args.degrees[k]}), "
                  f"L={args.degrees[-1]}: training {tr_ps[-1]:.2f} / test {te_ps[-1]:.2f} dB", flush=True)

    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "sh_oracle.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    plot_curves(res, res["conditions"]["gold"]["train_obs_per_point_median"], HERE / "figures" / "sh_oracle.png")
    plot_point(example, HERE / "figures" / "sh_oracle_point.png")
    print(f"-> {HERE / 'sh_oracle.json'}, figures/sh_oracle.png, figures/sh_oracle_point.png")


if __name__ == "__main__":
    main()
