# -*- coding: utf-8 -*-
"""
Colour experiment, stage 0: does PSNR hide colour error? (CPU only; reads the files written by render_tests.py)

Every pixel of every test view is converted to the colour a calibrated sRGB display shows (colorimetry.py), and
the render is compared with the ground truth in
    ΔE00            overall colour difference, split into its lightness / chroma / hue terms
    Δu'v', Δxy      chromaticity difference (only where the truth has L* >= 20; chromaticity is unstable when dark)
    signed ΔL*, ΔC*ab   systematic bias (ΔC*ab < 0: the render is less saturated than the truth)
Pixel classes are defined on the ground truth only, so they are the same for every model:
    object   mask > 0.5                       background   the rest
    edge     object pixels where a 4-neighbour differs by ΔE*ab > 10, or within 1 px of the silhouette
    flat     object pixels that are not edge
    dark     L* < 20       bright   L* > 90       saturated   C*ab > 40
    gamut    smallest display channel < 0.02 and L* >= 20 (saturated colours at the edge of the sRGB gamut, where the
             renderer's clamp of colour at 0 sits)
Classes overlap (a pixel can be edge and saturated); each row is a subset of the object pixels.

    python experiments/color/chroma_eval.py logs/lego_half_7k logs/gloss_gold logs/gloss_matte logs/toy_aa

-> experiments/color/results/stage0.json, experiments/color/figures/stage0_<run>.png, a table on stdout
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from colorimetry import ciede2000, rgb_to_xyz, srgb_decode, upvp, xy, xyz_to_lab  # noqa: E402

HERE = Path(__file__).resolve().parent
CLASSES = ("object", "edge", "flat", "dark", "bright", "saturated", "gamut", "background")
EDGE_DE, DARK_L, BRIGHT_L, SAT_C, GAMUT_MIN = 10.0, 20.0, 90.0, 40.0, 0.02


def per_pixel(gt, rend):
    """All per-pixel quantities for one view. gt / rend: (H,W,3) display values"""
    r = np.clip(rend, 0, 1)
    X_g, X_r = rgb_to_xyz(srgb_decode(gt)), rgb_to_xyz(srgb_decode(r))
    lab_g, lab_r = xyz_to_lab(X_g), xyz_to_lab(X_r)
    dE, tL, tC, tH = ciede2000(lab_g, lab_r)
    return {
        "lab_g": lab_g, "dE": dE, "tL": tL, "tC": tC, "tH": tH,
        "duv": np.linalg.norm(upvp(X_r) - upvp(X_g), axis=-1),
        "dxy": np.linalg.norm(xy(X_r) - xy(X_g), axis=-1),
        "dL": lab_r[..., 0] - lab_g[..., 0],
        "dC": np.hypot(lab_r[..., 1], lab_r[..., 2]) - np.hypot(lab_g[..., 1], lab_g[..., 2]),
        "se": ((r - gt) ** 2).mean(-1),
        "over": (rend > 1.0).any(-1),
    }


def classes(gt, lab_g, mask):
    obj = mask > 0.5
    L, C = lab_g[..., 0], np.hypot(lab_g[..., 1], lab_g[..., 2])
    jump = np.zeros(L.shape, bool)
    for ax in (0, 1):
        d = np.linalg.norm(np.diff(lab_g, axis=ax), axis=-1) > EDGE_DE
        s = np.diff(obj.astype(np.int8), axis=ax) != 0
        e = d | s
        if ax == 0:
            jump[1:] |= e
            jump[:-1] |= e
        else:
            jump[:, 1:] |= e
            jump[:, :-1] |= e
    edge = obj & jump
    return {
        "object": obj, "edge": edge, "flat": obj & ~edge,
        "dark": obj & (L < DARK_L), "bright": obj & (L > BRIGHT_L), "saturated": obj & (C > SAT_C),
        "gamut": obj & (gt.min(-1) < GAMUT_MIN) & (L >= DARK_L), "background": ~obj,
    }


def summarize(acc, n_obj):
    """acc: per-class lists of per-pixel arrays, concatenated over views"""
    out = {}
    for c in CLASSES:
        d = {k: np.concatenate(v) for k, v in acc[c].items()}
        n = len(d["dE"])
        if n == 0:
            out[c] = {"pixels": 0}
            continue
        chroma = d["Lg"] >= DARK_L
        e2 = d["tL"] ** 2 + d["tC"] ** 2 + d["tH"] ** 2
        s = max(e2.sum(), 1e-12)
        out[c] = {
            "pixels": int(n), "frac_of_object": n / n_obj,
            "psnr": -10 * math.log10(max(d["se"].mean(), 1e-12)),
            "dE00_mean": float(d["dE"].mean()), "dE00_p95": float(np.percentile(d["dE"], 95)),
            "frac_dE00_gt2": float((d["dE"] > 2).mean()), "frac_dE00_gt5": float((d["dE"] > 5).mean()),
            "share_lightness": float((d["tL"] ** 2).sum() / s), "share_chroma": float((d["tC"] ** 2).sum() / s),
            "share_hue": float((d["tH"] ** 2).sum() / s),
            "dL_mean": float(d["dL"].mean()), "dCab_mean": float(d["dC"].mean()),
            "duv_mean": float(d["duv"][chroma].mean()) if chroma.any() else None,
            "duv_p95": float(np.percentile(d["duv"][chroma], 95)) if chroma.any() else None,
            "dxy_mean": float(d["dxy"][chroma].mean()) if chroma.any() else None,
            "frac_over_1": float(d["over"].mean()),
        }
    return out


def figure(name, view, gt, rend, px, mask, path):
    ys, xs = np.nonzero(mask > 0.5)
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    cut = lambda a: a[y0:y1, x0:x1]  # noqa: E731
    obj = cut(mask > 0.5)
    panels = [
        ("truth", cut(gt), None, None),
        ("3DGS", cut(np.clip(rend, 0, 1)), None, None),
        ("ΔE00 (0–10)", np.where(obj, cut(px["dE"]), np.nan), "magma", (0, 10)),
        ("Δu'v' (0–0.02)", np.where(obj & (cut(px["lab_g"][..., 0]) >= DARK_L), cut(px["duv"]), np.nan), "magma", (0, 0.02)),
        ("ΔC*ab (render − truth)", np.where(obj, cut(px["dC"]), np.nan), "RdBu_r", (-15, 15)),
    ]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
    for ax, (title, img, cmap, lim) in zip(axs, panels):
        if cmap is None:
            ax.imshow(img.astype(np.float32), interpolation="nearest")
        else:
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("white")
            im = ax.imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1], interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{name}, test view {view} (largest mean ΔE00 on the object)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(run_dir):
    z = np.load(Path(run_dir) / "test_renders.npz")
    R, G, M = z["render"].astype(np.float64), z["gt"].astype(np.float64), z["mask"].astype(np.float64)
    keys = ("dE", "tL", "tC", "tH", "duv", "dxy", "dL", "dC", "se", "over", "Lg")
    acc = {c: {k: [] for k in keys} for c in CLASSES}
    per_view, worst = [], (-1.0, 0)
    for v in range(len(R)):
        px = per_pixel(G[v], R[v])
        px["Lg"] = px["lab_g"][..., 0]
        cl = classes(G[v], px["lab_g"], M[v])
        for c in CLASSES:
            for k in keys:
                acc[c][k].append(px[k][cl[c]])
        o = cl["object"]
        pv = {"psnr_object": -10 * math.log10(max(px["se"][o].mean(), 1e-12)), "dE00_object": float(px["dE"][o].mean()),
              "duv_object": float(px["duv"][o & (px["Lg"] >= DARK_L)].mean())}
        per_view.append(pv)
        worst = max(worst, (pv["dE00_object"], v))
    n_obj = sum(len(a) for a in acc["object"]["dE"])
    res = summarize(acc, n_obj)
    pvs = np.array([[p["psnr_object"], p["dE00_object"], p["duv_object"]] for p in per_view])
    corr = np.corrcoef(pvs.T) if len(pvs) > 2 else np.full((3, 3), np.nan)
    v = worst[1]
    name = Path(run_dir).name
    figure(name, v, G[v], R[v], per_pixel(G[v], R[v]), M[v], HERE / "figures" / f"stage0_{name}.png")
    return {"datadir": str(z["datadir"]), "views": len(R), "classes": res, "per_view": per_view,
            "corr_psnr_dE00": float(corr[0, 1]), "corr_psnr_duv": float(corr[0, 2]), "worst_view": int(v)}


def table(results):
    hdr = (f"{'run':<14}{'class':<11}{'%obj':>6}{'PSNR':>7}{'dE00':>6}{'p95':>6}{'>2':>6}"
           f"{'  L/C/H share':>16}{'dL*':>7}{'dCab':>7}{'duv':>8}{'dxy':>8}")
    print(hdr)
    for name, r in results.items():
        for c in CLASSES:
            s = r["classes"][c]
            if not s.get("pixels"):
                continue
            share = f"{s['share_lightness']:.2f}/{s['share_chroma']:.2f}/{s['share_hue']:.2f}"
            f = lambda x, p=4: "   -" if x is None else f"{x:.{p}f}"  # noqa: E731
            print(f"{name:<14}{c:<11}{100 * s['frac_of_object']:6.1f}{s['psnr']:7.2f}{s['dE00_mean']:6.2f}"
                  f"{s['dE00_p95']:6.2f}{s['frac_dE00_gt2']:6.2f}{share:>16}{s['dL_mean']:7.2f}{s['dCab_mean']:7.2f}"
                  f"{f(s['duv_mean']):>8}{f(s['dxy_mean']):>8}")
        print(f"{'':14}per-view correlation with object PSNR: dE00 {r['corr_psnr_dE00']:+.2f}, "
              f"du'v' {r['corr_psnr_duv']:+.2f}  ({r['views']} views)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories that contain test_renders.npz")
    ap.add_argument("--out", default=str(HERE / "results" / "stage0.json"))
    args = ap.parse_args()
    (HERE / "figures").mkdir(exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = {Path(r).name: evaluate(r) for r in args.runs}
    table(results)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {args.out}, {HERE / 'figures'}")


if __name__ == "__main__":
    main()
