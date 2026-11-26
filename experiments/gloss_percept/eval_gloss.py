# -*- coding: utf-8 -*-
"""
Gloss cues of 3DGS renders against the ground truth (GPU for rendering, a few seconds per model).

Each model is rendered on the test views of its data (linear radiance; experiments/color/stage1_eval.render_model) and
compared with the truth on the object (pixels fully covered by it: coverage > 0.99 in raw/test/*_mask.pfm). The scene is
achromatic, so everything is on the luminance Y. The image cues are the ones gloss psychophysics names:

    PSNR          on encode(linear), the space of the loss (the object only)
    Y ratio       summed linear Y of the render / of the truth: all object pixels, the highlights (the truth's brightest
                  5% of the object in each view) and the peaks (its brightest 0.5%)
The cues below are on display values (one exposure per data set, the truth's median Y on the object -> 0.18, clipped at
1, sRGB curve; the same for truth and render), because the truth's peaks (reflected lamps, Y up to ~100) would dominate
them on linear values and no SDR display shows them:
    skew          skewness over the object (Motoyoshi et al. 2007: positive skew goes with gloss), per view
    p99 - median  contrast of the highlights against the body (Marlow et al. 2012: highlight contrast)
    coverage      share of the object above the truth's 95th percentile of that view (5% for the truth itself;
                  Marlow et al.: highlight coverage)
    clip          share of the object clipped by the display (linear Y x exposure >= 1)
    sharpness     energy of D - blur(D, sigma 1 px) inside the truth's highlights, render / truth (highlight sharpness)
    found         local highlights (display value above its gaussian surround, sigma 3 px, by more than 0.08) of the
                  truth that the image has too (1 px tolerance), as a share of the truth's
    blobs         number of such highlight blobs (connected regions), image / truth

The same cues of the truth rendered with other seeds (test_seed2) give the noise floor.

    python experiments/gloss_percept/eval_gloss.py logs/gp_sphere_r0.05/ckpt_007000.pt ... --out experiments/gloss_percept/results/run1.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "color"))
from gs.color import REC2020_TO_SRGB, encode  # noqa: E402
from gs.dataset import read_pfm  # noqa: E402
from stage1_eval import render_model  # noqa: E402

LUM = np.array([0.2126729, 0.7151522, 0.0721750])
HL_Q, PEAK_Q = 95.0, 99.5
EXPOSURE_MEDIAN = 0.18     # display exposure: the truth's median Y on the object (all test views) -> 0.18 (mid grey)


def blur(Y, sigma=1.0):
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(Y, r, mode="edge")
    p = np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 0, p)
    return np.apply_along_axis(lambda v: np.convolve(v, k, "valid"), 1, p)


def skew(v):
    v = v - v.mean()
    return float((v ** 3).mean() / (v ** 2).mean() ** 1.5)


def display(Y, k):
    """what an SDR display shows: exposure k, clipped at 1, sRGB curve"""
    Y = np.clip(Y * k, 0, 1)
    return np.where(Y <= 0.0031308, 12.92 * Y, 1.055 * np.clip(Y, 0.0031308, None) ** (1 / 2.4) - 0.055)


def cues(x_lin, t_lin, obj, k):
    """x_lin, t_lin: (H,W,3) linear sRGB of the image and of the truth; obj: (H,W) bool; k: display exposure.
    The light amounts (Y ratios) are on linear Y. The image cues (skew, contrast, coverage, sharpness) are on display
    values, since the truth's peaks (reflected lamps, Y up to ~100) would otherwise dominate them and no SDR display
    shows them"""
    Y, T = x_lin @ LUM, t_lin @ LUM
    thr, pk = np.percentile(T[obj], HL_Q), np.percentile(T[obj], PEAK_Q)
    hl, peak = obj & (T >= thr), obj & (T >= pk)
    D, Dt = display(Y, k), display(T, k)
    dthr = np.percentile(Dt[obj], HL_Q)
    # local highlights: display value above its gaussian surround (sigma 3 px) by more than 0.08
    from scipy import ndimage
    loc = lambda A: obj & (A - ndimage.gaussian_filter(A, 3.0) > 0.08)  # noqa: E731
    H, Ht = loc(D), loc(Dt)
    se = ((encode(torch.from_numpy(x_lin)) - encode(torch.from_numpy(t_lin))) ** 2).mean(-1).numpy()
    hp = lambda A: ((A - blur(A)) ** 2)[hl].sum()  # noqa: E731
    return {"se": float(se[obj].mean()), "Y": float(Y[obj].sum()), "Y_t": float(T[obj].sum()),
            "Y_hl": float(Y[hl].sum()), "Y_hl_t": float(T[hl].sum()), "Y_peak": float(Y[peak].sum()), "Y_peak_t": float(T[peak].sum()),
            "skew": skew(D[obj]), "contrast": float(np.percentile(D[obj], 99) - np.median(D[obj])),
            "coverage": float((D[obj] >= dthr).mean()), "clipped": float((Y[obj] * k >= 1).mean()),
            "hp": float(hp(D)), "hp_t": float(hp(Dt)),
            "hl_found": int((ndimage.binary_dilation(H, iterations=1) & Ht).sum()), "hl_t": int(Ht.sum()),
            "hl_blobs": ndimage.label(H)[1], "hl_blobs_t": ndimage.label(Ht)[1]}


def summarize(per_view):
    s = lambda k: sum(d[k] for d in per_view)  # noqa: E731
    mean = lambda k: float(np.mean([d[k] for d in per_view]))  # noqa: E731
    return {"psnr": -10 * math.log10(max(mean("se"), 1e-12)),
            "Y_ratio": s("Y") / s("Y_t"), "Y_ratio_hl": s("Y_hl") / s("Y_hl_t"), "Y_ratio_peak": s("Y_peak") / s("Y_peak_t"),
            "skew": mean("skew"), "contrast": mean("contrast"), "coverage": mean("coverage"), "clipped": mean("clipped"),
            "sharpness": s("hp") / s("hp_t"), "hl_found": s("hl_found") / max(s("hl_t"), 1),
            "hl_blobs": s("hl_blobs") / max(s("hl_blobs_t"), 1)}


def load_truth(datadir, split):
    M = REC2020_TO_SRGB.double().numpy()
    d = Path(datadir)
    meta = json.loads((d / "transforms_test.json").read_text(encoding="utf-8"))
    out, masks = [], []
    for f in meta["frames"]:
        name = Path(f["file_path"]).name
        out.append(read_pfm(str(d / split / name)).astype(np.float64) @ M.T)
        masks.append(read_pfm(str(d / "raw" / "test" / name.replace(".pfm", "_mask.pfm")))[..., 0] > 0.99)
    return np.stack(out), np.stack(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, figs = {}, []
    truth_cache = {}
    for c in a.ckpts:
        name = Path(c).parent.name
        datadir, mode, rend, n = render_model(c)
        if datadir not in truth_cache:
            gt, obj = load_truth(datadir, "test")
            gt2, _ = load_truth(datadir, "test_seed2")
            k = EXPOSURE_MEDIAN / np.median(np.concatenate([(g @ LUM)[o] for g, o in zip(gt, obj)]))
            truth = {"exposure": float(k), "truth": summarize([cues(gt[v], gt[v], obj[v], k) for v in range(len(gt))]),
                     "noise (seed 2)": summarize([cues(gt2[v], gt[v], obj[v], k) for v in range(len(gt))])}
            truth_cache[datadir] = (gt, obj, truth)
        gt, obj, truth = truth_cache[datadir]
        k = truth["exposure"]
        r = summarize([cues(rend[v], gt[v], obj[v], k) for v in range(len(gt))])
        res[name] = {"datadir": datadir, "color_mode": mode, "n_gauss": n, "model": r, **truth}
        figs.append((name, gt, rend, obj, k))
        print(f"\n{name} ({datadir}, {mode}, {n} Gaussians); display exposure x{k:.2f}")
        print(f"{'':16}{'PSNR':>7}{'Y all':>7}{'Y hl':>7}{'Y peak':>8} | display:{'skew':>6}{'p99-med':>9}{'cover':>7}"
              f"{'clip':>7}{'sharp':>7}{'found':>7}{'blobs':>7}")
        for lab, s in (("truth", truth["truth"]), ("noise (seed 2)", truth["noise (seed 2)"]), ("3DGS", r)):
            print(f"{lab:16}{s['psnr']:7.2f}{s['Y_ratio']:7.3f}{s['Y_ratio_hl']:7.3f}{s['Y_ratio_peak']:8.3f} | "
                  f"{s['skew']:14.2f}{s['contrast']:9.3f}{s['coverage']:7.3f}{s['clipped']:7.3f}{s['sharpness']:7.3f}"
                  f"{s['hl_found']:7.2f}{s['hl_blobs']:7.2f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    # figure: two test views per model, truth and render on one exposure, and their luminance histograms
    fig, axs = plt.subplots(len(figs), 5, figsize=(14, 2.9 * len(figs)), squeeze=False)
    for i, (name, gt, rend, obj, k) in enumerate(figs):
        sh = lambda x: np.repeat(display(x @ LUM, k)[..., None], 3, -1)  # noqa: E731
        for j, (t, img) in enumerate((("truth, view 0", gt[0]), ("3DGS, view 0", rend[0]), ("truth, view 3", gt[3]),
                                      ("3DGS, view 3", rend[3]))):
            axs[i, j].imshow(sh(img))
            axs[i, j].set_title(f"{name}: {t}" if j == 0 else t, fontsize=8)
            axs[i, j].axis("off")
        bins = np.logspace(-3, 2, 80)
        axs[i, 4].hist(np.concatenate([(g @ LUM)[o] for g, o in zip(gt, obj)]), bins, histtype="step", label="truth", color="k")
        axs[i, 4].hist(np.concatenate([(x @ LUM)[o] for x, o in zip(rend, obj)]), bins, histtype="step", label="3DGS", color="C1")
        axs[i, 4].set_xscale("log")
        axs[i, 4].set_yscale("log")
        axs[i, 4].set_xlabel("Y (object, all test views)", fontsize=8)
        axs[i, 4].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=110)
    print(f"\n-> {a.out}, {Path(a.out).with_suffix('.png')}")


if __name__ == "__main__":
    main()
