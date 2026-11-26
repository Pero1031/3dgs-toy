# -*- coding: utf-8 -*-
"""
Colour experiment, stage 1: colour error against scene-linear ground truth (GPU for rendering, a few seconds per model).

Every model is rendered on the test views and brought to linear sRGB (gs/color.py, negative values kept). The ground
truth is the .pfm itself, converted from Rec.2020 to linear sRGB without clipping. Both go to CIE XYZ and are compared on
the statue in
    Δu'v'      chromaticity (only where the truth has L* >= 20 and the render's chromaticity is defined: Y > 0 and
               X + 15Y + 3Z above 1% of the truth's; a render with negative channels can be black or negative there.
               The number of pixels left out is duv_undefined in the json)
    Δu'v' agg  chromaticity of the summed XYZ of all pixels of a class: render against truth. Noise cancels in the sum,
               a systematic shift does not (per-pixel Δu'v' of the truth's own noise is as large as the effects here)
    ΔE00       with a fixed white: D65 at Y_w = the 99th percentile of the truth's Y on the statue over all test views
               (the same for every model), and the share of its chroma term
    ΔC*ab      signed: below 0, the render is less saturated than the truth
    Y ratio    summed Y of the render / of the truth (below 1: energy lost, mostly in highlights)
    out        share of the pixels the render puts outside sRGB (a negative linear-sRGB channel)
    PSNR       on encode(linear sRGB), the space the loss uses
Pixel classes (on the truth): statue, outside sRGB (a negative linear-sRGB channel), inside sRGB, edge / flat
(a 4-neighbour differs by ΔE*ab > 10, or within 1 px of the silhouette), and outside / inside split by the truth's L*:
out_dim, out_mid, out_hi (L* < 40, 40-70, >= 70), in_dim (inside, L* < 40), out2020 (outside Rec.2020 too), out_in2020 (outside sRGB, inside Rec.2020).
"statue" is the mask of the data: in stage 2 (color_pigment) it covers the statue and the balls.
Two floors on the same pixels:
    clip    the truth against the truth clipped to sRGB (negative channels -> 0): what a renderer whose colours cannot be
            negative in sRGB loses at least, per pixel
    clip2020  the same for Rec.2020 (only when some pixels are outside Rec.2020)
    noise   the truth against the same views rendered with other seeds (test_seed2/)

    python experiments/color/stage1_eval.py logs/cg_display/ckpt_007000.pt logs/cg_lin_srgb/ckpt_007000.pt ...

-> experiments/color/results/stage1.json, experiments/color/figures/stage1_views.png (--out, --fig)
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
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from colorimetry import WHITE_D65, ciede2000, rgb_to_xyz, upvp, xyz_to_lab  # noqa: E402
from gs.color import REC2020_TO_SRGB, SRGB_TO_2020, ColorMode, encode  # noqa: E402
from gs.dataset import load_cameras, read_pfm  # noqa: E402
from gs.gaussians import GaussianModel  # noqa: E402
from gs.render import render  # noqa: E402

CLASSES = ("statue", "outside", "inside", "edge", "flat", "out_dim", "out_mid", "out_hi", "in_dim", "out2020", "out_in2020")
BANDS = (40.0, 70.0)   # L* of the truth: dim < 40 <= mid < 70 <= hi (Y/Y_w = 0.11, 0.41)
EDGE_DE, DARK_L = 10.0, 20.0
SHOW_VIEW = 0


def test_frames(datadir, split="test"):
    meta = json.loads((Path(datadir) / f"transforms_{split}.json").read_text(encoding="utf-8"))
    return [Path(datadir) / f["file_path"] for f in meta["frames"]]


def load_truth(datadir, split="test"):
    """linear sRGB truth, second-seed truth, masks: (V,H,W,3), (V,H,W,3), (V,H,W)"""
    M = REC2020_TO_SRGB.double().numpy()
    gt, gt2, mask = [], [], []
    for p in test_frames(datadir, split):
        gt.append(read_pfm(p).astype(np.float64) @ M.T)
        p2 = p.parent.parent / "test_seed2" / p.name
        gt2.append(read_pfm(p2).astype(np.float64) @ M.T if p2.exists() else None)
        mask.append(read_pfm(p.with_name(p.name.replace("_denoised.pfm", "_mask.pfm")))[..., 0])
    return np.stack(gt), gt2, np.stack(mask)


def to2020(x):
    return x @ SRGB_TO_2020.double().numpy().T


def from2020(x):
    return x @ REC2020_TO_SRGB.double().numpy().T


def classes(gt_lin, lab_g, mask):
    st = mask > 0.5
    jump = np.zeros(st.shape, bool)
    for ax in (0, 1):
        e = (np.linalg.norm(np.diff(lab_g, axis=ax), axis=-1) > EDGE_DE) | (np.diff(st.astype(np.int8), axis=ax) != 0)
        if ax == 0:
            jump[1:] |= e
            jump[:-1] |= e
        else:
            jump[:, 1:] |= e
            jump[:, :-1] |= e
    out = (gt_lin < 0).any(-1)
    out2020 = (to2020(gt_lin) < -1e-6).any(-1)   # tolerance: a truth clipped at 0 in Rec.2020 comes back with -1e-17 after the round trip
    L = lab_g[..., 0]
    dim, hi = L < BANDS[0], L >= BANDS[1]
    return {"statue": st, "outside": st & out, "inside": st & ~out, "edge": st & jump, "flat": st & ~jump,
            "out_dim": st & out & dim, "out_mid": st & out & ~dim & ~hi, "out_hi": st & out & hi, "in_dim": st & ~out & dim,
            "out2020": st & out2020, "out_in2020": st & out & ~out2020}


def compare(gt_lin, x_lin, white):
    """per-pixel differences of x against the truth"""
    Xg, Xx = rgb_to_xyz(gt_lin), rgb_to_xyz(x_lin)
    lg, lx = xyz_to_lab(Xg, white), xyz_to_lab(Xx, white)
    dE, tL, tC, tH = ciede2000(lg, lx)
    den = lambda X: X[..., 0] + 15 * X[..., 1] + 3 * X[..., 2]  # noqa: E731
    ok = (Xx[..., 1] > 0) & (den(Xx) > 0.01 * den(Xg))   # chromaticity of the render is defined (not black / negative)
    return {"duv": np.linalg.norm(upvp(Xx) - upvp(Xg), axis=-1), "duv_ok": ok, "dE": dE, "tC2": tC ** 2, "Xg": Xg, "Xx": Xx,
            "e2": tL ** 2 + tC ** 2 + tH ** 2, "dC": np.hypot(lx[..., 1], lx[..., 2]) - np.hypot(lg[..., 1], lg[..., 2]),
            "se": ((encode(torch.from_numpy(x_lin)) - encode(torch.from_numpy(gt_lin))) ** 2).mean(-1).numpy(),
            "Lg": lg[..., 0], "neg": (x_lin < 0).any(-1).astype(np.float64)}


def summarize(per_view, cls):
    """per_view: list of compare() dicts; cls: list of class dicts"""
    res = {}
    for c in CLASSES:
        sel = [cl[c] for cl in cls]
        n = sum(int(s.sum()) for s in sel)
        if n == 0:
            res[c] = {"pixels": 0}
            continue
        cat = lambda k, extra=None: np.concatenate([d[k][s & (extra[i] if extra else True)] for i, (d, s) in enumerate(zip(per_view, sel))])  # noqa: E731
        bright = [(d["Lg"] >= DARK_L) & d["duv_ok"] for d in per_view]
        duv = cat("duv", bright)
        agg = float(np.linalg.norm(upvp(cat("Xx").sum(0)) - upvp(cat("Xg").sum(0))))
        res[c] = {"pixels": n, "psnr": -10 * math.log10(max(cat("se").mean(), 1e-12)), "duv_agg": agg,
                  "duv_mean": float(duv.mean()) if len(duv) else None,
                  "duv_p95": float(np.percentile(duv, 95)) if len(duv) else None,
                  "duv_undefined": int(sum(int((s & (d["Lg"] >= DARK_L) & ~d["duv_ok"]).sum()) for d, s in zip(per_view, sel))),
                  "dE00_mean": float(cat("dE").mean()),
                  "share_chroma": float(cat("tC2").sum() / max(cat("e2").sum(), 1e-12)),
                  "dCab_mean": float(cat("dC").mean()),
                  "energy": float(cat("Xx")[:, 1].sum() / cat("Xg")[:, 1].sum()),
                  "pred_outside": float(cat("neg").mean())}
    return res


@torch.no_grad()
def render_model(ckpt_path, split="test"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    cm = ColorMode(a.get("color_mode", "display"), ckpt.get("data_space", "display-srgb"))
    model = GaussianModel.from_state_dict(ckpt["model"])
    bg = ckpt["bg"].cuda()
    cams = load_cameras(a["datadir"], split, a["half_res"], ckpt["bg"].tolist(), "cuda", cm.image_fn)
    out = []
    for cam in cams:
        img = render(model, cam, bg, ckpt["sh_degree"], tile=a["tile"], clamp_color=cm.clamp)["image"]
        out.append(cm.to_linear_srgb(img.double()).cpu().numpy())
    return a["datadir"], cm.mode, np.stack(out), len(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", default=str(HERE / "results" / "stage1.json"))
    ap.add_argument("--split", default="test", help="views to evaluate on (train: how well the model fits its own views; the noise floor needs test_seed2 and exists for test only)")
    ap.add_argument("--fig", default=str(HERE / "figures" / "stage1_views.png"))
    args = ap.parse_args()

    models = [(Path(c).parent.name, *render_model(c, args.split)) for c in args.ckpts]
    datadir = models[0][1]
    assert all(m[1] == datadir for m in models), "all checkpoints must be trained on the same data"
    gt, gt2, mask = load_truth(datadir, args.split)
    Yg = rgb_to_xyz(gt)[..., 1]
    white = WHITE_D65 * np.percentile(Yg[mask > 0.5], 99)
    cls = [classes(gt[v], xyz_to_lab(rgb_to_xyz(gt[v]), white), mask[v]) for v in range(len(gt))]

    results = {"datadir": datadir, "split": args.split, "white_Y": float(white[1]), "views": len(gt),
               "frac_outside_srgb": sum(int(c["outside"].sum()) for c in cls) / sum(int(c["statue"].sum()) for c in cls),
               "floors": {}, "models": {}}
    results["floors"]["clip"] = summarize([compare(gt[v], np.clip(gt[v], 0, None), white) for v in range(len(gt))], cls)
    if any(c["out2020"].any() for c in cls):   # what a renderer that cannot leave Rec.2020 loses at least
        results["floors"]["clip2020"] = summarize([compare(gt[v], from2020(np.clip(to2020(gt[v]), 0, None)), white)
                                                   for v in range(len(gt))], cls)
    if all(g is not None for g in gt2):
        results["floors"]["noise"] = summarize([compare(gt[v], gt2[v], white) for v in range(len(gt))], cls)
    for name, _, mode, rend, n in models:
        results["models"][name] = {"color_mode": mode, "n_gauss": n,
                                   **summarize([compare(gt[v], rend[v], white) for v in range(len(gt))], cls)}

    print(f"{datadir}: {results['views']} {args.split} views, {100 * results['frac_outside_srgb']:.1f}% of the statue outside sRGB, "
          f"white Y {results['white_Y']:.3f}")
    print(f"{'':30}{'class':<9}{'pixels':>8}{'PSNR':>7}{'duv':>8}{'p95':>8}{'duv agg':>9}{'dE00':>6}{'C share':>8}{'dCab':>7}{'Y ratio':>8}{'out':>6}")
    rows = [(f"floor: {k}", v) for k, v in results["floors"].items()] + \
           [(f"{k} [{v['color_mode']}]", v) for k, v in results["models"].items()]
    for name, r in rows:
        for c in CLASSES:
            s = r[c]
            if not s.get("pixels"):
                continue
            f = lambda x: "     -" if x is None else f"{x:.4f}"  # noqa: E731
            print(f"{name[:29]:<30}{c:<9}{s['pixels']:8d}{s['psnr']:7.2f}{f(s['duv_mean']):>8}{f(s['duv_p95']):>8}{f(s['duv_agg']):>9}"
                  f"{s['dE00_mean']:6.2f}{s['share_chroma']:8.2f}{s['dCab_mean']:7.2f}{s['energy']:8.2f}{s['pred_outside']:6.2f}")
        print()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")

    # Figure: test view SHOW_VIEW, the truth and each model (display values of the clipped colour), and Δu'v' maps
    v = SHOW_VIEW
    ys, xs = np.nonzero(mask[v] > 0.5)
    cut = (slice(max(ys.min() - 6, 0), ys.max() + 7), slice(max(xs.min() - 6, 0), xs.max() + 7))
    show = lambda x: encode(torch.from_numpy(np.clip(x, 0, None))).numpy()[cut]  # noqa: E731
    st = (mask[v] > 0.5) & (xyz_to_lab(rgb_to_xyz(gt[v]), white)[..., 0] >= DARK_L)
    cols = [("truth", gt[v], None), ("truth clipped to sRGB", np.clip(gt[v], 0, None), gt[v])] + \
           [(f"{name} [{mode}]", rend[v], gt[v]) for name, _, mode, rend, _ in models]
    fig, axs = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.6))
    cm_ = plt.get_cmap("magma").copy()
    cm_.set_bad((1, 1, 1, 0))
    for j, (title, x, ref) in enumerate(cols):
        axs[0, j].imshow(show(x).astype(np.float32))
        axs[0, j].set_title(title, fontsize=8)
        if ref is not None:
            d = np.linalg.norm(upvp(rgb_to_xyz(x)) - upvp(rgb_to_xyz(ref)), axis=-1)
            axs[1, j].imshow(np.where((mask[v] > 0.5) & ~st, 0.75, np.nan)[cut], cmap="gray", vmin=0, vmax=1,
                             interpolation="nearest")   # dark pixels, left out of the per-pixel numbers
            im = axs[1, j].imshow(np.where(st, d, np.nan)[cut], cmap=cm_, vmin=0, vmax=0.01, interpolation="nearest")
        else:
            axs[1, j].imshow(np.where(mask[v] > 0.5, (gt[v] < 0).any(-1), np.nan)[cut], cmap="gray_r", vmin=0, vmax=1,
                             interpolation="nearest")
            axs[1, j].set_title("outside sRGB (black)", fontsize=8)
        axs[0, j].axis("off")
        axs[1, j].axis("off")
    fig.colorbar(im, ax=axs[1, -1], fraction=0.046, pad=0.03, label="Δu'v'")
    fig.suptitle(f"test view {v}: display values of the colour clipped to sRGB (top), Δu'v' to the truth (bottom, 0-0.01; grey: L* < {DARK_L:.0f}, left out)",
                 fontsize=9)
    fig.tight_layout()
    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=110)
    print(f"-> {args.out}, {args.fig}")


if __name__ == "__main__":
    main()
