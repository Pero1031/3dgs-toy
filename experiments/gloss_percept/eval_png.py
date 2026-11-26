# -*- coding: utf-8 -*-
"""
Gloss cues and the CLIP glossiness readout of models trained on the display-referred copy of a data set (to_png.py),
on its test views. Everything is on display values, the space these models are trained in: the truth is the test
PNG composited onto black, its other-seed render (test_seed2) gives the noise floor, and each model's render on black
is quantised to 8 bit like a saved PNG.

    3DGS-DR       a directory of its renders (eval.py --save_images: test/ours_<iter>/renders/rgb/00000.png, ...)
    this repo     a checkpoint of train.py trained on the PNGs (--color-mode display --black-bkgd)

The cues are eval_gloss.cues on these images: display values D are mapped back to linear through the sRGB curve and
the display exposure (to_png.EXPOSURE), so that eval_gloss.display gives D again. Light ratios (Y all / hl / peak) are
then on clipped values and are left out; PSNR is on display values over the object. The CLIP readout prepares the
images as gloss_loss.to_input does (needs transformers, i.e. image-to-3d-diffusion's environment; --clip).

    python experiments/gloss_percept/eval_png.py blob1__coat_r0.05 --dr D:/3dgsdr_work/3DGS-DR/output/gp_blob1_r0.05/test/ours_61000/renders/rgb \
        --ckpt logs/gpp_blob1_r0.05/ckpt_007000.pt --clip --out experiments/gloss_percept/results/png_blob1_r0.05.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from eval_gloss import cues  # noqa: E402
from gs.dataset import read_pfm  # noqa: E402
from to_png import EXPOSURE  # noqa: E402


def lin(D):
    """display value -> linear radiance that eval_gloss.display(., EXPOSURE) maps back to D"""
    D = np.asarray(D, np.float64)
    return np.where(D <= 0.04045, D / 12.92, ((D + 0.055) / 1.055) ** 2.4) / EXPOSURE


def load_png(p):
    a = imageio.imread(p).astype(np.float64) / 255.0
    return a[..., :3] * a[..., 3:4] if a.shape[-1] == 4 else a[..., :3]


@torch.no_grad()
def render_ckpt(path, names):
    from gs.color import ColorMode, data_color_space
    from gs.dataset import load_cameras
    from gs.gaussians import GaussianModel
    from gs.render import render
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    cm = ColorMode(a["color_mode"], data_color_space(a["datadir"]))
    assert a["color_mode"] == "display", "trained on the PNGs in display mode"
    cams = load_cameras(a["datadir"], "test", False, [0.0, 0.0, 0.0], "cuda", cm.image_fn)
    model = GaussianModel.from_state_dict(ck["model"])
    bg = torch.zeros(3, device="cuda")
    out = [render(model, c, bg, ck["sh_degree"], clamp_color=cm.clamp)["image"].clamp(0, 1) for c in cams]
    assert len(out) == len(names)
    return [np.round(o.cpu().numpy() * 255) / 255 for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--dr", nargs="*", default=[], help="3DGS-DR render directories (label=dir or dir)")
    ap.add_argument("--ckpt", nargs="*", default=[], help="train.py checkpoints trained on the PNGs")
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = ROOT / "data" / "gloss_percept_png" / a.name
    src = ROOT / "data" / "gloss_percept" / a.name
    names = [Path(f["file_path"]).name for f in json.loads((d / "transforms_test.json").read_text(encoding="utf-8"))["frames"]]
    cov = [np.clip(read_pfm(str(src / "raw" / "test" / f"{n}_mask.pfm"))[..., 0], 0, 1) for n in names]
    obj = [c > 0.99 for c in cov]
    imgs = {"truth": [load_png(d / "test" / f"{n}.png") for n in names],
            "noise (seed 2)": [load_png(d / "test_seed2" / f"{n}.png") for n in names]}
    for s in a.dr:
        lab, p = s.split("=", 1) if "=" in s else ("3DGS-DR " + Path(s).parents[2].name, s)
        files = sorted(Path(p).glob("*.png"))
        assert len(files) == len(names), f"{p}: {len(files)} renders for {len(names)} views"
        imgs[lab] = [load_png(f) for f in files]
    for c in a.ckpt:
        imgs[f"3DGS {Path(c).parent.name}"] = render_ckpt(c, names)
    gr = None
    if a.clip:
        from gloss_loss import GlossReadout, crop_box, to_input
        gr = GlossReadout("cuda")
    T = imgs["truth"]
    res = {}
    for lab, X in imgs.items():
        per = [cues(lin(X[v]), lin(T[v]), obj[v], EXPOSURE) for v in range(len(names))]
        se = np.mean([((X[v] - T[v]) ** 2).mean(-1)[obj[v]].mean() for v in range(len(names))])
        s = lambda k: sum(p[k] for p in per)  # noqa: E731
        m = lambda k: float(np.mean([p[k] for p in per]))  # noqa: E731
        r = {"psnr_display": -10 * math.log10(max(se, 1e-12)), "skew": m("skew"), "contrast": m("contrast"),
             "coverage": m("coverage"), "clipped": m("clipped"), "sharpness": s("hp") / s("hp_t"),
             "hl_found": s("hl_found") / max(s("hl_t"), 1), "hl_blobs": s("hl_blobs") / max(s("hl_blobs_t"), 1)}
        if gr is not None:
            vals = []
            for v in range(len(names)):
                cv = torch.from_numpy(cov[v]).float().cuda()
                x = torch.from_numpy(lin(X[v])).float().cuda()
                vals.append(gr(to_input(x, cv, crop_box(cv))).item())
            r["clip_gloss"] = float(np.mean(vals))
        res[lab] = r
    print(f"{a.name}: display exposure x{EXPOSURE}")
    print(f"{'':34}{'PSNR':>7}{'skew':>7}{'p99-med':>9}{'cover':>7}{'clip':>7}{'sharp':>7}{'found':>7}{'blobs':>7}"
          + ("  CLIP gloss" if gr is not None else ""))
    for lab, r in res.items():
        print(f"{lab:34}{r['psnr_display']:7.2f}{r['skew']:7.2f}{r['contrast']:9.3f}{r['coverage']:7.3f}{r['clipped']:7.3f}"
              f"{r['sharpness']:7.3f}{r['hl_found']:7.2f}{r['hl_blobs']:7.2f}"
              + (f"{r['clip_gloss']:12.2f}" if "clip_gloss" in r else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
