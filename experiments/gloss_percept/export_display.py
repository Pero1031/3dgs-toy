# -*- coding: utf-8 -*-
"""
Display images of the truth and of 3DGS renders, for the gloss models of D:\\project\\gloss-perception-lab (lab_cues.py).

Every image is shown as an SDR display would: linear luminance x EXPOSURE (one value for all conditions: same light,
same camera), clipped at 1, sRGB curve. The object is composited onto a mid grey (0.5 in display values, the background
the lab's networks were trained with) with the truth's coverage of each pixel. The scene is achromatic, so the PNG is grey.

    python experiments/gloss_percept/export_display.py --ckpts logs/gp_sphere_r0.05/ckpt_007000.pt ...

Truth: every data/gloss_percept/<shape>__<material>/ with test views (test/ and, where present, test_seed2/).
3DGS: each checkpoint rendered on the test views of its data (GPU, a few seconds each).
-> data/gloss_percept/lab_input/<label>/view_XXX.png, view_XXX_mask.png (coverage > 0.5), manifest.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "color"))
from gs.color import REC2020_TO_SRGB  # noqa: E402
from gs.dataset import read_pfm  # noqa: E402

DATA = ROOT / "data" / "gloss_percept"
OUT = DATA / "lab_input"
LUM = np.array([0.2126729, 0.7151522, 0.0721750])
EXPOSURE = 3.0          # the truth's median object luminance -> 0.18 (eval_gloss.py found x2.99-3.10 on every condition)
BG_DISPLAY = 0.5
BACKGROUND, CROP = "grey", 0.0   # set from the command line


def display(lin):
    Y = np.clip((lin @ LUM) * EXPOSURE, 0, 1)
    return np.where(Y <= 0.0031308, 12.92 * Y, 1.055 * np.clip(Y, 0.0031308, None) ** (1 / 2.4) - 0.055)


def test_names(d):
    meta = json.loads((d / "transforms_test.json").read_text(encoding="utf-8"))
    return [Path(f["file_path"]).name for f in meta["frames"]]


def crop_square(a, cov, fill):
    """square crop in which the object (coverage > 0.5) spans `CROP` of the side, centred on its bounding box,
    resized back to the input size (Serrano's images show the object at about 80% of the frame)"""
    from PIL import Image
    ys, xs = np.where(cov > 0.5)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(round(max(ys.max() - ys.min(), xs.max() - xs.min()) / CROP))
    y0, x0 = int(round(cy - side / 2)), int(round(cx - side / 2))
    H, W = a.shape[:2]
    canvas = np.empty((side, side) + a.shape[2:], a.dtype)
    canvas[...] = fill
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y0 + side, H), min(x0 + side, W)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
    return np.asarray(Image.fromarray(canvas).resize((W, H), Image.LANCZOS))


def write(label, kind, shape, material, d, names, images, rows):
    import imageio.v2 as imageio
    (OUT / label).mkdir(parents=True, exist_ok=True)
    for name, lin in zip(names, images):
        cov = np.clip(read_pfm(str(d / "raw" / "test" / name.replace(".pfm", "_mask.pfm")))[..., 0], 0, 1)
        if BACKGROUND == "env":    # the environment as the renderer saw it behind the object (from the raw truth)
            bg = display(read_pfm(str(d / "raw" / "test" / name)).astype(np.float64))
        else:
            bg = BG_DISPLAY
        img = (np.clip(cov * display(lin) + (1 - cov) * bg, 0, 1) * 255 + 0.5).astype(np.uint8)
        mask = ((cov > 0.5) * 255).astype(np.uint8)
        if CROP:
            fill = int(BG_DISPLAY * 255 + 0.5) if BACKGROUND != "env" else int(np.median(img[cov < 0.5]))
            img, mask = crop_square(img, cov, fill), (crop_square(mask, cov, 0) > 127).astype(np.uint8) * 255
        stem = name.replace(".pfm", "")
        imageio.imwrite(OUT / label / f"{stem}.png", img)
        imageio.imwrite(OUT / label / f"{stem}_mask.png", mask)
        rows.append({"path": f"{label}/{stem}.png", "mask": f"{label}/{stem}_mask.png", "label": label, "kind": kind,
                     "shape": shape, "material": material, "view": stem})


def main():
    global OUT, BACKGROUND, CROP
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--background", choices=("grey", "env"), default="grey",
                    help="grey: mid grey (the lab's networks); env: the environment map behind the object, as Serrano's images")
    ap.add_argument("--crop", type=float, default=0.0, help="crop so that the object spans this share of the frame (0: no crop)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT, BACKGROUND, CROP = Path(a.out), a.background, a.crop
    M = REC2020_TO_SRGB.double().numpy()
    rows = []
    for d in sorted(p for p in DATA.iterdir() if (p / "transforms_test.json").exists()):
        shape, material = d.name.split("__")
        names = test_names(d)
        for split, kind in (("test", "truth"), ("test_seed2", "truth_seed2")):
            if all((d / split / n).exists() for n in names):
                imgs = [read_pfm(str(d / split / n)).astype(np.float64) @ M.T for n in names]
                write(f"{kind}__{d.name}", kind, shape, material, d, names, imgs, rows)
    if a.ckpts:
        from stage1_eval import render_model
        for c in a.ckpts:
            datadir, mode, rend, n = render_model(c)
            d = ROOT / datadir
            shape, material = d.name.split("__")
            write(f"3dgs__{Path(c).parent.name}", "3dgs", shape, material, d, test_names(d), list(rend), rows)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images ({len({r['label'] for r in rows})} sets) -> {OUT}")


if __name__ == "__main__":
    main()
