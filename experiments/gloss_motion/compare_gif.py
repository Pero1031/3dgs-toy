# -*- coding: utf-8 -*-
"""
Side-by-side clips of the fixed test view over time: the truth (test PNGs composited onto black) and the renders of
each model (directories of renders named in the order of the test views, e.g. 3DGS-DR's eval.py --save_images
test/ours_<iter>/renders/rgb, or Deformable 3D Gaussians' render.py test/ours_<iter>/renders), in display values.

    python experiments/gloss_motion/compare_gif.py blob1__coat_r0.05__sweep60 \
        "Deformable 3DGS=D:/dgs_work/Deformable-3D-Gaussians/output/gm_blob1_r0.05/test/ours_40000/renders" \
        "4DGS-DR=D:/3dgsdr_work/output/gm_blob1_r0.05_dr/test/ours_61000/renders/rgb" --out experiments/gloss_motion/results/blob1_r0.05

-> <out>.gif (played forward and back) and <out>_frames.png (frames 0, 10, 18, 29; rows: truth, then each model)
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(p):
    a = imageio.imread(p).astype(np.float64) / 255.0
    rgb = a[..., :3] * a[..., 3:4] if a.shape[-1] == 4 else a[..., :3]
    return (rgb * 255 + 0.5).astype(np.uint8)


def label(img, text):
    """a strip with the label above the image (no font dependency: drawn with matplotlib)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = 22, img.shape[1]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor("black")
    fig.text(0.5, 0.5, text, color="white", ha="center", va="center", fontsize=9)
    fig.canvas.draw()
    strip = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return np.concatenate([strip[:h, :w], img], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("models", nargs="+", help="label=directory of renders")
    ap.add_argument("--png-root", default="data/gloss_motion_png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=12.0)
    a = ap.parse_args()
    d = ROOT / a.png_root / a.name
    names = sorted(p.stem for p in (d / "test").glob("*.png"))
    cols = {"truth": [load(d / "test" / f"{n}.png") for n in names]}
    for m in a.models:
        lab, p = m.split("=", 1)
        files = sorted(Path(p).glob("*.png"))
        assert len(files) == len(names), f"{p}: {len(files)} renders for {len(names)} test views"
        cols[lab] = [load(f) for f in files]
    frames = [np.concatenate([label(cols[k][i], k) for k in cols], 1) for i in range(len(names))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gif = out.parent / (out.name + ".gif")  # not with_suffix: names such as blob1_r0.05 have a dot
    imageio.mimsave(gif, frames + frames[::-1], duration=1000 / a.fps, loop=0)
    pick = [i for i in (0, 10, 18, len(names) - 1) if i < len(names)]
    rows = [np.concatenate([cols[k][i] for i in pick], 1) for k in cols]
    imageio.imwrite(out.parent / (out.name + "_frames.png"), np.concatenate(rows, 0))
    print(f"columns {list(cols)} -> {gif}, {out.parent / (out.name + '_frames.png')} (frames {pick})")


if __name__ == "__main__":
    main()
