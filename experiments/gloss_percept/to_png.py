# -*- coding: utf-8 -*-
"""
A display-referred copy of a gloss data set, for 3DGS codes that read NeRF-synthetic data (8-bit RGBA PNG composited
onto black or white, e.g. 3DGS-DR).

Each view is the raw render (raw/<split>/<view>.pfm: linear, the environment in the background) on the same display
as the CLIP readout and the figures (export_display.py, gloss_loss.py): linear sRGB x EXPOSURE, clipped at 1, sRGB
curve, 8 bit. Alpha is the object's coverage of the pixel (_mask.pfm), so the code composites the object onto its own
background. Highlights brighter than the display's white are clipped, as on the screen. points3d.ply holds the
initial points: random, inside the same cube as train.py --init-extent 0.3 (the codes default to a cube of side 2.6,
far larger than this object).

    python experiments/gloss_percept/to_png.py blob1__coat_r0.05
    -> data/gloss_percept_png/blob1__coat_r0.05/{train,test,test_seed2}/*.png, transforms_*.json, points3d.ply

The dynamic data of experiments/gloss_motion (D-NeRF layout; "time" is kept in every frame):

    python experiments/gloss_percept/to_png.py blob1__coat_r0.05__sweep60 --src-root data/gloss_motion --dst-root data/gloss_motion_png

Raw renders of Blender (motion.json "renderer": "blender") hold the object alone, premultiplied by its coverage; the
PNG holds straight colour, so they are divided by the coverage.
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from gs.color import REC2020_TO_SRGB  # noqa: E402
from gs.dataset import read_pfm  # noqa: E402

EXPOSURE = 3.0
INIT_POINTS, INIT_EXTENT = 20000, 0.3


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.clip(x, 0.0031308, None) ** (1 / 2.4) - 0.055)


def write_ply(path, xyz, rgb):
    """binary PLY with the fields 3DGS reads (x y z nx ny nz red green blue)"""
    v = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                                  ("nz", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = xyz.T
    v["nx"] = v["ny"] = v["nz"] = 0
    v["red"], v["green"], v["blue"] = rgb.T
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(v)
            + "".join(f"property float {n}\n" for n in ("x", "y", "z", "nx", "ny", "nz"))
            + "".join(f"property uchar {n}\n" for n in ("red", "green", "blue")) + "end_header\n")
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(v.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="data set under --src-root, e.g. blob1__coat_r0.05")
    ap.add_argument("--src-root", default="data/gloss_percept")
    ap.add_argument("--dst-root", default="data/gloss_percept_png")
    a = ap.parse_args()
    src = ROOT / a.src_root / a.name
    dst = ROOT / a.dst_root / a.name
    motion = src / "motion.json"
    premult = motion.exists() and json.loads(motion.read_text(encoding="utf-8")).get("renderer") == "blender"
    M = REC2020_TO_SRGB.double().numpy()
    clipped = []
    for split, meta_split in (("train", "train"), ("test", "test"), ("test_seed2", "test")):
        if not (src / "raw" / split).exists():
            continue
        meta = json.loads((src / f"transforms_{meta_split}.json").read_text(encoding="utf-8"))
        (dst / split).mkdir(parents=True, exist_ok=True)
        frames = []
        for f in meta["frames"]:
            name = Path(f["file_path"]).stem
            x = read_pfm(str(src / "raw" / split / f"{name}.pfm")).astype(np.float64) @ M.T
            m = np.clip(read_pfm(str(src / "raw" / split / f"{name}_mask.pfm"))[..., :1].astype(np.float64), 0, 1)
            x = np.where(np.isfinite(x), x, 0)
            if premult:
                x = np.where(m > 1e-3, x / np.maximum(m, 1e-3), 0)
            obj = m[..., 0] > 0.99
            clipped.append((x[obj].max(-1) * EXPOSURE >= 1).mean())
            rgba = np.concatenate([srgb(x * EXPOSURE), m], -1)
            imageio.imwrite(dst / split / f"{name}.png", np.round(rgba * 255).astype(np.uint8))
            frames.append({**f, "file_path": f"{split}/{name}"})
        (dst / f"transforms_{split}.json").write_text(json.dumps({**meta, "color_space": "display-srgb", "frames": frames}, indent=1), encoding="utf-8")
        print(f"{split}: {len(frames)} views")
    rng = np.random.default_rng(0)
    write_ply(dst / "points3d.ply", rng.uniform(-INIT_EXTENT, INIT_EXTENT, (INIT_POINTS, 3)).astype(np.float32),
              rng.integers(0, 2, (INIT_POINTS, 3)).astype(np.uint8))
    print(f"share of object pixels clipped by the display: {np.mean(clipped):.3f}"
          f"{' (premultiplied raw renders divided by the coverage)' if premult else ''} -> {dst}")


if __name__ == "__main__":
    main()
