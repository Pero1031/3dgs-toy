# -*- coding: utf-8 -*-
"""
Colour experiment, stage 0, GPU part: render every test view of trained checkpoints and save them with the ground
truth, so that the colour evaluation (chroma_eval.py) runs on the CPU and can be repeated freely.

    python experiments/color/render_tests.py logs/lego_half_7k/ckpt_007000.pt logs/gloss_gold/ckpt_007000.pt ...

-> <checkpoint dir>/test_renders.npz with
     render (V,H,W,3) float16  the render, not clamped (values above 1 are kept, to count them)
     gt     (V,H,W,3) float16  the ground truth exactly as training sees it (composited on the background, half-res)
     mask   (V,H,W)   float16  object weight: alpha of RGBA images, else <name>_mask.png, else 1
     bg     (3,)               background colour
"""

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from gs.dataset import load_cameras  # noqa: E402
from gs.gaussians import GaussianModel  # noqa: E402
from gs.render import render  # noqa: E402


def load_masks(datadir, split, half_res):
    """Object masks in the same order and at the same resolution as load_cameras"""
    datadir = Path(datadir)
    frames = json.loads((datadir / f"transforms_{split}.json").read_text(encoding="utf-8"))["frames"]
    out = []
    for f in frames:
        p = datadir / f["file_path"]
        if p.suffix == "":
            p = p.with_suffix(".png")
        img = imageio.imread(p)
        mp = p.with_name(p.stem.replace("_denoised", "") + "_mask.png")
        if img.ndim == 3 and img.shape[-1] == 4:
            m = img[..., 3].astype(np.float32) / 255.0
        elif mp.exists():
            m = imageio.imread(mp).astype(np.float32) / 255.0
            m = m[..., 0] if m.ndim == 3 else m
        else:
            m = np.ones(img.shape[:2], np.float32)
        t = torch.from_numpy(m)
        if half_res:  # same area averaging as gs/dataset.py
            t = F.interpolate(t[None, None], scale_factor=0.5, mode="area")[0, 0]
        out.append(t.numpy())
    return np.stack(out)


@torch.no_grad()
def run(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    model = GaussianModel.from_state_dict(ckpt["model"])
    bg = ckpt["bg"].cuda()
    cams = load_cameras(a["datadir"], "test", a["half_res"], ckpt["bg"].tolist())
    renders, gts = [], []
    t0 = time.time()
    for cam in cams:
        img = render(model, cam, bg, ckpt["sh_degree"], tile=a["tile"])["image"]
        renders.append(img.cpu().numpy().astype(np.float16))
        gts.append(cam.image.cpu().numpy().astype(np.float16))
    sec = (time.time() - t0) / len(cams)
    mask = load_masks(a["datadir"], "test", a["half_res"]).astype(np.float16)
    out = Path(ckpt_path).parent / "test_renders.npz"
    np.savez_compressed(out, render=np.stack(renders), gt=np.stack(gts), mask=mask,
                        bg=ckpt["bg"].numpy(), datadir=a["datadir"], ckpt=Path(ckpt_path).as_posix())
    print(f"{ckpt_path}: {len(cams)} views, {sec * 1000:.0f} ms/view -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    for c in ap.parse_args().ckpts:
        run(c)


if __name__ == "__main__":
    main()
