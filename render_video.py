# -*- coding: utf-8 -*-
"""
Render novel-view images that go once around the object, with trained Gaussians.

    python render_video.py --ckpt logs/toy/ckpt_007000.pt --gif
"""

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import torch

from gs.camera import Camera
from gs.gaussians import GaussianModel
from gs.poses import look_at_origin
from gs.render import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--elevation", type=float, default=30.0)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=1.0, help="resolution scale relative to the training images")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = GaussianModel.from_state_dict(ckpt["model"])
    bg = ckpt["bg"].cuda()
    W = round(ckpt["camera"]["W"] * args.scale)
    H = round(ckpt["camera"]["H"] * args.scale)
    focal = ckpt["camera"]["focal"] * args.scale
    print(f"Gaussians {len(model)}, {W}x{H}")

    out_dir = Path(args.ckpt).parent / "orbit"
    out_dir.mkdir(exist_ok=True)
    frames, times = [], []
    with torch.no_grad():
        for k in range(args.n_frames):
            c2w = look_at_origin(360.0 * k / args.n_frames, args.elevation, args.radius)
            cam = Camera.from_blender(c2w, W, H, focal)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = render(model, cam, bg, ckpt["sh_degree"])["image"]
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            frame = (img.clamp(0, 1) * 255).round().byte().cpu().numpy()
            imageio.imwrite(out_dir / f"frame_{k:03d}.png", frame)
            frames.append(frame)

    t = sorted(times)[len(times) // 2]
    print(f"render {t * 1000:.1f} ms/image (median, {1 / t:.0f} fps) -> {out_dir}")
    if args.gif:
        imageio.mimsave(out_dir / "orbit.gif", frames, duration=1000 / 30, loop=0)
        print(f"GIF -> {out_dir / 'orbit.gif'}")


if __name__ == "__main__":
    main()
