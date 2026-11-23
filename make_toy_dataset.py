# -*- coding: utf-8 -*-
"""
Generate a toy dataset for checking that things work (no external data download needed). Same one as nerf-toy.

A sphere of radius 1 at the origin (with a striped pattern of meridians and parallels on its surface) and a small
cube beside it are ray-traced analytically with numpy and saved in Blender format (transforms_*.json + RGBA PNG).

    python make_toy_dataset.py --out data/toy_sphere
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from gs.poses import look_at_origin

LIGHT = np.array([0.4, -0.5, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
BOX_C, BOX_H = np.array([0.0, 1.4, -0.4]), 0.4  # center and half side length of the cube


def render(c2w, H, W, focal):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dirs = np.stack([(i + 0.5 - W / 2) / focal, -(j + 0.5 - H / 2) / focal, -np.ones_like(i, float)], -1)
    d = dirs @ c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(c2w[:3, 3], d.shape)

    # Intersection with the sphere
    b = (o * d).sum(-1)
    c = (o * o).sum(-1) - 1.0
    disc = b * b - c
    t_s = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
    t_s = np.where(t_s > 0, t_s, np.inf)

    # Intersection with the axis-aligned cube (slab method)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (BOX_C - BOX_H - o) / d
        t2 = (BOX_C + BOX_H - o) / d
    tmin = np.minimum(t1, t2).max(-1)
    tmax = np.maximum(t1, t2).min(-1)
    t_b = np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)

    rgb = np.ones((H, W, 3))
    alpha = np.zeros((H, W))

    hit_s = (t_s < t_b) & np.isfinite(t_s)
    p = o + d * np.where(hit_s, t_s, 0.0)[..., None]
    n = p
    lon = np.arctan2(p[..., 1], p[..., 0])
    lat = np.arcsin(np.clip(p[..., 2], -1, 1))
    stripe = ((np.floor(lon / (np.pi / 6)) + np.floor(lat / (np.pi / 8))) % 2)[..., None]
    base = np.where(stripe > 0, [0.85, 0.25, 0.2], [0.95, 0.85, 0.3])
    shade = 0.25 + 0.75 * np.clip((n * LIGHT).sum(-1, keepdims=True), 0, 1)
    rgb[hit_s] = (base * shade)[hit_s]
    alpha[hit_s] = 1.0

    hit_b = np.isfinite(t_b) & ~hit_s
    p = o + d * np.where(hit_b, t_b, 0.0)[..., None]
    q = (p - BOX_C) / BOX_H
    axis = np.abs(q).argmax(-1)
    n = np.zeros_like(q)
    np.put_along_axis(n, axis[..., None], np.sign(np.take_along_axis(q, axis[..., None], -1)), -1)
    shade = 0.25 + 0.75 * np.clip((n * LIGHT).sum(-1, keepdims=True), 0, 1)
    rgb[hit_b] = (np.array([0.2, 0.45, 0.9]) * shade)[hit_b]
    alpha[hit_b] = 1.0

    return np.concatenate([rgb, alpha[..., None]], -1)


def render_aa(c2w, H, W, focal, s, *args):
    """Render one pixel with s×s rays and average them (removes the aliased (staircase) edges at silhouettes and
    pattern boundaries). Same as render when s=1.

    Colors are multiplied by opacity before averaging, then divided back by the averaged opacity
    (because PNG RGBA holds non-premultiplied color).
    """
    if s == 1:
        return render(c2w, H, W, focal, *args)
    img = render(c2w, H * s, W * s, focal * s, *args).reshape(H, s, W, s, 4)
    a = img[..., 3].mean((1, 3))
    pm = (img[..., :3] * img[..., 3:]).mean((1, 3))
    rgb = np.where(a[..., None] > 0, pm / np.maximum(a[..., None], 1e-12), 1.0)
    return np.concatenate([rgb, a[..., None]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/toy_sphere"))
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--supersample", type=int, default=4,
                    help="render each pixel with N×N rays and average them (with 1, silhouettes get aliased (staircase) edges that Gaussians cannot reproduce)")
    ap.add_argument("--radius", type=float, default=4.0)
    args = ap.parse_args()

    fov = np.deg2rad(40.0)
    focal = 0.5 * args.size / np.tan(0.5 * fov)
    rng = np.random.default_rng(0)

    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_val)]:
        (args.out / split).mkdir(parents=True, exist_ok=True)
        frames = []
        for k in range(n):
            if split == "train":
                az, el = rng.uniform(0, 360), rng.uniform(5, 80)  # random over the upper hemisphere
            else:
                az, el = 360.0 * k / n + 15.0, 30.0
            c2w = look_at_origin(az, el, args.radius)
            img = render_aa(c2w, args.size, args.size, focal, args.supersample)
            imageio.imwrite(args.out / split / f"r_{k}.png", (img * 255).round().astype(np.uint8))
            frames.append({"file_path": f"./{split}/r_{k}", "transform_matrix": c2w.tolist()})
        with open(args.out / f"transforms_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"camera_angle_x": float(fov), "frames": frames}, f, indent=2)
        print(f"{split}: {n} images -> {args.out / split}")


if __name__ == "__main__":
    main()
