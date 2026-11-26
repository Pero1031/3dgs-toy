# -*- coding: utf-8 -*-
"""
Loader for the Blender synthetic dataset format (same as nerf_synthetic and nerf-toy).

    <datadir>/transforms_{train,val,test}.json  +  images
    {"camera_angle_x": horizontal field of view [rad],
     "frames": [{"file_path": "./train/r_0", "transform_matrix": 4x4 camera-to-world}, ...]}

If file_path has no extension, .png is appended. RGBA images are composited onto the background color with alpha.
A .pfm image is linear float radiance and is kept as it is (values above 1 and negative values included; it has no
alpha, the background is part of the rendering).
Dynamic scenes (D-NeRF format) have "time": time [0,1] in each frame, which goes into Camera.time.
"""

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .camera import Camera


def read_pfm(path):
    """Portable float map -> (H,W,C) float32, top row first"""
    with open(path, "rb") as f:
        kind = f.readline().strip()
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        a = np.fromfile(f, "<f4" if scale < 0 else ">f4").reshape(h, w, 3 if kind == b"PF" else 1)
    return np.ascontiguousarray(a[::-1])  # PFM rows go bottom to top


def _load_image(path, half_res, bg):
    if Path(path).suffix.lower() == ".pfm":
        img = read_pfm(path)
        img = np.repeat(img, 3, -1) if img.shape[-1] == 1 else img
    else:
        img = imageio.imread(path).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        rgb, a = img[..., :3], img[..., 3:]
        img = rgb * a + np.asarray(bg, dtype=np.float32) * (1.0 - a)
    t = torch.from_numpy(np.ascontiguousarray(img[..., :3]))
    if half_res:
        t = torch.nn.functional.interpolate(t.permute(2, 0, 1)[None], scale_factor=0.5,
                                            mode="area")[0].permute(1, 2, 0)
    return t


def load_cameras(datadir, split, half_res=False, bg=(1.0, 1.0, 1.0), device="cuda", image_fn=None):
    """Load one split as a list of Camera (None if it does not exist).
    bg is the background color that RGBA images are composited onto.
    image_fn: applied to each (H,W,3) image tensor after loading (e.g. a change of colour space, gs/color.py)."""
    datadir = Path(datadir)
    meta_path = datadir / f"transforms_{split}.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    cams = []
    for frame in meta["frames"]:
        p = datadir / frame["file_path"]
        if p.suffix == "":
            p = p.with_suffix(".png")
        img = _load_image(p, half_res, bg)
        if image_fn is not None:
            img = image_fn(img)
        H, W = img.shape[:2]
        focal = 0.5 * W / math.tan(0.5 * float(meta["camera_angle_x"]))
        cams.append(Camera.from_blender(frame["transform_matrix"], W, H, focal, img, device,
                                        time=frame.get("time")))
    return cams
