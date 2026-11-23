# -*- coding: utf-8 -*-
"""Pinhole camera.

World coordinates are +Z up, same as Blender. The dataset's transform_matrix is a Blender/OpenGL-convention
camera-to-world matrix (the camera looks along -Z, +Y is up on screen), so it is converted to and stored in
the OpenCV convention (+Z forward, +Y down on screen, +X right on screen) for rasterization.

The center of pixel (i, j) is at image coordinates (i + 0.5, j + 0.5), and the principal point is (W/2, H/2).
(Same convention as the ray generation in make_toy_dataset.py.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    R: torch.Tensor  # (3,3) world-to-camera rotation   p_cam = R @ p_world + t
    t: torch.Tensor  # (3,)  world-to-camera translation
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    image: torch.Tensor | None = None  # (H,W,3) ground-truth image [0,1]
    time: float | None = None          # capture time [0,1] (dynamic scenes only)

    @property
    def center(self) -> torch.Tensor:
        """World coordinates of the camera center = -Rᵀ t"""
        return -self.R.T @ self.t

    @staticmethod
    def from_blender(c2w, W, H, focal, image=None, device="cuda", time=None) -> "Camera":
        c2w = torch.as_tensor(c2w, dtype=torch.float32).clone()
        c2w[:3, 1:3] *= -1  # OpenGL (+Y up, -Z forward) -> OpenCV (+Y down, +Z forward): flip the camera's y and z axes
        w2c = torch.linalg.inv(c2w)
        return Camera(
            R=w2c[:3, :3].contiguous().to(device),
            t=w2c[:3, 3].contiguous().to(device),
            fx=float(focal), fy=float(focal), cx=W / 2.0, cy=H / 2.0, W=int(W), H=int(H),
            image=None if image is None else image.to(device), time=time,
        )
