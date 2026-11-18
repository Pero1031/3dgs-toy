# -*- coding: utf-8 -*-
"""ピンホールカメラ。

ワールド座標は Blender と同じ +Z が上。データセットの transform_matrix は
Blender/OpenGL 流儀の camera-to-world（カメラは -Z を向き、+Y が画面上）なので、
ラスタライズ用に OpenCV 流儀（+Z が前方、+Y が画面下、+X が画面右）に変換して持つ。

画素 (i, j) の中心は画像座標 (i + 0.5, j + 0.5)、主点は (W/2, H/2)。
（make_toy_dataset.py のレイ生成と同じ約束）
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Camera:
    R: torch.Tensor  # (3,3) ワールド→カメラの回転   p_cam = R @ p_world + t
    t: torch.Tensor  # (3,)  ワールド→カメラの平行移動
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int
    image: torch.Tensor | None = None  # (H,W,3) 正解画像 [0,1]
    time: float | None = None          # 撮影した時刻 [0,1]（動くシーンのときだけ）

    @property
    def center(self) -> torch.Tensor:
        """カメラ中心のワールド座標 = -Rᵀ t"""
        return -self.R.T @ self.t

    @staticmethod
    def from_blender(c2w, W, H, focal, image=None, device="cuda", time=None) -> "Camera":
        c2w = torch.as_tensor(c2w, dtype=torch.float32).clone()
        c2w[:3, 1:3] *= -1  # OpenGL (+Y上, -Z前) -> OpenCV (+Y下, +Z前): カメラの y, z 軸を反転
        w2c = torch.linalg.inv(c2w)
        return Camera(
            R=w2c[:3, :3].contiguous().to(device),
            t=w2c[:3, 3].contiguous().to(device),
            fx=float(focal), fy=float(focal), cx=W / 2.0, cy=H / 2.0, W=int(W), H=int(H),
            image=None if image is None else image.to(device), time=time,
        )
