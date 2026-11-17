# -*- coding: utf-8 -*-
"""カメラ姿勢のユーティリティ（原点を注視する球面上のカメラ）。nerf-toy と同じ。"""

import numpy as np


def look_at_origin(azimuth_deg, elevation_deg, radius):
    """方位角・仰角・距離から、原点を向く camera-to-world 行列 (4,4) を作る。

    ワールドは +Z が上（Blenderと同じ）。カメラは -Z 方向を向く（OpenGL 流儀）。
    """
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    pos = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    forward = -pos / np.linalg.norm(pos)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -forward, pos
    return c2w
