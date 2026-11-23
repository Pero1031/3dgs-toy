# -*- coding: utf-8 -*-
"""Camera pose utilities (cameras on a sphere looking at the origin). Same as nerf-toy."""

import numpy as np


def look_at_origin(azimuth_deg, elevation_deg, radius):
    """Build a camera-to-world matrix (4,4) that looks at the origin, from azimuth, elevation and distance.

    The world is +Z up (same as Blender). The camera looks along -Z (OpenGL convention).
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
