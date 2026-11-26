# -*- coding: utf-8 -*-
"""
Colorimetry used by the colour experiment (numpy only).

Display values are interpreted as sRGB (IEC 61966-2-1): the sRGB curve, sRGB primaries, D65 white.
That is what a calibrated sRGB display shows, so the quantities below are "the colour an observer sees on that
display", not the colour of the scene (the renderer's tone mapping sits in between; see README).

    srgb_decode   display value -> linear RGB
    rgb_to_xyz    linear sRGB -> CIE XYZ (Y of the display white = 1)
    xy, upvp      chromaticity (CIE 1931 xy, CIE 1976 u'v')
    xyz_to_lab    CIELAB, reference white D65
    ciede2000     ΔE00 and its three weighted terms (lightness, chroma, hue), Sharma, Wu & Dalal (2005)

Run this file to check ΔE00 against the test pairs published by Sharma et al.
"""

import numpy as np

# linear sRGB -> XYZ, D65 (IEC 61966-2-1)
M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
WHITE_D65 = M_RGB2XYZ.sum(1)  # XYZ of RGB = (1,1,1): (0.95047, 1.0, 1.08883)


def srgb_decode(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(lin):
    return lin @ M_RGB2XYZ.T


def xy(XYZ, eps=1e-12):
    s = XYZ.sum(-1, keepdims=True)
    return XYZ[..., :2] / np.maximum(s, eps)


def upvp(XYZ, eps=1e-12):
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    d = np.maximum(X + 15 * Y + 3 * Z, eps)
    return np.stack([4 * X / d, 9 * Y / d], -1)


def xyz_to_lab(XYZ, white=WHITE_D65):
    t = XYZ / white
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def srgb_to_lab(v):
    return xyz_to_lab(rgb_to_xyz(srgb_decode(v)))


def ciede2000(lab1, lab2):
    """Returns (ΔE00, tL, tC, tH), where tL = ΔL'/(kL SL), tC = ΔC'/(kC SC), tH = ΔH'/(kH SH), kL = kC = kH = 1.
    ΔE00² = tL² + tC² + tH² + RT tC tH, so tL² : tC² : tH² tells whether an error is in lightness, chroma, or hue.
    tL and tC are signed (second minus first): tC < 0 means the second colour is less saturated."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cm7 / (Cm7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    zero = (C1p * C2p) == 0

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(zero, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    Lpm = (L1 + L2) / 2
    Cpm = (C1p + C2p) / 2
    hsum = h1p + h2p
    hpm = np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hpm = np.where(zero, hsum, hpm)
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Cpm7 = Cpm ** 7
    RC = 2 * np.sqrt(Cpm7 / (Cpm7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    SC = 1 + 0.045 * Cpm
    SH = 1 + 0.015 * Cpm * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    tL, tC, tH = dLp / SL, dCp / SC, dHp / SH
    dE = np.sqrt(np.maximum(tL ** 2 + tC ** 2 + tH ** 2 + RT * tC * tH, 0.0))
    return dE, tL, tC, tH


def _self_test():
    # A subset of the 34 test pairs of Sharma, Wu & Dalal (2005), Table 1: (Lab1, Lab2, ΔE00)
    pairs = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
        ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
        ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
        ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ]
    l1 = np.array([p[0] for p in pairs])
    l2 = np.array([p[1] for p in pairs])
    ref = np.array([p[2] for p in pairs])
    got = ciede2000(l1, l2)[0]
    err = np.abs(got - ref).max()
    print("ΔE00 vs Sharma et al.:", np.round(got, 4), "max abs error", f"{err:.1e}")
    assert err < 1e-4
    # white maps to L* = 100, a* = b* = 0, and to the D65 chromaticity
    w = srgb_to_lab(np.ones(3))
    print("sRGB white -> Lab", np.round(w, 4), " xy", np.round(xy(rgb_to_xyz(np.ones(3))), 4))
    assert np.allclose(w, [100, 0, 0], atol=1e-3)


if __name__ == "__main__":
    _self_test()
