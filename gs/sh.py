# -*- coding: utf-8 -*-
"""View-dependent color with spherical harmonics (SH).

Each Gaussian has (deg+1)^2 SH coefficients per RGB channel, and for the view direction d
(the unit vector from the camera center to the Gaussian center) the color is
    color(d) = Σ_lm k_lm Y_lm(d) + 0.5
With deg=0 this is a constant color regardless of direction (diffuse color).
Constants and formulas are the same real SH as the official implementation (utils/sh_utils.py).
"""

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def eval_sh(deg, sh, dirs):
    """sh: (N, K, 3) [K >= (deg+1)^2],  dirs: (N, 3) unit vectors  ->  (N, 3)"""
    out = C0 * sh[:, 0]
    if deg < 1:
        return out
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    out = out - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if deg < 2:
        return out
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    out = (out
           + C2[0] * xy * sh[:, 4]
           + C2[1] * yz * sh[:, 5]
           + C2[2] * (2 * zz - xx - yy) * sh[:, 6]
           + C2[3] * xz * sh[:, 7]
           + C2[4] * (xx - yy) * sh[:, 8])
    if deg < 3:
        return out
    out = (out
           + C3[0] * y * (3 * xx - yy) * sh[:, 9]
           + C3[1] * xy * z * sh[:, 10]
           + C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
           + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
           + C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
           + C3[5] * z * (xx - yy) * sh[:, 14]
           + C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return out


def rgb_to_sh0(rgb):
    """Degree-0 coefficients that give the constant color rgb"""
    return (rgb - 0.5) / C0


def sh_basis(deg, dirs):
    """Real SH basis up to an arbitrary degree (N, (deg+1)^2).
    Order and signs are the same as eval_sh (they agree up to degree 3).

    Y_l^m = √2 K_l^m cos(mφ) P_l^m(cosθ)    (m > 0)
          = K_l^0 P_l^0(cosθ)              (m = 0)
          = √2 K_l^|m| sin(|m|φ) P_l^|m|(cosθ)  (m < 0)
    K_l^m = √((2l+1)/(4π) · (l-m)!/(l+m)!). P_l^m is the associated Legendre function with the
    Condon-Shortley phase, computed by recurrence.
    The index is l² + l + m. float64 internally for precision.
    """
    import math

    import torch

    d = dirs.double()
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    phi = torch.atan2(y, x)
    s = torch.sqrt((1 - z * z).clamp(min=0))  # sinθ
    out = torch.zeros(d.shape[0], (deg + 1) ** 2, dtype=torch.float64, device=d.device)
    pmm = torch.ones_like(z)  # P_m^m
    for m in range(deg + 1):
        if m > 0:
            pmm = -(2 * m - 1) * s * pmm
            cm, sm = math.sqrt(2) * torch.cos(m * phi), math.sqrt(2) * torch.sin(m * phi)
        p_prev, p = None, pmm  # P_{l-1}^m, P_l^m (start from l = m)
        for l in range(m, deg + 1):
            if l == m + 1:
                p_prev, p = p, z * (2 * m + 1) * pmm
            elif l > m + 1:
                p_prev, p = p, ((2 * l - 1) * z * p - (l + m - 1) * p_prev) / (l - m)
            k = math.sqrt((2 * l + 1) / (4 * math.pi) * math.exp(math.lgamma(l - m + 1) - math.lgamma(l + m + 1)))
            if m == 0:
                out[:, l * l + l] = k * p
            else:
                kp = k * p
                out[:, l * l + l + m] = cm * kp
                out[:, l * l + l - m] = sm * kp
    return out.to(dirs.dtype)
