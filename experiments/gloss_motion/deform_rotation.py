# -*- coding: utf-8 -*-
"""
Does a trained deformation field turn the object as the truth does, and does it turn the Gaussians' orientations
(and so the normals that 4DGS-DR reflects about) with their positions? Runs on the CPU.

The Gaussians are moved to time t as the renderers of Deformable 3D Gaussians and 4DGS-DR do (x + dx,
normalize(q) + dq, s + ds) and compared with the same Gaussians at the reference time t_ref = 0.5:

    positions     Kabsch rotation of the positions (opacity > 0.5, weighted by opacity), component about +y
    orientations  median of the per-Gaussian rotations R_t R_ref^T, component about +y
    normal error  median angle between the shortest axis at t and the one at t_ref turned with the positions
                  (vs pos.), or turned by the truth's rotation (vs truth)

The truth turns about +y through the object's centre by A (t - t_ref) degrees (sweep, A = 60).
A breakdown at t = 1 by distance from the centroid follows (the outer shell is the visible surface).

    python experiments/gloss_motion/deform_rotation.py D:/3dgsdr_work/output/gm_blob1_r0.05_dr 61000
    python experiments/gloss_motion/deform_rotation.py D:/dgs_work/Deformable-3D-Gaussians/output/gm_blob1_r0.05 40000 \
        --code D:/dgs_work/Deformable-3D-Gaussians

Needs the code's DeformNetwork (utils/time_utils.py) and plyfile: run with the environment of the official code.
"""

import argparse
import os
import sys

import numpy as np
import torch
from plyfile import PlyData


def quat_to_mat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    r, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def kabsch(p, q, w):
    """rotation R minimising sum w |R (p - pc) - (q - qc)|^2"""
    w = w / w.sum()
    pc, qc = (w[:, None] * p).sum(0), (w[:, None] * q).sum(0)
    H = ((p - pc) * w[:, None]).T @ (q - qc)
    U, _, Vt = np.linalg.svd(H.numpy().astype(np.float64))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def rotvec(R):
    """rotation vectors (radians) of a batch of 3x3 matrices"""
    th = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1))
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(th)
    return v * np.where(s > 1e-6, th / (2 * np.maximum(s, 1e-6)), 0.5)[:, None]


def wmedian(x, w):
    i = np.argsort(x)
    c = np.cumsum(w[i])
    return x[i][np.searchsorted(c, c[-1] / 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("iteration", type=int)
    ap.add_argument("--code", default=r"D:\3dgsdr_work\3DGS-DR-deform", help="clone that has utils/time_utils.py")
    ap.add_argument("--amplitude", type=float, default=60.0, help="truth: degrees per unit of time")
    ap.add_argument("--t-ref", type=float, default=0.5)
    ap.add_argument("--min-opacity", type=float, default=0.5,
                    help="Gaussians used; 0.0099 for a model saved right after an opacity reset (all at 0.01 or less)")
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    from utils.time_utils import DeformNetwork

    v = PlyData.read(f"{a.model}/point_cloud/iteration_{a.iteration}/point_cloud.ply")["vertex"]
    X = torch.from_numpy(np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32))
    Q = torch.nn.functional.normalize(torch.from_numpy(np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)), dim=-1)
    S = torch.from_numpy(np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32))
    op = torch.from_numpy(1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32))))
    w = op * (op > a.min_opacity)
    W = w.numpy().astype(np.float64)

    net = DeformNetwork(is_blender=True)
    net.load_state_dict(torch.load(f"{a.model}/deform/iteration_{a.iteration}/deform.pth", map_location="cpu"))
    net.eval()
    # models trained with --no_deform_scale ignore the deformation's size output
    cfg = open(f"{a.model}/cfg_args").read() if os.path.exists(f"{a.model}/cfg_args") else ""
    fixed_size = "no_deform_scale=True" in cfg

    @torch.no_grad()
    def deform(t):
        dx, dq, ds = net(X, torch.full((len(X), 1), float(t)))
        x, s, R = X + dx, (S if fixed_size else S + ds), quat_to_mat(Q + dq)
        k = torch.argmin(s.abs(), -1)
        n = R[torch.arange(len(R)), :, k]
        axis[float(t)] = k.numpy()
        ss = np.sort(s.abs().numpy(), 1)
        flat[float(t)] = wmedian(ss[:, 0] / ss[:, 1], W)
        return x, R, n

    axis, flat = {}, {}

    x0, R0, n0 = deform(a.t_ref)
    print(f"{len(X)} Gaussians, {int((W > 0).sum())} with opacity > {a.min_opacity}; reference time {a.t_ref}")
    print(f"{'t':>5} {'truth':>6} {'positions':>10} {'orientations':>13} {'normal err vs pos.':>19} {'vs truth':>9}"
          "   (degrees, turns about +y)")
    for t in np.linspace(0, 1, 11):
        x, R, n = deform(t)
        Rp = kabsch(x0, x, w)
        rv = np.degrees(rotvec((R @ R0.transpose(1, 2)).numpy().astype(np.float64)))
        n_exp = (torch.from_numpy(Rp.astype(np.float32)) @ n0.T).T
        err = np.degrees(np.arccos(np.clip(np.abs((n_exp * n).sum(-1)).numpy(), 0, 1)))
        th = np.radians(a.amplitude * (t - a.t_ref))
        Rt = torch.tensor([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], dtype=torch.float32)
        err_t = np.degrees(np.arccos(np.clip(np.abs(((Rt @ n0.T).T * n).sum(-1)).numpy(), 0, 1)))
        print(f"{t:5.2f} {a.amplitude * (t - a.t_ref):6.1f} {np.degrees(rotvec(Rp[None])[0])[1]:10.1f} "
              f"{wmedian(rv[:, 1], W):13.1f} {wmedian(err, W):19.1f} {wmedian(err_t, W):9.1f}")

    sw = max(np.sum(W * (axis[t] != axis[a.t_ref])) / W.sum() for t in (0.0, 1.0))
    print(f"\nsizes {'fixed over time' if fixed_size else 'changed by the deformation'}; shortest / middle size "
          f"(median) {flat[a.t_ref]:.3f} at t = {a.t_ref}, {flat[0.0]:.3f} at 0, {flat[1.0]:.3f} at 1; "
          f"shortest axis differs from t = {a.t_ref} for up to {100 * sw:.1f}% (opacity-weighted)")

    # per Gaussian at t = 1: the turn of the position about the vertical axis through the centroid, by distance
    c = (w[:, None] * x0).sum(0) / w.sum()
    r0 = x0 - c
    x, R, _ = deform(1.0)
    r = x - c
    da = torch.atan2(r0[:, 2], r0[:, 0]) - torch.atan2(r[:, 2], r[:, 0])
    d = np.degrees((torch.remainder(da + np.pi, 2 * np.pi) - np.pi).numpy())
    oy = np.degrees(rotvec((R @ R0.transpose(1, 2)).numpy().astype(np.float64)))[:, 1]
    rad = r0.norm(dim=-1).numpy()
    ok = (W > 0) & ((r0[:, 0] ** 2 + r0[:, 2] ** 2).sqrt() > 0.1).numpy()
    print(f"\nt = 1 (truth {a.amplitude * (1 - a.t_ref):.1f}), Gaussians 0.1+ from the axis, by distance from the centroid:")
    edges = np.percentile(rad[ok], [0, 25, 50, 75, 90, 100])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (rad >= lo) & (rad <= hi)
        print(f"  r {lo:.3f}-{hi:.3f}: position turn {np.median(d[m]):6.1f} (10-90% {np.percentile(d[m], 10):6.1f}"
              f"..{np.percentile(d[m], 90):6.1f}), orientation turn {np.median(oy[m]):6.1f}  (n={m.sum()})")


if __name__ == "__main__":
    main()
