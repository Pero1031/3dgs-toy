# -*- coding: utf-8 -*-
"""
Isolate why the shared lighting (env_oracle.py) did not work on this gold statue.

Candidates:
    1. Sub-pixel bumps: when the normal changes a lot within one pixel (about 2.7 mm), the pixel colour is an average
       of reflections from many directions, and cannot be predicted from the reflection direction of the single
       normal at the point
    2. Interreflection and occlusion: what lies in the reflection direction is not a light source but the statue
       itself, and the lighting is not visible
    3. The lighting is not at infinity (even for the same light source, the direction in which it is seen shifts
       with the position of the point)

For each observation, check geometrically whether the ray cast from the point in the reflection direction
    - hits a light source (the 4 spheres and the softbox) (computed at their true positions, so the error of 3 does
      not enter)
    - is occluded by the statue itself on the way (an approximation that represents the statue on a 2 mm grid and
      marches the ray)
and, for each point, measure the normal variation within the extent of a pixel.
Then compare the per-point SH and the shared lighting again, restricted to unoccluded observations and to points
with few bumps.

    python experiments/gloss/env_diagnose.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env_oracle as eo  # noqa: E402
import sh_oracle as so  # noqa: E402

DEV = so.DEV
VOXEL = 0.002        # grid size for the occlusion test [m]
T_START = 0.008      # distance within which the occlusion test is not started, until the ray has left
                     # the grid cells of its own surface [m]
SPREAD_R = 0.00135   # radius for measuring the normal variation [m] (half the width of one pixel)
SPREAD_LOW = 5.0     # points below this count as "points with few bumps" [degrees]
SIDE = [np.array([1.9 * math.sin(math.radians(45 + 90 * k)), 1.2, 1.9 * math.cos(math.radians(45 + 90 * k))])
        for k in range(4)]


def light_hit(P, r):
    """Whether the ray from point P in direction r hits a sphere light (radius 0.35 m) or the softbox
    (y=1.9, 1.8 m square)"""
    hit = torch.zeros(r.shape[:-1], dtype=torch.bool, device=DEV)
    for c in SIDE:
        oc = torch.tensor(c, dtype=torch.float32, device=DEV) - P
        t = (oc * r).sum(-1)
        d2 = (oc * oc).sum(-1) - t * t
        hit |= (t > 0) & (d2 < 0.35 ** 2)
    t = (1.9 - P[..., 1]) / r[..., 1].clamp(min=1e-6)
    x, z = P[..., 0] + t * r[..., 0], P[..., 2] + t * r[..., 2]
    hit |= (r[..., 1] > 0) & (x.abs() < 0.9) & (z.abs() < 0.9)
    return hit


def occupancy(D, lo, hi):
    """Build the grid of the statue's shell from the dense surface points D (dilated by one cell to close gaps)"""
    shape = tuple(int(x) for x in np.ceil((hi - lo) / VOXEL))
    idx = ((D - torch.tensor(lo, device=DEV)) / VOXEL).long()
    for k in range(3):
        idx[:, k] = idx[:, k].clamp(0, shape[k] - 1)
    occ = torch.zeros(shape, device=DEV)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    occ = torch.nn.functional.max_pool3d(occ[None, None], 3, 1, 1)[0, 0] > 0
    return occ, torch.tensor(lo, dtype=torch.float32, device=DEV), shape


def occluded(P, r, cos_rn, occ, lo, shape, step=0.001, t_max=0.8):
    """March from point P in direction r; the ray counts as occluded if it enters the grid of the statue's shell
    after it has moved away from its own surface. The test starts at the larger of T_START and "the distance at which
    the ray is 5 mm away from the surface" (prevents false detections for rays that leave at a grazing angle)"""
    out = torch.zeros(r.shape[0], dtype=torch.bool, device=DEV)
    dims = torch.tensor(shape, device=DEV)
    t0 = torch.maximum(torch.full_like(cos_rn, T_START), 0.005 / cos_rn.clamp(min=0.05))
    for t in np.arange(T_START, t_max, step):
        q = ((P + t * r - lo) / VOXEL).long()
        inside = ((q >= 0) & (q < dims)).all(-1) & (t >= t0)
        if not inside.any():
            if t > t0.max().item():
                break
            continue
        qi = q[inside]
        out[inside] |= occ[qi[:, 0], qi[:, 1], qi[:, 2]]
    return out


def main():
    v, f = so.load_mesh()
    P, N = so.sample_surface(v, f, 10000, seed=1)
    Ns = eo.smooth_normals(10000, seed=1)
    D, ND = so.sample_surface(v, f, 2_000_000, seed=2)

    # For each point, the variation of the normals of the surface points within the width of a pixel
    # (converted from the length of the mean vector to an angle)
    tree = cKDTree(D)
    spread = np.zeros(len(P))
    for i, nb in enumerate(tree.query_ball_point(P, SPREAD_R)):
        m = ND[nb].mean(0) if len(nb) else N[i]
        spread[i] = math.degrees(math.acos(min(1.0, np.linalg.norm(m))))
    print(f"Normal variation within the extent of a pixel: median {np.median(spread):.1f}°, 90th percentile {np.percentile(spread, 90):.1f}°")

    lo, hi = v.min(0) - 0.01, v.max(0) + 0.01
    P, N, Ns, Dt = (torch.tensor(x, dtype=torch.float32, device=DEV) for x in (P, N, Ns, D))
    occ, lo_t, shape = occupancy(Dt, lo, hi)
    masks = {s: so.load_masks(so.ROOT / "data" / "gloss_matte", s) for s in ("train", "test")}

    obs = {}
    for split in ("train", "test"):
        cams = so.load_cameras(so.ROOT / "data" / "gloss_gold", split, False, so.BG, DEV)
        obs[split] = so.observe(cams, masks[split], P, N, Dt)
    keep = obs["train"][0].sum(0) >= so.MIN_TRAIN_OBS
    P, N, Ns = P[keep], N[keep], Ns[keep]
    spread = torch.tensor(spread[keep.cpu().numpy()], device=DEV)
    obs = {s: [x[:, keep] for x in o] for s, o in obs.items()}

    # Reflected ray of each observation: does it hit a light source, is it occluded by the statue
    flags = {}
    for split, (vis, col, dirs) in obs.items():
        r = eo.reflect(dirs, Ns)
        V, n = vis.shape
        Pb = P[None].expand(V, n, 3)
        hit = light_hit(Pb, r)
        cos_rn = (r * Ns[None]).sum(-1)
        occl = occluded(Pb.reshape(-1, 3), r.reshape(-1, 3), cos_rn.reshape(-1), occ, lo_t, shape).reshape(V, n)
        flags[split] = (hit, occl)

    # Compare the brightness of the observations by where the ray goes and by bumps
    vis, col, _ = obs["train"]
    hit, occl = flags["train"]
    lum = col @ so.LUMA.to(DEV)
    low = (spread < SPREAD_LOW)[None].expand_as(vis)
    stats = {}
    print(f"\nLuminance (image values) of the training-view observations ({int(vis.sum())} in total)")
    print(f"{'Where the ray goes':<16} {'Bumps':<10} {'Share':>6} {'Median':>7} {'25%':>6} {'75%':>6}")
    for name, m_dest in (("Light (unoccluded)", hit & ~occl), ("Dark (unoccluded)", ~hit & ~occl), ("Occluded by the statue", occl)):
        for sname, m_sp in (("few", low), ("many", ~low)):
            m = vis & m_dest & m_sp
            x = lum[m].cpu().numpy()
            if len(x) == 0:
                continue
            q = np.percentile(x, [25, 50, 75])
            stats[f"{name}/{sname}"] = {"frac": float(m.sum() / vis.sum()), "q25": q[0], "median": q[1], "q75": q[2]}
            print(f"{name:<16} {sname:<10} {100 * m.sum().item() / vis.sum().item():5.1f}% {q[1]:7.3f} {q[0]:6.3f} {q[2]:6.3f}")

    # Restriction: compare the per-point SH and the shared lighting on unoccluded observations only,
    # and on points with few bumps only
    def subset(pts, drop_occl):
        o = {}
        for split, (v_, c_, d_) in obs.items():
            vv = v_ & ~flags[split][1] if drop_occl else v_.clone()
            o[split] = [vv[:, pts], c_[:, pts], d_[:, pts]]
        ok = o["train"][0].sum(0) >= so.MIN_TRAIN_OBS
        return {s: [x[:, ok] for x in o[s]] for s in o}, Ns[pts][ok]

    def per_point_best(o):
        best = -1.0
        for L in (2, 3, 4, 6):
            c = so.fit(L, *o["train"], smooth=so.SMOOTH_FIG)
            best = max(best, so.psnr(so.predict(L, c, o["test"][2]), o["test"][1], o["test"][0]))
        return best

    def shared(o, Nn, Lp):
        es = eo.fit(o["train"], Nn, Lp, 32, [1.0])
        return eo.evaluate(o["train"], o["test"], Nn, Lp, 32, es)[1.0]["test"]

    allp = torch.ones(len(P), dtype=torch.bool, device=DEV)
    cases = [("All points, all observations", allp, False),
             ("All points, unoccluded observations only", allp, True),
             (f"Points with few bumps (< {SPREAD_LOW:g}°), unoccluded observations only", spread < SPREAD_LOW, True)]
    comp = {}
    print(f"\n{'Set':<36} {'Points':>5} {'Per-point SH':>10} {'Const+shared':>9} {'SH2+shared':>9}  (test-view PSNR [dB])")
    for name, pts, drop in cases:
        o, Nn = subset(pts, drop)
        row = {"n_points": int(Nn.shape[0]), "per_point_sh": per_point_best(o),
               "const_plus_shared": shared(o, Nn, 0), "sh2_plus_shared": shared(o, Nn, 2)}
        comp[name] = row
        print(f"{name:<36} {row['n_points']:>5} {row['per_point_sh']:>10.2f} {row['const_plus_shared']:>9.2f} "
              f"{row['sh2_plus_shared']:>9.2f}", flush=True)

    out = {"spread_deg": {"median": float(spread.median()), "p90": float(torch.quantile(spread, 0.9))},
           "occluded_frac_train": float((occl & vis).sum() / vis.sum()), "lum_stats": stats, "subsets": comp}
    (HERE / "env_diagnose.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {HERE / 'env_diagnose.json'}")


if __name__ == "__main__":
    main()
