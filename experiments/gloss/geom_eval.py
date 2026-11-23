# -*- coding: utf-8 -*-
"""
Compare the position error of the Gaussians against the true mesh (Erato). A simple metric that uses only the
Gaussian centres.
Same computation as D:/GoLD_rayt/out/studies/gs_gloss/geom_eval.py, and the results are also written to JSON.

    python experiments/gloss/geom_eval.py logs/gloss_gold/point_cloud.ply logs/gloss_matte/point_cloud.ply

For Gaussians whose opacity exceeds the threshold:
    accuracy  distance from the centre -> the nearest point on the true surface
    signed    the same distance with a sign (positive = outside the surface, negative = inside. Decided by the normal
              of the nearest triangle)
    floaters  fraction that is 2 cm or more away from the surface
    coverage  distance from a point on the true surface -> the nearest Gaussian centre (is the surface covered?)
Units are mm (the scene is in m, and the statue is 500 mm tall).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OBJ = r"D:\GoLD_rayt\GoLD_rayt\assets\Meshes\erato\erato_closed.obj"
SCALE, ROT_DEG, TRANS = 0.0133, 90.0, np.array([0.0011, 0.0169, 0.0314])  # same placement as the scene in make_views.py
N_SAMPLES = 400000
THRESHOLDS = (0.3, 0.5, 0.7)


def load_mesh():
    v, f = [], []
    with open(OBJ, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    v = np.array(v) * SCALE
    c, s = math.cos(math.radians(ROT_DEG)), math.sin(math.radians(ROT_DEG))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # rotation about the +y axis
    return v @ R.T + TRANS, np.array(f)


def sample_surface(v, f, n, seed=0):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    pts = (1 - r1)[:, None] * a[idx] + (r1 * (1 - r2))[:, None] * b[idx] + (r1 * r2)[:, None] * c[idx]
    nrm = cross[idx] / np.maximum(np.linalg.norm(cross[idx], axis=1), 1e-20)[:, None]
    return pts, nrm


def load_ply(path):
    raw = open(path, "rb").read()
    h = raw.index(b"end_header\n") + len(b"end_header\n")
    hdr = raw[:h].decode("ascii").splitlines()
    n = int([l for l in hdr if l.startswith("element vertex")][0].split()[2])
    props = [l.split()[2] for l in hdr if l.startswith("property")]
    a = np.frombuffer(raw[h:], dtype=np.float32).reshape(n, len(props))
    d = {k: a[:, i] for i, k in enumerate(props)}
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(float), 1.0 / (1.0 + np.exp(-d["opacity"].astype(float)))


def main():
    pts, nrm = sample_surface(*load_mesh(), N_SAMPLES)
    tree = cKDTree(pts)
    mm = 1000.0
    out = {}
    print(f"{'file':<34} {'thr':>4} {'N':>6} | {'acc.mean':>8} {'median':>7} {'p90':>7} | {'signed':>7} | "
          f"{'floaters':>8} | {'cov.mean':>8} {'p90':>7}")
    for path in sys.argv[1:]:
        P, op = load_ply(path)
        name = Path(path).parent.name
        out[name] = {}
        for thr in THRESHOLDS:
            Q = P[op > thr]
            dist, j = tree.query(Q)
            signed = ((Q - pts[j]) * nrm[j]).sum(1)
            cov, _ = cKDTree(Q).query(pts)
            r = {"n": int(len(Q)), "acc_mean": dist.mean() * mm, "acc_median": float(np.median(dist)) * mm,
                 "acc_p90": float(np.percentile(dist, 90)) * mm, "signed_mean": signed[dist < 0.02].mean() * mm,
                 "floaters_pct": 100 * float((dist > 0.02).mean()), "cov_mean": cov.mean() * mm,
                 "cov_p90": float(np.percentile(cov, 90)) * mm}
            out[name][str(thr)] = r
            print(f"{name:<34} {thr:4.1f} {r['n']:6d} | {r['acc_mean']:8.2f} {r['acc_median']:7.2f} {r['acc_p90']:7.2f} | "
                  f"{r['signed_mean']:+7.2f} | {r['floaters_pct']:7.1f}% | {r['cov_mean']:8.2f} {r['cov_p90']:7.2f}")
    dst = Path(__file__).resolve().parent / "geometry.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
