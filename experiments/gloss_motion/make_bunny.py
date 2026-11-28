# -*- coding: utf-8 -*-
"""
The Stanford bunny as a stimulus shape: data/gloss_percept/mesh/bunny.ply, next to the blobs.

The scan (Stanford 3D Scanning Repository, bunny.tar.gz: bunny/reconstruction/bun_zipper.ply, 35,947 vertices) is
not in this repository; download it and pass its path. Its holes (four in the base, one under the chin) are closed with
a fan around each hole's centroid. The bunny is y-up like the scenes; it is centred on the camera target (the middle of
its bounding box), so the vertical rotation axis of make_motion.py runs through the middle of the box, and scaled to a
bounding-sphere radius of --radius (default: largest side 0.43 m, as blob1).

    python experiments/gloss_motion/make_bunny.py D:/assets/stanford_bunny/bunny/reconstruction/bun_zipper.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gloss_percept"))
sys.path.insert(0, str(HERE.parent / "gloss"))
import make_stimuli as ms  # noqa: E402
from make_sphere import write_ply  # noqa: E402


def read_ascii_ply(path):
    with open(path, "r") as fh:
        n_v = n_f = 0
        while True:
            line = fh.readline().split()
            if line[:2] == ["element", "vertex"]:
                n_v = int(line[2])
            elif line[:2] == ["element", "face"]:
                n_f = int(line[2])
            elif line == ["end_header"]:
                break
        V = np.loadtxt(fh, max_rows=n_v)[:, :3]
        F = np.loadtxt(fh, max_rows=n_f, dtype=np.int64)
    assert (F[:, 0] == 3).all(), "triangles only"
    return V, F[:, 1:]


def fill_holes(V, F):
    """close every boundary loop with a fan around its centroid, keeping the faces' orientation"""
    he = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    have = set(map(tuple, he))
    nxt = {a: b for a, b in he if (b, a) not in have}          # boundary half-edges a -> b
    V, F = list(V), [F]
    loops = []
    while nxt:
        a0 = next(iter(nxt))
        loop = [a0]
        while True:
            b = nxt.pop(loop[-1])
            if b == a0:
                break
            loop.append(b)
        c = len(V)
        V.append(np.mean([V[i] for i in loop], axis=0))
        F.append(np.array([[loop[(k + 1) % len(loop)], loop[k], c] for k in range(len(loop))]))
        loops.append(len(loop))
    return np.array(V), np.concatenate(F), loops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="bun_zipper.ply of the Stanford 3D Scanning Repository")
    ap.add_argument("--radius", type=float, default=0.29, help="bounding-sphere radius about the box centre, m (0.29: box 0.43 m, as blob1)")
    a = ap.parse_args()

    V, F = read_ascii_ply(a.scan)
    V, F, loops = fill_holes(V, F)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert (cnt == 2).all(), "not closed after filling"
    V = V - (V.min(0) + V.max(0)) / 2
    V *= a.radius / np.linalg.norm(V, axis=1).max()
    vol = np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6
    if vol < 0:                                                   # make the faces face outwards
        F = F[:, ::-1]
    N = ms.vertex_normals(V, F)
    out = ms.DATA / "mesh" / "bunny.ply"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, V, N, F)
    print(f"filled {len(loops)} holes ({loops} boundary vertices); {len(V)} vertices, {len(F)} faces, closed; "
          f"box {np.ptp(V, 0).round(3)} m, volume {abs(vol):.4f} m^3 -> {out}")


if __name__ == "__main__":
    main()
