# -*- coding: utf-8 -*-
"""
Preparation for stage 3: can a surface represented by Gaussians hold the normal variation within a pixel (a check
without training)?

In stage 2 (sphere_rough.py), the blur σ = 2√(1 − |n̄|) set from the length |n̄| of the normal averaged within the
pixel footprint (Toksvig's conversion) agreed well with the roughness optimized per point. To do deferred shading
in 3DGS, the straightforward implementation is to composite the normals of the Gaussians that overlap a pixel by
α blending and use the length as |n̄|.

However, α blending is not "the average within the footprint" but "compositing that covers in order from the front",
so when opaque Gaussians overlap, only the normals of the few in front remain. So small disc-shaped Gaussians lying
along the surface are placed at the vertices of the true mesh of the bumpy sphere (make_sphere.py) (about 10 per
bump wavelength), the normals are composited with the 3dgs-toy rasterizer, and the per-pixel |n̄| is compared with
the ground truth (the analytic normals averaged within the box footprint of the pixel).

    Tiling          Gaussians with standard deviation S_TAN along the surface and S_NRM in the normal direction.
                    The opacity is each of OPACITIES
    Compositing     rasterize of 3dgs-toy (normals are composited instead of colour). |n̄| is measured after
                    dividing the composited normal by the accumulated opacity
    Supersampling   render the same Gaussians at SS times the resolution, sum the composited normals over each
                    SS×SS pixels, and then measure |n̄|
                    (averages the footprint on screen. The same idea as mipmapping a normal map)

    python experiments/gloss/splat_normals.py

Output: experiments/gloss/splat_normals.json (the figures are redrawn on the test views by plot_deferred.py)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import make_sphere as ms  # noqa: E402
import sh_oracle as so  # noqa: E402
import sphere_rough as sr  # noqa: E402
from gs.camera import Camera  # noqa: E402
from gs.rasterize import conic_and_radius, project_gaussians, rasterize  # noqa: E402

DEV = sr.DEV
S_TAN = 0.3e-3            # standard deviation along the surface [m] (the vertex spacing is about 0.41 mm)
S_NRM = 0.01e-3           # standard deviation in the normal direction [m] (an almost flat disc)
OPACITIES = [0.99, 0.3]   # opaque (typical when 3DGS represents a surface) and semi-transparent
VIEWS = [f"view_1_{i:03d}" for i in range(0, 24, 4)]  # 6 views on the equator (longitude 4–304°)
SCALES = {512: 2, 256: 4, 128: 4}   # pixel size -> supersampling factor
TRUTH_SAMPLES = 64        # ground truth: number of samples within the pixel box
OUT_JSON = HERE / "splat_normals.json"


def read_ply(path):
    """Read vertices and normals from the format of make_sphere.write_ply (float xyz + nxnynz, little-endian)"""
    with open(path, "rb") as fh:
        n = None
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        v = np.frombuffer(fh.read(n * 24), dtype="<f4").reshape(n, 6)
    return v[:, :3].copy(), v[:, 3:].copy()


def to_camera(frm, scale):
    """Make a 3dgs-toy Camera (world-to-camera) from sphere_rough.Cam (camera-to-world, OpenCV convention)"""
    c = sr.Cam(frm, scale)
    R = c.Rc2w.T.contiguous()
    return Camera(R=R, t=-(R @ c.o), fx=c.f, fy=c.f, cx=c.c, cy=c.c, W=scale, H=scale), c


@torch.no_grad()
def splat_normals(P, N, cam, opacity):
    """Composite normals with disc-shaped Gaussians lying along the surface -> (H, W, 3) Σ w n, (H, W) accumulated
    opacity"""
    face = ((cam.center - P) * N).sum(-1) > 0          # drop those facing away (the far side of the sphere)
    P, N = P[face], N[face]
    cov3d = S_TAN ** 2 * torch.eye(3, device=DEV) + (S_NRM ** 2 - S_TAN ** 2) * N[:, :, None] * N[:, None, :]
    means2d, cov2d, depth = project_gaussians(P, cov3d, cam)
    op = torch.full((len(P),), opacity, device=DEV)
    conic, _, tile_radius = conic_and_radius(cov2d, op)
    img, alpha, _ = rasterize(means2d, conic, op, N, tile_radius, depth, cam.W, cam.H,
                              torch.zeros(3, device=DEV), tile=8, max_entries=2 ** 25)
    return img, alpha


def pool_normals(img, alpha, k):
    """Sum Σ w n and the accumulated opacity over each k×k pixels (averages the footprint on screen)"""
    if k == 1:
        return img, alpha
    s = F.avg_pool2d(img.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)
    a = F.avg_pool2d(alpha[None, None], k)[0, 0]
    return s, a


def sigma_from(sum_n, alpha):
    ln = (sum_n.norm(dim=-1) / alpha.clamp(min=1e-6)).clamp(max=1)
    return 2 * (1 - ln).sqrt()


@torch.no_grad()
def truth_and_select(c, scale, offs):
    """Indices and sectors of the pixels whose centre hits the sphere and lies in the range used (excluding the range
    where the bumps are weakened, the sector boundaries, and places seen too obliquely), and the ground-truth blur σ
    (the analytic normals averaged within the box footprint of the pixel)"""
    j, i = torch.meshgrid(torch.arange(scale, device=DEV), torch.arange(scale, device=DEV), indexing="ij")
    u, v = i.reshape(-1).float() + 0.5, j.reshape(-1).float() + 0.5
    n0 = c.hit_sphere(u, v)
    X = ms.R0 * n0
    cosv = (F.normalize(c.o - X, dim=-1) * n0).sum(-1)
    q = n0.cpu().numpy().astype(np.float64)
    lon, lat = ms.lon_lat(q)
    off = lon % ms.SECTOR_DEG
    sec = (lon // ms.SECTOR_DEG).astype(int)
    ok = ((np.abs(lat) <= sr.LAT_MAX) & (off >= sr.BORDER_MARGIN) & (off <= ms.SECTOR_DEG - sr.BORDER_MARGIN)
          & np.isin(sec, sr.GEOM) & (cosv.cpu().numpy() > so.COS_MIN))
    idx = torch.tensor(np.nonzero(ok)[0], device=DEV)
    amp = torch.tensor(ms.amplitude(ms.R0 * q[ok]), dtype=torch.float32, device=DEV)
    sig = torch.empty(len(idx), device=DEV)
    for b in range(0, len(idx), 2048):
        jj = idx[b:b + 2048]
        n0s = c.hit_sphere(u[jj, None] + offs[None, :, 0], v[jj, None] + offs[None, :, 1])
        sig[b:b + 2048] = sr.mean_normal(n0s, amp[b:b + 2048])[1]
    return idx, sec[ok], sig


def main():
    t0 = time.time()
    P_np, N_np = read_ply(ms.DATA / "mesh" / "au.ply")
    P = torch.tensor(P_np, device=DEV)
    N = F.normalize(torch.tensor(N_np, device=DEV), dim=-1)
    print(f"Gaussians: {len(P)} (vertices of the au sectors of the bumpy sphere)", flush=True)
    s = torch.quasirandom.SobolEngine(2, scramble=True, seed=sr.SEED).draw(TRUTH_SAMPLES).to(DEV) - 0.5
    frm = {n: f for sp, n, f in ms.views() if sp == "train"}
    rows = {}   # (scale, method) -> sector -> list of σ
    for scale, ss in SCALES.items():
        acc = {}
        for name in VIEWS:
            cam, c = to_camera(frm[name], scale)
            idx, sec, sig_true = truth_and_select(c, scale, s)
            res = {"truth": sig_true}
            for op in OPACITIES:
                img, alpha = splat_normals(P, N, cam, op)
                a = alpha.reshape(-1)[idx]
                res[f"splat_o{op}"] = sigma_from(img.reshape(-1, 3)[idx], a)
                res[f"alpha_o{op}"] = a
                cam_ss, _ = to_camera(frm[name], scale * ss)
                img2, alpha2 = pool_normals(*splat_normals(P, N, cam_ss, op), ss)
                res[f"ss{ss}_o{op}"] = sigma_from(img2.reshape(-1, 3)[idx], alpha2.reshape(-1)[idx])
            for k, v in res.items():
                for kk in sr.GEOM:
                    m = torch.tensor(sec == kk, device=DEV)
                    acc.setdefault(k, {}).setdefault(ms.SECTORS[kk][0], []).append(v[m].cpu().numpy())
            print(f"  {scale} px {name}: pixels {len(idx)} ({time.time() - t0:.0f} s)", flush=True)
        out = {}
        for k, d in acc.items():
            out[k] = {}
            for sname, lst in d.items():
                x = np.concatenate(lst)
                if k.startswith("alpha"):
                    out[k][sname] = {"median": float(np.median(x)), "p10": float(np.percentile(x, 10))}
                else:
                    x = np.degrees(x)
                    out[k][sname] = {"mean_deg": float(x.mean()), "median_deg": float(np.median(x)), "n": int(len(x))}
        rows[str(scale)] = {"supersample": ss, "sectors": out}
        print(f"[{scale} px] mean blur per sector [degrees] (ground truth / composited o=0.99 / supersampled o=0.99 / composited o=0.3 / supersampled o=0.3)")
        for kk in sr.GEOM:
            sname = ms.SECTORS[kk][0]
            vals = [out[k][sname]["mean_deg"] for k in
                    ("truth", "splat_o0.99", f"ss{ss}_o0.99", "splat_o0.3", f"ss{ss}_o0.3")]
            print(f"    {sname:<9}" + "".join(f" {x:6.2f}" for x in vals)
                  + f"   median accumulated opacity {out['alpha_o0.99'][sname]['median']:.3f} / {out['alpha_o0.3'][sname]['median']:.3f}")
    OUT_JSON.write_text(json.dumps({"settings": {"s_tan_mm": S_TAN * 1e3, "s_nrm_mm": S_NRM * 1e3,
                                                 "opacities": OPACITIES, "views": VIEWS,
                                                 "truth_samples": TRUTH_SAMPLES, "n_gaussians": len(P)},
                                    "scales": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT_JSON} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
