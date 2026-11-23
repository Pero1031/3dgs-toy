# -*- coding: utf-8 -*-
"""
Stage 2: make sphere data with controlled conditions using GoLD_rayt (do bumps finer than a pixel act as roughness?).

In the statue results (env_rough.py), the roughness optimized per point hardly correlated with the normal variation
within a pixel. On the statue, interreflection, the parallax of near light sources, folds with aligned orientation,
and the denoiser all overlap and cannot be separated, so the same question is posed again with those removed:

    Shape     a sphere of radius 0.2 m (convex, so no interreflection and no self-occlusion). Divided into 8 sectors
              of 45° of longitude each
    Lighting  an HDR environment map only (distant lighting, so no parallax)
    Material  gold (GGX, roughness 0.1 = α 0.01. Same as the statue)
    Sectors   smooth_a / smooth_b : no bumps
              bump3 / bump6 / bump12 : isotropic bumps. The tilt of the normal has a standard deviation of
                                      3° / 6° / 12° per axis, wavelength about 4 mm
              brdf3 / brdf6 / brdf12 : no bumps. The material roughness is set to the GGX α whose normal
                                      distribution has the same half width at half maximum as bump
                                      (a control where the blur comes from material roughness, not from bumps)
    Images    rendered at 512 px (1 pixel = 1.36 mm), then averaged over 2x2 and 4x4 in radiance to also make
              256 px (2.72 mm) and 128 px (5.44 mm). The views stay the same and only the pixel size changes,
              so there are 3 cases: the bumps are larger than, about the same as, or smaller than a pixel

The bumps are an isotropic Gaussian field g, made as a sum of 3D plane waves (K waves, wavelength λ, random direction
and phase) evaluated on the sphere, multiplied by a per-sector amplitude. The normals are computed analytically from
its gradient and written to the PLY as vertex normals (GoLD_rayt interpolates vertex normals within triangles).
GoLD_rayt has no normal maps, so the bumps are baked into a fine mesh (an icosahedron subdivided 9 times, about
5.2 million triangles).

    python experiments/gloss/make_sphere.py mesh                 # write the mesh and check the tilt of the bumps
    python experiments/gloss/make_sphere.py views [--render] [--limit N] [--spp N]
    python experiments/gloss/make_sphere.py post                 # PNGs and transforms_*.json for training 3DGS

-> in data/gloss_sphere/: mesh/*.ply and mesh/check.json, scenes/*.json, render/{train,test}/*.pfm (512 px, linear
   radiance. *_denoised.pfm is after denoising, *_mask.pfm is the sphere mask). The reduction to 256 and 128 px is
   done on the analysis side (sphere_rough.py).
   post writes s{512,256,128}/{train,test}/*.png (display values. The background is replaced with a uniform grey
   BG_DISPLAY), s*/masks/{train,test}/*.png (sphere masks), and s*/transforms_{train,test}.json
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from make_views import CWD, EXE  # noqa: E402

DATA = ROOT / "data" / "gloss_sphere"

# ---- Shape ----
R0 = 0.2                 # radius of the sphere [m]
LAMBDA = 0.004           # wavelength of the plane waves [m] (on the sphere the wavelength is this or longer)
K_WAVES = 256            # number of plane waves
FIELD_SEED = 7
ICO_LEVEL = 9            # number of icosahedron subdivisions (edge length about 0.41 mm, about 10 edges per wavelength)
SECTOR_DEG = 45.0
RAMP_DEG = 4.0           # width over which the bump amplitude is blended smoothly at sector boundaries (one side) [degrees]
LAT_FULL, LAT_ZERO = 60.0, 70.0  # above this latitude the bumps are weakened, reaching 0 at LAT_ZERO (the sectors meet at the poles)
GGX_PER_SLOPE = 1.829    # α of the GGX whose normal distribution has the same half width as a Gaussian normal tilt with standard deviation s (tan) per axis: α = 1.829 s
SECTORS = [              # (name, bump tilt σ_n [degrees], material)
    ("smooth_a", 0.0, "au"),
    ("bump3", 3.0, "au"),
    ("bump6", 6.0, "au"),
    ("bump12", 12.0, "au"),
    ("smooth_b", 0.0, "au"),
    ("brdf3", 0.0, "brdf3"),
    ("brdf6", 0.0, "brdf6"),
    ("brdf12", 0.0, "brdf12"),
]
BRDF_SIGMA = {"brdf3": 3.0, "brdf6": 6.0, "brdf12": 12.0}
AU_ROUGHNESS = 0.1


def brdf_roughness(sig_deg):
    """GoLD_rayt roughness (α = roughness²)"""
    return math.sqrt(GGX_PER_SLOPE * math.tan(math.radians(sig_deg)))


MATERIALS = {"au": {"type": "conductor", "metal": "Au", "roughness": AU_ROUGHNESS}}
for _name, _sig in BRDF_SIGMA.items():
    MATERIALS[_name] = {"type": "conductor", "metal": "Au", "roughness": round(brdf_roughness(_sig), 4)}

# ---- Lighting and cameras ----
ENV_FILE = "assets/env/grace-new.hdr"
ENV_SCALE = 2.0
TARGET = [0.0, 0.0, 0.0]
RADIUS = 1.3             # camera distance (same as the statue data)
FOV = 30.0
SIZE = 512
SCALES = [512, 256, 128]
RINGS = [-30.0, 0.0, 30.0, 60.0]  # elevations of the training views
PER_RING = 24
N_TEST = 12


# ---- Bump field ----

def _waves():
    rng = np.random.default_rng(FIELD_SEED)
    u = rng.normal(size=(K_WAVES, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u / LAMBDA, rng.uniform(0, 2 * math.pi, K_WAVES)


FREQ, PHASE = _waves()


def lon_lat(q):
    """Longitude (from +z toward +x, 0–360°) and latitude (+y is north) [degrees]"""
    lon = np.degrees(np.arctan2(q[..., 0], q[..., 2])) % 360.0
    lat = np.degrees(np.arcsin(np.clip(q[..., 1] / np.linalg.norm(q, axis=-1), -1, 1)))
    return lon, lat


def sector_index(q):
    return (lon_lat(q)[0] // SECTOR_DEG).astype(int) % len(SECTORS)


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def slope_target(q):
    """Target bump tilt at point q (standard deviation per axis, tan). Blended smoothly at sector boundaries and
    at high latitudes"""
    lon, lat = lon_lat(q)
    s = np.array([math.tan(math.radians(sig)) for _, sig, _ in SECTORS])
    k = (lon // SECTOR_DEG).astype(int) % len(SECTORS)
    off = lon - k * SECTOR_DEG                      # position within the sector [0, 45)
    val = s[k].copy()
    lo = off < RAMP_DEG                             # near the left boundary: blend with the sector on the left
    t = (off[lo] + RAMP_DEG) / (2 * RAMP_DEG)
    val[lo] = s[(k[lo] - 1) % len(SECTORS)] + (s[k[lo]] - s[(k[lo] - 1) % len(SECTORS)]) * _smoothstep(t)
    hi = off > SECTOR_DEG - RAMP_DEG                # near the right boundary: blend with the sector on the right
    t = (off[hi] - (SECTOR_DEG - RAMP_DEG)) / (2 * RAMP_DEG)
    val[hi] = s[k[hi]] + (s[(k[hi] + 1) % len(SECTORS)] - s[k[hi]]) * _smoothstep(t)
    return val * (1 - _smoothstep((np.abs(lat) - LAT_FULL) / (LAT_ZERO - LAT_FULL)))


def amplitude(q):
    """Amplitude [m] that multiplies the field g (variance 1). The variance of the tilt per axis is a²(2π/λ)²/3,
    so from the target tilt s, a = s λ √3 / 2π"""
    return slope_target(q) * LAMBDA * math.sqrt(3) / (2 * math.pi)


def field(q, chunk=65536):
    """g(q) and its 3D gradient. q is points on the sphere (n, 3) [m]"""
    g = np.empty(len(q))
    dg = np.empty((len(q), 3))
    c = math.sqrt(2.0 / K_WAVES)
    for i in range(0, len(q), chunk):
        ph = 2 * math.pi * q[i:i + chunk] @ FREQ.T + PHASE
        g[i:i + chunk] = c * np.cos(ph).sum(1)
        dg[i:i + chunk] = -c * 2 * math.pi * np.sin(ph) @ FREQ
    return g, dg


def surface(n0):
    """Position and normal of the bumpy surface at the reference-sphere normals n0 (n, 3).
    Position p = (R0 + h) n0; the normal is n0 − ∇_t h, normalized (h ≪ R0, so the correction for the curvature of
    the sphere is ignored)"""
    n0 = n0 / np.linalg.norm(n0, axis=1, keepdims=True)
    q = R0 * n0
    g, dg = field(q)
    a = amplitude(q)
    eps = 1e-5
    da = np.stack([(amplitude(q + eps * e) - amplitude(q - eps * e)) / (2 * eps) for e in np.eye(3)], 1)
    dh = a[:, None] * dg + g[:, None] * da
    dt = dh - (dh * n0).sum(1, keepdims=True) * n0
    n = n0 - dt
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return q + (a * g)[:, None] * n0, n


def tangent_frame(n0):
    a = np.where(np.abs(n0[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
    t1 = np.cross(a, n0)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(n0, t1)


# ---- Mesh ----

def icosphere(level):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(level):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        key = np.minimum(e[:, 0], e[:, 1]) * len(v) + np.maximum(e[:, 0], e[:, 1])
        uk, inv = np.unique(key, return_inverse=True)
        a, b = uk // len(v), uk % len(v)
        m = v[a] + v[b]
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        mid = (len(v) + inv).reshape(3, -1).T          # indices of the midpoints of the 3 edges of each face
        v = np.concatenate([v, m])
        i0, i1, i2 = f.T
        m01, m12, m20 = mid.T
        f = np.concatenate([np.stack([i0, m01, m20], 1), np.stack([i1, m12, m01], 1),
                            np.stack([i2, m20, m12], 1), np.stack([m01, m12, m20], 1)])
    return v, f


def write_ply(path, P, N, F):
    used, Fi = np.unique(F, return_inverse=True)
    Fi = Fi.reshape(F.shape)
    vert = np.empty(len(used), dtype=[("p", "<f4", 3), ("n", "<f4", 3)])
    vert["p"], vert["n"] = P[used], N[used]
    face = np.empty(len(F), dtype=[("c", "u1"), ("i", "<i4", 3)])
    face["c"], face["i"] = 3, Fi
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(vert)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property float nx\nproperty float ny\nproperty float nz\n"
            f"element face {len(face)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(vert.tobytes())
        fh.write(face.tobytes())


def mesh_groups():
    """Mesh per material (vertices on sector boundaries are duplicated, but their position and normal are the same,
    so no gaps appear)"""
    groups = {}
    for k, (_, _, mat) in enumerate(SECTORS):
        groups.setdefault(mat, []).append(k)
    return groups


def cmd_mesh(a):
    t0 = time.time()
    v, f = icosphere(ICO_LEVEL)
    P, N = surface(v)
    print(f"{len(v)} vertices, {len(f)} triangles ({time.time() - t0:.0f} s)", flush=True)
    sec = sector_index(v[f].mean(1))
    (DATA / "mesh").mkdir(parents=True, exist_ok=True)
    for mat, ks in mesh_groups().items():
        Fm = f[np.isin(sec, ks)]
        write_ply(DATA / "mesh" / f"{mat}.ply", P, N, Fm)
        print(f"  {mat}: sectors {ks}, {len(Fm)} triangles", flush=True)

    # Check 1: near the middle of each sector, is the standard deviation of the normal tilt (per axis) as targeted
    # (analytic normals)
    rng = np.random.default_rng(0)
    rows = {}
    for k, (name, sig, _) in enumerate(SECTORS):
        if sig == 0:
            continue
        lon = np.radians(k * SECTOR_DEG + rng.uniform(10, SECTOR_DEG - 10, 200000))
        lat = np.arcsin(rng.uniform(-math.sin(math.radians(50)), math.sin(math.radians(50)), 200000))
        n0 = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], 1)
        _, n = surface(n0)
        t1, t2 = tangent_frame(n0)
        cz = (n * n0).sum(1)
        sx, sy = (n * t1).sum(1) / cz, (n * t2).sum(1) / cz
        ln = float(np.mean(cz))  # length of the normal averaged over a wide area (≈ E[cos])
        rows[name] = {"target_deg": sig, "slope_std_deg": [float(np.degrees(np.arctan(sx.std()))),
                                                           float(np.degrees(np.arctan(sy.std())))],
                      "mean_cos": ln, "sigma_from_length_deg": float(np.degrees(2 * math.sqrt(max(1 - ln, 0))))}
        print(f"  {name}: target {sig:.1f}°, standard deviation of the tilt {rows[name]['slope_std_deg'][0]:.2f}° / "
              f"{rows[name]['slope_std_deg'][1]:.2f}°, blur converted from E[cos] {rows[name]['sigma_from_length_deg']:.1f}°"
              f" (twice the target = {2 * sig:.0f}°)", flush=True)

    # Check 2: difference between "the vertex normals interpolated within triangles", which the renderer uses, and
    # the analytic normals
    fi = rng.integers(0, len(f), 400000)
    fi = fi[np.isin(sec[fi], [k for k, (_, s, _) in enumerate(SECTORS) if s > 0])]
    b = rng.dirichlet([1, 1, 1], len(fi))
    ni = (N[f[fi]] * b[:, :, None]).sum(1)
    ni /= np.linalg.norm(ni, axis=1, keepdims=True)
    n0 = (v[f[fi]] * b[:, :, None]).sum(1)
    _, na = surface(n0)
    n0 /= np.linalg.norm(n0, axis=1, keepdims=True)
    tilt_i = np.degrees(np.arccos(np.clip((ni * n0).sum(1), -1, 1)))
    tilt_a = np.degrees(np.arccos(np.clip((na * n0).sum(1), -1, 1)))
    err = np.degrees(np.arccos(np.clip((ni * na).sum(1), -1, 1)))
    interp = {"rms_tilt_interpolated_deg": float(np.sqrt((tilt_i ** 2).mean())),
              "rms_tilt_analytic_deg": float(np.sqrt((tilt_a ** 2).mean())),
              "angle_interp_vs_analytic_median_deg": float(np.median(err)),
              "angle_interp_vs_analytic_p99_deg": float(np.percentile(err, 99))}
    print(f"  interpolated vs analytic normals: RMS tilt {interp['rms_tilt_interpolated_deg']:.2f}° / "
          f"{interp['rms_tilt_analytic_deg']:.2f}°, difference in direction: median {interp['angle_interp_vs_analytic_median_deg']:.3f}°, "
          f"99th percentile {interp['angle_interp_vs_analytic_p99_deg']:.3f}°", flush=True)
    disp = np.abs(np.linalg.norm(P, axis=1) - R0)
    out = {"vertices": int(len(v)), "faces": int(len(f)), "sectors": rows, "interpolation": interp,
           "max_displacement_mm": float(disp.max() * 1000), "materials": MATERIALS}
    (DATA / "mesh" / "check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"max displacement {disp.max() * 1000:.2f} mm ({time.time() - t0:.0f} s)")


# ---- Views and scenes ----

def cam_from(elev_deg, azim_deg):
    e, az = math.radians(elev_deg), math.radians(azim_deg)
    return [TARGET[0] + RADIUS * math.cos(e) * math.sin(az), TARGET[1] + RADIUS * math.sin(e),
            TARGET[2] + RADIUS * math.cos(e) * math.cos(az)]


def c2w(frm):
    """camera-to-world (OpenGL convention: the camera looks along -z and y is up on screen. Same as make_views.py)"""
    z = np.subtract(frm, TARGET)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, frm
    return m.tolist()


def views():
    out = []
    for ring, elev in enumerate(RINGS):
        for i in range(PER_RING):
            azim = 360.0 * i / PER_RING + ring * (360.0 / PER_RING) / len(RINGS)
            out.append(("train", f"view_{ring}_{i:03d}", cam_from(elev, azim)))
    rng = random.Random(1)
    for i in range(N_TEST):
        out.append(("test", f"view_{i:03d}", cam_from(rng.uniform(-35.0, 65.0), rng.uniform(0.0, 360.0))))
    return out


def scene(frm, spp, seed, out):
    objs = [{"shape": "mesh", "file": (DATA / "mesh" / f"{mat}.ply").as_posix(), "material": mat}
            for mat in mesh_groups()]
    return {
        "options": {"width": SIZE, "height": SIZE, "spp": spp, "seed": seed, "denoise": True, "out": out,
                    "mask": ",".join(mesh_groups())},
        "camera": {"from": frm, "at": TARGET, "fov": FOV},
        "environment": {"type": "hdr", "file": ENV_FILE, "scale": ENV_SCALE},
        "materials": MATERIALS,
        "objects": objs,
    }


def cmd_views(a):
    for sub in ("scenes", "render/train", "render/test"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, (split, name, frm) in enumerate(views()):
        out = (DATA / "render" / split / f"{name}.pfm").as_posix()
        sc = DATA / "scenes" / f"{split}_{name}.json"
        sc.write_text(json.dumps(scene(frm, a.spp, k + 1, out), indent=1), encoding="utf-8")
        jobs.append((sc, Path(out)))
    print(f"{len(jobs)} view scenes -> {DATA / 'scenes'}")
    if not a.render:
        return
    todo = [j for j in jobs if not j[1].exists()][: a.limit]
    t0 = time.time()
    with open(DATA / "render.log", "a", encoding="utf-8") as log:
        for i, (sc, out) in enumerate(todo):
            t1 = time.time()
            r = subprocess.run([EXE, str(sc)], cwd=CWD, capture_output=True)
            msg = f"{i + 1}/{len(todo)} {sc.name} rc={r.returncode} {time.time() - t1:.1f}s"
            log.write(msg + "\n")
            log.flush()
            print(msg, flush=True)
            if r.returncode != 0 or not out.exists():
                log.write(r.stdout.decode("utf-8", "replace")[-2000:] + "\n" + r.stderr.decode("utf-8", "replace")[-2000:] + "\n")
                raise SystemExit(f"rendering failed: {sc}")
    print(f"{len(todo)} views rendered in {(time.time() - t0) / 60:.1f} min")


BG_DISPLAY = 0.396        # uniform colour placed in the background by post (display value. Same as the statue data)


def cmd_post(a):
    """Average the 512 px radiance down to each pixel size, replace the background with a uniform colour, and write
    PNGs in display values.
    The background is blended in radiance (edge pixels of the sphere keep a little of the environment map that was
    in the original pixels)"""
    import imageio.v2 as imageio
    import torch
    import torch.nn.functional as F

    import env_rough as er
    import sphere_rough as sr

    l_bg = float(er.to_radiance(torch.tensor(BG_DISPLAY, dtype=torch.float64)))
    for scale in SCALES:
        k = SIZE // scale
        root = DATA / f"s{scale}"
        for split in ("train", "test"):
            (root / split).mkdir(parents=True, exist_ok=True)
            (root / "masks" / split).mkdir(parents=True, exist_ok=True)
            frames = []
            for sp, name, frm in views():
                if sp != split:
                    continue
                base = DATA / "render" / split / name
                x = torch.from_numpy(sr.read_pfm(f"{base}.pfm")).double()
                m = torch.from_numpy(sr.read_pfm(f"{base}_mask.pfm")).double()
                x = F.avg_pool2d(x.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)
                m = F.avg_pool2d(m[None, None], k)[0, 0].clamp(0, 1)
                y = er.radiance_to_display(x * m[..., None] + l_bg * (1 - m[..., None]))
                imageio.imwrite(root / split / f"{name}.png", (y * 255).round().byte().numpy())
                imageio.imwrite(root / "masks" / split / f"{name}.png", (m * 255).round().byte().numpy())
                frames.append({"file_path": f"./{split}/{name}", "transform_matrix": c2w(frm)})
            meta = {"camera_angle_x": math.radians(FOV), "frames": frames}
            (root / f"transforms_{split}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{scale} px -> {root}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mesh")
    p = sub.add_parser("views")
    p.add_argument("--spp", type=int, default=512)
    p.add_argument("--render", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    sub.add_parser("post")
    a = ap.parse_args()
    {"mesh": cmd_mesh, "views": cmd_views, "post": cmd_post}[a.cmd](a)


if __name__ == "__main__":
    main()
