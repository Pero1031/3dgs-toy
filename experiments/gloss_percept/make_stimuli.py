# -*- coding: utf-8 -*-
"""
Gloss perception of 3DGS reconstructions: stimuli rendered with GoLD_rayt (RGB build, CPU).

The question is how much the errors of 3DGS (highlights reproduced with too little energy, view dependence that is too
weak) change perceived gloss. The stimuli therefore follow gloss psychophysics rather than the earlier gold statue:

    colour     none. Achromatic materials under an achromatic environment (the luminance of an HDR light probe), so
               that colour cannot carry or confound the gloss cues. The pipeline stays the one of the colour
               experiment, so colour can be added back as a factor later.
    shape      a factor, from simple to complex: a sphere and smooth "blobs" (a sphere displaced along its normal by a
               low-frequency random field, two amplitudes). Convex enough to have little interreflection.
    light      a distant HDR environment only (no parallax), as in Fleming, Dror & Adelson 2003.
    material   a matte reference, a clear coat over a grey diffuse base at several coat roughnesses (the physical
               ruler of gloss), and measured MERL tables that are available as .hpbpca.

    python experiments/gloss_percept/make_stimuli.py env        # achromatic copy of the environment map
    python experiments/gloss_percept/make_stimuli.py mesh       # sphere and blob meshes
    python experiments/gloss_percept/make_stimuli.py trial [--size 256 --spp 512]   # one view per shape x material
    python experiments/gloss_percept/make_stimuli.py views SHAPE MATERIAL          # multi-view data for 3DGS

-> data/gloss_percept/ (env/, mesh/, trial/, <shape>__<material>/ in the layout of the colour experiment:
   {train,test,test_seed2}/*.pfm on a uniform background, raw/ with the environment, transforms_{train,test}.json)
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "gloss"))
from make_sphere import icosphere, write_ply  # noqa: E402
from make_views import CWD, EXE  # noqa: E402  (RGB build)

DATA = ROOT / "data" / "gloss_percept"

# ---- Environment ----
ENV_SRC = "assets/env/grace-new.hdr"      # relative to CWD (GoLD_rayt)
ENV_GRAY = DATA / "env" / "grace_gray.hdr"
ENV_SCALE = 2.0

# ---- Shapes (radius and field in metres; the sphere and camera are those of the gloss sphere data) ----
R = 0.2
MESH_LEVEL = 7                            # icosphere subdivisions: 163,842 vertices
BLOB_WAVES = 10                           # plane waves of the displacement field
BLOB_FREQ = 0.9                           # cycles per unit of the unit sphere (one or two bulges across)
BLOB_SEED = 3
SHAPES = {"sphere": 0.0, "blob1": 0.10, "blob2": 0.20}   # standard deviation of the displacement / R

# ---- Materials ----
BASE_ALBEDO = 0.2
COAT_IOR = 1.5
MATERIALS = {"matte": {"type": "diffuse", "albedo": [BASE_ALBEDO] * 3}}
for _r in (0.05, 0.15, 0.3, 0.5):         # GoLD_rayt roughness (alpha = roughness^2)
    MATERIALS[f"coat_r{_r:g}"] = {"type": "layered", "coat": {"ior": COAT_IOR, "roughness": _r},
                                  "base": {"albedo": [BASE_ALBEDO] * 3}}
# Measured MERL tables available in GoLD_rayt. white-marble is not achromatic (the measured table is yellowish, visible
# in the trial), so it is kept only for the trial.
for _m in ("white-marble", "black-obsidian"):
    MATERIALS[f"merl_{_m}"] = {"type": "measured", "file": f"assets/hpbrdf/merl/{_m}.hpbpca"}

# ---- Multi-view data (the views of the colour experiment: 3 rings of 30 training views, 10 random test views) ----
TRAIN_RINGS = (15.0, 35.0, 55.0)
PER_RING = 30
N_TEST = 10
SPP_TRAIN = 512
SPP_TEST = 1024                           # the test views are the ground truth of the evaluation: less noise
BG_LINEAR = 0.15                          # uniform background replacing the environment (display 0.3965)

# ---- Camera ----
TARGET = [0.0, 0.0, 0.0]
RADIUS = 1.3
FOV = 30.0
TRIAL_VIEW = (15.0, 30.0)                 # elevation, azimuth (degrees)


# ---------------------------------------------------------------- Radiance .hdr (RGBE) read / write

def read_hdr(path):
    data = Path(path).read_bytes()
    i = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", i)
    dims = data[i:line_end].split()
    h, w = int(dims[1]), int(dims[3])
    assert dims[0] == b"-Y" and dims[2] == b"+X", dims
    p = line_end + 1
    out = np.empty((h, w, 4), np.uint8)
    buf = np.frombuffer(data, np.uint8)
    for y in range(h):
        if buf[p] == 2 and buf[p + 1] == 2 and (int(buf[p + 2]) << 8 | int(buf[p + 3])) == w:   # new RLE
            p += 4
            for c in range(4):
                x = 0
                while x < w:
                    n = int(buf[p])
                    p += 1
                    if n > 128:
                        n -= 128
                        out[y, x:x + n, c] = buf[p]
                        p += 1
                    else:
                        out[y, x:x + n, c] = buf[p:p + n]
                        p += n
                    x += n
        else:                                                                           # flat
            out[y] = buf[p:p + 4 * w].reshape(w, 4)
            p += 4 * w
    e = out[..., 3].astype(np.int32)
    f = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return out[..., :3].astype(np.float64) * f[..., None]


def write_hdr(path, img):
    img = np.clip(np.asarray(img, np.float64), 0, None)
    m = img.max(axis=-1)
    mant, e = np.frexp(m)
    scale = np.where(m > 1e-32, mant * 256.0 / np.maximum(m, 1e-300), 0.0)
    rgbe = np.zeros(img.shape[:2] + (4,), np.uint8)
    rgbe[..., :3] = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    h, w = img.shape[:2]
    with open(path, "wb") as fh:
        fh.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode("ascii"))
        fh.write(rgbe.tobytes())       # flat scanlines (readers take them when a line does not start with 2 2)


def cmd_env(a):
    src = Path(CWD) / ENV_SRC
    img = read_hdr(src)
    Y = img @ np.array([0.2126729, 0.7151522, 0.0721750])
    ENV_GRAY.parent.mkdir(parents=True, exist_ok=True)
    write_hdr(ENV_GRAY, np.repeat(Y[..., None], 3, axis=-1))
    back = read_hdr(ENV_GRAY)[..., 0]
    print(f"{src.name} {img.shape[1]}x{img.shape[0]}: mean Y {Y.mean():.4f}, max {Y.max():.1f}; "
          f"written {ENV_GRAY} (round trip: max rel. error {np.max(np.abs(back - Y) / np.maximum(Y, 1e-6)):.4f})")


# ---------------------------------------------------------------- Meshes

def blob_field(u):
    """smooth random field on the unit sphere, zero mean and unit standard deviation over the sphere"""
    rng = np.random.default_rng(BLOB_SEED)
    k = rng.normal(size=(BLOB_WAVES, 3))
    k *= BLOB_FREQ * 2 * math.pi / np.linalg.norm(k, axis=1, keepdims=True)
    ph = rng.uniform(0, 2 * math.pi, BLOB_WAVES)
    g = np.cos(u @ k.T + ph).sum(axis=1)
    return (g - g.mean()) / g.std()


def vertex_normals(P, F):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])   # area-weighted
    N = np.zeros_like(P)
    for i in range(3):
        np.add.at(N, F[:, i], n)
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def cmd_mesh(a):
    (DATA / "mesh").mkdir(parents=True, exist_ok=True)
    u, F = icosphere(MESH_LEVEL)
    g = blob_field(u)
    for name, amp in SHAPES.items():
        r = R * (1 + amp * g)
        P = u * r[:, None]
        N = u if amp == 0 else vertex_normals(P, F)
        write_ply(DATA / "mesh" / f"{name}.ply", P, N, F)
        tilt = np.degrees(np.arccos(np.clip((N * u).sum(1), -1, 1)))
        print(f"{name}: radius {r.min():.3f}-{r.max():.3f} m, normal tilt from radial: median {np.median(tilt):.1f} deg, "
              f"95% {np.percentile(tilt, 95):.1f} deg, max {tilt.max():.1f} deg; {len(P)} vertices")


# ---------------------------------------------------------------- Scenes and renders

def cam_from(elev_deg, azim_deg):
    e, az = math.radians(elev_deg), math.radians(azim_deg)
    return [RADIUS * math.cos(e) * math.sin(az), RADIUS * math.sin(e), RADIUS * math.cos(e) * math.cos(az)]


def scene(shape, mat, frm, size, spp, seed, out):
    return {
        "options": {"width": size, "height": size, "spp": spp, "seed": seed, "denoise": False, "out": out,
                    "mask": "obj"},
        "camera": {"from": frm, "at": TARGET, "fov": FOV},
        "environment": {"type": "hdr", "file": ENV_GRAY.as_posix(), "scale": ENV_SCALE},
        "materials": {"obj": MATERIALS[mat]},
        "objects": [{"shape": "mesh", "file": (DATA / "mesh" / f"{shape}.ply").as_posix(), "material": "obj"}],
    }


def render(sc_path, out):
    t = time.time()
    r = subprocess.run([EXE, str(sc_path)], cwd=CWD, capture_output=True)
    if r.returncode != 0 or not Path(out).exists():
        print(r.stdout.decode("utf-8", "replace")[-1500:], r.stderr.decode("utf-8", "replace")[-1500:])
        raise SystemExit(f"rendering failed: {sc_path}")
    return time.time() - t


def cmd_trial(a):
    from gs.dataset import read_pfm
    import imageio.v2 as imageio
    from gs.color import srgb_curve
    import torch

    d = DATA / "trial"
    d.mkdir(parents=True, exist_ok=True)
    frm = cam_from(*TRIAL_VIEW)
    imgs, times = {}, []
    for shape in SHAPES:
        for mat in MATERIALS:
            out = (d / f"{shape}__{mat}.pfm").as_posix()
            if not Path(out).exists() or a.force:
                sc = d / f"{shape}__{mat}.json"
                sc.write_text(json.dumps(scene(shape, mat, frm, a.size, a.spp, 1, out), indent=1), encoding="utf-8")
                times.append(render(sc, out))
                print(f"{shape:7} {mat:22} {times[-1]:6.1f} s", flush=True)
            imgs[shape, mat] = read_pfm(out).astype(np.float64)
    if times:
        print(f"{len(times)} renders, {np.mean(times):.1f} s each at {a.size} px, {a.spp} spp")
    # display: one exposure for all (the matte sphere's median luminance on the object -> 0.18), sRGB curve, clip
    m = read_pfm(str(d / "sphere__matte_mask.pfm"))[..., 0] > 0.5 if (d / "sphere__matte_mask.pfm").exists() else None
    Y = imgs["sphere", "matte"] @ np.array([0.2126729, 0.7151522, 0.0721750])
    k = 0.18 / np.median(Y[m] if m is not None else Y)
    rows = [np.concatenate([imgs[s, mt] for mt in MATERIALS], 1) for s in SHAPES]
    disp = srgb_curve(torch.from_numpy(np.clip(np.concatenate(rows, 0) * k, 0, 1))).numpy()
    imageio.imwrite(d / "trial.png", (disp * 255 + 0.5).astype(np.uint8))
    print(f"exposure x{k:.3f}; columns {list(MATERIALS)}, rows {list(SHAPES)} -> {d / 'trial.png'}")


def c2w(frm):
    """camera-to-world (OpenGL convention: the camera looks along -z and y is up on screen)"""
    z = np.array(frm, float) - np.array(TARGET, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, frm
    return m.tolist()


def views():
    import random
    out = []
    for ring, elev in enumerate(TRAIN_RINGS):
        for i in range(PER_RING):
            azim = 360.0 * i / PER_RING + ring * (360.0 / PER_RING) / len(TRAIN_RINGS)
            out.append(("train", f"view_{ring}_{i:03d}", cam_from(elev, azim)))
    rng = random.Random(1)
    tests = [("test", f"view_{i:03d}", cam_from(rng.uniform(12.0, 65.0), rng.uniform(0.0, 360.0))) for i in range(N_TEST)]
    return out + tests + [("test_seed2", n, f) for _, n, f in tests]


def write_pfm(path, img):
    img = np.ascontiguousarray(np.asarray(img, "<f4")[::-1])
    with open(path, "wb") as fh:
        fh.write(f"PF\n{img.shape[1]} {img.shape[0]}\n-1.0\n".encode("ascii"))
        fh.write(img.tobytes())


def cmd_views(a):
    """Render one condition (shape x material) from every view, then composite onto a uniform background.
    raw/<split>/<view>.pfm is linear Rec.2020 (--gamut rec2020; the scene is achromatic, so this equals linear sRGB) with
    the environment in the background; <split>/<view>.pfm replaces the background by BG_LINEAR in radiance, weighted by
    the object's coverage of the pixel (_mask.pfm), so edge pixels keep their share of the environment."""
    from gs.dataset import read_pfm

    name = f"{a.shape}__{a.material}"
    root = DATA / name
    for sub in ("scenes", "raw/train", "raw/test", "raw/test_seed2", "train", "test", "test_seed2"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    jobs, frames = [], {"train": [], "test": []}
    for k, (split, vname, frm) in enumerate(views()):
        if a.test_only and split != "test":
            continue                      # truth on the test views only (the ladder of gloss levels; no 3DGS)
        spp = SPP_TRAIN if split == "train" else SPP_TEST
        raw = (root / "raw" / split / f"{vname}.pfm").as_posix()
        sc = root / "scenes" / f"{split}_{vname}.json"
        sc.write_text(json.dumps(scene(a.shape, a.material, frm, a.size, spp, k + 1, raw), indent=1), encoding="utf-8")
        jobs.append((sc, Path(raw), root / split / f"{vname}.pfm"))
        if split in frames:
            frames[split].append({"file_path": f"{split}/{vname}.pfm", "transform_matrix": c2w(frm)})
    for split, fr in frames.items():
        if not fr:
            continue
        (root / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": math.radians(FOV), "color_space": "linear-rec2020", "frames": fr}, indent=1),
            encoding="utf-8")
    todo = [j for j in jobs if not j[1].exists()]
    print(f"{name}: {len(jobs)} views, {len(todo)} to render", flush=True)
    t0 = time.time()
    with open(root / "render.log", "a", encoding="utf-8") as log:
        for i, (sc, raw, _) in enumerate(todo):
            t1 = time.time()
            r = subprocess.run([EXE, str(sc), "--gamut", "rec2020"], cwd=CWD, capture_output=True)
            msg = f"{i + 1}/{len(todo)} {sc.name} rc={r.returncode} {time.time() - t1:.1f}s"
            log.write(msg + "\n")
            log.flush()
            if i % 10 == 0 or i == len(todo) - 1:
                print(msg, flush=True)
            if r.returncode != 0 or not raw.exists():
                log.write(r.stderr.decode("utf-8", "replace")[-1500:] + "\n")
                raise SystemExit(f"rendering failed: {sc}")
    if todo:
        print(f"{len(todo)} views rendered in {(time.time() - t0) / 60:.1f} min", flush=True)
    # composite onto the uniform background, and report anything odd
    for _, raw, dst in jobs:
        x = read_pfm(str(raw)).astype(np.float64)
        m = np.clip(read_pfm(str(raw).replace(".pfm", "_mask.pfm"))[..., :1].astype(np.float64), 0, 1)
        y = x * m + BG_LINEAR * (1 - m)
        bad = ~np.isfinite(x).all(-1)
        if bad.any() or (x < -1e-4).any():
            print(f"  warning {raw.name}: {bad.sum()} non-finite pixels, min {np.nanmin(x):.4g}")
        write_pfm(dst, np.where(np.isfinite(y), y, BG_LINEAR))
    print(f"composited {len(jobs)} views onto {BG_LINEAR} -> {root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    sub.add_parser("mesh")
    t = sub.add_parser("trial")
    t.add_argument("--size", type=int, default=256)
    t.add_argument("--spp", type=int, default=512)
    t.add_argument("--force", action="store_true")
    v = sub.add_parser("views")
    v.add_argument("shape", choices=list(SHAPES))
    v.add_argument("material", choices=list(MATERIALS))
    v.add_argument("--size", type=int, default=256)
    v.add_argument("--test-only", action="store_true", help="only the 10 test views (truth for the ladder of gloss levels)")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    {"env": cmd_env, "mesh": cmd_mesh, "trial": cmd_trial, "views": cmd_views}[a.cmd](a)


if __name__ == "__main__":
    main()
