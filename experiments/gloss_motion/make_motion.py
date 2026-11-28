# -*- coding: utf-8 -*-
"""
A glossy object that rotates under a fixed environment: dynamic (4D) data for the gloss experiments, rendered with
Blender Cycles on the GPU (blender_render.py; the default) or GoLD_rayt (RGB build, CPU; --renderer gold).

Blender is matched to the GoLD_rayt scenes of gloss_percept (bcompare, bcalib). The world, camera, object, environment
and a 1-pixel box filter are the same. GoLD_rayt's coat sums the walks between coat and base exactly (pbrt-v4's
LayeredBxDF), which darkens the base, and Blender's Principled coat does not: with the base albedo 0.2, Principled's
base colour is set to BLENDER_BASE = 0.092 (fitted on 3 coated conditions x 2 views, 0.089-0.094). The result matches
GoLD_rayt within its own noise: mean luminance 0.99-1.03, highlights 0.99-1.05, peaks 0.99-1.10, the per-pixel
difference at the level of two GoLD_rayt seeds; the matte object matches to 1.000. A training view takes 2.3 s on the
Quadro M4000 (CUDA, 512 spp) against about 20 s with GoLD_rayt.

Why motion: in a static scene (gloss_percept), the camera moves and the highlights move over the surface because the
view changes, which 3DGS models as view dependence. Here the object itself turns in front of the environment, so the
highlights slide over the surface while the surface texture and shading move with it. Gloss psychophysics treats
this relative motion as a cue to material, and a deformation-field 4DGS whose colour is SH attached to each Gaussian
(evaluated with the world-space view direction, never rotated) cannot represent it: a point that turns changes its
radiance over time even when the camera does not move.

Everything except the motion is the static gloss stimulus: shapes, materials, environment, background and the
camera distance and field of view of experiments/gloss_percept/make_stimuli.py.

    motion     a rotation about the vertical axis (y) through the object's centre, angle(t) for t in [0, 1]:
                 sweep   -A/2 + A t         (constant angular speed; default A = 60 degrees)
                 swing   (A/2) sin(2 pi t)  (back and forth)
    train      D-NeRF layout: one view per time (N_TRAIN times, evenly spaced), the camera at a random position in the
               elevation band of the static training rings (15-55 degrees), 512 spp
    test       a fixed camera (the observer's view: elevation 25, azimuth 20 degrees) at the N_TEST intermediate times
               (k + 0.5) / N_TEST, which are not training times, 1024 spp; test_seed2 repeats them with other seeds
               (noise floor)

    python experiments/gloss_motion/make_motion.py check                       # GoLD_rayt rotation checks (about a minute)
    python experiments/gloss_motion/make_motion.py bcompare --device CUDA [--base 0.092 --tag _b092]   # Blender vs GoLD
    python experiments/gloss_motion/make_motion.py bcalib                      # fit the Principled base colour
    python experiments/gloss_motion/make_motion.py bcheck                      # Blender's rotation against GoLD_rayt's
    python experiments/gloss_motion/make_motion.py preview [--frames 24]       # low-spp clip of the fixed view, GIF
    python experiments/gloss_motion/make_motion.py views SHAPE MATERIAL [--motion sweep --amplitude 60]

-> data/gloss_motion/<shape>__<material>__<motion><amplitude>/ (with --renderer gold: ...__gold/) in the layout of
   gloss_percept: {train,test,test_seed2}/*.pfm composited onto the uniform background, raw/ with the renders and the
   coverage masks, transforms_{train,test}.json with "time" and "rotation_deg" in each frame, motion.json. The raw
   renders of GoLD_rayt have the environment behind the object; Blender's hold the object alone, premultiplied by its
   coverage.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "gloss_percept"))
sys.path.insert(0, str(HERE.parent / "gloss"))
import make_stimuli as ms  # noqa: E402
from gs.dataset import read_pfm  # noqa: E402

DATA = ROOT / "data" / "gloss_motion"
AXIS = [0.0, 1.0, 0.0]
SHAPES = list(ms.SHAPES) + ["bunny"]   # bunny.ply: make_bunny.py
N_TRAIN = 150
N_TEST = 30
ELEV_BAND = (15.0, 55.0)
TEST_VIEW = (25.0, 20.0)          # elevation, azimuth (degrees)
LUM = np.array([0.2126729, 0.7151522, 0.0721750])


def angle(t, motion, amp):
    if motion == "sweep":
        return -amp / 2 + amp * t
    if motion == "swing":
        return amp / 2 * math.sin(2 * math.pi * t)
    raise ValueError(motion)


def scene(shape, mat, frm, deg, size, spp, seed, out):
    sc = ms.scene(shape, mat, frm, size, spp, seed, out)
    sc["objects"][0]["rotate"] = {"axis": AXIS, "degrees": float(deg)}
    return sc


def run(sc_path, raw):
    """render one scene; returns the time taken"""
    t = time.time()
    r = subprocess.run([ms.EXE, str(sc_path), "--gamut", "rec2020"], cwd=ms.CWD, capture_output=True)
    if r.returncode != 0 or not Path(raw).exists():
        print(r.stdout.decode("utf-8", "replace")[-1500:], r.stderr.decode("utf-8", "replace")[-1500:])
        raise SystemExit(f"rendering failed: {sc_path}")
    return time.time() - t


def render_one(d, name, shape, mat, frm, deg, size, spp, seed):
    raw = d / f"{name}.pfm"
    if not raw.exists():
        sc = d / f"{name}.json"
        sc.write_text(json.dumps(scene(shape, mat, frm, deg, size, spp, seed, raw.as_posix()), indent=1), encoding="utf-8")
        run(sc, raw)
    x = read_pfm(str(raw)).astype(np.float64)
    m = np.clip(read_pfm(str(raw).replace(".pfm", "_mask.pfm"))[..., 0].astype(np.float64), 0, 1)
    return x, m


# ---------------------------------------------------------------- Checks of the renderer's rotation

def cmd_check(a):
    """1. A rotated sphere renders like the unrotated one (positions and normals turn together, the material does not
          depend on the frame): the difference must be at the level of the noise of another seed.
       2. The blob turned by +theta, seen from azimuth az, has the silhouette of the unturned blob seen from
          az - theta (the sign convention of the rotation used to write the ground-truth motion)."""
    d = DATA / "check"
    d.mkdir(parents=True, exist_ok=True)
    size, spp, th = 128, 64, 40.0
    frm = ms.cam_from(*TEST_VIEW)
    s0, m0 = render_one(d, "sphere_0", "sphere", "coat_r0.05", frm, 0.0, size, spp, 1)
    s1, _ = render_one(d, "sphere_rot", "sphere", "coat_r0.05", frm, th, size, spp, 1)
    s2, _ = render_one(d, "sphere_0_seed2", "sphere", "coat_r0.05", frm, 0.0, size, spp, 2)
    obj = m0 > 0.99
    Y = lambda x: x @ LUM  # noqa: E731
    rel = lambda x, y: float(np.abs(Y(x) - Y(y))[obj].mean() / Y(y)[obj].mean())  # noqa: E731
    print(f"1. sphere, coat 0.05: mean |dY| / mean Y on the object: rotated {th:g} deg vs not {rel(s1, s0):.4f}; "
          f"other seed vs not {rel(s2, s0):.4f} (noise)")
    _, mb = render_one(d, "blob_rot", "blob1", "matte", frm, th, size, 16, 1)
    ious = {}
    for sign in (-1, 1):
        _, mc = render_one(d, f"blob_cam{sign:+d}", "blob1", "matte", ms.cam_from(TEST_VIEW[0], TEST_VIEW[1] + sign * th),
                           0.0, size, 16, 1)
        A, B = mb > 0.5, mc > 0.5
        ious[sign] = float((A & B).sum() / (A | B).sum())
    print(f"2. blob turned +{th:g} deg at azimuth {TEST_VIEW[1]:g} vs unturned blob at azimuth {TEST_VIEW[1]:g} -/+ {th:g}: "
          f"silhouette IoU {ious[-1]:.4f} / {ious[1]:.4f} (expected: the first is ~1)")


# ---------------------------------------------------------------- Blender Cycles (blender_render.py)

BLENDER = r"D:\Blender\blender.exe"
BLENDER_SCRIPT = HERE / "blender_render.py"
BLENDER_BASE = 0.092              # Principled base colour that gives GoLD_rayt's coat over a base of albedo 0.2 (bcalib)
# --texture noise: a grey pattern on the base, fixed to the object (Blender only): noise in object coordinates (scale per
# metre: about 15 features across the 0.43 m objects), its Fac mapped from fac_range to base x [1 - contrast, 1 + contrast]
TEXTURE = {"type": "noise", "scale": 15.0, "detail": 2.0, "roughness": 0.5, "fac_range": [0.3, 0.7], "contrast": 0.8}


def blender_cfg(shape, mat, device, size, jobs, base=None, texture=None):
    """base: the Principled base colour of a layered material (default: the albedo of make_stimuli.MATERIALS);
    texture: a pattern on the base (TEXTURE)"""
    m = dict(ms.MATERIALS[mat])
    if base is not None:
        m["blender_base_albedo"] = [float(base)] * 3
    if texture:
        m["texture"] = texture
    return {"env": str(ms.ENV_GRAY.resolve()), "env_scale": ms.ENV_SCALE,
            "mesh": str((ms.DATA / "mesh" / f"{shape}.ply").resolve()), "material": m,
            "device": device, "size": size, "fov": ms.FOV, "jobs": jobs}


def run_blender(cfg, cfg_path, every=0, log=None):
    """render the jobs of cfg in one Blender session; returns the seconds of each render. every: print every n-th
    render; log: an open file that gets every render's line"""
    Path(cfg_path).write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    p = subprocess.Popen([BLENDER, "-b", "--factory-startup", "--python", str(BLENDER_SCRIPT), "--", str(cfg_path)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    times, tail = [], []
    for raw in p.stdout:
        line = raw.decode("utf-8", "replace").rstrip()
        tail = (tail + [line])[-60:]
        if line.startswith("[blender]") and "/" in line.split()[1]:
            times.append(float(line.rsplit(" ", 1)[1][:-1]))
            if log is not None:
                log.write(line + "\n")
                log.flush()
            if every and (len(times) % every == 1 or len(times) == len(cfg["jobs"])):
                print(line, flush=True)
    if p.wait() != 0 or len(times) != len(cfg["jobs"]):
        print("\n".join(tail))
        raise SystemExit(f"Blender failed: {cfg_path}")
    return times


def cmd_bcompare(a):
    """Blender against GoLD_rayt on the static test views of gloss_percept (1024 spp truth and its second seed), on the
    pixels fully covered by the object in both: mean luminance, highlight (the GoLD render's brightest 5%) and peak
    (0.5%) energy, and the mean |dY| relative to the mean, against the same between GoLD's two seeds (noise).
    With --devices, also the time per render on each device."""
    from gs.color import srgb_curve
    import imageio.v2 as imageio
    import torch
    d = DATA / "bcompare"
    d.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    for cond in a.conds:
        shape, mat = cond.split("__")
        src = ms.DATA / cond
        meta = json.loads((src / "transforms_test.json").read_text(encoding="utf-8"))
        views = [f for f in meta["frames"]][:a.views]
        jobs = [{"c2w": f["transform_matrix"], "rotate_deg": 0.0, "spp": a.spp, "seed": k + 1,
                 "out": (d / f"{cond}_{Path(f['file_path']).stem}{a.tag}.pfm").as_posix()} for k, f in enumerate(views)]
        base = None if mat == "matte" else a.base
        times = run_blender(blender_cfg(shape, mat, a.device, 256, jobs, base), d / f"{cond}{a.tag}.json")
        for f, job, t in zip(views, jobs, times):
            name = Path(f["file_path"]).stem
            g = read_pfm(str(src / "raw" / "test" / f"{name}.pfm")).astype(np.float64)
            gm = read_pfm(str(src / "raw" / "test" / f"{name}_mask.pfm"))[..., 0]
            b = read_pfm(job["out"]).astype(np.float64)
            bm = read_pfm(job["out"].replace(".pfm", "_mask.pfm"))[..., 0]
            obj = (gm > 0.99) & (bm > 0.99)
            Yg, Yb = (g @ LUM)[obj], (b @ LUM)[obj]
            s2 = src / "raw" / "test_seed2" / f"{name}.pfm"
            noise = float(np.abs((read_pfm(str(s2)).astype(np.float64) @ LUM)[obj] - Yg).mean() / Yg.mean()) if s2.exists() else float("nan")
            hl, pk = Yg >= np.percentile(Yg, 95), Yg >= np.percentile(Yg, 99.5)
            iou = float(((gm > 0.5) & (bm > 0.5)).sum() / ((gm > 0.5) | (bm > 0.5)).sum())
            rows.append({"cond": cond, "view": name, "sec": t, "silhouette_iou": iou,
                         "mean_ratio": float(Yb.mean() / Yg.mean()), "highlight_ratio": float(Yb[hl].sum() / Yg[hl].sum()),
                         "peak_ratio": float(Yb[pk].sum() / Yg[pk].sum()), "rel_abs_diff": float(np.abs(Yb - Yg).mean() / Yg.mean()),
                         "rel_abs_diff_gold_seeds": noise})
            r = rows[-1]
            print(f"{cond:20} {name}: {t:5.1f} s  IoU {iou:.4f}  mean {r['mean_ratio']:.3f}  highlights {r['highlight_ratio']:.3f}  "
                  f"peaks {r['peak_ratio']:.3f}  |dY| {r['rel_abs_diff']:.3f} (GoLD seeds {noise:.3f})", flush=True)
            k = 0.18 / np.median(Yg)
            comp = lambda x, m: x * m[..., None] + ms.BG_LINEAR * (1 - m[..., None])  # noqa: E731
            pair = np.concatenate([comp(g, gm), b + ms.BG_LINEAR * (1 - bm[..., None])], 1)
            panels.append((srgb_curve(torch.from_numpy(np.clip(pair * k, 0, 1))).numpy() * 255 + 0.5).astype(np.uint8))
    imageio.imwrite(d / f"bcompare_{a.device}{a.tag}.png", np.concatenate(panels, 0))
    (d / f"bcompare_{a.device}{a.tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"left GoLD_rayt, right Blender ({a.device}) -> {d / f'bcompare_{a.device}{a.tag}.png'}")


def cmd_bcalib(a):
    """The Principled base colour that gives the body of GoLD_rayt's coated objects. Blender is linear in the base
    colour c: B(c) = S + (c / 0.2) (B(0.2) - S), where S is the render with a black base (the coat's reflection
    alone). On the object outside GoLD's brightest 5% (the highlights), c = 0.2 sum(G - S) / sum(B(0.2) - S). Needs the
    renders of bcompare (default base) on the same views."""
    d = DATA / "bcompare"
    out = {}
    for cond in a.conds:
        shape, mat = cond.split("__")
        src = ms.DATA / cond
        meta = json.loads((src / "transforms_test.json").read_text(encoding="utf-8"))
        views = meta["frames"][:a.views]
        jobs = [{"c2w": f["transform_matrix"], "rotate_deg": 0.0, "spp": a.spp, "seed": k + 1,
                 "out": (d / f"{cond}_{Path(f['file_path']).stem}_b0.pfm").as_posix()} for k, f in enumerate(views)]
        run_blender(blender_cfg(shape, mat, a.device, 256, jobs, 0.0), d / f"{cond}_b0.json")
        num = den = 0.0
        for f, job in zip(views, jobs):
            name = Path(f["file_path"]).stem
            G = read_pfm(str(src / "raw" / "test" / f"{name}.pfm")).astype(np.float64) @ LUM
            gm = read_pfm(str(src / "raw" / "test" / f"{name}_mask.pfm"))[..., 0]
            S = read_pfm(job["out"]).astype(np.float64) @ LUM
            B = read_pfm(str(d / f"{cond}_{name}.pfm")).astype(np.float64) @ LUM
            bm = read_pfm(str(d / f"{cond}_{name}_mask.pfm"))[..., 0]
            obj = (gm > 0.99) & (bm > 0.99)
            body = obj & (G < np.percentile(G[obj], 95))
            c = 0.2 * (G - S)[body].sum() / (B - S)[body].sum()
            num, den = num + (G - S)[body].sum(), den + (B - S)[body].sum()
            print(f"{cond:20} {name}: coat alone / GoLD on the body {S[body].mean() / G[body].mean():.3f}; base colour {c:.4f}")
        out[cond] = 0.2 * num / den
        print(f"{cond:20} base colour {out[cond]:.4f}", flush=True)
    (d / "bcalib.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_btime(a):
    """time per render of one training view (512 spp, 256 px) on each Blender device; the first render of a session
    includes loading the kernels, so it is reported apart"""
    d = DATA / "bcompare"
    d.mkdir(parents=True, exist_ok=True)
    frm = ms.cam_from(*TEST_VIEW)
    for dev in a.devices:
        jobs = [{"c2w": ms.c2w(frm), "rotate_deg": 10.0 * k, "spp": a.spp, "seed": k + 1,
                 "out": (d / f"time_{dev}_{k}.pfm").as_posix()} for k in range(a.n)]
        t = run_blender(blender_cfg("blob1", "coat_r0.05", dev, 256, jobs), d / f"time_{dev}.json")
        print(f"{dev:6}: first render {t[0]:.1f} s, then {np.median(t[1:]):.1f} s per render "
              f"({a.spp} spp, 256 px; GoLD_rayt CPU: about 20 s)", flush=True)


def cmd_bcheck(a):
    """Blender's turn against GoLD_rayt's: the blob turned by +40 degrees, matte, at the test view (128 px), against
    the GoLD_rayt render of the check subcommand (blob_rot): silhouette IoU and mean luminance"""
    d = DATA / "check"
    g = d / "blob_rot.pfm"
    if not g.exists():
        raise SystemExit("run make_motion.py check first (GoLD_rayt renders)")
    out = (d / "blob_rot_blender.pfm").as_posix()
    run_blender(blender_cfg("blob1", "matte", a.device, 128,
                            [{"c2w": ms.c2w(ms.cam_from(*TEST_VIEW)), "rotate_deg": 40.0, "spp": 256, "seed": 1, "out": out}]),
                d / "blob_rot_blender.json")
    gm = read_pfm(str(d / "blob_rot_mask.pfm"))[..., 0]
    bm = read_pfm(out.replace(".pfm", "_mask.pfm"))[..., 0]
    obj = (gm > 0.99) & (bm > 0.99)
    Yg = read_pfm(str(g)).astype(np.float64) @ LUM
    Yb = read_pfm(out).astype(np.float64) @ LUM
    iou = float(((gm > 0.5) & (bm > 0.5)).sum() / ((gm > 0.5) | (bm > 0.5)).sum())
    print(f"blob turned +40 deg, Blender vs GoLD_rayt: silhouette IoU {iou:.4f}, mean Y ratio {Yb[obj].mean() / Yg[obj].mean():.3f}")


# ---------------------------------------------------------------- Preview

def to_display(x, k):
    from gs.color import srgb_curve
    import torch
    return srgb_curve(torch.from_numpy(np.clip(x * k, 0, 1))).numpy()


def cmd_preview(a):
    """the fixed test view over time at low spp, glossy and matte side by side (one exposure), as a GIF"""
    import imageio.v2 as imageio
    d = DATA / "preview" / f"{a.shape}__{a.motion}{a.amplitude:g}"
    d.mkdir(parents=True, exist_ok=True)
    frm = ms.cam_from(*TEST_VIEW)
    frames, t0 = {m: [] for m in a.materials}, time.time()
    for k in range(a.frames):
        t = (k + 0.5) / a.frames
        for mat in a.materials:
            x, m = render_one(d, f"{mat}_{k:03d}", a.shape, mat, frm, angle(t, a.motion, a.amplitude), a.size, a.spp, k + 1)
            frames[mat].append(x * m[..., None] + ms.BG_LINEAR * (1 - m[..., None]))
    print(f"{a.frames} frames x {len(a.materials)} materials in {time.time() - t0:.0f} s", flush=True)
    first = frames[a.materials[-1]][0]
    obj = read_pfm(str(d / f"{a.materials[-1]}_000_mask.pfm"))[..., 0] > 0.99
    k = 0.18 / np.median((first @ LUM)[obj])          # the last material's median Y on the object -> 0.18
    gif = [(to_display(np.concatenate([frames[m][i] for m in a.materials], 1), k) * 255 + 0.5).astype(np.uint8)
           for i in range(a.frames)]
    out = d / "preview.gif"
    imageio.mimsave(out, gif, duration=1000 / a.fps, loop=0)
    imageio.imwrite(d / "preview_first_last.png", np.concatenate([gif[0], gif[-1]], 0))
    print(f"columns {a.materials}, exposure x{k:.3f} -> {out}")


# ---------------------------------------------------------------- Multi-view dynamic data

def views(motion, amp):
    rng = random.Random(1)
    out = []
    for k in range(N_TRAIN):
        t = k / (N_TRAIN - 1)
        frm = ms.cam_from(rng.uniform(*ELEV_BAND), rng.uniform(0.0, 360.0))
        out.append(("train", f"t_{k:03d}", frm, t))
    frm = ms.cam_from(*TEST_VIEW)
    tests = [("test", f"t_{k:03d}", frm, (k + 0.5) / N_TEST) for k in range(N_TEST)]
    return out + tests + [("test_seed2", n, f, t) for _, n, f, t in tests]


def cmd_views(a):
    blender = a.renderer == "blender"
    texture = {**TEXTURE, "contrast": a.texture_contrast} if a.texture == "noise" else None
    assert blender or not texture, "the pattern is drawn by Blender only"
    name = (f"{a.shape}__{a.material}__{a.motion}{a.amplitude:g}" + ("__tex" if texture else "")
            + ("" if blender else "__gold"))
    root = DATA / name
    for sub in ("scenes", "raw/train", "raw/test", "raw/test_seed2", "train", "test", "test_seed2"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "motion.json").write_text(json.dumps(
        {"shape": a.shape, "material": a.material, "motion": a.motion, "amplitude_deg": a.amplitude,
         "axis": AXIS, "centre": ms.TARGET, "angle": "sweep: -A/2 + A t; swing: A/2 sin(2 pi t); positive = right-handed "
                                                       "about the axis (azimuth of object points increases)",
         "test_view_elev_azim": TEST_VIEW, "renderer": a.renderer,
         "blender_base_albedo": BLENDER_BASE if blender and ms.MATERIALS[a.material]["type"] == "layered" else None,
         "texture": texture},
        indent=1), encoding="utf-8")
    jobs, bjobs, frames = [], [], {"train": [], "test": []}
    for k, (split, vname, frm, t) in enumerate(views(a.motion, a.amplitude)):
        deg = angle(t, a.motion, a.amplitude)
        spp = ms.SPP_TRAIN if split == "train" else ms.SPP_TEST
        raw = (root / "raw" / split / f"{vname}.pfm").as_posix()
        sc = root / "scenes" / f"{split}_{vname}.json"
        if not blender:
            sc.write_text(json.dumps(scene(a.shape, a.material, frm, deg, a.size, spp, k + 1, raw), indent=1), encoding="utf-8")
        jobs.append((sc, Path(raw), root / split / f"{vname}.pfm"))
        bjobs.append({"c2w": ms.c2w(frm), "rotate_deg": deg, "spp": spp, "seed": k + 1, "out": raw})
        if split in frames:
            frames[split].append({"file_path": f"{split}/{vname}.pfm", "time": t, "rotation_deg": deg,
                                  "transform_matrix": ms.c2w(frm)})
    for split, fr in frames.items():
        (root / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": math.radians(ms.FOV), "color_space": "linear-rec2020", "frames": fr}, indent=1),
            encoding="utf-8")
    todo = [i for i, j in enumerate(jobs) if not j[1].exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{name}: {len(jobs)} views, {len(todo)} to render ({a.renderer})", flush=True)
    t0 = time.time()
    with open(root / "render.log", "a", encoding="utf-8") as log:
        if blender and todo:
            base = BLENDER_BASE if ms.MATERIALS[a.material]["type"] == "layered" else None
            run_blender(blender_cfg(a.shape, a.material, a.device, a.size, [bjobs[i] for i in todo], base, texture),
                        root / "scenes" / "blender_jobs.json", every=10, log=log)
        for i, (sc, raw, _) in enumerate([] if blender else [jobs[i] for i in todo]):
            dt = run(sc, raw)
            msg = f"{i + 1}/{len(todo)} {sc.name} {dt:.1f}s"
            log.write(msg + "\n")
            log.flush()
            if i % 10 == 0 or i == len(todo) - 1:
                print(msg, flush=True)
    if todo:
        print(f"{len(todo)} views rendered in {(time.time() - t0) / 60:.1f} min", flush=True)
    done = 0
    for _, raw, dst in jobs:
        if not raw.exists():
            continue
        x = read_pfm(str(raw)).astype(np.float64)
        m = np.clip(read_pfm(str(raw).replace(".pfm", "_mask.pfm"))[..., :1].astype(np.float64), 0, 1)
        if not np.isfinite(x).all() or (x < -1e-4).any():
            print(f"  warning {raw.name}: {(~np.isfinite(x).all(-1)).sum()} non-finite pixels, min {np.nanmin(x):.4g}")
        # GoLD_rayt: the environment is behind the object (edge pixels keep their share of it); Blender: premultiplied
        y = (x if blender else x * m) + ms.BG_LINEAR * (1 - m)
        ms.write_pfm(dst, np.where(np.isfinite(y), y, ms.BG_LINEAR))
        done += 1
    print(f"composited {done} of {len(jobs)} views onto {ms.BG_LINEAR} -> {root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("preview")
    p.add_argument("--shape", default="blob1", choices=SHAPES)
    p.add_argument("--materials", nargs="+", default=["coat_r0.05", "matte"], choices=list(ms.MATERIALS))
    p.add_argument("--motion", default="sweep", choices=("sweep", "swing"))
    p.add_argument("--amplitude", type=float, default=60.0)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--spp", type=int, default=128)
    p.add_argument("--fps", type=float, default=12.0)
    b = sub.add_parser("bcompare")
    b.add_argument("--conds", nargs="+", default=["sphere__coat_r0.05", "blob1__coat_r0.05", "blob1__coat_r0.15",
                                                  "blob1__matte"])
    b.add_argument("--views", type=int, default=2)
    b.add_argument("--spp", type=int, default=1024)
    b.add_argument("--device", default="CPU", choices=("CPU", "CUDA", "OPTIX"))
    b.add_argument("--base", type=float, default=None, help="Principled base colour of the coated materials")
    b.add_argument("--tag", default="", help="suffix of the output names")
    c = sub.add_parser("bcalib")
    c.add_argument("--conds", nargs="+", default=["sphere__coat_r0.05", "blob1__coat_r0.05", "blob1__coat_r0.15"])
    c.add_argument("--views", type=int, default=2)
    c.add_argument("--spp", type=int, default=1024)
    c.add_argument("--device", default="CUDA", choices=("CPU", "CUDA", "OPTIX"))
    bt = sub.add_parser("btime")
    bt.add_argument("--devices", nargs="+", default=["CPU", "CUDA", "OPTIX"])
    bt.add_argument("--n", type=int, default=3)
    bt.add_argument("--spp", type=int, default=512)
    v = sub.add_parser("views")
    v.add_argument("shape", choices=SHAPES)
    v.add_argument("material", choices=list(ms.MATERIALS))
    v.add_argument("--motion", default="sweep", choices=("sweep", "swing"))
    v.add_argument("--amplitude", type=float, default=60.0, help="degrees (sweep: total turn; swing: peak to peak)")
    v.add_argument("--size", type=int, default=256)
    v.add_argument("--limit", type=int, default=0, help="render at most this many views (0: all), e.g. to time a few")
    v.add_argument("--renderer", default="blender", choices=("blender", "gold"))
    v.add_argument("--device", default="CUDA", choices=("CPU", "CUDA", "OPTIX"), help="--renderer blender")
    v.add_argument("--texture", default="none", choices=("none", "noise"), help="a pattern on the base (TEXTURE)")
    v.add_argument("--texture-contrast", type=float, default=TEXTURE["contrast"])
    k = sub.add_parser("bcheck")
    k.add_argument("--device", default="CUDA", choices=("CPU", "CUDA", "OPTIX"))
    a = ap.parse_args()
    {"check": cmd_check, "preview": cmd_preview, "views": cmd_views, "bcompare": cmd_bcompare, "bcalib": cmd_bcalib, "btime": cmd_btime,
     "bcheck": cmd_bcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
