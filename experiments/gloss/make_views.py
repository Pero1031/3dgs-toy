# -*- coding: utf-8 -*-
"""
Make the data for testing whether 3DGS breaks down on gloss, using GoLD_rayt (Erato statue, gold vs a diffuse
surface of a similar colour).

Two conditions with exactly the same geometry, lighting, and cameras, differing only in the material of the statue:
    gold  : gold (conductor Au, roughness 0.1). Most of the appearance is reflection of the lighting, and it changes
            greatly with the view
    matte : diffuse, with a colour close to gold. The appearance does not depend on the view (control)

The lighting and camera layout follow v2 of D:/GoLD_rayt/out/studies/gs_gloss/make_views.py
(a softbox above + 4 sphere lights around, no floor, 3 rings of cameras around the statue), with dim uniform
ambient light added.
With the ambient light, there are no parts where the reflection on the gold becomes pure black and cannot be told
apart from the background, and the background is the same uniform colour in every view (in training, that colour is
given with --bg).

    python experiments/gloss/make_views.py gold  --render
    python experiments/gloss/make_views.py matte --render

-> in data/gloss_<variant>/: scenes/*.json, {train,test}/*.png (*_denoised.png is used for training),
   statue masks (*_mask.png. They are determined by geometry and cameras only, so they are the same for every
   condition), transforms_{train,test}.json
"""

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

EXE = r"D:\GoLD_rayt\out\cli-dev\GoLD_rayt.exe"   # RGB version
CWD = r"D:\GoLD_rayt\GoLD_rayt"                     # base for relative paths in the scene
ROOT = Path(__file__).resolve().parents[2]          # 3dgs-toy/

TARGET = [0.0, 0.25, 0.0]   # centre of the statue (height 0.50 m, y = 0..0.50). The world has +Y up
RADIUS = 1.3                # camera distance
FOV = 30.0                  # degrees. The image is square, so horizontal and vertical are the same
ENV_SCALE = 0.15            # brightness of the uniform ambient light

MATERIALS = {
    "gold": {"type": "conductor", "metal": "Au", "roughness": 0.1},
    "matte": {"type": "diffuse", "albedo": [0.80, 0.55, 0.22]},
}


def cam_from(elev_deg, azim_deg):
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return [TARGET[0] + RADIUS * math.cos(e) * math.sin(a),
            TARGET[1] + RADIUS * math.sin(e),
            TARGET[2] + RADIUS * math.cos(e) * math.cos(a)]


def c2w(frm):
    """camera-to-world (OpenGL convention: the camera looks along -z and y is up on screen)"""
    z = [frm[i] - TARGET[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in z))
    z = [v / n for v in z]
    up = [0.0, 1.0, 0.0]
    x = [up[1] * z[2] - up[2] * z[1], up[2] * z[0] - up[0] * z[2], up[0] * z[1] - up[1] * z[0]]
    n = math.sqrt(sum(v * v for v in x))
    x = [v / n for v in x]
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    return [[x[0], y[0], z[0], frm[0]], [x[1], y[1], z[1], frm[1]],
            [x[2], y[2], z[2], frm[2]], [0.0, 0.0, 0.0, 1.0]]


def scene(variant, frm, size, spp, seed, out_png, mask):
    objs = [{"shape": "mesh", "file": "assets/Meshes/erato/erato_closed.obj", "scale": 0.0133,
             "rotate": {"axis": [0, 1, 0], "degrees": 90},
             "translate": [0.0011, 0.0169, 0.0314], "material": "statue"},
            {"shape": "quad", "corner": [-0.9, 1.9, -0.9], "edge1": [1.8, 0, 0], "edge2": [0, 0, 1.8],
             "material": "top"}]
    for k in range(4):
        az = math.radians(45.0 + 90.0 * k)
        objs.append({"shape": "sphere", "center": [1.9 * math.sin(az), 1.2, 1.9 * math.cos(az)],
                     "radius": 0.35, "material": "side"})
    options = {"width": size, "height": size, "spp": spp, "seed": seed, "denoise": True, "out": out_png}
    if mask:
        options["mask"] = "statue"
    return {
        "options": options,
        "camera": {"from": frm, "at": TARGET, "fov": FOV},
        "environment": {"type": "uniform", "scale": ENV_SCALE},
        "materials": {"statue": MATERIALS[variant],
                      "top": {"type": "emission", "spectrum": "d65", "radiance": 8},
                      "side": {"type": "emission", "spectrum": "d65", "radiance": 7}},
        "objects": objs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(MATERIALS))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--spp", type=int, default=512)
    ap.add_argument("--train-per-ring", type=int, default=30)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--render", action="store_true", help="render all views after writing the scenes (views already rendered are skipped)")
    ap.add_argument("--limit", type=int, default=None, help="upper limit on the number of views to render (for trial renders)")
    a = ap.parse_args()

    root = ROOT / "data" / f"gloss_{a.variant}"
    for sub in ("scenes", "train", "test"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    views = []  # (split, name, frm)
    for ring, elev in enumerate((15.0, 35.0, 55.0)):
        for i in range(a.train_per_ring):
            azim = 360.0 * i / a.train_per_ring + ring * (360.0 / a.train_per_ring) / 3.0
            views.append(("train", f"view_{ring}_{i:03d}", cam_from(elev, azim)))
    rng = random.Random(1)
    for i in range(a.test):  # test views are random directions between the rings (no overlap with the training views)
        views.append(("test", f"view_{i:03d}", cam_from(rng.uniform(12.0, 65.0), rng.uniform(0.0, 360.0))))

    frames = {"train": [], "test": []}
    jobs = []
    for k, (split, name, frm) in enumerate(views):
        png = (root / split / f"{name}.png").as_posix()
        sc = root / "scenes" / f"{split}_{name}.json"
        sc.write_text(json.dumps(scene(a.variant, frm, a.size, a.spp, k + 1, png, True), indent=1),
                      encoding="utf-8")
        frames[split].append({"file_path": f"{split}/{name}_denoised", "transform_matrix": c2w(frm)})
        jobs.append((sc, Path(png.replace(".png", "_denoised.png"))))
    for split in ("train", "test"):
        (root / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": math.radians(FOV), "frames": frames[split]}, indent=1), encoding="utf-8")
    print(f"{len(frames['train'])} train + {len(frames['test'])} test -> {root}")

    if not a.render:
        return
    todo = [j for j in jobs if not j[1].exists()][: a.limit]
    t0 = time.time()
    with open(root / "render.log", "a", encoding="utf-8") as log:
        for i, (sc, out) in enumerate(todo):
            t1 = time.time()
            r = subprocess.run([EXE, str(sc)], cwd=CWD, capture_output=True)
            msg = f"{i + 1}/{len(todo)} {sc.name} rc={r.returncode} {time.time() - t1:.1f}s"
            log.write(msg + "\n")
            log.flush()
            print(msg, flush=True)
            if r.returncode != 0:
                log.write(r.stderr.decode("utf-8", "replace")[-1000:] + "\n")
                raise SystemExit(f"rendering failed: {sc}")
    print(f"{len(todo)} views rendered in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
