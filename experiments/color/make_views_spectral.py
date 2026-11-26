# -*- coding: utf-8 -*-
"""
Colour experiment, stage 1: the gold statue of the gloss experiment, rendered with the spectral build of GoLD_rayt as
linear radiance, under a light whose colour lies outside the sRGB gamut.

Same geometry, light layout, cameras and seeds as experiments/gloss/make_views.py (gold). Three things change:
    lights     the softbox and the four sphere lights emit a candle spectrum (Planck 1850 K) instead of D65.
               The candle itself is outside sRGB (its sRGB blue is negative), and so is part of its reflection on gold
    ambient    uniform ambient 0.05 instead of 0.15 (it stays D65; dimmer, so that more of the statue leaves sRGB:
               9% at 0.15, 32% at 0.05, in a 48 px trial)
    output     .pfm in linear Rec.2020 (--gamut rec2020). The film is CIE 1931 XYZ turned into RGB by one exact matrix,
               without clipping; the denoiser works in Rec.2020, so colours outside sRGB but inside Rec.2020 survive it

    python experiments/color/make_views_spectral.py --render

-> data/color_gold_candle/: scenes/*.json, {train,test}/*_denoised.pfm (used), *.pfm (noisy), *_mask.pfm,
   test_seed2/ (the test views again with other seeds: the noise floor of the ground truth),
   transforms_{train,test}.json with "color_space": "linear-rec2020"
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gloss"))
import make_views as gloss  # noqa: E402

EXE = r"D:\GoLD_rayt\out\cli-dev-spectral\GoLD_rayt.exe"   # spectral build
CWD = gloss.CWD
LIGHT = "candle"
AMBIENT = 0.05
NAME = "color_gold_candle"


def scene(frm, size, spp, seed, out_pfm):
    sc = gloss.scene("gold", frm, size, spp, seed, out_pfm, True)
    for k in ("top", "side"):
        sc["materials"][k]["spectrum"] = LIGHT
    sc["environment"]["scale"] = AMBIENT
    return sc


def main(name=NAME, scene_fn=scene):
    """name: the folder under data/; scene_fn(frm, size, spp, seed, out_pfm) -> scene dict (make_views_pigment.py uses both)"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--spp", type=int, default=512)
    ap.add_argument("--render", action="store_true", help="render all views (views already rendered are skipped)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    root = gloss.ROOT / "data" / name
    for sub in ("scenes", "train", "test", "test_seed2"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # The same view list as the gloss data (3 rings of 30 training views, 10 random test views)
    views = []
    for ring, elev in enumerate((15.0, 35.0, 55.0)):
        for i in range(30):
            azim = 360.0 * i / 30 + ring * (360.0 / 30) / 3.0
            views.append(("train", f"view_{ring}_{i:03d}", gloss.cam_from(elev, azim)))
    import random
    rng = random.Random(1)
    for i in range(10):
        views.append(("test", f"view_{i:03d}", gloss.cam_from(rng.uniform(12.0, 65.0), rng.uniform(0.0, 360.0))))
    tests = [v for v in views if v[0] == "test"]
    views += [("test_seed2", name, frm) for _, name, frm in tests]

    frames = {"train": [], "test": []}
    jobs = []
    for k, (split, name, frm) in enumerate(views):
        out = (root / split / f"{name}.pfm").as_posix()
        sc = root / "scenes" / f"{split}_{name}.json"
        sc.write_text(json.dumps(scene_fn(frm, a.size, a.spp, k + 1, out), indent=1), encoding="utf-8")
        if split in frames:
            frames[split].append({"file_path": f"{split}/{name}_denoised.pfm", "transform_matrix": gloss.c2w(frm)})
        jobs.append((sc, Path(out.replace(".pfm", "_denoised.pfm"))))
    for split in frames:
        (root / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": math.radians(gloss.FOV), "color_space": "linear-rec2020", "frames": frames[split]},
            indent=1), encoding="utf-8")
    print(f"{len(frames['train'])} train + {len(frames['test'])} test + {len(tests)} test_seed2 -> {root}")

    if not a.render:
        return
    todo = [j for j in jobs if not j[1].exists()][: a.limit]
    t0 = time.time()
    with open(root / "render.log", "a", encoding="utf-8") as log:
        for i, (sc, out) in enumerate(todo):
            t1 = time.time()
            r = subprocess.run([EXE, str(sc), "--gamut", "rec2020"], cwd=CWD, capture_output=True)
            msg = f"{i + 1}/{len(todo)} {sc.name} rc={r.returncode} {time.time() - t1:.1f}s"
            log.write(msg + "\n")
            log.flush()
            print(msg, flush=True)
            if r.returncode != 0 or not out.exists():
                log.write(r.stderr.decode("utf-8", "replace")[-1000:] + "\n")
                raise SystemExit(f"rendering failed: {sc}")
    print(f"{len(todo)} views rendered in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
