# -*- coding: utf-8 -*-
"""
光沢で 3DGS が破綻するかの検証用データを GoLD_rayt で作る（Erato 像、金 vs 同系色の拡散面）。

形状・照明・カメラはまったく同じで、像の素材だけが違う 2 条件:
    gold  : 金（conductor Au, roughness 0.1）。見え方の大半が照明の映り込みで、視点で大きく変わる
    matte : 金に近い色の拡散面。見え方は視点によらない（対照）

照明とカメラの配置は D:/GoLD_rayt/out/studies/gs_gloss/make_views.py の v2
（上のソフトボックス + 周りの球光源 4 つ、床なし、像の周り 3 段のリング）にそろえ、暗い一様な環境光を足した。
環境光があるので金の映り込みが真っ黒になって背景と見分けられなくなる部分がなく、
背景はどの視点でも同じ一様な色になる（学習では --bg でその色を与える）。

    python experiments/gloss/make_views.py gold  --render
    python experiments/gloss/make_views.py matte --render

-> data/gloss_<variant>/ に scenes/*.json、{train,test}/*.png（学習には *_denoised.png を使う）、
   像のマスク（*_mask.png。形状とカメラだけで決まるので、どの条件のものでも同じ）、transforms_{train,test}.json
"""

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

EXE = r"D:\GoLD_rayt\out\cli-dev\GoLD_rayt.exe"   # RGB 版
CWD = r"D:\GoLD_rayt\GoLD_rayt"                     # シーン内の相対パスの基準
ROOT = Path(__file__).resolve().parents[2]          # 3dgs-toy/

TARGET = [0.0, 0.25, 0.0]   # 像の中心（高さ 0.50 m、y = 0..0.50）。ワールドは +Y が上
RADIUS = 1.3                # カメラの距離
FOV = 30.0                  # 度。正方形の画像なので水平・垂直とも同じ
ENV_SCALE = 0.15            # 一様な環境光の明るさ

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
    """camera-to-world（OpenGL 流儀: カメラは -z を向き、y が画面上）"""
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
    ap.add_argument("--render", action="store_true", help="シーンを書いたあと全視点を描画する（描画済みは飛ばす）")
    ap.add_argument("--limit", type=int, default=None, help="描画する視点数の上限（試し描き用）")
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
    for i in range(a.test):  # テスト視点はリングの間のランダムな方向（学習視点と重ならない）
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
                raise SystemExit(f"描画に失敗: {sc}")
    print(f"{len(todo)} 視点を {(time.time() - t0) / 60:.1f} 分で描画")


if __name__ == "__main__":
    main()
