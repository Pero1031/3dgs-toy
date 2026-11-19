# -*- coding: utf-8 -*-
"""
動くシーンの動作確認用データセットを生成する（外部データのダウンロード不要）。

make_toy_dataset.py の縞模様の球と立方体を、時刻 t ∈ [0,1] で動かす:
    球     上下に弾み（中心の高さ 0.35 sin 2πt）、半径が脈打つ（1 + 0.15 sin 4πt）。模様は面に付いて動く
    立方体 球のまわりの円周上で、±15° 往復する（軸に平行な向きのまま）。--orbit なら半周する
--orbit では立方体が大きく動く（約 5 単位）。基準の形を 1 つ持つ変形モデルの苦手な例として残している。
照明は固定した平行光で、拡散反射だけ。球は回転しないので、面の各点の明るさは時刻によらない
（＝色を各ガウシアンに持たせる変形モデルで、誤差なく表せるシーン）。

D-NeRF と同じ形式（transforms_*.json の各フレームに "time"）で保存する。学習用は 1 時刻に 1 視点
（カメラは毎回ランダムな位置に飛ぶ）、テスト用は学習に無い中間の時刻を、赤道より少し上の一周から見る。

    python make_toy_dynamic.py --out data/toy_dynamic
    python make_toy_dynamic.py --out data/toy_dynamic_orbit --orbit
"""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from gs.poses import look_at_origin

LIGHT = np.array([0.4, -0.5, 0.75])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
BOX_H, ORBIT_R, BOX_Z = 0.3, 1.7, -0.4   # 立方体の半辺長、周回の半径、高さ


def scene(t, orbit=False):
    """時刻 t の球の中心・半径と、立方体の中心"""
    c = np.array([0.0, 0.0, 0.35 * np.sin(2 * np.pi * t)])
    r = 1.0 + 0.15 * np.sin(4 * np.pi * t)
    phi = np.pi / 2 + (np.pi * t if orbit else np.radians(15) * np.sin(2 * np.pi * t))
    b = np.array([ORBIT_R * np.cos(phi), ORBIT_R * np.sin(phi), BOX_Z])
    return c, r, b


def render(c2w, H, W, focal, t, orbit=False):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dirs = np.stack([(i + 0.5 - W / 2) / focal, -(j + 0.5 - H / 2) / focal, -np.ones_like(i, float)], -1)
    d = dirs @ c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(c2w[:3, 3], d.shape)
    c, r, box_c = scene(t, orbit)

    # 球との交差
    oc = o - c
    b = (oc * d).sum(-1)
    cc = (oc * oc).sum(-1) - r * r
    disc = b * b - cc
    t_s = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
    t_s = np.where(t_s > 0, t_s, np.inf)

    # 軸平行な立方体との交差（slab 法）
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box_c - BOX_H - o) / d
        t2 = (box_c + BOX_H - o) / d
    tmin = np.minimum(t1, t2).max(-1)
    tmax = np.maximum(t1, t2).min(-1)
    t_b = np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)

    rgb = np.ones((H, W, 3))
    alpha = np.zeros((H, W))

    hit_s = (t_s < t_b) & np.isfinite(t_s)
    n = (o + d * np.where(hit_s, t_s, 0.0)[..., None] - c) / r      # 単位法線 = 球の中心から見た向き
    lon = np.arctan2(n[..., 1], n[..., 0])
    lat = np.arcsin(np.clip(n[..., 2], -1, 1))
    stripe = ((np.floor(lon / (np.pi / 6)) + np.floor(lat / (np.pi / 8))) % 2)[..., None]
    base = np.where(stripe > 0, [0.85, 0.25, 0.2], [0.95, 0.85, 0.3])
    shade = 0.25 + 0.75 * np.clip((n * LIGHT).sum(-1, keepdims=True), 0, 1)
    rgb[hit_s] = (base * shade)[hit_s]
    alpha[hit_s] = 1.0

    hit_b = np.isfinite(t_b) & ~hit_s
    p = o + d * np.where(hit_b, t_b, 0.0)[..., None]
    q = (p - box_c) / BOX_H
    axis = np.abs(q).argmax(-1)
    n = np.zeros_like(q)
    np.put_along_axis(n, axis[..., None], np.sign(np.take_along_axis(q, axis[..., None], -1)), -1)
    shade = 0.25 + 0.75 * np.clip((n * LIGHT).sum(-1, keepdims=True), 0, 1)
    rgb[hit_b] = (np.array([0.2, 0.45, 0.9]) * shade)[hit_b]
    alpha[hit_b] = 1.0

    return np.concatenate([rgb, alpha[..., None]], -1)


def render_aa(c2w, H, W, focal, s, *args):
    """1 画素を s×s 本のレイで描いて平均する（輪郭と模様の境目のギザギザを消す）。s=1 なら render と同じ。

    色は不透明度を掛けてから平均し、平均した不透明度で割り戻す（PNG の RGBA は掛けていない色を持つので）。
    """
    if s == 1:
        return render(c2w, H, W, focal, *args)
    img = render(c2w, H * s, W * s, focal * s, *args).reshape(H, s, W, s, 4)
    a = img[..., 3].mean((1, 3))
    pm = (img[..., :3] * img[..., 3:]).mean((1, 3))
    rgb = np.where(a[..., None] > 0, pm / np.maximum(a[..., None], 1e-12), 1.0)
    return np.concatenate([rgb, a[..., None]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/toy_dynamic"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--supersample", type=int, default=4,
                    help="1 画素を何×何本のレイで描いて平均するか（1 だと輪郭がギザギザになり、ガウシアンでは再現できない）")
    ap.add_argument("--radius", type=float, default=5.5)
    ap.add_argument("--orbit", action="store_true", help="立方体を半周させる（大きく動く例）")
    args = ap.parse_args()

    fov = np.deg2rad(40.0)
    focal = 0.5 * args.size / np.tan(0.5 * fov)
    rng = np.random.default_rng(0)

    for split, n in [("train", args.n_train), ("val", 8), ("test", args.n_test)]:
        (args.out / split).mkdir(parents=True, exist_ok=True)
        frames = []
        for k in range(n):
            if split == "train":
                t = k / (n - 1)
                az, el = rng.uniform(0, 360), rng.uniform(5, 80)   # 上半球からランダム
            else:
                t = (k + 0.5) / n                                  # 学習に無い中間の時刻
                az, el = 360.0 * k / n + (15.0 if split == "test" else 40.0), 30.0
            c2w = look_at_origin(az, el, args.radius)
            img = render_aa(c2w, args.size, args.size, focal, args.supersample, t, args.orbit)
            imageio.imwrite(args.out / split / f"r_{k:03d}.png", (img * 255).round().astype(np.uint8))
            frames.append({"file_path": f"./{split}/r_{k:03d}", "time": float(t),
                           "transform_matrix": c2w.tolist()})
        with open(args.out / f"transforms_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"camera_angle_x": float(fov), "frames": frames}, f, indent=2)
        print(f"{split}: {n} 枚 -> {args.out / split}")


if __name__ == "__main__":
    main()
