# -*- coding: utf-8 -*-
"""
学習したガウシアンの動きを、おもちゃデータの正解の動きと突き合わせる。

make_toy_dynamic.py のシーンは動きの式が分かっている。物体に付いた点 x の、時刻 t0 から t1 への移動 Φ は:
    球     中心からの向きを保って、中心と半径だけ変わる   Φ(x) = c(t1) + (r(t1)/r(t0)) (x - c(t0))
    立方体 平行移動                                       Φ(x) = x + b(t1) - b(t0)
画像の再現度（PSNR）が高くても、ガウシアンが物体と一緒に動いているとは限らない。ここでは位置を直接比べる。

時刻 t に面の近く（|符号付き距離| < --tol）にあるガウシアン i について、Δ だけ後の位置を比べる:
    err   = |p_i(t+Δ) - Φ(p_i(t))|       動きの誤差
    move  = |Φ(p_i(t)) - p_i(t)|         正解の移動量（＝まったく動かさないモデルの誤差）
    ratio = Σ w err / Σ w move           0 なら正解どおり、1 なら動かさないのと同じ
重み w は 2 つの時刻の実効不透明度の幾何平均。時刻を持つガウシアン（--spacetime）は Δ が時間幅より
長いと消えているので、残っている割合 alive = Σ w / Σ w(t) も出す（変形場のモデルは常に 1）。

err と move は、面の法線方向と接線方向にも分けて出す（ratio_n, ratio_t）。一様な色の面の上では、
接線方向に滑っても画像は変わらないので、接線方向の動きは画像からは決まらない。

物体がそもそも表せているかは cover で見る: 正解の面の上の点から、いちばん近い（実効不透明度 > 0.1 の）
ガウシアンまでの距離の平均。物体が消えていると大きくなる。

    python experiments/dynamic/motion_eval.py --ckpts logs/dyn_toy/ckpt_007000.pt logs/dyn_toy_st/ckpt_007000.pt
    python experiments/dynamic/motion_eval.py --orbit --ckpts logs/dyn_orbit/ckpt_005000.pt ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import make_toy_dynamic as toy  # noqa: E402
from gs.deform import DeformNet  # noqa: E402
from gs.gaussians import GaussianModel  # noqa: E402
from gs.spacetime import SpacetimeModel  # noqa: E402

OBJS = ("sphere", "cube")
DELTAS = (0.025, 0.1, 0.25, 0.5)


# ---------------------------------------------------------------- 正解の形と動き
def sdf(x, t, orbit):
    """(N,3) -> 球と立方体への符号付き距離 (N,2)"""
    c, r, b = toy.scene(t, orbit)
    d_s = np.linalg.norm(x - c, axis=1) - r
    q = np.abs(x - b) - toy.BOX_H
    d_b = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    return np.stack([d_s, d_b], 1)


def normal(x, obj, t, orbit, eps=1e-4):
    """いちばん近い物体の符号付き距離の勾配（面の法線）"""
    g = np.zeros_like(x)
    for a in range(3):
        e = np.zeros(3)
        e[a] = eps
        g[:, a] = np.take_along_axis(sdf(x + e, t, orbit) - sdf(x - e, t, orbit), obj[:, None], 1)[:, 0]
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)


def flow(x, obj, t0, t1, orbit):
    """物体 obj（0: 球, 1: 立方体）に付いた点 x の、t0 から t1 への移動先"""
    c0, r0, b0 = toy.scene(t0, orbit)
    c1, r1, b1 = toy.scene(t1, orbit)
    on_sphere = c1 + (r1 / r0) * (x - c0)
    return np.where((obj == 0)[:, None], on_sphere, x + b1 - b0)


def surface_points(t, orbit, n=2000, seed=0):
    """正解の面の上の点（球, 立方体）"""
    rng = np.random.default_rng(seed)
    c, r, b = toy.scene(t, orbit)
    v = rng.normal(size=(n, 3))
    s = c + r * v / np.linalg.norm(v, axis=1, keepdims=True)
    q = rng.uniform(-1, 1, size=(n, 3))
    ax = rng.integers(0, 3, n)
    q[np.arange(n), ax] = rng.choice([-1.0, 1.0], n)
    return s, b + toy.BOX_H * q


# ---------------------------------------------------------------- 学習したモデル
class Run:
    def __init__(self, path, dev):
        ck = torch.load(path, map_location=dev, weights_only=False)
        a = ck["args"]
        self.name = Path(path).parent.name
        self.net = None
        if a.get("spacetime"):
            self.kind = "spacetime"
            self.model = SpacetimeModel.from_state_dict(ck["model"], dev)
        else:
            self.kind = "static" if ck["deform"] is None else "deform"
            self.model = GaussianModel.from_state_dict(ck["model"], dev)
            if ck["deform"] is not None:
                self.net = DeformNet(a["deform_depth"], a["deform_width"]).to(dev)
                self.net.load_state_dict(ck["deform"])

    @torch.no_grad()
    def at(self, t):
        """時刻 t の位置 (N,3) と実効不透明度 (N,)"""
        m = self.model
        x, w = m.params["means"], m.opacities
        if self.kind == "spacetime":
            d = m.delta_at(t)
            x, w = x + d["means"], w * d["opacity"]
        elif self.net is not None:
            x = x + self.net(x, t)["means"]
        return x.double().cpu().numpy(), w.double().cpu().numpy()


def evaluate(run, orbit, tol, n_t=41):
    ts = np.linspace(0, 1, n_t)
    state = [run.at(t) for t in ts]
    res = {"kind": run.kind, "n_gauss": len(run.model)}

    # 物体が表せているか: 正解の面からいちばん近いガウシアンまでの距離
    cover = np.zeros((n_t, 2))
    for k, t in enumerate(ts):
        x, w = state[k]
        x = torch.as_tensor(x[w > 0.1])
        for o, pts in enumerate(surface_points(t, orbit)):
            cover[k, o] = torch.cdist(torch.as_tensor(pts), x).min(1).values.mean() if len(x) else np.inf
    res["cover"] = dict(zip(OBJS, cover.mean(0).tolist()))

    # 動きの誤差
    res["motion"] = {}
    for delta in DELTAS:
        step = int(round(delta * (n_t - 1)))
        acc = np.zeros((2, 8))   # 物体ごとに Σ w {err, move, 1}, Σ w(t), Σ w {err, move} の法線・接線成分
        for k in range(n_t - step):
            (x0, w0), (x1, w1) = state[k], state[k + step]
            d = sdf(x0, ts[k], orbit)
            obj = np.abs(d).argmin(1)
            near = np.abs(d).min(1) < tol
            target = flow(x0, obj, ts[k], ts[k + step], orbit)
            n = normal(x0, obj, ts[k], orbit)
            split = lambda v: (np.abs((v * n).sum(1)), np.linalg.norm(v - (v * n).sum(1, keepdims=True) * n, axis=1))
            err, move = np.linalg.norm(x1 - target, axis=1), np.linalg.norm(target - x0, axis=1)
            (err_n, err_t), (move_n, move_t) = split(x1 - target), split(target - x0)
            w = np.sqrt(w0 * w1)
            for o in range(2):
                m = near & (obj == o)
                acc[o] += [(w * err)[m].sum(), (w * move)[m].sum(), w[m].sum(), w0[m].sum()]                     + [(w * v)[m].sum() for v in (err_n, move_n, err_t, move_t)]
        res["motion"][str(delta)] = {
            name: {"err": a[0] / max(a[2], 1e-12), "move": a[1] / max(a[2], 1e-12),
                   "ratio": a[0] / max(a[1], 1e-12), "alive": a[2] / max(a[3], 1e-12),
                   "ratio_n": a[4] / max(a[5], 1e-12), "ratio_t": a[6] / max(a[7], 1e-12)}
            for name, a in zip(OBJS, acc)}

    if run.kind == "spacetime":
        p = run.model.params
        sig = run.model.t_scales.squeeze(-1)
        res["t_scale_median"] = sig.median().item()
        res["t_scale_quartiles"] = torch.quantile(sig, torch.tensor([0.25, 0.75], device=sig.device)).tolist()
        res["speed_median"] = p["velocity"].norm(dim=1).median().item()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--orbit", action="store_true", help="--orbit で作ったデータ（立方体が半周する）で学習したもの")
    ap.add_argument("--tol", type=float, default=0.1, help="面からこの距離までのガウシアンを数える")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for path in args.ckpts:
        run = Run(path, dev)
        results[run.name] = evaluate(run, args.orbit, args.tol)

    out = args.out or HERE / ("motion_orbit.json" if args.orbit else "motion_toy.json")
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    print(f"{'':22s} {'cover':>13s} | " + " | ".join(f"Δ={d:<5} ratio (alive)".center(29) for d in DELTAS))
    print(f"{'':22s} {'球':>6s} {'立方体':>5s} | " + " | ".join(f"{'球':^13s} {'立方体':^12s}" for _ in DELTAS))
    for name, r in results.items():
        cells = []
        for d in DELTAS:
            m = r["motion"][str(d)]
            cells.append(" ".join(f"{m[o]['ratio']:5.2f} ({m[o]['alive']:4.2f}) " for o in OBJS))
        print(f"{name:22s} {r['cover']['sphere']:6.3f} {r['cover']['cube']:6.3f} | " + " | ".join(cells))
        m = r["motion"]["0.1"]
        print(f"{'':22s} Δ=0.1 の法線 / 接線: " + "、".join(
            f"{j} {m[o]['ratio_n']:.2f} / {m[o]['ratio_t']:.2f}" for o, j in zip(OBJS, ("球", "立方体"))))
        if "t_scale_median" in r:
            print(f"{'':22s} 時間幅 σ_t の中央値 {r['t_scale_median']:.3f}、速さの中央値 {r['speed_median']:.2f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
