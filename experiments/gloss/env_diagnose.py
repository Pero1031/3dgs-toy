# -*- coding: utf-8 -*-
"""
共有の照明（env_oracle.py）がこの金の像で効かなかった理由を切り分ける。

候補:
    1. 画素より細かい凹凸: 1 画素（約 2.7 mm）の中で法線が大きく変わると、画素の色はいろいろな方向の映り込みの平均になり、
       点の法線 1 本の反射方向では予測できない
    2. 相互反射・遮蔽: 反射方向の先が光源ではなく像自身で、照明が見えていない
    3. 照明が無限遠ではない（同じ光源でも点の位置によって見える方向がずれる）

各観測について、点から反射方向に飛ばした光線が
    - 光源（4 つの球とソフトボックス）に当たるか（真の位置で計算するので 3 の誤差は入らない）
    - 途中で像自身に遮られるか（像を 2 mm の格子にして光線を進める近似）
を幾何的に調べ、各点について画素の大きさの範囲での法線のばらつきを測る。
そのうえで、遮られていない観測・凹凸の少ない点だけに絞って、点ごとの SH と共有の照明を比べ直す。

    python experiments/gloss/env_diagnose.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env_oracle as eo  # noqa: E402
import sh_oracle as so  # noqa: E402

DEV = so.DEV
VOXEL = 0.002        # 遮蔽判定の格子の大きさ [m]
T_START = 0.008      # 自分の表面の格子を抜けるまで、遮蔽の判定を始めない距離 [m]
SPREAD_R = 0.00135   # 法線のばらつきを測る半径 [m]（1 画素の幅の半分）
SPREAD_LOW = 5.0     # これ未満を「凹凸の少ない点」とする [度]
SIDE = [np.array([1.9 * math.sin(math.radians(45 + 90 * k)), 1.2, 1.9 * math.cos(math.radians(45 + 90 * k))])
        for k in range(4)]


def light_hit(P, r):
    """点 P から方向 r の光線が、球光源（半径 0.35 m）かソフトボックス（y=1.9、1.8 m 四方）に当たるか"""
    hit = torch.zeros(r.shape[:-1], dtype=torch.bool, device=DEV)
    for c in SIDE:
        oc = torch.tensor(c, dtype=torch.float32, device=DEV) - P
        t = (oc * r).sum(-1)
        d2 = (oc * oc).sum(-1) - t * t
        hit |= (t > 0) & (d2 < 0.35 ** 2)
    t = (1.9 - P[..., 1]) / r[..., 1].clamp(min=1e-6)
    x, z = P[..., 0] + t * r[..., 0], P[..., 2] + t * r[..., 2]
    hit |= (r[..., 1] > 0) & (x.abs() < 0.9) & (z.abs() < 0.9)
    return hit


def occupancy(D, lo, hi):
    """表面の密な点 D から、像の殻の格子（1 格子ぶん膨らませて隙間をふさぐ）を作る"""
    shape = tuple(int(x) for x in np.ceil((hi - lo) / VOXEL))
    idx = ((D - torch.tensor(lo, device=DEV)) / VOXEL).long()
    for k in range(3):
        idx[:, k] = idx[:, k].clamp(0, shape[k] - 1)
    occ = torch.zeros(shape, device=DEV)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    occ = torch.nn.functional.max_pool3d(occ[None, None], 3, 1, 1)[0, 0] > 0
    return occ, torch.tensor(lo, dtype=torch.float32, device=DEV), shape


def occluded(P, r, cos_rn, occ, lo, shape, step=0.001, t_max=0.8):
    """点 P から方向 r に進めて、自分の表面から離れた先で像の殻の格子に入れば遮られたとする。
    判定を始める距離は T_START と「表面から 5 mm 離れる距離」の大きい方（すれすれに出ていく光線の誤判定を防ぐ）"""
    out = torch.zeros(r.shape[0], dtype=torch.bool, device=DEV)
    dims = torch.tensor(shape, device=DEV)
    t0 = torch.maximum(torch.full_like(cos_rn, T_START), 0.005 / cos_rn.clamp(min=0.05))
    for t in np.arange(T_START, t_max, step):
        q = ((P + t * r - lo) / VOXEL).long()
        inside = ((q >= 0) & (q < dims)).all(-1) & (t >= t0)
        if not inside.any():
            if t > t0.max().item():
                break
            continue
        qi = q[inside]
        out[inside] |= occ[qi[:, 0], qi[:, 1], qi[:, 2]]
    return out


def main():
    v, f = so.load_mesh()
    P, N = so.sample_surface(v, f, 10000, seed=1)
    Ns = eo.smooth_normals(10000, seed=1)
    D, ND = so.sample_surface(v, f, 2_000_000, seed=2)

    # 各点で、画素の幅の範囲にある表面の点の法線のばらつき（平均ベクトルの長さから角度に直す）
    tree = cKDTree(D)
    spread = np.zeros(len(P))
    for i, nb in enumerate(tree.query_ball_point(P, SPREAD_R)):
        m = ND[nb].mean(0) if len(nb) else N[i]
        spread[i] = math.degrees(math.acos(min(1.0, np.linalg.norm(m))))
    print(f"画素の範囲での法線のばらつき: 中央値 {np.median(spread):.1f}°, 90% 点 {np.percentile(spread, 90):.1f}°")

    lo, hi = v.min(0) - 0.01, v.max(0) + 0.01
    P, N, Ns, Dt = (torch.tensor(x, dtype=torch.float32, device=DEV) for x in (P, N, Ns, D))
    occ, lo_t, shape = occupancy(Dt, lo, hi)
    masks = {s: so.load_masks(so.ROOT / "data" / "gloss_matte", s) for s in ("train", "test")}

    obs = {}
    for split in ("train", "test"):
        cams = so.load_cameras(so.ROOT / "data" / "gloss_gold", split, False, so.BG, DEV)
        obs[split] = so.observe(cams, masks[split], P, N, Dt)
    keep = obs["train"][0].sum(0) >= so.MIN_TRAIN_OBS
    P, N, Ns = P[keep], N[keep], Ns[keep]
    spread = torch.tensor(spread[keep.cpu().numpy()], device=DEV)
    obs = {s: [x[:, keep] for x in o] for s, o in obs.items()}

    # 各観測の反射光線: 光源に当たるか、像に遮られるか
    flags = {}
    for split, (vis, col, dirs) in obs.items():
        r = eo.reflect(dirs, Ns)
        V, n = vis.shape
        Pb = P[None].expand(V, n, 3)
        hit = light_hit(Pb, r)
        cos_rn = (r * Ns[None]).sum(-1)
        occl = occluded(Pb.reshape(-1, 3), r.reshape(-1, 3), cos_rn.reshape(-1), occ, lo_t, shape).reshape(V, n)
        flags[split] = (hit, occl)

    # 観測の明るさを、光線の行き先と凹凸ごとに比べる
    vis, col, _ = obs["train"]
    hit, occl = flags["train"]
    lum = col @ so.LUMA.to(DEV)
    low = (spread < SPREAD_LOW)[None].expand_as(vis)
    stats = {}
    print(f"\n学習視点の観測（全 {int(vis.sum())} 個）の輝度（画像の値）")
    print(f"{'光線の行き先':<16} {'点の凹凸':<10} {'割合':>6} {'中央値':>7} {'25%':>6} {'75%':>6}")
    for name, m_dest in (("光源（遮られない）", hit & ~occl), ("暗い所（遮られない）", ~hit & ~occl), ("像自身に遮られる", occl)):
        for sname, m_sp in (("少ない", low), ("多い", ~low)):
            m = vis & m_dest & m_sp
            x = lum[m].cpu().numpy()
            if len(x) == 0:
                continue
            q = np.percentile(x, [25, 50, 75])
            stats[f"{name}/{sname}"] = {"frac": float(m.sum() / vis.sum()), "q25": q[0], "median": q[1], "q75": q[2]}
            print(f"{name:<16} {sname:<10} {100 * m.sum().item() / vis.sum().item():5.1f}% {q[1]:7.3f} {q[0]:6.3f} {q[2]:6.3f}")

    # 絞り込み: 遮られない観測だけ、凹凸の少ない点だけで、点ごとの SH と共有の照明を比べる
    def subset(pts, drop_occl):
        o = {}
        for split, (v_, c_, d_) in obs.items():
            vv = v_ & ~flags[split][1] if drop_occl else v_.clone()
            o[split] = [vv[:, pts], c_[:, pts], d_[:, pts]]
        ok = o["train"][0].sum(0) >= so.MIN_TRAIN_OBS
        return {s: [x[:, ok] for x in o[s]] for s in o}, Ns[pts][ok]

    def per_point_best(o):
        best = -1.0
        for L in (2, 3, 4, 6):
            c = so.fit(L, *o["train"], smooth=so.SMOOTH_FIG)
            best = max(best, so.psnr(so.predict(L, c, o["test"][2]), o["test"][1], o["test"][0]))
        return best

    def shared(o, Nn, Lp):
        es = eo.fit(o["train"], Nn, Lp, 32, [1.0])
        return eo.evaluate(o["train"], o["test"], Nn, Lp, 32, es)[1.0]["test"]

    allp = torch.ones(len(P), dtype=torch.bool, device=DEV)
    cases = [("すべての点・すべての観測", allp, False),
             ("すべての点・遮られない観測だけ", allp, True),
             (f"凹凸の少ない点（< {SPREAD_LOW:g}°）・遮られない観測だけ", spread < SPREAD_LOW, True)]
    comp = {}
    print(f"\n{'集合':<36} {'点':>5} {'点ごとの SH':>10} {'定数+共有':>9} {'SH2+共有':>9}  （テスト視点 PSNR [dB]）")
    for name, pts, drop in cases:
        o, Nn = subset(pts, drop)
        row = {"n_points": int(Nn.shape[0]), "per_point_sh": per_point_best(o),
               "const_plus_shared": shared(o, Nn, 0), "sh2_plus_shared": shared(o, Nn, 2)}
        comp[name] = row
        print(f"{name:<36} {row['n_points']:>5} {row['per_point_sh']:>10.2f} {row['const_plus_shared']:>9.2f} "
              f"{row['sh2_plus_shared']:>9.2f}", flush=True)

    out = {"spread_deg": {"median": float(spread.median()), "p90": float(torch.quantile(spread, 0.9))},
           "occluded_frac_train": float((occl & vis).sum() / vis.sum()), "lum_stats": stats, "subsets": comp}
    (HERE / "env_diagnose.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {HERE / 'env_diagnose.json'}")


if __name__ == "__main__":
    main()
