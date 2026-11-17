# -*- coding: utf-8 -*-
"""
「共有の照明」版の当てはめ: 反射方向で引く 1 つの照明関数を全点で共有すると、
点ごとの SH（sh_oracle.py）の天井を越えられるかを、3DGS を使わずに測る。

モデル（点 p、視点 v の観測色 y）:
    y(p, v) ≈ Σ_k c_{p,k} Y_k(d)   +   Σ_j e_j Y_j(r)
              点ごとの低次の項          全点で共有する照明（反射方向の高次 SH）
    d: カメラ → 点の向き、r = 2(ω·n)n − ω（ω = −d、n は真の法線）: 視線を法線で折り返した方向

    - 金の映り込みは「反射方向に何があるか」で決まり、どの点も同じ照明を見ているので、照明 E を全点で共有する。
      点ごとには 26 回しかない観測が、全点では 20 万回の照明の観測になる。
    - 拡散の成分は反射方向ではなく法線で決まるので、点ごとの項に任せる（拡散と鏡面の分解）。

解き方: 点ごとの係数 c_p は、共有の係数 e を決めれば点ごとの小さな最小二乗で閉じた形に決まるので、
先に消去する（Schur 補元）。残る e の正規方程式（最大 2401 × 2401）を 1 回解き、c_p は後から求める。

    python experiments/gloss/env_oracle.py

出力: experiments/gloss/env_oracle.json, figures/env_oracle.png, figures/env_map.png
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sh_oracle as so  # noqa: E402  観測の集め方・評価・図の色は前回の検証と共通
from gs.sh import sh_basis  # noqa: E402

DEV = so.DEV
S_LOC = 0.03                          # 点ごとの項のなめらかさの罰（sh_oracle で最も良かった値）
LOCAL_DEGREES = [-1, 0, 2]            # 点ごとの項の次数（-1 = 点ごとの項なし、0 = 点ごとの定数）
ENV_DEGREES = [16, 32, 48]            # 共有の照明の SH の次数（解像度はおよそ 180°/L）
ENV_SMOOTH = [0.1, 1.0, 10.0]         # 共有の照明のなめらかさの罰
CHUNK = 128                           # 一度に処理する点の数


def smooth_normals(n, seed):
    """sample_surface(…, n, seed) と同じ点での、なめらかな（頂点法線を補間した）法線。

    GoLD_rayt は三角形の 3 頂点の法線を重心座標で補間して反射を計算する（TriangleMesh.hpp の fill）。
    ほぼ鏡面の金では、面の法線との数度の差が反射方向では 2 倍の差になるので、レンダラーと同じ法線を使う。
    """
    from geom_eval import OBJ, ROT_DEG, SCALE, TRANS
    v, vn, f = [], [], []
    with open(OBJ, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vn "):
                vn.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([[int(i) - 1 for i in p.split("/")[::2]] for p in line.split()[1:4]])  # v//vn
    v, vn, f = np.array(v) * SCALE, np.array(vn), np.array(f)  # f: (F, 3 頂点, [v, vn])
    c, s = math.cos(math.radians(ROT_DEG)), math.sin(math.radians(ROT_DEG))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    v, vn = v @ R.T + TRANS, vn @ R.T
    a, b, cc = v[f[:, 0, 0]], v[f[:, 1, 0]], v[f[:, 2, 0]]
    cross = np.cross(b - a, cc - a)
    area = np.linalg.norm(cross, axis=1) / 2
    rng = np.random.default_rng(seed)  # sample_surface と同じ乱数の使い方で、同じ三角形・同じ重心座標を引く
    idx = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    ns = (w[:, :, None] * vn[f[idx, :, 1]]).sum(1)
    ns /= np.linalg.norm(ns, axis=1, keepdims=True)
    ng = cross[idx] / np.linalg.norm(cross[idx], axis=1, keepdims=True)
    ns *= np.sign((ns * ng).sum(1, keepdims=True))  # 面の法線と同じ側（外向き）にそろえる
    return ns


def reflect(dirs, N):
    """視線 d（カメラ → 点）を法線 n で折り返した方向 r = 2(ω·n)n − ω、ω = −d"""
    w = -dirs
    return 2 * (w * N[None]).sum(-1, keepdim=True) * N[None] - w


def penalty(L, s):
    l = torch.tensor([math.isqrt(k) for k in range((L + 1) ** 2)], dtype=torch.float64, device=DEV)
    return torch.diag(so.LAMBDA + s * l * (l + 1))


def blocks(obs, N, Lp, Le, sl):
    """点の範囲 sl について、点ごとの基底 Bl (V,m,Kl)、共有の基底 Be (V,m,Ke)、重み w、観測 y を作る"""
    vis, col, dirs = obs
    V = vis.shape[0]
    d, n = dirs[:, sl], N[sl]
    m = d.shape[1]
    Be = sh_basis(Le, reflect(d, n).reshape(-1, 3)).float().reshape(V, m, -1)
    Bl = sh_basis(Lp, d.reshape(-1, 3)).double().reshape(V, m, -1) if Lp >= 0 else None
    return Bl, Be, vis[:, sl].double(), col[:, sl]


def fit(obs, N, Lp, Le, smooths):
    """共有の照明 e（なめらかさの罰ごと）を解く -> {s: e (Ke,3)}"""
    n = obs[0].shape[1]
    Ke = (Le + 1) ** 2
    S = torch.zeros(Ke, Ke, dtype=torch.float64, device=DEV)
    rhs = torch.zeros(Ke, 3, dtype=torch.float64, device=DEV)
    Rl = penalty(Lp, S_LOC) if Lp >= 0 else None
    for i in range(0, n, CHUNK):
        Bl, Be, w, y = blocks(obs, N, Lp, Le, slice(i, i + CHUNK))
        Ew = (Be * w[..., None].float()).reshape(-1, Ke)
        S += (Ew.T @ Be.reshape(-1, Ke)).double()            # Σ Eᵀ W E（全観測）
        rhs += (Ew.T @ y.reshape(-1, 3).float()).double()     # Σ Eᵀ W y
        if Bl is None:
            continue
        # 点ごとの項を消去: 各点で A_p = Blᵀ W Bl + R、G_p = L_p⁻¹ (Blᵀ W E)、g_p = L_p⁻¹ (Blᵀ W y)
        # とすると、e の正規方程式から Σ G_pᵀ G_p と Σ G_pᵀ g_p を引けばよい（L_p は A_p のコレスキー因子）
        Blw = Bl * w[..., None]
        L = torch.linalg.cholesky(torch.einsum("vmk,vmj->mkj", Blw, Bl) + Rl)
        M = torch.einsum("vmk,vmj->mkj", Blw.float(), Be).double()
        G = torch.linalg.solve_triangular(L, M, upper=False).reshape(-1, Ke)
        g = torch.linalg.solve_triangular(L, torch.einsum("vmk,vmc->mkc", Blw, y), upper=False).reshape(-1, 3)
        S -= (G.T.float() @ G.float()).double()
        rhs -= G.T @ g
    return {s: torch.linalg.solve(S + penalty(Le, s), rhs) for s in smooths}


def evaluate(obs_tr, obs_te, N, Lp, Le, es):
    """点ごとの係数を求めて（共有の e を固定した点ごとの最小二乗）、学習視点・テスト視点の PSNR を出す"""
    n = obs_tr[0].shape[1]
    Rl = penalty(Lp, S_LOC) if Lp >= 0 else None
    E = torch.cat([es[s] for s in es], 1)  # (Ke, 3 × 罰の数): 罰ごとの照明をまとめて評価する
    se = {s: {"train": [0.0, 0.0], "test": [0.0, 0.0]} for s in es}
    for i in range(0, n, CHUNK):
        sl = slice(i, i + CHUNK)
        tr, te = blocks(obs_tr, N, Lp, Le, sl), blocks(obs_te, N, Lp, Le, sl)
        env_tr = (tr[1] @ E.float()).double()
        env_te = (te[1] @ E.float()).double()
        for k, s in enumerate(es):
            ptr, pte = env_tr[..., 3 * k:3 * k + 3], env_te[..., 3 * k:3 * k + 3]
            if Lp >= 0:
                Bl, _, w, y = tr
                Blw = Bl * w[..., None]
                c = torch.linalg.solve(torch.einsum("vmk,vmj->mkj", Blw, Bl) + Rl,
                                       torch.einsum("vmk,vmc->mkc", Blw, y - ptr))
                ptr = ptr + torch.einsum("vmk,mkc->vmc", Bl, c)
                pte = pte + torch.einsum("vmk,mkc->vmc", te[0], c)
            for split, (pred, (_, _, w, y)) in (("train", (ptr, tr)), ("test", (pte, te))):
                err = ((pred.clamp(0, 1) - y) ** 2).mean(-1)
                se[s][split][0] += (err * w).sum().item()
                se[s][split][1] += w.sum().item()
    return {s: {sp: -10 * math.log10(a / b) for sp, (a, b) in v.items()} for s, v in se.items()}


def light_outlines():
    """像の中心から見た光源の輪郭（方位・仰角、度）。シーンは make_views.py と同じ"""
    c = np.array([0.0, 0.25, 0.0])
    to_ae = lambda v: (np.degrees(np.arctan2(v[..., 0], v[..., 2])),  # noqa: E731
                       np.degrees(np.arcsin(v[..., 1] / np.linalg.norm(v, axis=-1))))
    out = []
    t = np.linspace(0, 2 * np.pi, 120)
    for k in range(4):  # 球光源: 中心方向のまわりの角半径 asin(r / 距離) の円
        a = np.radians(45 + 90 * k)
        p = np.array([1.9 * np.sin(a), 1.2, 1.9 * np.cos(a)]) - c
        dist = np.linalg.norm(p)
        z = p / dist
        x = np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rad = np.arcsin(0.35 / dist)
        v = np.cos(rad) * z + np.sin(rad) * (np.cos(t)[:, None] * x + np.sin(t)[:, None] * y)
        out.append(to_ae(v))
    s = np.linspace(-0.9, 0.9, 60)  # ソフトボックス: 1.8 m 四方の正方形の縁
    edge = np.concatenate([np.stack([s, np.full_like(s, 0.9)], 1), np.stack([np.full_like(s, 0.9), -s], 1),
                           np.stack([-s, np.full_like(s, -0.9)], 1), np.stack([np.full_like(s, -0.9), s], 1)])
    v = np.stack([edge[:, 0], np.full(len(edge), 1.9), edge[:, 1]], 1) - c
    out.append(to_ae(v))
    return out


def plot_env(e, Le, refl_tr, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9})
    az = torch.linspace(-180, 180, 721, dtype=torch.float64)
    el = torch.linspace(-90, 90, 361, dtype=torch.float64)
    E_, A_ = torch.meshgrid(torch.deg2rad(el), torch.deg2rad(az), indexing="ij")
    dirs = torch.stack([torch.cos(E_) * torch.sin(A_), torch.sin(E_), torch.cos(E_) * torch.cos(A_)], -1)
    img = (sh_basis(Le, dirs.reshape(-1, 3).to(DEV)) @ e).clamp(0, 1).reshape(len(el), len(az), 3).cpu().numpy()

    # 学習視点の観測の反射方向がどこに落ちたか（そこ以外の照明は観測から決まらない）
    r = refl_tr.cpu().double()
    ia = ((torch.rad2deg(torch.atan2(r[:, 0], r[:, 2])) + 180) / 360 * 180).long().clamp(0, 179)
    ie = ((torch.rad2deg(torch.asin(r[:, 1].clamp(-1, 1))) + 90) / 180 * 90).long().clamp(0, 89)
    cov = torch.zeros(90, 180).index_put_((ie, ia), torch.ones(len(ia)), accumulate=True).numpy() > 0

    fig, ax = plt.subplots(figsize=(8.0, 4.3), dpi=200, facecolor=so.C_SURF, constrained_layout=True)
    ax.imshow(img, extent=[-180, 180, -90, 90], origin="lower", aspect="auto", interpolation="bilinear")
    ax.imshow(np.where(cov, np.nan, 1.0), extent=[-180, 180, -90, 90], origin="lower", aspect="auto",
              cmap="Greys", vmin=0, vmax=3, alpha=0.6, interpolation="nearest")
    for k, (a, e_) in enumerate(light_outlines()):
        ax.plot(a, e_, color=so.C_SERIES["matte"], lw=1.4, ls=(0, (3, 2)),
                label="真の光源の位置（像の中心から見た輪郭）" if k == 0 else None)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("反射方向の方位（度）")
    ax.set_ylabel("反射方向の仰角（度）")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    ax.set_title(f"金の像の映り込みから復元した共有の照明 E(r)（SH {Le} 次）。灰色の帯 = 学習視点の映り込みが一度も落ちていない方向",
                 fontsize=8.5, loc="left")
    fig.savefig(path, facecolor=so.C_SURF)
    plt.close(fig)


def plot_bars(rows, path):
    """rows: [(モデル名, {cond: test PSNR})]。横向きの棒を条件ごとに並べる"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9,
                         "axes.edgecolor": so.C_TEXT2, "xtick.color": so.C_TEXT2, "ytick.color": so.C_TEXT})
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(rows) + 1.0), dpi=200, facecolor=so.C_SURF)
    ax.set_facecolor(so.C_SURF)
    h = 0.36
    for k, cond in enumerate(("gold", "matte")):
        ys = [i + (k - 0.5) * (h + 0.04) for i in range(len(rows))]
        vals = [r[1][cond] for r in rows]
        ax.barh(ys, vals, height=h, color=so.C_SERIES[cond], label=so.LABEL[cond])
        for y, v in zip(ys, vals):
            ax.text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=7.5, color=so.C_TEXT)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("テスト視点の観測色の PSNR [dB]")
    ax.set_xlim(0, max(v for r in rows for v in r[1].values()) + 6)
    ax.grid(axis="x", color=so.C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("点ごとの SH と共有の照明: テスト視点での再現度（各モデルで最も良い設定）", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=so.C_SURF)
    plt.close(fig)


def main():
    v, f = so.load_mesh()
    P, N = so.sample_surface(v, f, 10000, seed=1)  # sh_oracle.py と同じ点（N は面の法線: 可視判定に使う）
    Ns = smooth_normals(10000, seed=1)             # なめらかな法線: 反射方向に使う
    ang = np.degrees(np.arccos(np.clip((N * Ns).sum(1), -1, 1)))
    print(f"面の法線となめらかな法線の差: 中央値 {np.median(ang):.1f}°, 90% 点 {np.percentile(ang, 90):.1f}°")
    D, _ = so.sample_surface(v, f, 2_000_000, seed=2)
    P, N, Ns, D = (torch.tensor(x, dtype=torch.float32, device=DEV) for x in (P, N, Ns, D))
    masks = {s: so.load_masks(so.ROOT / "data" / "gloss_matte", s) for s in ("train", "test")}
    prev = json.loads((HERE / "sh_oracle.json").read_text(encoding="utf-8"))

    res = {"local_smooth": S_LOC, "env_degrees": ENV_DEGREES, "env_smooth": ENV_SMOOTH,
           "normal_diff_deg": {"median": float(np.median(ang)), "p90": float(np.percentile(ang, 90))}, "conditions": {}}
    best_gold = None
    for cond in ("gold", "matte"):
        obs = {}
        for split in ("train", "test"):
            cams = so.load_cameras(so.ROOT / "data" / f"gloss_{cond}", split, False, so.BG, DEV)
            obs[split] = so.observe(cams, masks[split], P, N, D)
        keep = obs["train"][0].sum(0) >= so.MIN_TRAIN_OBS
        tr = [x[:, keep] for x in obs["train"]]
        te = [x[:, keep] for x in obs["test"]]
        Nk = Ns[keep]
        out = {}
        for Lp in LOCAL_DEGREES:
            for Le in ENV_DEGREES:
                es = fit(tr, Nk, Lp, Le, ENV_SMOOTH)
                ps = evaluate(tr, te, Nk, Lp, Le, es)
                for s in ENV_SMOOTH:
                    out[f"Lp{Lp}_Le{Le}_s{s:g}"] = ps[s]
                    print(f"{cond:<6} 点ごと L={Lp:2d}  照明 L={Le}  罰 {s:5g}:  学習 {ps[s]['train']:6.2f}  "
                          f"テスト {ps[s]['test']:6.2f} dB", flush=True)
                    if cond == "gold" and Lp == 0 and (best_gold is None or ps[s]["test"] > best_gold[0]):
                        best_gold = (ps[s]["test"], Le, s, es[s])
        res["conditions"][cond] = out
        if cond == "gold":
            refl_tr = reflect(tr[2], Nk)[tr[0]]  # 学習視点で見えている観測の反射方向

    # 図: 各モデルで最も良い設定のテスト視点 PSNR
    def best(cond, Lp):
        return max(v["test"] for k, v in res["conditions"][cond].items() if k.startswith(f"Lp{Lp}_"))

    pp = {c: max(max(prev["conditions"][c]["test_psnr"]),
                 max(x["best_test_psnr"] for x in prev["conditions"][c]["smooth_check"].values()))
          for c in ("gold", "matte")}
    rows = [("点ごとの SH（前回の最良）", pp),
            ("共有の照明だけ", {c: best(c, -1) for c in ("gold", "matte")}),
            ("点ごとの定数 + 共有の照明", {c: best(c, 0) for c in ("gold", "matte")}),
            ("点ごとの SH 2 次 + 共有の照明", {c: best(c, 2) for c in ("gold", "matte")})]
    res["summary_test_psnr"] = {name: vals for name, vals in rows}
    res["env_map"] = {"local_degree": 0, "env_degree": best_gold[1], "env_smooth": best_gold[2],
                      "test_psnr": best_gold[0]}
    (HERE / "env_oracle.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    plot_bars(rows, HERE / "figures" / "env_oracle.png")
    plot_env(best_gold[3], best_gold[1], refl_tr, HERE / "figures" / "env_map.png")
    for name, vals in rows:
        print(f"{name:<24} 金 {vals['gold']:6.2f}  拡散面 {vals['matte']:6.2f} dB")
    print(f"-> {HERE / 'env_oracle.json'}, figures/env_oracle.png, figures/env_map.png")


if __name__ == "__main__":
    main()
