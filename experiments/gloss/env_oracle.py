# -*- coding: utf-8 -*-
"""
Fit with "shared lighting": measure, without 3DGS, whether sharing one lighting function looked up by reflection
direction across all points gets past the ceiling of the per-point SH (sh_oracle.py).

Model (observed colour y of point p from view v):
    y(p, v) ≈ Σ_k c_{p,k} Y_k(d)   +   Σ_j e_j Y_j(r)
              per-point low-degree term   lighting shared by all points (high-degree SH in the reflection direction)
    d: direction from the camera to the point, r = 2(ω·n)n − ω (ω = −d, n is the true normal): the view direction
       mirrored about the normal

    - The reflection on gold is determined by "what lies in the reflection direction", and every point sees the same
      lighting, so the lighting E is shared by all points. Each point has only 26 observations, but over all points
      they become 200,000 observations of the lighting.
    - The diffuse component is determined by the normal, not by the reflection direction, so it is left to the
      per-point term (split into diffuse and specular).

Solver: once the shared coefficients e are fixed, the per-point coefficients c_p are determined in closed form by a
small per-point least squares, so they are eliminated first (Schur complement). The remaining normal equations of e
(at most 2401 × 2401) are solved once, and c_p is computed afterwards.

    python experiments/gloss/env_oracle.py

Output: experiments/gloss/env_oracle.json, figures/env_oracle.png, figures/env_map.png
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sh_oracle as so  # noqa: E402  observation collection, evaluation and figure colours are shared with the previous check
from gs.sh import sh_basis  # noqa: E402

DEV = so.DEV
S_LOC = 0.03                          # smoothness penalty of the per-point term (the best value in sh_oracle)
LOCAL_DEGREES = [-1, 0, 2]            # degree of the per-point term (-1 = no per-point term, 0 = per-point constant)
ENV_DEGREES = [16, 32, 48]            # SH degree of the shared lighting (resolution is about 180°/L)
ENV_SMOOTH = [0.1, 1.0, 10.0]         # smoothness penalty of the shared lighting
CHUNK = 128                           # number of points processed at a time


def smooth_normals(n, seed):
    """Smooth normals (interpolated from vertex normals) at the same points as sample_surface(…, n, seed).

    GoLD_rayt computes reflections by interpolating the normals of the 3 vertices of a triangle with barycentric
    coordinates (fill in TriangleMesh.hpp). For near-mirror gold, a difference of a few degrees from the face normal
    is doubled in the reflection direction, so the same normals as the renderer are used.
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
    v, vn, f = np.array(v) * SCALE, np.array(vn), np.array(f)  # f: (F, 3 vertices, [v, vn])
    c, s = math.cos(math.radians(ROT_DEG)), math.sin(math.radians(ROT_DEG))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    v, vn = v @ R.T + TRANS, vn @ R.T
    a, b, cc = v[f[:, 0, 0]], v[f[:, 1, 0]], v[f[:, 2, 0]]
    cross = np.cross(b - a, cc - a)
    area = np.linalg.norm(cross, axis=1) / 2
    rng = np.random.default_rng(seed)  # use the random numbers in the same way as sample_surface, to draw the same
                                       # triangles and the same barycentric coordinates
    idx = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    w = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    ns = (w[:, :, None] * vn[f[idx, :, 1]]).sum(1)
    ns /= np.linalg.norm(ns, axis=1, keepdims=True)
    ng = cross[idx] / np.linalg.norm(cross[idx], axis=1, keepdims=True)
    ns *= np.sign((ns * ng).sum(1, keepdims=True))  # align to the same side as the face normal (outward)
    return ns


def reflect(dirs, N):
    """Direction r = 2(ω·n)n − ω, ω = −d: the view direction d (camera → point) mirrored about the normal n"""
    w = -dirs
    return 2 * (w * N[None]).sum(-1, keepdim=True) * N[None] - w


def penalty(L, s):
    l = torch.tensor([math.isqrt(k) for k in range((L + 1) ** 2)], dtype=torch.float64, device=DEV)
    return torch.diag(so.LAMBDA + s * l * (l + 1))


def blocks(obs, N, Lp, Le, sl):
    """For the point range sl, build the per-point basis Bl (V,m,Kl), the shared basis Be (V,m,Ke), the weights w
    and the observations y"""
    vis, col, dirs = obs
    V = vis.shape[0]
    d, n = dirs[:, sl], N[sl]
    m = d.shape[1]
    Be = sh_basis(Le, reflect(d, n).reshape(-1, 3)).float().reshape(V, m, -1)
    Bl = sh_basis(Lp, d.reshape(-1, 3)).double().reshape(V, m, -1) if Lp >= 0 else None
    return Bl, Be, vis[:, sl].double(), col[:, sl]


def fit(obs, N, Lp, Le, smooths):
    """Solve the shared lighting e (for each smoothness penalty) -> {s: e (Ke,3)}"""
    n = obs[0].shape[1]
    Ke = (Le + 1) ** 2
    S = torch.zeros(Ke, Ke, dtype=torch.float64, device=DEV)
    rhs = torch.zeros(Ke, 3, dtype=torch.float64, device=DEV)
    Rl = penalty(Lp, S_LOC) if Lp >= 0 else None
    for i in range(0, n, CHUNK):
        Bl, Be, w, y = blocks(obs, N, Lp, Le, slice(i, i + CHUNK))
        Ew = (Be * w[..., None].float()).reshape(-1, Ke)
        S += (Ew.T @ Be.reshape(-1, Ke)).double()            # Σ Eᵀ W E (all observations)
        rhs += (Ew.T @ y.reshape(-1, 3).float()).double()     # Σ Eᵀ W y
        if Bl is None:
            continue
        # Eliminate the per-point term: with A_p = Blᵀ W Bl + R, G_p = L_p⁻¹ (Blᵀ W E), g_p = L_p⁻¹ (Blᵀ W y) at
        # each point, it is enough to subtract Σ G_pᵀ G_p and Σ G_pᵀ g_p from the normal equations of e
        # (L_p is the Cholesky factor of A_p)
        Blw = Bl * w[..., None]
        L = torch.linalg.cholesky(torch.einsum("vmk,vmj->mkj", Blw, Bl) + Rl)
        M = torch.einsum("vmk,vmj->mkj", Blw.float(), Be).double()
        G = torch.linalg.solve_triangular(L, M, upper=False).reshape(-1, Ke)
        g = torch.linalg.solve_triangular(L, torch.einsum("vmk,vmc->mkc", Blw, y), upper=False).reshape(-1, 3)
        S -= (G.T.float() @ G.float()).double()
        rhs -= G.T @ g
    return {s: torch.linalg.solve(S + penalty(Le, s), rhs) for s in smooths}


def evaluate(obs_tr, obs_te, N, Lp, Le, es):
    """Compute the per-point coefficients (per-point least squares with the shared e fixed) and return the PSNR
    on the training and test views"""
    n = obs_tr[0].shape[1]
    Rl = penalty(Lp, S_LOC) if Lp >= 0 else None
    E = torch.cat([es[s] for s in es], 1)  # (Ke, 3 × number of penalties): evaluate the lighting of all penalties together
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
    """Outlines of the light sources seen from the centre of the statue (azimuth and elevation, degrees).
    The scene is the same as in make_views.py"""
    c = np.array([0.0, 0.25, 0.0])
    to_ae = lambda v: (np.degrees(np.arctan2(v[..., 0], v[..., 2])),  # noqa: E731
                       np.degrees(np.arcsin(v[..., 1] / np.linalg.norm(v, axis=-1))))
    out = []
    t = np.linspace(0, 2 * np.pi, 120)
    for k in range(4):  # sphere lights: a circle of angular radius asin(r / distance) around the direction to the centre
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
    s = np.linspace(-0.9, 0.9, 60)  # softbox: the edge of a 1.8 m square
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

    # Where the reflection directions of the training-view observations fell
    # (the lighting anywhere else is not determined by the observations)
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
                label="True positions of the light sources (outlines seen from the centre of the statue)" if k == 0 else None)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("Azimuth of the reflection direction (degrees)")
    ax.set_ylabel("Elevation of the reflection direction (degrees)")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    ax.set_title(f"Shared lighting E(r) recovered from the reflections on the gold statue (SH degree {Le}). Grey band = directions where no reflection from the training views ever fell",
                 fontsize=8.5, loc="left")
    fig.savefig(path, facecolor=so.C_SURF)
    plt.close(fig)


def plot_bars(rows, path):
    """rows: [(model name, {cond: test PSNR})]. Horizontal bars, laid out side by side for each condition"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9,
                         "axes.edgecolor": so.C_TEXT, "xtick.color": so.C_TEXT, "ytick.color": so.C_TEXT})
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
    ax.set_xlabel("Test-view PSNR of the observed colour [dB]")
    ax.set_xlim(0, max(v for r in rows for v in r[1].values()) + 6)
    ax.grid(axis="x", color=so.C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Per-point SH and shared lighting: reconstruction quality on the test views (best setting for each model)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=so.C_SURF)
    plt.close(fig)


def main():
    v, f = so.load_mesh()
    P, N = so.sample_surface(v, f, 10000, seed=1)  # same points as sh_oracle.py (N is the face normal: used for the visibility test)
    Ns = smooth_normals(10000, seed=1)             # smooth normals: used for the reflection direction
    ang = np.degrees(np.arccos(np.clip((N * Ns).sum(1), -1, 1)))
    print(f"Difference between face normals and smooth normals: median {np.median(ang):.1f}°, 90th percentile {np.percentile(ang, 90):.1f}°")
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
                    print(f"{cond:<6} per-point L={Lp:2d}  lighting L={Le}  penalty {s:5g}:  train {ps[s]['train']:6.2f}  "
                          f"test {ps[s]['test']:6.2f} dB", flush=True)
                    if cond == "gold" and Lp == 0 and (best_gold is None or ps[s]["test"] > best_gold[0]):
                        best_gold = (ps[s]["test"], Le, s, es[s])
        res["conditions"][cond] = out
        if cond == "gold":
            refl_tr = reflect(tr[2], Nk)[tr[0]]  # reflection directions of the observations visible in the training views

    # Figure: test-view PSNR of the best setting for each model
    def best(cond, Lp):
        return max(v["test"] for k, v in res["conditions"][cond].items() if k.startswith(f"Lp{Lp}_"))

    pp = {c: max(max(prev["conditions"][c]["test_psnr"]),
                 max(x["best_test_psnr"] for x in prev["conditions"][c]["smooth_check"].values()))
          for c in ("gold", "matte")}
    rows = [("Per-point SH (best of the previous fit)", pp),
            ("Shared lighting only", {c: best(c, -1) for c in ("gold", "matte")}),
            ("Per-point constant + shared lighting", {c: best(c, 0) for c in ("gold", "matte")}),
            ("Per-point degree-2 SH + shared lighting", {c: best(c, 2) for c in ("gold", "matte")})]
    res["summary_test_psnr"] = {name: vals for name, vals in rows}
    res["env_map"] = {"local_degree": 0, "env_degree": best_gold[1], "env_smooth": best_gold[2],
                      "test_psnr": best_gold[0]}
    (HERE / "env_oracle.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    plot_bars(rows, HERE / "figures" / "env_oracle.png")
    plot_env(best_gold[3], best_gold[1], refl_tr, HERE / "figures" / "env_map.png")
    for name, vals in rows:
        print(f"{name:<24} gold {vals['gold']:6.2f}  diffuse {vals['matte']:6.2f} dB")
    print(f"-> {HERE / 'env_oracle.json'}, figures/env_oracle.png, figures/env_map.png")


if __name__ == "__main__":
    main()
