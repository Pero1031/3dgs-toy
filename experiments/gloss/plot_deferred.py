# -*- coding: utf-8 -*-
"""
Summary and figures for Follow-up 5 (stage 3): compare the 3 modes of train_deferred.py with vanilla 3DGS
(eval_sphere_vanilla.py).

1. Collect the reconstruction quality and the blur per sector from logs/dr_*/eval.json and
   logs/sphere_vanilla/eval.json.
2. Normal variation when the pixel size is changed: render the 12 test views at 512, 256, 128 px, and on the pixels
   of the smooth and bump sectors compare σ = 2√(1 − |n̄|) (no anisotropy correction) among the following 3
   (on the same pixels).
       Ground truth   average the analytic normals within the box footprint of the pixel
                      (splat_normals.truth_and_select)
       Placed discs   composite small discs placed at the vertices of the true mesh (splat_normals.py,
                      opacity 0.99. No training)
       Trained        composite the Gaussians of the 3 modes of train_deferred.py
3. Size, opacity, and offset from the reference sphere of the trained Gaussians.

    python experiments/gloss/plot_deferred.py

Output: experiments/gloss/deferred_sphere.json, figures/deferred_sphere.png, figures/deferred_maps.png
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import deferred as dg  # noqa: E402
import make_sphere as ms  # noqa: E402
import sh_oracle as so  # noqa: E402
import sphere_rough as sr  # noqa: E402
import splat_normals as sn  # noqa: E402
import train_deferred as td  # noqa: E402
from gs.dataset import load_cameras  # noqa: E402
from gs.gaussians import GaussianModel  # noqa: E402
from gs.render import render  # noqa: E402

DEV = "cuda"
MODES = ["learned", "toksvig", "both"]
NBAR = r"$|\bar{n}|$"   # |n̄| in the figures (the combining overline breaks in the Japanese font, so it is written as math)
MODE_LABEL = {"learned": "Deferred: roughness learned", "toksvig": f"Deferred: converted (from {NBAR})", "both": "Deferred: both",
              "vanilla": "Vanilla 3DGS (colour of SH degree 3)"}
C_MODE = {"learned": sr.C_CAT[0], "toksvig": sr.C_CAT[1], "both": sr.C_CAT[2], "vanilla": sr.C_CAT[3],
          "splat": sr.C_CAT[4]}
M_MODE = {"learned": "o", "toksvig": "s", "both": "^", "vanilla": "D", "splat": "v"}
SCALES = ["512", "256", "128"]
SECTOR_GROUPS = [("smooth", ["smooth_a", "smooth_b"], "Smooth"), ("bump3", ["bump3"], "Bumps 3°"),
                 ("bump6", ["bump6"], "Bumps 6°"), ("bump12", ["bump12"], "Bumps 12°"),
                 ("brdf3", ["brdf3"], "GGX 3°"), ("brdf6", ["brdf6"], "GGX 6°"), ("brdf12", ["brdf12"], "GGX 12°")]
MAP_VIEWS = [(2, "bump sectors in front"), (8, "GGX sectors in front")]   # indices of the test views (longitude 162° and 340°)
OUT_JSON = HERE / "deferred_sphere.json"


def logdir(name):
    return ROOT / "logs" / name


def load_deferred(mode):
    ck = torch.load(logdir(f"dr_{mode}") / "ckpt.pt", map_location=DEV)
    model = dg.DeferredModel.from_state_dict(ck["model"], device=DEV)
    env = dg.EnvSH(device=DEV)
    with torch.no_grad():
        env.coef.copy_(ck["env"].to(DEV))
    return model, env


def group_mean(sectors, key, names):
    """Merge sectors (smooth A and B are averaged with weights by pixel count)"""
    w = np.array([sectors[n]["n_pixels"] for n in names], dtype=float)
    return float(np.sum([sectors[n][key] * wi for n, wi in zip(names, w)]) / w.sum())


def collect():
    res = {"modes": {}, "vanilla": json.loads((logdir("sphere_vanilla") / "eval.json").read_text(encoding="utf-8"))}
    for m in MODES:
        ev = json.loads((logdir(f"dr_{m}") / "eval.json").read_text(encoding="utf-8"))
        row = {"n_gauss": ev["n_gauss"], "train_minutes": ev["train_minutes"],
               "train_psnr_sphere": ev["train_psnr_sphere"], "eval": {}}
        for s in SCALES:
            sec = ev["eval"][s]["sectors"]
            row["eval"][s] = {"psnr_sphere": ev["eval"][s]["psnr_sphere"],
                              "sectors": {g: {k: group_mean(sec, k, names) for k in ("len_n", "sig_geo", "sig_mat", "sigma", "alpha")}
                                          for g, names, _ in SECTOR_GROUPS}}
        res["modes"][m] = row
    st2 = json.loads((HERE / "sphere_rough.json").read_text(encoding="utf-8"))
    res["stage2_fitted_all"] = {s: {g: float(np.mean([st2["scales"][s]["sectors"][n]["fitted_all"]["mean_deg"] for n in names]))
                                    for g, names, _ in SECTOR_GROUPS} for s in SCALES}
    res["pixel_width_mm"] = {s: st2["scales"][s]["pixel_width_mm_median"] for s in SCALES}
    return res


@torch.no_grad()
def gaussian_stats():
    out = {}
    q = torch.tensor([0.05, 0.1, 0.5, 0.9, 0.95])
    for m in MODES:
        p = torch.load(logdir(f"dr_{m}") / "ckpt.pt", map_location="cpu")["model"]["params"]
        s = torch.exp(p["log_scales"]).sort(dim=1).values * 1e3
        op = torch.sigmoid(p["opacity_logits"]).reshape(-1)
        dr = p["means"].norm(dim=1) * 1e3 - ms.R0 * 1e3
        out[m] = {"quantiles": q.tolist(),
                  "s_max_mm": s[:, 2].quantile(q).tolist(), "s_min_mm": s[:, 0].quantile(q).tolist(),
                  "opacity": op.quantile(q).tolist(), "frac_opacity_below_0.5": float((op < 0.5).float().mean()),
                  "radial_offset_mm": dr.quantile(q).tolist(),
                  "hist_edges_mm": np.linspace(-14, 6, 41).tolist(),
                  "hist_opacity_weighted": np.histogram(dr.numpy(), bins=np.linspace(-14, 6, 41),
                                                        weights=op.numpy())[0].tolist()}
    return out


@torch.no_grad()
def resolution_study():
    """On the 12 test views, σ = 2√(1 − |n̄|) (no anisotropy correction) for each pixel size, for the ground truth,
    the placed discs, and the 3 trained modes"""
    t0 = time.time()
    P_np, N_np = sn.read_ply(ms.DATA / "mesh" / "au.ply")
    P = torch.tensor(P_np, device=DEV)
    N = F.normalize(torch.tensor(N_np, device=DEV), dim=-1)
    offs = torch.quasirandom.SobolEngine(2, scramble=True, seed=sr.SEED).draw(sn.TRUTH_SAMPLES).to(DEV) - 0.5
    models = {m: load_deferred(m) for m in MODES}
    tex = {m: env.textures() for m, (_, env) in models.items()}
    bg_rad = float(dg.display_to_radiance(torch.tensor(td.BG)))
    frm = [f for sp, n, f in ms.views() if sp == "test"]
    acc = {}
    for s in SCALES:
        scale = int(s)
        for k, f in enumerate(frm):
            cam, c = sn.to_camera(f, scale)
            idx, sec, sig_true = sn.truth_and_select(c, scale, offs)
            vals = {"truth": sig_true}
            img, alpha = sn.splat_normals(P, N, cam, 0.99)
            vals["splat"] = sn.sigma_from(img.reshape(-1, 3)[idx], alpha.reshape(-1)[idx])
            for m, (model, env) in models.items():
                o = dg.render_deferred(model, env, cam, m, bg_rad, T=tex[m])
                vals[m] = 2 * torch.sqrt((1 - o["len_n"].reshape(-1)[idx]).clamp(min=0))
            for key, v in vals.items():
                v = np.degrees(v.cpu().numpy())
                for kk in sr.GEOM:
                    acc.setdefault(s, {}).setdefault(key, {}).setdefault(ms.SECTORS[kk][0], []).append(v[sec == kk])
        print(f"  {s} px done ({time.time() - t0:.0f} s)", flush=True)
    out = {}
    for s, d in acc.items():
        out[s] = {}
        for key, dd in d.items():
            x = {n: np.concatenate(v) for n, v in dd.items()}
            out[s][key] = {"smooth": float(np.concatenate([x["smooth_a"], x["smooth_b"]]).mean()),
                           **{n: float(x[n].mean()) for n in ("bump3", "bump6", "bump12")},
                           "n_pixels": {n: int(len(v)) for n, v in x.items()}}
    return out


# ---- Figures ----

def _dodge(i, n, w=0.16):
    return (i - (n - 1) / 2) * w


def plot_summary(R, path):
    plt = sr.er.setup_plt()
    fig = plt.figure(figsize=(11.6, 11.2), dpi=200, facecolor=so.C_SURF)
    gs_ = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.02], hspace=0.55, wspace=1.15, top=0.95, bottom=0.07)
    xg = np.arange(len(SECTOR_GROUPS))
    glab = [g[2] for g in SECTOR_GROUPS]

    # (a) blur per sector (256 px, the blur actually used)
    ax = fig.add_subplot(gs_[0, :4])
    sr._axes_style(ax)
    ref = [R["stage2_fitted_all"]["256"][g] for g, _, _ in SECTOR_GROUPS]
    series = [("ref", "Stage 2: blur optimized per point with the true shape", so.C_TEXT, "_", ref)] + [
        (m, MODE_LABEL[m], C_MODE[m], M_MODE[m], [R["modes"][m]["eval"]["256"]["sectors"][g]["sigma"] for g, _, _ in SECTOR_GROUPS])
        for m in MODES]
    for i, (key, lab, col, mk, y) in enumerate(series):
        x = xg + _dodge(i, len(series))
        if key == "ref":
            ax.scatter(x, y, marker=mk, s=140, color=col, lw=2.2, label=lab, zorder=3)
        else:
            ax.scatter(x, y, marker=mk, s=34, color=col, edgecolor=so.C_SURF, lw=0.8, label=lab, zorder=3)
    ax.set_xticks(xg)
    ax.set_xticklabels(glab, fontsize=8)
    ax.set_ylim(0, 32)
    ax.set_ylabel("Blur σ in the reflection direction [degrees]", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False, ncol=2)
    ax.set_title("(a) Blur per sector (at the trained 256 px, mean over the test-view pixels)", fontsize=9, loc="left")

    # (b) reconstruction quality and pixel size
    ax = fig.add_subplot(gs_[0, 4:])
    sr._axes_style(ax)
    xs = np.arange(len(SCALES))
    rows = [("vanilla", [R["vanilla"]["eval"][s]["psnr_sphere"] for s in SCALES])] + [
        (m, [R["modes"][m]["eval"][s]["psnr_sphere"] for s in SCALES]) for m in MODES]
    for key, y in rows:
        ax.plot(xs, y, color=C_MODE[key], marker=M_MODE[key], ms=5, lw=1.4, mec=so.C_SURF, mew=0.8,
                label=MODE_LABEL[key])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s} px" + ("\n(trained)" if s == "256" else "") for s in SCALES], fontsize=7.5)
    ax.set_xlim(-0.3, len(SCALES) - 0.7)
    ax.set_ylim(21.0, 24.0)
    ax.set_ylabel("PSNR over the sphere region [dB] (12 test views)", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.8, frameon=False)
    ax.set_title("(b) Reconstruction quality", fontsize=9, loc="left")

    # (c) breakdown of the both mode
    ax = fig.add_subplot(gs_[1, :4])
    sr._axes_style(ax)
    sec = R["modes"]["both"]["eval"]["256"]["sectors"]
    parts = [("sig_geo", f"Geometric blur (converted from {NBAR})", C_MODE["toksvig"], M_MODE["toksvig"]),
             ("sig_mat", "Material blur (learned roughness)", C_MODE["learned"], M_MODE["learned"])]
    for i, (key, lab, col, mk) in enumerate(parts):
        ax.scatter(xg + _dodge(i, 2, 0.2), [sec[g][key] for g, _, _ in SECTOR_GROUPS], marker=mk, s=34, color=col,
                   edgecolor=so.C_SURF, lw=0.8, label=lab, zorder=3)
    ax.scatter(xg + 0.3, ref, marker="_", s=110, color=so.C_TEXT, lw=2.0, label="Stage 2 (same as (a))", zorder=3)
    ax.set_xticks(xg)
    ax.set_xticklabels(glab, fontsize=8)
    ax.set_ylim(0, 32)
    ax.set_ylabel("Blur σ in the reflection direction [degrees]", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False, ncol=3)
    ax.set_title("(c) Both mode: whether the blur went to geometry or to material (the blur actually used is the root sum of squares of the two)",
                 fontsize=9, loc="left")

    # (d) depth of the Gaussians
    ax = fig.add_subplot(gs_[1, 4:])
    sr._axes_style(ax)
    for m in MODES:
        g = R["gaussians"][m]
        e = np.array(g["hist_edges_mm"])
        h = np.array(g["hist_opacity_weighted"])
        ax.step(np.r_[e[:-1], e[-1]], np.r_[h, h[-1]] / h.sum() / np.diff(e)[0], where="post", color=C_MODE[m], lw=1.4,
                label=MODE_LABEL[m])
    ax.axvspan(-1.3, 1.3, color=so.C_GRID, alpha=0.8, lw=0, zorder=0)
    ax.axvline(0, color=so.C_TEXT2, lw=1.0, ls=(0, (4, 3)))
    ax.text(1.6, ax.get_ylim()[1] * 0.92, "range of the bump\ndisplacement", fontsize=6.8, color=so.C_TEXT2, va="top")
    ax.set_xlabel("Offset from the reference sphere [mm] (negative is inside)", fontsize=8)
    ax.set_ylabel("Fraction [1/mm] (weighted by opacity)", fontsize=8)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False)
    ax.set_title(f"(d) Position of the trained Gaussians (1 pixel ≈ {R['pixel_width_mm']['256']:.1f} mm)", fontsize=9, loc="left")

    # (e)–(g) normal variation when the pixel size is changed
    width = np.array([R["pixel_width_mm"][s] for s in SCALES])
    RS = R["resolution"]
    lines = [("truth", "Ground truth (mean of the normals within the footprint)", so.C_TEXT, "o", (0, (4, 3)), 1.2),
             ("splat", "Small discs placed on the true surface (no training)", C_MODE["splat"], M_MODE["splat"], "-", 1.4)] + [
        (m, f"Trained Gaussians ({MODE_LABEL[m]})", C_MODE[m], M_MODE[m], "-", 1.4) for m in MODES]
    first = None
    for j, name in enumerate(["bump3", "bump6", "bump12"]):
        ax = fig.add_subplot(gs_[2, 2 * j:2 * j + 2])
        first = first or ax
        sr._axes_style(ax)
        top = 0
        for key, lab, col, mk, ls, lw in lines:
            y = [RS[s][key][name] for s in SCALES]
            top = max(top, max(y))
            ax.plot(width, y, color=col, marker=mk, ms=5, lw=lw, ls=ls, mec=so.C_SURF, mew=0.8,
                    label=lab if j == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(width)
        ax.set_xticklabels([f"{w:.2f}\n({s} px)" for w, s in zip(width, SCALES)], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlim(width.min() / 1.25, width.max() * 1.25)
        ax.set_ylim(0, top * 1.12)
        ax.set_xlabel("Width of one pixel [mm] (on the sphere surface, median)", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"σ = 2√(1 − {NBAR}) [degrees] (no anisotropy correction)", fontsize=8)
        ax.set_title(f"({'efg'[j]}) {sr.SECTOR_LABEL[name]}", fontsize=9, loc="left")
    fig.legend(*first.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               fontsize=7.2, frameon=False)
    fig.suptitle("Bumpy sphere: the 3 modes of deferred shading added to 3dgs-toy, and vanilla 3DGS (trained at 256 px)",
                 fontsize=10.5, x=0.01, y=0.995, ha="left")
    fig.savefig(path, facecolor=so.C_SURF, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


@torch.no_grad()
def plot_maps(path):
    """2 test views (256 px): ground truth / vanilla 3DGS / deferred: both / geometric blur / material blur"""
    plt = sr.er.setup_plt()
    import matplotlib.colors as mcolors
    d256 = ROOT / "data" / "gloss_sphere" / "s256"
    cams = load_cameras(d256, "test", bg=(td.BG,) * 3)
    frm = [f for sp, n, f in ms.views() if sp == "test"]
    ck = torch.load(logdir("sphere_vanilla") / "ckpt_007000.pt", map_location=DEV)
    van = GaussianModel.from_state_dict(ck["model"])
    model, env = load_deferred("both")
    bg_rad = float(dg.display_to_radiance(torch.tensor(td.BG)))
    lo, hi = 48, 208
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(so.C_SURF)
    norm = mcolors.Normalize(0, 30)
    cols = ["Ground truth", "Vanilla 3DGS", "Deferred: both", f"Geometric blur (converted from {NBAR})", "Material blur (learned roughness)"]
    fig = plt.figure(figsize=(11.6, 5.3), dpi=200, facecolor=so.C_SURF)
    gs_ = fig.add_gridspec(len(MAP_VIEWS), 6, width_ratios=[1, 1, 1, 1, 1, 0.06], hspace=0.16, wspace=0.05,
                           left=0.01, right=0.95, top=0.9, bottom=0.02)
    for r, (k, note) in enumerate(MAP_VIEWS):
        cam = cams[k]
        o = dg.render_deferred(model, env, cam, "both", bg_rad)
        vimg = render(van, cam, torch.tensor([td.BG] * 3, device=DEV), ck["sh_degree"])["image"]
        fg = (o["alpha"] > 0.5).cpu().numpy()
        panels = [cam.image, vimg, o["image"], o["sig_geo"], o["sig_mat"]]
        c = sr.Cam(frm[k], 256)
        for j, x in enumerate(panels):
            ax = fig.add_subplot(gs_[r, j])
            if j < 3:
                ax.imshow(x.clamp(0, 1).cpu().numpy()[lo:hi, lo:hi], extent=(lo, hi, hi, lo), interpolation="nearest")
            else:
                v = np.degrees(x.cpu().numpy())
                v = np.where(fg, v, np.nan)
                im = ax.imshow(v[lo:hi, lo:hi], cmap=cmap, norm=norm, extent=(lo, hi, hi, lo), interpolation="nearest")
            line = "#fcfcfb" if j < 3 else so.C_TEXT2
            for kk, (sname, _, _) in enumerate(ms.SECTORS):
                u, v_, cv = sr._meridian(c, frm[k], ms.SECTOR_DEG * kk, np.linspace(-62, 62, 80))
                ax.plot(np.where(cv > 0.3, u, np.nan), np.where(cv > 0.3, v_, np.nan), color=line, lw=0.7,
                        ls=(0, (3, 3)), alpha=0.85)
                if j == 0:
                    u, v_, cv = sr._meridian(c, frm[k], ms.SECTOR_DEG * (kk + 0.5), [0.0])
                    if cv[0] > 0.45:
                        ax.text(u[0], hi - 5, sr.SECTOR_LABEL[sname], ha="center", va="bottom", fontsize=7,
                                color=so.C_TEXT, bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.9))
            ax.set_xlim(lo, hi)
            ax.set_ylim(hi, lo)
            ax.set_axis_off()
            if r == 0:
                ax.set_title(cols[j], fontsize=8.5, loc="left")
            if j == 0:
                ax.text(-0.02, 0.5, f"Test view {k}\n{note}", transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=7.5, color=so.C_TEXT2)
    cax = fig.add_subplot(gs_[:, 5])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Blur σ in the reflection direction [degrees]", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Deferred shading (both mode) on test views (256 px, cropped to the centre of the sphere. Dotted lines are sector boundaries)",
                 fontsize=9.5, x=0.01, y=0.995, ha="left")
    fig.savefig(path, facecolor=so.C_SURF, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    if "--plot-only" in sys.argv:   # redraw only the figures from deferred_sphere.json (the maps are rendered again)
        R = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    else:
        R = collect()
        R["gaussians"] = gaussian_stats()
        print("Variation when the pixel size is changed (12 test views)", flush=True)
        R["resolution"] = resolution_study()
        OUT_JSON.write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    for s in SCALES:
        print(f"[{s} px] σ (no anisotropy correction): " + "  ".join(
            f"{k}: " + "/".join(f"{R['resolution'][s][k][n]:.1f}" for n in ("smooth", "bump3", "bump6", "bump12"))
            for k in ["truth", "splat"] + MODES))
    fig_dir = HERE / "figures"
    plot_summary(R, fig_dir / "deferred_sphere.png")
    plot_maps(fig_dir / "deferred_maps.png")
    print(f"-> {OUT_JSON}, {fig_dir / 'deferred_sphere.png'}, {fig_dir / 'deferred_maps.png'}")


if __name__ == "__main__":
    main()
