# -*- coding: utf-8 -*-
"""
Isolate why the correlation between "the roughness optimized per point" and "the measured normal variation" in
env_rough.py was weak (Spearman ρ ≈ 0.14).

    1. Split-half reliability: split the training views into even-numbered and odd-numbered ones, optimize the
       per-point roughness again on each, and take the rank correlation between the two. This shows how well the
       optimized roughness itself reproduces (the size of the measurement noise), and gives a rough upper limit for ρ
       (it is converted to the reliability for all views with the Spearman–Brown formula, and the correlation is
       corrected for attenuation).
    2. Recovery test: apply the same procedure to synthetic observations made with "the blur converted from the
       measured variation" (same views, same points, fitted lighting), and see whether the blur width is recovered.
       This shows whether the procedure itself has the power to discriminate per-point roughness.
       Two cases: no noise, and independent noise of the same size as the fit residual on the real data.

    python experiments/gloss/env_rough_check.py

Output: experiments/gloss/env_rough_check.json
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env_rough as er  # noqa: E402

DEV = er.DEV
LP, S = 0, 0.1  # settings used for the roughness comparison in env_rough.py
                # (per-point constant, smoothness penalty of the lighting 0.1)


def identifiable(loss):
    rmse = loss.sqrt()
    return ((rmse.max(1).values - rmse.min(1).values) > er.ID_CONTRAST).cpu().numpy()


def rho(a, b, m):
    return float(spearmanr(a[m], b[m])[0])


def main():
    t0 = time.time()
    grid = torch.tensor(np.radians(er.SIGMA_GRID_DEG), dtype=torch.float32, device=DEV)
    obs, Np, sel, radii, ln_smooth, ln_face, sig_meas = er.prepare(t0)
    meas = sig_meas[er.R_NOMINAL_MM / 1000]
    meas_deg = np.degrees(meas.cpu().numpy())
    o_tr, o_te = er.Obs(*obs["train"], Np, "radiance"), er.Obs(*obs["test"], Np, "radiance")
    out = {"local_degree": LP, "env_smooth": S, "radius_mm": er.R_NOMINAL_MM}

    # All views (same as env_rough.py)
    sig, _, _, _, loss, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
    full_deg, id_full = np.degrees(sig.cpu().numpy()), identifiable(loss)
    out["full"] = {"rho_measured": rho(meas_deg, full_deg, id_full), "n_identifiable": int(id_full.sum()),
                   "median_fitted_deg": float(np.median(full_deg[id_full]))}
    print(f"All views: ρ (measured variation, optimized roughness) = {out['full']['rho_measured']:.3f} ({time.time() - t0:.0f} s)",
          flush=True)

    # 1. Split-half reliability
    halves = {}
    for name, start in (("even", 0), ("odd", 1)):
        o_h = er.Obs(*[x[start::2] for x in obs["train"]], Np, "radiance")
        s_h, _, _, _, l_h, _ = er.run_fitted(o_h, o_te, LP, S, grid)
        halves[name] = (np.degrees(s_h.cpu().numpy()), identifiable(l_h))
        del o_h
        torch.cuda.empty_cache()
        print(f"  fitted the half ({name}) ({time.time() - t0:.0f} s)", flush=True)
    (se, ie), (so_, io) = halves["even"], halves["odd"]
    both = ie & io
    r_half = rho(se, so_, both)
    r_full = 2 * r_half / (1 + r_half)  # Spearman–Brown: reliability when the observations are doubled
    out["split_half"] = {
        "n_both_identifiable": int(both.sum()), "rho_even_odd": r_half, "reliability_full_spearman_brown": r_full,
        "rho_measured_even": rho(meas_deg, se, both), "rho_measured_odd": rho(meas_deg, so_, both),
        "rho_measured_full_corrected": out["full"]["rho_measured"] / math.sqrt(max(r_full, 1e-6))}
    print(f"Split-half: ρ (even, odd) = {r_half:.3f} → reliability for all views {r_full:.3f}, "
          f"ρ (measured variation, optimized roughness) corrected for attenuation = {out['split_half']['rho_measured_full_corrected']:.3f}",
          flush=True)

    # 2. Recovery test: can the blur be recovered from synthetic observations made with the blur of the
    # measured variation
    _, ptr, _, _ = er.run_fixed(o_tr, o_te, meas, LP, S)
    clean = er.radiance_to_display(ptr)
    noise = math.sqrt(er.mse(o_tr, ptr))  # fit residual on the real data (RMSE of display values)
    real = o_tr.ydisp
    gen = torch.Generator(device=DEV).manual_seed(0)
    out["recovery"] = {"residual_rmse_real": noise}
    rec = {}
    for name, sd in (("no_noise", 0.0), ("noise_as_real_residual", noise)):
        o_tr.ydisp = (clean + sd * torch.randn(clean.shape, generator=gen, device=DEV, dtype=clean.dtype)).clamp(0, 1)
        s_r, _, _, _, l_r, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
        rec_deg, id_r = np.degrees(s_r.cpu().numpy()), identifiable(l_r)
        out["recovery"][name] = {"noise_rmse": sd, "n_identifiable": int(id_r.sum()),
                                 "rho_true_recovered": rho(meas_deg, rec_deg, id_r),
                                 "median_true_deg": float(np.median(meas_deg[id_r])),
                                 "median_recovered_deg": float(np.median(rec_deg[id_r]))}
        r = out["recovery"][name]
        rec[name] = (rec_deg, id_r)
        print(f"Recovery ({name}, noise {sd:.3f}): ρ (true blur, recovered blur) = {r['rho_true_recovered']:.3f}, "
              f"median: true {r['median_true_deg']:.1f}° / recovered {r['median_recovered_deg']:.1f}°"
              f" ({time.time() - t0:.0f} s)", flush=True)
    o_tr.ydisp = real

    # 3. What does the optimized roughness correlate with (rank correlation with per-point candidates,
    # only on points where the roughness is determined)
    v, f = er.so.load_mesh()
    Ps = er.so.sample_surface(v, f, 10000, seed=1)[0][sel]
    center = np.array([0.0, 0.25, 0.0])  # centre of the statue (TARGET in make_views.py)
    lights = [np.asarray(c, dtype=float) for c in er.ed.SIDE] + [np.array([0.0, 1.9, 0.0])]

    def angle(a, b):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))

    # Parallax: difference between the direction to a light source seen from the centre of the statue and seen
    # from the point (the shift when it is treated as distant lighting). Maximum over the 5 light sources
    parallax = np.max([angle(L - Ps, L - center) for L in lights], axis=0)
    vis, _, dirs = obs["train"]
    r = er.eo.reflect(dirs, Np)
    Pb = torch.tensor(Ps, dtype=torch.float32, device=DEV)[None].expand(*vis.shape, 3)
    n_obs = vis.sum(0).double()
    hit = (er.ed.light_hit(Pb, r) & vis).sum(0).double() / n_obs
    cosv = ((-dirs * Np[None]).sum(-1) * vis).sum(0).double() / n_obs
    cands = {"Measured normal variation (radius 2.72 mm)": meas_deg,
             "Light parallax (difference from the direction seen from the statue centre, max)": parallax,
             "Distance from the statue centre": np.linalg.norm(Ps - center, axis=1),
             "Height": Ps[:, 1],
             "Fraction of observations whose reflected ray hits a light": hit.cpu().numpy(),
             "Number of training-view observations": n_obs.cpu().numpy(),
             "Mean cos between view direction and normal": cosv.cpu().numpy()}
    out["correlates"] = {k: rho(full_deg, x, id_full) for k, x in cands.items()}
    out["parallax_deg"] = {"median": float(np.median(parallax)), "p90": float(np.percentile(parallax, 90))}
    print(f"Parallax of the light sources: median {np.median(parallax):.1f}°, 90th percentile {np.percentile(parallax, 90):.1f}°")
    for k, x in out["correlates"].items():
        print(f"  ρ (optimized roughness, {k}) = {x:+.3f}")

    (HERE / "env_rough_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # Figure: results of the same procedure on the real data and on synthetic data made according to the
    # hypothesis (with noise), side by side
    plt = er.setup_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.7), dpi=200, facecolor=er.so.C_SURF, sharey=True)
    rr = out["recovery"]["noise_as_real_residual"]
    rec_deg, id_r = rec["noise_as_real_residual"]
    er.scatter_panel(ax1, meas_deg[id_full], full_deg[id_full],
                     f"(a) Real images ({int(id_full.sum())} points, ρ = {out['full']['rho_measured']:.2f}, "
                     f"split-half reliability {r_full:.2f})")
    er.scatter_panel(ax2, meas_deg[id_r], rec_deg[id_r],
                     f"(b) Synthetic observations blurred as the horizontal axis says + noise {noise:.3f} ({rr['n_identifiable']} points, "
                     f"ρ = {rr['rho_true_recovered']:.2f})", ylabel="")
    fig.suptitle("Per-point blur fitted with the same procedure: the procedure can recover a blur that follows the conversion, but the blur of the real images does not follow the conversion",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "env_rough_check.png", facecolor=er.so.C_SURF)
    plt.close(fig)
    print(f"-> {HERE / 'env_rough_check.json'}, figures/env_rough_check.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
