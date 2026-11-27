# -*- coding: utf-8 -*-
"""
Oracle for stage 2: on a sphere with controlled conditions (make_sphere.py), compare the blur (roughness) optimized
per point with the blur converted from the normal variation within a pixel (a check that does not use 3DGS. The sphere
version of env_rough.py).

Questions
    1. When the bumps are finer than a pixel, does the blur optimized per point agree with Toksvig's conversion
       σ_r = 2σ_n?
    2. When the pixel size (512 / 256 / 128 px) is changed, does the optimal blur follow the change in "the variation
       within a pixel"?
    3. What width of Gaussian does the blur due to material roughness (GGX α) correspond to? (the brdf sectors)

Models (all of them: shared lighting of SH degree 48 + per-point constant. The solver is the same as in env_rough.py)
    material      σ = 0. The normal is the direction averaged over the per-point footprint (the models below use the
                  same normal, except measured_obs)
    measured_pt   Average the normals over the per-point footprint (isotropic on the tangent plane. Per axis, the
                  spread that combines the pixel box and the triangle of the bilinear interpolation used when picking
                  up observations; the width is one pixel at the mean depth of the training views that see the point),
                  and set σ = 2√(1 − |n̄|). Same as measured for the statue
    measured_obs  For each observation, average the normals over the footprint of that pixel (box ⊗ triangle on the
                  image plane, projected onto the sphere. It stretches at oblique viewing), reflect about that
                  direction, and blur with σ = 2√(1 − |n̄|). The ideal form of Toksvig-style prefiltering
    measured_obs_aniso  Multiply the σ of measured_obs by √((1 + cos²θ)/2) (θ is the angle between the view direction
                  and the mean normal). Tilting the normal by δ within the plane of incidence turns the reflection
                  direction by 2δ, but tilting it perpendicular to the plane of incidence turns it by only 2δ cosθ.
                  This is the isotropic blur that corresponds to the mean of the variances of those 2 axes
    supersampled  An exact model that does not use the blur approximation. For each observation, reflect about the
                  normal of each footprint sample (SUPER_SAMPLES of them), average the SH of those directions, and
                  multiply by the lighting. The upper limit for models that blur with an isotropic Gaussian
    fitted        Choose σ per point from a grid (solved alternately with the lighting). A stand-in for the roughness
                  that existing methods learn
    sh6           Per-point SH of degree 6 (reference)

The normals are computed from the analytic expression for the bumps (the renderer interpolates vertex normals, so the
tilt is about 3% smaller. See the check in make_sphere.py mesh).
The position shift due to the bumps (at most 1.3 mm) is ignored in the projection, and the reference sphere is used.

Handling of the sectors: the brdf sectors are smooth in shape, and measured does not know the material roughness, so
the models are compared only on the points of the 5 smooth and bump sectors. The optimal blur of the brdf sectors is
obtained by solving fitted again on the points of all sectors.

    python experiments/gloss/sphere_rough.py [--scales 512 256 128] [--no-denoised-check]

In addition, the error when the σ of measured_obs is multiplied uniformly by k (the best scale factor) and the
equivalent σ obtained by assuming that the averaged SH of supersampled corresponds to Y(n̄)·exp(−l(l+1)σ²/2)
(l = 4 and 12) are also recorded.

Output: experiments/gloss/sphere_rough.json, data/gloss_sphere/sphere_rough_points.npz (per-point values, for
redrawing the figures)
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env_rough as er  # noqa: E402
import make_sphere as ms  # noqa: E402
import sh_oracle as so  # noqa: E402

DEV = er.DEV
N_PER_SECTOR = 800       # number of points per sector
LAT_MAX = 50.0           # points with latitude above this are not used (avoids the range where the bumps are
                         # weakened, and the poles)
BORDER_MARGIN = 7.0      # use only points at least this far from the sector boundaries [degrees] (avoids the 4°
                         # blend width and the spread of the footprint)
FP_SAMPLES = 512         # number of samples for averaging the normals within the footprint
SUPER_SAMPLES = 64       # number of samples for averaging the SH of the reflection directions in supersampled
                         # (the first ones of FP_SAMPLES. It is a Sobol sequence, so they are not biased)
K_SCAN = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # candidate scale factors for the σ of measured_obs
EQUIV_L = [4, 12]        # SH degrees at which the equivalent σ is read
LP, S = 0, 0.1           # degree of the per-point term, smoothness penalty of the lighting (settings used for the
                         # roughness comparison in env_rough.py)
SH_DEG = 6
GEOM = [k for k, (_, _, mat) in enumerate(ms.SECTORS) if mat == "au"]  # the smooth and bump sectors
SEED = 3
OUT_NPZ = ms.DATA / "sphere_rough_points.npz"


# ---- Images and cameras ----

def read_pfm(path):
    with open(path, "rb") as fh:
        kind = fh.readline().strip()
        w, h = map(int, fh.readline().split())
        sc = float(fh.readline())
        a = np.frombuffer(fh.read(), dtype="<f4" if sc < 0 else ">f4")
    return a.reshape(h, w, 3 if kind == b"PF" else 1)[::-1].squeeze().copy()


def load_split(split, scale, denoised):
    """Display-value images (V, S, S, 3) and masks eroded by 3x3 (V, S, S). Downscaling averages in radiance and then
    converts to display values (the same as shooting with a camera with larger pixels).
    The views are in the order of make_sphere.views()"""
    k = ms.SIZE // scale
    names = [(n, frm) for s, n, frm in ms.views() if s == split]
    img, msk = [], []
    for name, _ in names:
        base = ms.DATA / "render" / split / name
        x = torch.from_numpy(read_pfm(f"{base}{'_denoised' if denoised else ''}.pfm")).to(DEV)
        m = torch.from_numpy(read_pfm(f"{base}_mask.pfm")).to(DEV)
        x = F.avg_pool2d(x.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)
        m = F.avg_pool2d(m[None, None], k)[0, 0]
        img.append(er.radiance_to_display(x).float())
        msk.append(-F.max_pool2d(-m[None, None], 3, 1, 1)[0, 0])
    return [frm for _, frm in names], torch.stack(img), torch.stack(msk)


class Cam:
    def __init__(self, frm, scale):
        c2w = np.array(ms.c2w(frm))
        Rcv = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])   # OpenCV convention: x right, y down, z forward
        self.Rc2w = torch.tensor(Rcv, dtype=torch.float32, device=DEV)
        self.o = torch.tensor(frm, dtype=torch.float32, device=DEV)
        self.f = (scale / 2) / math.tan(math.radians(ms.FOV / 2))
        self.c = scale / 2
        self.S = scale

    def project(self, X):
        pc = (X - self.o) @ self.Rc2w
        return self.f * pc[..., 0] / pc[..., 2] + self.c, self.f * pc[..., 1] / pc[..., 2] + self.c, pc[..., 2]

    def hit_sphere(self, u, v):
        """Normal n0 at the near intersection of the ray through pixel position (u, v) with the reference sphere"""
        d = torch.stack([(u - self.c) / self.f, (v - self.c) / self.f, torch.ones_like(u)], -1) @ self.Rc2w.T
        d = F.normalize(d, dim=-1)
        b = (d * self.o).sum(-1)
        disc = (b * b - (self.o @ self.o - ms.R0 ** 2)).clamp(min=0)
        X = self.o + (-b - disc.sqrt())[..., None] * d
        return F.normalize(X, dim=-1)


# ---- Normals of the bumps (torch version of make_sphere.surface. Only the middle of each sector is used,
# so the amplitude is a per-point constant) ----

FREQ = torch.tensor(ms.FREQ, dtype=torch.float32, device=DEV)
PHASE = torch.tensor(ms.PHASE, dtype=torch.float32, device=DEV)


def normals(n0, a, chunk=1 << 17):
    """Normal of the bumpy surface at the reference-sphere normal n0 (m, 3). a (m,) is the amplitude [m]"""
    out = torch.empty_like(n0)
    c = math.sqrt(2.0 / ms.K_WAVES)
    for i in range(0, len(n0), chunk):
        nn = n0[i:i + chunk]
        ph = 2 * math.pi * (ms.R0 * nn) @ FREQ.T + PHASE
        dg = -c * 2 * math.pi * (torch.sin(ph) @ FREQ)
        dt = dg - (dg * nn).sum(-1, keepdim=True) * nn
        out[i:i + chunk] = F.normalize(nn - a[i:i + chunk, None] * dt, dim=-1)
    return out


def footprint_offsets():
    """Sample positions within the footprint (in pixels; per axis, the sum of 3 uniform variables = box ⊗ triangle
    of the bilinear interpolation) (M, 2)"""
    s = torch.quasirandom.SobolEngine(6, scramble=True, seed=SEED).draw(FP_SAMPLES).to(DEV)
    return torch.stack([s[:, :3].sum(1) - 1.5, s[:, 3:].sum(1) - 1.5], 1)


def mean_normal(n0s, a):
    """Compute the bump normals at the reference normals n0s (m, M, 3) of the footprint samples and average them
    -> direction (m, 3), σ = 2√(1 − |n̄|) (m,)"""
    m, M, _ = n0s.shape
    n = normals(n0s.reshape(-1, 3), a[:, None].expand(m, M).reshape(-1)).reshape(m, M, 3).mean(1)
    ln = n.norm(dim=-1).clamp(max=1)
    return n / ln[:, None], 2 * (1 - ln).sqrt()


# ---- Points and observations ----

def sample_points():
    """N_PER_SECTOR points per sector (uniform on the sphere, excluding boundaries and high latitudes)
    -> reference normals (n, 3), sector index (n,)"""
    rng = np.random.default_rng(SEED)
    out, sec = [], []
    need = [N_PER_SECTOR] * len(ms.SECTORS)
    while any(need):
        u = rng.normal(size=(200000, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        lon, lat = ms.lon_lat(u)
        off = lon % ms.SECTOR_DEG
        k = (lon // ms.SECTOR_DEG).astype(int)
        ok = (np.abs(lat) <= LAT_MAX) & (off >= BORDER_MARGIN) & (off <= ms.SECTOR_DEG - BORDER_MARGIN)
        for kk in range(len(ms.SECTORS)):
            take = u[ok & (k == kk)][:need[kk]]
            out.append(take)
            sec += [kk] * len(take)
            need[kk] -= len(take)
    return np.concatenate(out), np.array(sec)


@torch.no_grad()
def observe(cams, img, msk, n0, a, offs):
    """For each view: whether the point is visible, the observation (display values, bilinear interpolation),
    the view direction, the depth, and the mean normal and blur of the per-observation footprint"""
    P = ms.R0 * n0
    V, n = len(cams), len(n0)
    vis = torch.zeros(V, n, dtype=torch.bool, device=DEV)
    col = torch.zeros(V, n, 3, dtype=torch.float64, device=DEV)
    dirs = torch.zeros(V, n, 3, device=DEV)
    depth = torch.zeros(V, n, device=DEV)
    n_obs = torch.zeros(V, n, 3, device=DEV)
    s_obs = torch.zeros(V, n, device=DEV)
    for i, cam in enumerate(cams):
        u, v, z = cam.project(P)
        to_cam = F.normalize(cam.o - P, dim=-1)
        inb = (u >= 0) & (u < cam.S) & (v >= 0) & (v < cam.S)
        pix = v.clamp(0, cam.S - 1).long() * cam.S + u.clamp(0, cam.S - 1).long()
        ok = inb & ((n0 * to_cam).sum(-1) > so.COS_MIN) & (msk[i].reshape(-1)[pix] > 0.999)
        grid = torch.stack([2 * u / cam.S - 1, 2 * v / cam.S - 1], -1)[None, None]
        col[i] = F.grid_sample(img[i].permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T.double()
        vis[i], dirs[i], depth[i] = ok, -to_cam, z
        idx = ok.nonzero()[:, 0]
        for j in range(0, len(idx), 1024):
            jj = idx[j:j + 1024]
            n0s = cam.hit_sphere(u[jj, None] + offs[None, :, 0], v[jj, None] + offs[None, :, 1])
            n_obs[i, jj], s_obs[i, jj] = mean_normal(n0s, a[jj])
    return vis, col, dirs, depth, n_obs, s_obs


@torch.no_grad()
def point_footprint(n0, a, width, offs):
    """Average the normals over the isotropic per-point footprint (tangent plane; per axis, box ⊗ triangle with
    pixel width width [m])"""
    t1, t2 = (torch.tensor(t, dtype=torch.float32, device=DEV) for t in ms.tangent_frame(n0.cpu().numpy().astype(np.float64)))
    out_n, out_s = torch.empty_like(n0), torch.empty(len(n0), device=DEV)
    for j in range(0, len(n0), 1024):
        sl = slice(j, j + 1024)
        X = ms.R0 * n0[sl, None] + width[sl, None, None] * (offs[None, :, 0, None] * t1[sl, None]
                                                             + offs[None, :, 1, None] * t2[sl, None])
        out_n[sl], out_s[sl] = mean_normal(F.normalize(X, dim=-1), a[sl])
    return out_n, out_s


@torch.no_grad()
def supersampled_Y(cams, vis, dirs, n0, a, offs):
    """For each observation (in the order of vis.nonzero), average the SH of the directions reflected about the
    normal of each footprint sample (n_obs, Ke)"""
    P = ms.R0 * n0
    M = SUPER_SAMPLES
    out = []
    for i, cam in enumerate(cams):
        jj_all = vis[i].nonzero()[:, 0]
        u, v, _ = cam.project(P)
        for j in range(0, len(jj_all), 128):
            jj = jj_all[j:j + 128]
            n0s = cam.hit_sphere(u[jj, None] + offs[None, :M, 0], v[jj, None] + offs[None, :M, 1])
            m = len(jj)
            ns = normals(n0s.reshape(-1, 3), a[jj, None].expand(m, M).reshape(-1)).reshape(m, M, 3)
            dd = dirs[i, jj][:, None, :]
            r = dd - 2 * (dd * ns).sum(-1, keepdim=True) * ns
            out.append(er.sh_basis(er.LE, r.reshape(-1, 3)).float().reshape(m, M, -1).mean(1))
    return torch.cat(out)


def equivalent_sigma(Y_sup, Y_mean, l):
    """Assuming that the degree-l component of the averaged SH is a_l times the SH in the direction of the mean
    normal, read σ from a_l = exp(−l(l+1)σ²/2) (n_obs,)"""
    m = er.L_OF == l
    a_l = (Y_sup[:, m] * Y_mean[:, m]).sum(1) / (Y_mean[:, m] ** 2).sum(1).clamp(min=1e-12)
    return (-2 * a_l.clamp(1e-4, 1).log() / (l * (l + 1))).clamp(min=0).sqrt()


def per_point_rms(x, pid, n_pts):
    s2 = torch.zeros(n_pts, device=DEV).index_add_(0, pid, x.float() ** 2)
    cnt = torch.zeros(n_pts, device=DEV).index_add_(0, pid, torch.ones_like(x, dtype=torch.float32))
    return (s2 / cnt.clamp(min=1)).sqrt()


# ---- Analysis ----

def identifiable(loss):
    rmse = loss.sqrt()
    return ((rmse.max(1).values - rmse.min(1).values) > er.ID_CONTRAST).cpu().numpy()


def sector_psnr(o, pred, sec_pts):
    """PSNR of display values, per sector"""
    err = ((o.display(pred) - o.ydisp) ** 2).mean(1)
    sec = torch.tensor(sec_pts, device=DEV)[o.pid]
    out = {}
    for k in np.unique(sec_pts):
        m = sec == int(k)
        out[ms.SECTORS[k][0]] = -10 * math.log10(err[m].mean().item())
    return out


def run_scale(scale, n0_all, sec_all, a_all, offs, grid, t0, denoised=False, quick=False):
    data, cams = {}, {}
    for split in ("train", "test"):
        frms, img, msk = load_split(split, scale, denoised)
        cams[split] = [Cam(f, scale) for f in frms]
        data[split] = observe(cams[split], img, msk, n0_all, a_all, offs)
        del img, msk
    torch.cuda.empty_cache()
    keep = (data["train"][0].sum(0) >= so.MIN_TRAIN_OBS).cpu().numpy()
    vis_tr, _, _, depth_tr, _, s_obs_tr = data["train"]
    z_mean = (depth_tr * vis_tr).sum(0) / vis_tr.sum(0).clamp(min=1)
    width = z_mean * 2 * math.tan(math.radians(ms.FOV / 2)) / scale            # width of one pixel at the point [m]
    n_pt, s_pt = point_footprint(n0_all, a_all, width, offs)
    s_obs_rms = ((s_obs_tr ** 2 * vis_tr).sum(0) / vis_tr.sum(0).clamp(min=1)).sqrt()
    print(f"[{scale} px{', denoised' if denoised else ''}] observations and footprints: points {keep.sum()} / {len(keep)}, "
          f"training observations {int(vis_tr[:, keep].sum())} ({time.time() - t0:.0f} s)", flush=True)

    res = {"pixel_width_mm_median": float(width.median() * 1000)}
    pts = {"keep": keep, "width_mm": width.cpu().numpy() * 1000,
           "measured_pt_deg": np.degrees(s_pt.cpu().numpy()), "measured_obs_rms_deg": np.degrees(s_obs_rms.cpu().numpy())}

    def subset(sel):
        idx = torch.tensor(np.nonzero(sel)[0], device=DEV)
        return {sp: [x[:, idx] for x in d] for sp, d in data.items()}, idx

    def fitted_run(sel, tag):
        d, idx = subset(sel)
        o_tr, o_te = er.Obs(*d["train"][:3], n_pt[idx], "radiance"), er.Obs(*d["test"][:3], n_pt[idx], "radiance")
        sig, _, ptr, pte, loss, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
        out = (np.degrees(sig.cpu().numpy()), identifiable(loss), er.psnr(o_tr, ptr), er.psnr(o_te, pte),
               sector_psnr(o_te, pte, sec_all[sel]))
        print(f"  fitted ({tag}): train {out[2]:.2f} / test {out[3]:.2f} dB ({time.time() - t0:.0f} s)", flush=True)
        del o_tr, o_te
        torch.cuda.empty_cache()
        return out

    geom = keep & np.isin(sec_all, GEOM)
    if quick:  # check after denoising: fitted only
        f_deg, f_id, *_ = fitted_run(geom, "smooth and bump sectors")
        pts["fitted_geom_deg"], pts["fitted_geom_ident"] = np.full(len(keep), np.nan), np.zeros(len(keep), bool)
        pts["fitted_geom_deg"][geom], pts["fitted_geom_ident"][geom] = f_deg, f_id
        return res, pts

    # Model comparison (smooth and bump sectors)
    d, idx = subset(geom)
    o_tr, o_te = er.Obs(*d["train"][:3], n_pt[idx], "radiance"), er.Obs(*d["test"][:3], n_pt[idx], "radiance")
    res["n_points_geom"], res["n_obs_geom"] = int(geom.sum()), {"train": len(o_tr.pid), "test": len(o_te.pid)}
    res["models"] = {}

    def record(key, ptr, pte, o_tr, o_te):
        res["models"][key] = {"train": er.psnr(o_tr, ptr), "test": er.psnr(o_te, pte),
                              "test_by_sector": sector_psnr(o_te, pte, sec_all[geom])}
        print(f"  {key:<13} train {res['models'][key]['train']:6.2f}  test {res['models'][key]['test']:6.2f} dB"
              f" ({time.time() - t0:.0f} s)", flush=True)

    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, None, SH_DEG, None)
    record("sh6", ptr, pte, o_tr, o_te)
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, torch.zeros(o_tr.n_pts, device=DEV), LP, S)
    record("material", ptr, pte, o_tr, o_te)
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, s_pt[idx], LP, S)
    record("measured_pt", ptr, pte, o_tr, o_te)
    sig, _, ptr, pte, loss, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
    record("fitted", ptr, pte, o_tr, o_te)
    f_deg, f_id = np.degrees(sig.cpu().numpy()), identifiable(loss)
    del o_tr, o_te, ptr, pte
    torch.cuda.empty_cache()

    o_tr = er.Obs(*d["train"][:3], d["train"][4], "radiance")
    o_te = er.Obs(*d["test"][:3], d["test"][4], "radiance")
    o_tr.sig_obs, o_te.sig_obs = d["train"][5][d["train"][0]], d["test"][5][d["test"][0]]
    s_tr, s_te = o_tr.sig_obs, o_te.sig_obs
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
    record("measured_obs", ptr, pte, o_tr, o_te)

    # Isotropic blur corrected for the anisotropy of the reflection
    def cos_obs(sp):
        vis, dirs, n_obs = d[sp][0], d[sp][2], d[sp][4]
        return (-(dirs * n_obs).sum(-1))[vis]
    f_tr, f_te = ((1 + cos_obs("train") ** 2) / 2).sqrt(), ((1 + cos_obs("test") ** 2) / 2).sqrt()
    o_tr.sig_obs, o_te.sig_obs = s_tr * f_tr, s_te * f_te
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
    record("measured_obs_aniso", ptr, pte, o_tr, o_te)
    aniso_rms = per_point_rms(s_tr * f_tr, o_tr.pid, o_tr.n_pts)

    # When σ is multiplied uniformly by k
    res["scan_measured_obs"] = {}
    for k in K_SCAN:
        o_tr.sig_obs, o_te.sig_obs = k * s_tr, k * s_te
        _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
        res["scan_measured_obs"][str(k)] = {"train": er.psnr(o_tr, ptr), "test": er.psnr(o_te, pte)}
    best = max(res["scan_measured_obs"].items(), key=lambda kv: kv[1]["train"])
    print(f"  scale factor for measured_obs: best k on training = {best[0]} (train {best[1]['train']:.2f} / test "
          f"{best[1]['test']:.2f} dB) ({time.time() - t0:.0f} s)", flush=True)

    # Exact model that averages the SH of the reflection directions within the footprint, and the equivalent σ
    Ys_tr = supersampled_Y(cams["train"], d["train"][0], d["train"][2], n0_all[idx], a_all[idx], offs)
    Ys_te = supersampled_Y(cams["test"], d["test"][0], d["test"][2], n0_all[idx], a_all[idx], offs)
    equiv = {l: per_point_rms(equivalent_sigma(Ys_tr, o_tr.Y, l), o_tr.pid, o_tr.n_pts) for l in EQUIV_L}
    o_tr.Y, o_te.Y = Ys_tr, Ys_te
    o_tr.sig_obs, o_te.sig_obs = torch.zeros_like(s_tr), torch.zeros_like(s_te)
    _, ptr, pte, _ = er.run_fixed(o_tr, o_te, "obs", LP, S)
    record("supersampled", ptr, pte, o_tr, o_te)
    del o_tr, o_te, ptr, pte, d, Ys_tr, Ys_te
    torch.cuda.empty_cache()

    def scatter_geom(x):
        out = np.full(len(keep), np.nan)
        out[geom] = np.degrees(x.cpu().numpy())
        return out
    pts["measured_obs_aniso_rms_deg"] = scatter_geom(aniso_rms)
    for l in EQUIV_L:
        pts[f"equiv_l{l}_deg"] = scatter_geom(equiv[l])

    pts["fitted_geom_deg"], pts["fitted_geom_ident"] = np.full(len(keep), np.nan), np.zeros(len(keep), bool)
    pts["fitted_geom_deg"][geom], pts["fitted_geom_ident"][geom] = f_deg, f_id

    # Solve fitted again on the points of all sectors (optimal blur of the brdf sectors)
    a_deg, a_id, tr, te, by_sec = fitted_run(keep, "all sectors")
    res["fitted_all"] = {"train": tr, "test": te, "test_by_sector": by_sec}
    pts["fitted_all_deg"], pts["fitted_all_ident"] = np.full(len(keep), np.nan), np.zeros(len(keep), bool)
    pts["fitted_all_deg"][keep], pts["fitted_all_ident"][keep] = a_deg, a_id
    return res, pts


def summarize(res, pts, sec_all):
    """Median and mean per sector (mean: σ takes only grid values and the median jumps, so the figures and tables
    use the mean), and rank correlations"""
    keep = pts["keep"]
    rows = {}
    for k, (name, sig, mat) in enumerate(ms.SECTORS):
        m = keep & (sec_all == k)
        r = {"n_points": int(m.sum()), "design_blur_deg": 2 * (sig if mat == "au" else ms.BRDF_SIGMA[mat]),
             "measured_pt_deg": float(np.median(pts["measured_pt_deg"][m])),
             "measured_obs_rms_deg": float(np.median(pts["measured_obs_rms_deg"][m]))}
        r["mean"] = {"measured_pt_deg": float(np.mean(pts["measured_pt_deg"][m])),
                     "measured_obs_rms_deg": float(np.mean(pts["measured_obs_rms_deg"][m]))}
        for key in ["measured_obs_aniso_rms_deg"] + [f"equiv_l{l}_deg" for l in EQUIV_L]:
            if key in pts and np.isfinite(pts[key][m]).any():
                r[key] = float(np.nanmedian(pts[key][m]))
                r["mean"][key] = float(np.nanmean(pts[key][m]))
        for key in ("fitted_geom", "fitted_all"):
            if f"{key}_deg" not in pts:
                continue
            mi = m & pts[f"{key}_ident"]
            if mi.sum() == 0:
                continue
            q = np.percentile(pts[f"{key}_deg"][mi], [25, 50, 75])
            r[key] = {"n_identifiable": int(mi.sum()), "median_deg": float(q[1]), "q25_deg": float(q[0]),
                      "q75_deg": float(q[2]), "mean_deg": float(np.mean(pts[f"{key}_deg"][mi]))}
        rows[name] = r
    res["sectors"] = rows
    if "fitted_geom_deg" in pts:
        m = keep & np.isin(sec_all, GEOM) & pts["fitted_geom_ident"]
        bump = m & np.isin(sec_all, [k for k, (_, s, _) in enumerate(ms.SECTORS) if s > 0])
        res["spearman"] = {
            "geom_measured_pt_vs_fitted": float(spearmanr(pts["measured_pt_deg"][m], pts["fitted_geom_deg"][m])[0]),
            "geom_measured_obs_rms_vs_fitted": float(spearmanr(pts["measured_obs_rms_deg"][m],
                                                               pts["fitted_geom_deg"][m])[0])}
        for k in GEOM:  # rank correlation within a sector only (the effect of per-point variation,
                        # with the differences between sectors removed)
            mk = m & (sec_all == k)
            if mk.sum() > 30 and ms.SECTORS[k][1] > 0:
                res["spearman"][f"within_{ms.SECTORS[k][0]}"] = float(
                    spearmanr(pts["measured_pt_deg"][mk], pts["fitted_geom_deg"][mk])[0])
        res["spearman"]["n_bump_identifiable"] = int(bump.sum())


# ---- Figures ----

SECTOR_LABEL = {"smooth_a": "Smooth A", "bump3": "Bumps 3°", "bump6": "Bumps 6°", "bump12": "Bumps 12°",
                "smooth_b": "Smooth B", "brdf3": "GGX 3°", "brdf6": "GGX 6°", "brdf12": "GGX 12°"}
C_CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # category colours for the figures (assigned in this order)
SCALE_STYLE = {"512": ("o", "#a3a29e"), "256": ("^", "#6f6e6a"), "128": ("s", "#0b0b0b")}  # pixel size (marker shape and darkness)
VIEW_IDS = ["view_1_003", "view_1_009", "view_1_015", "view_1_021"]  # equatorial views (longitude 49, 139, 229, 319°)


def _axes_style(ax):
    ax.set_facecolor(so.C_SURF)
    ax.grid(color=so.C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_results(out, path):
    """(a)–(c) Blur width and pixel size for each bump sector, (d) test-view PSNR, (e) optimal blur of the GGX sectors"""
    plt = er.setup_plt()
    sc = [k for k in ("512", "256", "128") if k in out["scales"]]
    R = {k: out["scales"][k] for k in sc}
    width = np.array([R[k]["pixel_width_mm_median"] for k in sc])
    fig = plt.figure(figsize=(11.6, 8.4), dpi=200, facecolor=so.C_SURF)
    gs_ = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.08], hspace=0.42, wspace=1.1)
    series = [("fitted", "Optimized per point (the band is 25–75%)", C_CAT[0], "o", 2.2),
              ("measured_pt_deg", "Conversion: per-point footprint (same as the analysis of the statue)", C_CAT[1], "s", 1.4),
              ("measured_obs_rms_deg", "Conversion: per-observation footprint (stretches at oblique viewing)", C_CAT[2], "^", 1.4),
              ("measured_obs_aniso_rms_deg", "Conversion: per-observation × anisotropy correction of the reflection", C_CAT[3], "D", 1.4)]
    for j, name in enumerate(["bump3", "bump6", "bump12"]):
        ax = fig.add_subplot(gs_[0, 2 * j:2 * j + 2])
        _axes_style(ax)
        rows = [R[k]["sectors"][name] for k in sc]
        design = rows[0]["design_blur_deg"]
        ax.axhline(design, color=so.C_TEXT, lw=1.0, ls=(0, (4, 3)),
                   label="Conversion: sufficiently large footprint (2 × design tilt)" if j == 0 else None)
        for key, lab, col, mk, lw in series:
            if key == "fitted":
                f = [r.get("fitted_geom") for r in rows]
                y = np.array([x["mean_deg"] for x in f])
                ax.fill_between(width, [x["q25_deg"] for x in f], [x["q75_deg"] for x in f], color=col, alpha=0.15, lw=0)
            else:
                y = np.array([r["mean"].get(key, np.nan) for r in rows])
            ax.plot(width, y, color=col, lw=lw, marker=mk, ms=5, mec=so.C_SURF, mew=1.0, label=lab if j == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(width)
        ax.set_xticklabels([f"{w:.2f}\n({k} px)" for w, k in zip(width, sc)], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlim(width.min() / 1.25, width.max() * 1.25)
        ax.set_ylim(0, design * 1.12)
        ax.set_xlabel("Width of one pixel [mm] (at the point, median)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Blur width σ in reflection direction [degrees] (sector mean)")
        ax.set_title(f"({'abc'[j]}) {SECTOR_LABEL[name]} (standard deviation of the tilt per axis)", fontsize=9, loc="left")
    fig.legend(loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=7.5, frameon=False)

    # (d) Test-view PSNR
    ax = fig.add_subplot(gs_[1, :4])
    _axes_style(ax)
    models = [("material", "No blur (σ = 0)"), ("measured_pt", "Conversion: per-point footprint"),
              ("measured_obs", "Conversion: per-observation footprint"), ("measured_obs_aniso", "Conversion: per-observation × anisotropy correction"),
              ("scan", "Conversion: per-observation × best scale factor k"), ("fitted", "Optimized per point"),
              ("supersampled", "Average the reflection within the footprint (no blur approximation)")]
    ys = np.arange(len(models))
    for k in sc:
        mk, col = SCALE_STYLE[k]
        xs, ks = [], None
        for key, _ in models:
            if key == "scan":
                best = max(R[k]["scan_measured_obs"].items(), key=lambda kv: kv[1]["train"])
                xs.append(best[1]["test"])
                ks = best[0]
            else:
                xs.append(R[k]["models"][key]["test"])
        ax.scatter(xs, ys, marker=mk, s=34, color=col, edgecolor=so.C_SURF, lw=0.8, zorder=3,
                   label=f"{k} px (best k = {ks})")
    ax.set_yticks(ys)
    ax.set_yticklabels([m[1] for m in models], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Test-view PSNR of the observed colour [dB] (points of the 5 smooth and bump sectors)", fontsize=8)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False, title="Pixel size", title_fontsize=7.5)
    ax.set_title("(d) Reconstruction quality of the models that share the lighting (all of them: shared lighting of SH degree 48 + per-point constant)", fontsize=9, loc="left")

    # (e) GGX sectors
    ax = fig.add_subplot(gs_[1, 4:])
    _axes_style(ax)
    top = 34
    ax.plot([0, top], [0, top], color=so.C_TEXT, lw=1.0, ls=(0, (4, 3)), label="y = x")
    groups = [(["brdf3", "brdf6", "brdf12"], "fitted_all", C_CAT[4], "GGX sectors (smooth shape)"),
              (["bump3", "bump6", "bump12"], "fitted_geom", C_CAT[0], "Bump sectors")]
    for names, key, col, lab in groups:
        for i, k in enumerate(sc):
            mk, _ = SCALE_STYLE[k]
            x = np.array([R[k]["sectors"][n]["design_blur_deg"] for n in names]) + (i - 1) * 0.7
            f = [R[k]["sectors"][n][key] for n in names]
            y = np.array([v["mean_deg"] for v in f])
            err = np.array([[max(v["mean_deg"] - v["q25_deg"], 0) for v in f], [max(v["q75_deg"] - v["mean_deg"], 0) for v in f]])
            ax.errorbar(x, y, yerr=err, fmt=mk, ms=5, color=col, mec=so.C_SURF, mew=0.8, elinewidth=1.0, capsize=0,
                        label=f"{lab}" if i == 0 else None)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xticks([0, 6, 12, 24])
    ax.set_xlabel("Design blur [degrees] (for GGX, a Gaussian with matched half width)", fontsize=8)
    ax.set_ylabel("Optimal per-point blur σ [degrees] (mean, the lines are 25–75%)", fontsize=8)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)
    ax.text(0.98, 0.03, "Marker shape is the pixel size (○ 512 / △ 256 / □ 128 px)", transform=ax.transAxes, ha="right",
            fontsize=7, color=so.C_TEXT)
    ax.set_title("(e) Blur corresponding to the material roughness", fontsize=9, loc="left")
    fig.suptitle("Bumpy sphere: blur optimized per point, and the conversion from the normal variation within a pixel", fontsize=10.5, x=0.01,
                 y=0.995, ha="left")
    fig.savefig(path, facecolor=so.C_SURF, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _meridian(cam, frm, lon, lats_deg):
    """Project points on the meridian of longitude lon -> u, v, whether they face the camera (cosine of the
    facing angle > 0.3)"""
    lat = np.radians(np.asarray(lats_deg, dtype=float))
    q = np.stack([np.cos(lat) * np.sin(np.radians(lon)), np.sin(lat), np.cos(lat) * np.cos(np.radians(lon))], -1)
    X = torch.tensor(ms.R0 * q, dtype=torch.float32, device=DEV)
    o = torch.tensor(frm, dtype=torch.float32, device=DEV)
    cosv = (F.normalize(o - X, dim=-1) * F.normalize(X, dim=-1)).sum(-1)
    u, v, _ = cam.project(X)
    return u.cpu().numpy(), v.cpu().numpy(), cosv.cpu().numpy()


def plot_views(path):
    """The 4 equatorial views (centre of the 512 px images, with sector names and boundaries), and the centre of
    one view at the 3 pixel sizes"""
    plt = er.setup_plt()
    frm = {n: f for s, n, f in ms.views() if s == "train"}
    lo, hi = 96, 416                       # crop for the top row (the range that contains the sphere)
    name2, c, h = VIEW_IDS[1], 256, 96     # bottom row: the central square of 2h pixels of this view
    fig = plt.figure(figsize=(11.6, 7.7), dpi=200, facecolor=so.C_SURF)
    gs_ = fig.add_gridspec(2, 12, height_ratios=[2.9, 3.87], hspace=0.1, wspace=0.12, left=0.01, right=0.99,
                           top=0.95, bottom=0.01)
    rad = {}
    for j, name in enumerate(VIEW_IDS):
        rad[name] = torch.from_numpy(read_pfm(ms.DATA / "render" / "train" / f"{name}.pfm"))
        ax = fig.add_subplot(gs_[0, 3 * j:3 * j + 3])
        ax.imshow(er.radiance_to_display(rad[name][lo:hi, lo:hi]).numpy().clip(0, 1),
                  extent=(lo, hi, hi, lo))
        cam = Cam(frm[name], ms.SIZE)
        for k, (sname, _, _) in enumerate(ms.SECTORS):
            u, v, cv = _meridian(cam, frm[name], ms.SECTOR_DEG * k, np.linspace(-62, 62, 60))
            ax.plot(np.where(cv > 0.3, u, np.nan), np.where(cv > 0.3, v, np.nan), color="#fcfcfb", lw=0.8,
                    ls=(0, (3, 3)), alpha=0.85)
            u, v, cv = _meridian(cam, frm[name], ms.SECTOR_DEG * (k + 0.5), [0.0])
            if cv[0] > 0.6:  # show the name only for sectors whose centre is close to head-on (2 per view)
                ax.text(u[0], lo + 6, SECTOR_LABEL[sname], ha="center", va="top", fontsize=8, color=so.C_TEXT,
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9))
        ax.set_xlim(lo, hi)
        ax.set_ylim(hi, lo)
        ax.set_axis_off()
    for j, k in enumerate(ms.SCALES):
        f = ms.SIZE // k
        x = F.avg_pool2d(rad[name2].permute(2, 0, 1)[None], f)[0].permute(1, 2, 0)
        a0, a1 = (c - h) // f, (c + h) // f
        ax = fig.add_subplot(gs_[1, 4 * j:4 * j + 4])
        ax.imshow(er.radiance_to_display(x[a0:a1, a0:a1]).numpy().clip(0, 1), interpolation="nearest",
                  extent=(c - h, c + h, c + h, c - h))  # draw in 512 px pixel coordinates (to line up with the boundary lines)
        cam = Cam(frm[name2], ms.SIZE)
        u, v, cv = _meridian(cam, frm[name2], 135.0, np.linspace(-62, 62, 200))
        ax.plot(u, v, color="#fcfcfb", lw=1.0, ls=(0, (3, 3)), alpha=0.9)
        ax.set_xlim(c - h, c + h)
        ax.set_ylim(c + h, c - h)
        ax.set_axis_off()
        ax.set_title(f"{k} px (1 pixel ≈ {1.36 * f:.2f} mm)", fontsize=8, loc="left")
        if j == 0:
            for dx, lab in ((-50, "Bumps 6°"), (50, "Bumps 12°")):
                ax.text(c + dx, c + h - 6, lab, ha="center", va="bottom", fontsize=8, color=so.C_TEXT,
                        bbox=dict(boxstyle="round,pad=0.25", fc="#fcfcfb", ec="none", alpha=0.9))
    lon2 = ms.lon_lat(np.array(frm[name2]))[0]
    fig.suptitle(f"Bumpy sphere (radius {ms.R0 * 100:.0f} cm, bump wavelength {ms.LAMBDA * 1000:.0f} mm, dotted lines are sector boundaries): "
                 f"top: the 4 equatorial views; bottom: the centre of the view at longitude {lon2:.0f}°, at different pixel sizes",
                 fontsize=9.5, x=0.01, y=0.995, ha="left")
    fig.savefig(path, facecolor=so.C_SURF, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=ms.SCALES)
    ap.add_argument("--no-denoised-check", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="only redraw the figures from sphere_rough.json")
    a = ap.parse_args()
    fig_dir = HERE / "figures"
    if a.plot_only:  # recompute the sector summaries from the per-point values (npz), and draw the figures
        out = json.loads((HERE / "sphere_rough.json").read_text(encoding="utf-8"))
        z = np.load(OUT_NPZ)
        sec_all = z["sector"]
        for prefix, res in [(f"s{k}_", v) for k, v in out["scales"].items()] + [("dn256_", {})]:
            pts = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
            if not pts:
                continue
            summarize(res, pts, sec_all)
            if prefix == "dn256_":
                out["denoised_256"] = {k: {"fitted_geom": v.get("fitted_geom")} for k, v in res["sectors"].items()
                                       if ms.SECTORS[[x[0] for x in ms.SECTORS].index(k)][2] == "au"}
        (HERE / "sphere_rough.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        plot_results(out, fig_dir / "sphere_rough.png")
        plot_views(fig_dir / "sphere_views.png")
        return
    t0 = time.time()
    grid = torch.tensor(np.radians(er.SIGMA_GRID_DEG), dtype=torch.float32, device=DEV)
    n0_np, sec_all = sample_points()
    amp = ms.amplitude(ms.R0 * n0_np)
    n0 = torch.tensor(n0_np, dtype=torch.float32, device=DEV)
    a_t = torch.tensor(amp, dtype=torch.float32, device=DEV)
    offs = footprint_offsets()
    out = {"settings": {"n_per_sector": N_PER_SECTOR, "lat_max_deg": LAT_MAX, "border_margin_deg": BORDER_MARGIN,
                        "footprint_samples": FP_SAMPLES, "super_samples": SUPER_SAMPLES, "local_degree": LP, "env_smooth": S, "env_degree": er.LE,
                        "identifiable_contrast": er.ID_CONTRAST, "sectors": ms.SECTORS,
                        "materials": ms.MATERIALS}, "scales": {}}
    arrays = {"n0": n0_np, "sector": sec_all}
    for scale in a.scales:
        res, pts = run_scale(scale, n0, sec_all, a_t, offs, grid, t0)
        summarize(res, pts, sec_all)
        out["scales"][str(scale)] = res
        arrays.update({f"s{scale}_{k}": v for k, v in pts.items()})
        for name, r in res["sectors"].items():
            fg = r.get("fitted_geom", {}).get("median_deg", float("nan"))
            fa = r.get("fitted_all", {}).get("median_deg", float("nan"))
            ex = "".join(f"  {lab} {r[k]:5.1f}°" for lab, k in
                         [("aniso. corr.", "measured_obs_aniso_rms_deg"), ("equiv. l4", "equiv_l4_deg"),
                          ("equiv. l12", "equiv_l12_deg")] if k in r)
            print(f"    {name:<9} design {r['design_blur_deg']:4.0f}°  measured (point) {r['measured_pt_deg']:5.1f}°  "
                  f"measured (obs.) {r['measured_obs_rms_deg']:5.1f}°{ex}  optimized {fg:5.1f}° / on all sectors {fa:5.1f}°")
        print(f"    rank correlations: {res.get('spearman')}", flush=True)
        (HERE / "sphere_rough.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(OUT_NPZ, **arrays)
    if not a.no_denoised_check and 256 in a.scales:  # as for the statue: does using the denoised images change the optimal blur?
        res, pts = run_scale(256, n0, sec_all, a_t, offs, grid, t0, denoised=True, quick=True)
        summarize(res, pts, sec_all)
        out["denoised_256"] = {k: {"fitted_geom": v.get("fitted_geom")} for k, v in res["sectors"].items()
                               if ms.SECTORS[[s[0] for s in ms.SECTORS].index(k)][2] == "au"}
        arrays.update({f"dn256_{k}": v for k, v in pts.items()})
        for name, r in out["denoised_256"].items():
            print(f"    denoised {name:<9} optimized {r['fitted_geom']['median_deg'] if r['fitted_geom'] else float('nan'):5.1f}°")
    (HERE / "sphere_rough.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT_NPZ, **arrays)
    if len(out["scales"]) == len(ms.SCALES):
        plot_results(out, fig_dir / "sphere_rough.png")
        plot_views(fig_dir / "sphere_views.png")
    print(f"-> {HERE / 'sphere_rough.json'}, {OUT_NPZ}, figures/sphere_*.png ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
