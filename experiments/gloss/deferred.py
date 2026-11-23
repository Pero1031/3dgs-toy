# -*- coding: utf-8 -*-
"""
Stage 3: add deferred shading to 3dgs-toy (look up shared environment lighting in the reflection direction.
A simplified version in the same family as 3DGS-DR and Ref-GS).

Instead of colour SH, each Gaussian holds:
    normal  the shortest axis (turned toward the camera). It has no parameters and is determined by rotation and size
    base    constant term of the radiance (3) (softplus). Corresponds to the "per-point constant" of the oracle
    rough   width of the blur in the reflection direction, corresponding to material roughness (1) (sigmoid × SIG_MAX)

Rendering has 2 steps:
    1. The rasterizer α-blends the normal, base, and rough in the same way as colour (gs/rasterize.py. Any number
       of channels)
    2. For each pixel, reflect the view direction about the composited normal Σwn, and look up the shared environment
       lighting blurred with width σ:
           radiance = Σw·base + A·L_env(r, σ) + (1 − A)·background    (A: accumulated opacity)
       How σ is set (mode):
           learned  σ = composited rough / A
                    roughness is learned, as in existing methods
           toksvig  σ = 2√(1 − |n̄|) · √((1 + cos²θ)/2), |n̄| = |Σwn| / A
                    roughness is not learned (same as the conversion of stage 2)
           both     σ = √(toksvig² + learned²)
                    the geometric blur is converted, and only the material roughness is learned
       The conversion to display values (Reinhard → sRGB curve) is also the same as for the images, and the error
       is taken in display values.

The environment lighting is held as SH coefficients (degree LE). To look up blurred lighting, the SH blurred at each
width of SIG_LEVELS is expanded into equirectangular textures (the basis is computed in advance), and 3D linear
interpolation is done over reflection direction and σ (grid_sample).
Evaluating high SH degrees differentiably per pixel is too heavy, so gradients flow to all of the lighting, the
direction, and σ through the textures.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from gs.gaussians import GaussianModel, inverse_sigmoid, knn_mean_sq_dist  # noqa: E402
from gs.rasterize import conic_and_radius, covariance_3d, project_gaussians, quat_to_rotmat, rasterize  # noqa: E402
from gs.sh import sh_basis  # noqa: E402

LE = 48                                   # SH degree of the environment lighting (same as the oracle)
SIG_MAX = math.radians(40.0)              # upper limit of the blur width
SIG_STEP = math.radians(2.0)              # spacing of the widths at which blurred lighting is prepared (linear interpolation in between)
TEX_H, TEX_W = 128, 256                   # equirectangular texture into which the lighting is expanded (1 pixel = 1.4°)
MODES = ("learned", "toksvig", "both")


# ---- Conversion between display values and radiance (same as env_rough.py. Images: Reinhard → sRGB curve) ----

def lin_to_srgb(t):
    t = t.clamp(min=0)
    return torch.where(t <= 0.0031308, 12.92 * t, 1.055 * t.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def srgb_to_lin(y):
    return torch.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def radiance_to_display(x):
    x = x.clamp(min=0)
    return lin_to_srgb(x / (1 + x)).clamp(0, 1)


def display_to_radiance(y):
    t = srgb_to_lin(torch.as_tensor(y)).clamp(max=0.995)
    return t / (1 - t)


# ---- Environment lighting ----

def dir_to_uv(r):
    """Direction -> equirectangular (u, v) ∈ [0,1)×[0,1]. u is the longitude from +z toward +x, v is the angle
    from +y (north)"""
    u = torch.atan2(r[..., 0], r[..., 2]) / (2 * math.pi) % 1.0
    v = torch.acos(r[..., 1].clamp(-1 + 1e-6, 1 - 1e-6)) / math.pi
    return u, v


class EnvSH:
    """Environment lighting (radiance) held as SH coefficients. lookup(r, σ) returns the value in reflection
    direction r blurred by a Gaussian of width σ"""

    def __init__(self, Le=LE, init=0.3, device="cuda"):
        self.Le, self.Ke = Le, (Le + 1) ** 2
        coef = torch.zeros(self.Ke, 3, device=device)
        coef[0] = init / 0.28209479177387814  # start from a uniform brightness init
        self.coef = nn.Parameter(coef)
        v = (torch.arange(TEX_H, device=device) + 0.5) / TEX_H
        u = (torch.arange(TEX_W, device=device) + 0.5) / TEX_W
        vv, uu = torch.meshgrid(v * math.pi, u * 2 * math.pi, indexing="ij")
        d = torch.stack([vv.sin() * uu.sin(), vv.cos(), vv.sin() * uu.cos()], -1).reshape(-1, 3)
        self.B = sh_basis(Le, d).float()                                     # (texels, Ke)
        self.l = torch.tensor([math.isqrt(k) for k in range(self.Ke)], dtype=torch.float32, device=device)
        sig = torch.arange(0, SIG_MAX + 1e-6, SIG_STEP, device=device)
        self.n_levels = len(sig)
        self.att = torch.exp(-self.l * (self.l + 1) * sig[:, None] ** 2 / 2)  # (levels, Ke)

    def textures(self):
        """Textures of the lighting blurred at each width (1, 3, levels, TEX_H, TEX_W + 2). One column is wrapped
        around and added on each side for the seam in longitude"""
        E = self.att[:, :, None] * self.coef[None]                            # (levels, Ke, 3)
        T = (self.B @ E.permute(1, 0, 2).reshape(self.Ke, -1))                # (texels, levels·3)
        T = T.reshape(TEX_H, TEX_W, self.n_levels, 3).permute(3, 2, 0, 1)    # (3, levels, H, W)
        T = torch.cat([T[..., -1:], T, T[..., :1]], -1)
        return T[None]

    def lookup(self, r, sig, T=None):
        """r (M,3) unit vectors, sig (M,) radians -> (M,3). T is the result of textures() (reused within the same iteration)"""
        T = self.textures() if T is None else T
        u, v = dir_to_uv(r)
        x = -1 + 2 * (u * TEX_W + 0.5) / (TEX_W + 1)                        # coordinate including the wrapped columns (align_corners=True)
        y = (-1 + 2 * (v * TEX_H - 0.5) / (TEX_H - 1)).clamp(-1, 1)
        z = (-1 + 2 * (sig / SIG_STEP) / (self.n_levels - 1)).clamp(-1, 1)
        grid = torch.stack([x, y, z], -1).reshape(1, 1, 1, -1, 3)
        return F.grid_sample(T, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, 0].T

    def smoothness(self):
        """Penalty that is heavier for higher degrees, Σ l(l+1)|e|² / Ke (suppresses noise in the lighting)"""
        return ((self.l * (self.l + 1))[:, None] * self.coef ** 2).mean()


# ---- Gaussians ----

class DeferredModel(GaussianModel):
    names = ("means", "log_scales", "quats", "opacity_logits", "base", "rough")

    def __init__(self):
        super().__init__(sh_degree=0)

    def init_on_sphere(self, n, radius, center=(0.0, 0.0, 0.0), flat=0.1, rough_deg=2.0, device="cuda"):
        """Place n Gaussians uniformly on a sphere of radius radius, as discs lying along the surface (the size in
        the normal direction is flat times the size along the surface).
        A rough initial shape (corresponding to the SfM points in 3DGS) is assumed to be known"""
        g = torch.Generator(device="cpu").manual_seed(0)
        nrm = F.normalize(torch.randn(n, 3, generator=g), dim=-1).to(device)
        xyz = radius * nrm + torch.tensor(center, device=device)
        dist = torch.sqrt(knn_mean_sq_dist(xyz).clamp(min=1e-10))
        # rotation that turns the local z axis toward the normal (by acos(n_z) about the axis z × n). [w, x, y, z]
        q = F.normalize(torch.stack([1 + nrm[:, 2], -nrm[:, 1], nrm[:, 0], torch.zeros_like(nrm[:, 0])], -1), dim=-1)
        ls = torch.log(dist)[:, None].repeat(1, 3)
        ls[:, 2] += math.log(flat)
        self._set({
            "means": xyz,
            "log_scales": ls,
            "quats": q,
            "opacity_logits": inverse_sigmoid(torch.full((n, 1), 0.5, device=device)),
            "base": torch.full((n, 3), math.log(math.expm1(0.01)), device=device),   # 0.01 after softplus
            "rough": inverse_sigmoid(torch.full((n, 1), rough_deg / math.degrees(SIG_MAX), device=device)),
        })

    def state_dict(self):
        return {"params": {k: v.detach().cpu() for k, v in self.params.items()}}

    @staticmethod
    def from_state_dict(sd, device="cuda"):
        m = DeferredModel()
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m


def gaussian_normals(scales, quats, means, cam_center):
    """Direction of the shortest axis (turned toward the camera) (N,3)"""
    R = quat_to_rotmat(quats)
    idx = scales.argmin(dim=1)
    n = R.gather(2, idx[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    flip = ((cam_center - means) * n).sum(-1, keepdim=True) < 0
    return torch.where(flip, -n, n)


def pixel_dirs(cam):
    """World-space direction of the view ray (camera → object) through each pixel centre (H, W, 3)"""
    dev = cam.R.device
    j, i = torch.meshgrid(torch.arange(cam.H, device=dev), torch.arange(cam.W, device=dev), indexing="ij")
    d = torch.stack([(i + 0.5 - cam.cx) / cam.fx, (j + 0.5 - cam.cy) / cam.fy, torch.ones_like(i, dtype=torch.float32)], -1)
    return F.normalize(d @ cam.R, dim=-1)   # cam.R is world -> camera, so multiplying a row vector from the right gives camera -> world


def render_deferred(model, env, cam, mode, bg_rad, tile=8, max_entries=2 ** 24, near=0.2, T=None):
    """
    Returns dict:
        image (H,W,3) display values, alpha (H,W), means2d / ids / on_screen / radius (for densification. Same as
        render.py)
        len_n (H,W) |n̄|, sig_geo (H,W) converted blur, sig_mat (H,W) composited rough, sigma (H,W) blur actually used
    """
    p = model.params
    with torch.no_grad():
        z = p["means"] @ cam.R[2] + cam.t[2]
        ids = torch.nonzero(z > near).squeeze(1)
    means = p["means"][ids]
    scales = torch.exp(p["log_scales"][ids])
    quats = F.normalize(p["quats"][ids], dim=-1)
    opacity = torch.sigmoid(p["opacity_logits"][ids]).squeeze(-1)
    means2d, cov2d, depth = project_gaussians(means, covariance_3d(scales, quats), cam)
    conic, radius, tile_radius = conic_and_radius(cov2d, opacity)
    if means2d.requires_grad:
        means2d.retain_grad()

    nrm = gaussian_normals(scales, quats, means, cam.center)
    feat = torch.cat([nrm, F.softplus(p["base"][ids]), torch.sigmoid(p["rough"][ids]) * SIG_MAX], -1)
    buf, A, on_screen = rasterize(means2d, conic, opacity, feat, tile_radius, depth, cam.W, cam.H,
                                  torch.zeros(feat.shape[-1], device=feat.device), tile=tile, max_entries=max_entries)
    nsum, base, rsum = buf[..., :3], buf[..., 3:6], buf[..., 6]
    nlen = nsum.norm(dim=-1).clamp(min=1e-8)
    nbar = nsum / nlen[..., None]
    len_n = (nlen / A.clamp(min=1e-6)).clamp(max=1.0)
    d = pixel_dirs(cam)
    dn = (d * nbar).sum(-1, keepdim=True)
    r = d - 2 * dn * nbar
    cos2 = (dn[..., 0] ** 2).clamp(max=1.0)
    sig_geo = 2 * torch.sqrt((1 - len_n).clamp(min=0) + 1e-6) * torch.sqrt((1 + cos2) / 2)
    sig_mat = rsum / A.clamp(min=1e-6)
    sigma = {"learned": sig_mat, "toksvig": sig_geo,
             "both": torch.sqrt(sig_geo ** 2 + sig_mat ** 2)}[mode]
    L = env.lookup(r.reshape(-1, 3), sigma.reshape(-1), T).reshape(cam.H, cam.W, 3)
    rad = base + A[..., None] * L + (1 - A[..., None]) * bg_rad
    return {"image": radiance_to_display(rad), "alpha": A, "means2d": means2d, "ids": ids,
            "on_screen": on_screen, "radius": radius, "len_n": len_n, "sig_geo": sig_geo,
            "sig_mat": sig_mat, "sigma": sigma}
