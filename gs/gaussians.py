# -*- coding: utf-8 -*-
"""
Parameters of the set of Gaussians, and the operations that change their number (densification).

Trainable parameters of each Gaussian (activation used at render time in parentheses):
    means          (N,3)    center position
    log_scales     (N,3)    log of the per-axis standard deviation   (exp)
    quats          (N,4)    rotation quaternion [w,x,y,z]            (normalize)
    opacity_logits (N,1)    logit of the opacity                     (sigmoid)
    sh0            (N,1,3)  degree-0 SH coefficients (diffuse color)
    shN            (N,K,3)  SH coefficients of degree 1 and above (view-dependent color), K = (deg+1)^2 - 1

densification (paper Sec. 5.2):
    A large screen-space positional gradient is taken to mean that the reconstruction around it is insufficient, and
      - small Gaussians are duplicated (clone): fills regions where the shape is not fully covered
      - large Gaussians are split in two (split): refines regions where details cannot be represented
    In addition, those with opacity near 0 and those that are too large are removed (prune).
    The number of parameters changes, so the Adam moments must also be rebuilt to the same shape.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rasterize import quat_to_rotmat
from .sh import rgb_to_sh0

NAMES = ("means", "log_scales", "quats", "opacity_logits", "sh0", "shN")


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@torch.no_grad()
def knn_mean_sq_dist(x, k=3, chunk=2048):
    """Mean squared distance from each point to its k nearest neighbors (used to decide the initial scale)"""
    out = []
    for i in range(0, len(x), chunk):
        d2 = torch.cdist(x[i:i + chunk], x).pow(2)
        out.append(d2.topk(k + 1, dim=1, largest=False).values[:, 1:].mean(1))  # the first one is the point itself
    return torch.cat(out)


class GaussianModel:
    # Names of the trainable parameters. Subclasses can add to them
    # (densification and the Adam rebuild follow this list).
    names = NAMES

    def __init__(self, sh_degree=3):
        self.max_sh_degree = sh_degree
        self.params = {}
        self.optimizer = None

    def __len__(self):
        return self.params["means"].shape[0]

    # ---------------------------------------------------------------- activated values
    @property
    def means(self):
        return self.params["means"]

    @property
    def scales(self):
        return torch.exp(self.params["log_scales"])

    @property
    def quats(self):
        return F.normalize(self.params["quats"], dim=-1)

    @property
    def opacities(self):
        return torch.sigmoid(self.params["opacity_logits"]).squeeze(-1)

    # ---------------------------------------------------------------- initialization
    def init_random(self, n, half_extent, device="cuda", center=(0.0, 0.0, 0.0)):
        """Place points at random inside a cube of side 2*half_extent centered at center
        (same approach as the official implementation's initialization for synthetic data)"""
        xyz = (torch.rand(n, 3, device=device) * 2 - 1) * half_extent + torch.tensor(center, device=device)
        rgb = torch.rand(n, 3, device=device)
        self.init_from_points(xyz, rgb)

    def init_from_points(self, xyz, rgb):
        n, dev = len(xyz), xyz.device
        dist = torch.sqrt(knn_mean_sq_dist(xyz).clamp(min=1e-7))  # make each a sphere about as large as the spacing to its neighbors
        K = (self.max_sh_degree + 1) ** 2 - 1
        self._set({
            "means": xyz,
            "log_scales": torch.log(dist)[:, None].repeat(1, 3),
            "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1),
            "opacity_logits": inverse_sigmoid(torch.full((n, 1), 0.1, device=dev)),
            "sh0": rgb_to_sh0(rgb)[:, None, :],
            "shN": torch.zeros(n, K, 3, device=dev),
        })

    def _set(self, tensors):
        self.params = {k: nn.Parameter(tensors[k].contiguous()) for k in self.names}
        self._reset_stats()

    def _reset_stats(self):
        n, dev = len(self), self.means.device
        self.grad_accum = torch.zeros(n, device=dev)   # accumulated norm of the screen-space positional gradient
        self.denom = torch.zeros(n, device=dev)        # number of times it appeared on screen
        self.max_radii2d = torch.zeros(n, device=dev)  # maximum screen-space radius [px]

    # ---------------------------------------------------------------- optimization
    def setup_optimizer(self, spatial_scale, lr_means=1.6e-4, lr_means_final=1.6e-6,
                        lr_sh=2.5e-3, lr_opacity=0.05, lr_scale=5e-3, lr_rot=1e-3, extra_lrs=None):
        """Learning rates are the defaults of the official implementation as first released. Only the position
        learning rate is made proportional to the scene size and decayed exponentially.
        extra_lrs gives the learning rates of parameters added by subclasses {name: learning rate}"""
        self.lr_means_init = lr_means * spatial_scale
        self.lr_means_final = lr_means_final * spatial_scale
        lrs = {"means": self.lr_means_init, "sh0": lr_sh, "shN": lr_sh / 20,
               "opacity_logits": lr_opacity, "log_scales": lr_scale, "quats": lr_rot, **(extra_lrs or {})}
        groups = [{"params": [self.params[k]], "lr": lrs[k], "name": k} for k in self.names]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

    def update_lr(self, step, max_steps):
        t = min(step / max_steps, 1.0)
        lr = math.exp((1 - t) * math.log(self.lr_means_init) + t * math.log(self.lr_means_final))
        for g in self.optimizer.param_groups:
            if g["name"] == "means":
                g["lr"] = lr

    def _replace(self, name, tensor, state_fn):
        """Replace parameter name with tensor, and bring Adam's first and second moments to the same shape
        with state_fn."""
        new = nn.Parameter(tensor.contiguous())
        if self.optimizer is not None:
            group = next(g for g in self.optimizer.param_groups if g["name"] == name)
            state = self.optimizer.state.pop(group["params"][0], None)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    state[key] = state_fn(state[key])
                self.optimizer.state[new] = state
            group["params"][0] = new
        self.params[name] = new

    def _append(self, extra):
        for k in self.names:
            e = extra[k]
            self._replace(k, torch.cat([self.params[k].detach(), e]),
                          lambda s, e=e: torch.cat([s, torch.zeros_like(e)]))  # moments of the new points are 0

    def _keep(self, mask):
        for k in self.names:
            self._replace(k, self.params[k].detach()[mask], lambda s: s[mask])

    # ---------------------------------------------------------------- densification
    @torch.no_grad()
    def add_densification_stats(self, grad2d, ids, radius, W, H):
        """Accumulate the statistics of one render. grad2d is ∂L/∂μ' in pixel coordinates (len(ids), 2).

        The official implementation looks at the gradient in NDC coordinates (screen width is 2) and compares it
        with the threshold 0.0002, so we multiply by the pixel -> NDC conversion (W/2, H/2) to match.
        """
        scale = torch.tensor([0.5 * W, 0.5 * H], device=grad2d.device)
        self.grad_accum[ids] += (grad2d * scale).norm(dim=-1)
        self.denom[ids] += 1
        self.max_radii2d[ids] = torch.maximum(self.max_radii2d[ids], radius)

    @torch.no_grad()
    def densify_and_prune(self, grad_thresh, min_opacity, extent, max_screen_size=None,
                          percent_dense=0.01):
        n0 = len(self)
        grads = self.grad_accum / self.denom.clamp(min=1)
        size_thresh = percent_dense * extent
        is_big = self._is_big(size_thresh)
        clone_mask = (grads >= grad_thresh) & ~is_big
        split_mask = (grads >= grad_thresh) & is_big
        big_screen = self.max_radii2d > max_screen_size if max_screen_size else None

        # clone: duplicate as is at the same position (depth order decides which one is in front,
        # so later gradients move them apart)
        new = {k: v[clone_mask] for k, v in self.params.items()}

        split = self._split_children(split_mask, size_thresh)

        self._append({k: torch.cat([new[k], split[k]]) for k in self.names})

        # prune: the originals of a split, nearly transparent ones,
        # and (after an opacity reset) ones that are too large
        pad = torch.zeros(len(self) - n0, dtype=torch.bool, device=self.means.device)
        prune = torch.cat([split_mask, pad]) | (self.opacities < min_opacity)
        if big_screen is not None:
            prune |= torch.cat([big_screen, pad])
            prune |= self.scales.max(dim=1).values > 0.1 * extent
        self._keep(~prune)
        self._reset_stats()
        return int(clone_mask.sum()), int(split_mask.sum()), int(prune.sum())

    def _is_big(self, size_thresh):
        """Those to split rather than clone (subclasses can add conditions)"""
        return self.scales.max(dim=1).values > size_thresh

    def _split_children(self, split_mask, size_thresh, ns=2):
        """split: sample ns positions from the original Gaussian's distribution and place copies shrunk by 1/1.6"""
        split = {k: v[split_mask].repeat(ns, *[1] * (v.dim() - 1)) for k, v in self.params.items()}
        std = self.scales[split_mask].repeat(ns, 1)
        R = quat_to_rotmat(self.quats[split_mask]).repeat(ns, 1, 1)
        split["means"] = split["means"] + (R @ (torch.randn_like(std) * std)[..., None]).squeeze(-1)
        split["log_scales"] = torch.log(std / (0.8 * ns))
        return split

    @torch.no_grad()
    def reset_opacity(self, value=0.01):
        """Lower all opacities at once (paper Sec. 5.2).

        "Junk" Gaussians floating near the cameras (floaters) get their opacity back through optimization if they
        are really needed. If not, it does not come back and they are pruned.
        """
        cap = inverse_sigmoid(torch.tensor(value, device=self.means.device))
        self._replace("opacity_logits", torch.minimum(self.params["opacity_logits"].detach(), cap),
                      torch.zeros_like)

    # ---------------------------------------------------------------- save / load
    def state_dict(self):
        return {"max_sh_degree": self.max_sh_degree,
                "params": {k: v.detach().cpu() for k, v in self.params.items()}}

    @classmethod
    def from_state_dict(cls, sd, device="cuda"):
        m = cls(sd["max_sh_degree"])
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m

    @torch.no_grad()
    def save_ply(self, path):
        """.ply in the same format as the official implementation (opens in 3DGS viewers such as SuperSplat)"""
        p = {k: v.detach().cpu() for k, v in self.params.items()}
        n = len(self)
        f_dc = p["sh0"].transpose(1, 2).reshape(n, -1)
        f_rest = p["shN"].transpose(1, 2).reshape(n, -1)  # grouped per channel (R...,G...,B...)
        cols = [p["means"], torch.zeros(n, 3), f_dc, f_rest, p["opacity_logits"], p["log_scales"], p["quats"]]
        data = torch.cat(cols, 1).numpy().astype("<f4")
        names = (["x", "y", "z", "nx", "ny", "nz"]
                 + [f"f_dc_{i}" for i in range(f_dc.shape[1])]
                 + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
                 + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
        header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
        header += "".join(f"property float {s}\n" for s in names) + "end_header\n"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(np.ascontiguousarray(data).tobytes())
