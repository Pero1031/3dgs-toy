# -*- coding: utf-8 -*-
"""
Gaussians that carry their own time, i.e. spacetime Gaussians
(a simplified version of Spacetime Gaussians, Li et al. CVPR 2024 / 4D Gaussians, Yang et al. ICLR 2024).

gs/deform.py "offsets one canonical shape with an MLP". Here each Gaussian has its own time,
appears only around that time, and moves in a straight line:

    t_center    (N,1)  time at which it looks most opaque, μ_t
    log_t_scale (N,1)  log of the temporal width σ_t
    velocity    (N,3)  velocity v

    position at time t  x + v (t - μ_t)
    opacity at time t   o · exp(-½ ((t - μ_t) / σ_t)²)

Curved motion is represented by a polyline of Gaussians with short temporal width lined up in time order. No canonical
shape is needed, so objects may move a long way, or appear and disappear midway. In exchange, the frames that can see
one Gaussian are only those inside its temporal width (with one-view-per-time data, the number of views
that see it goes down).

Differences from the papers:
    - Motion is first-order (STG uses a cubic polynomial). Rotation and scale do not depend on time
    - Color stays a time-independent SH (STG turns features into color with a small MLP)
    - Splitting along time applies only to those whose temporal width is longer than t_split_thresh
      (shrunk by 1/1.6, same as the spatial split)
    - A lower bound can be put on the temporal width: σ_t = min_t_scale + exp(log_t_scale). With one-view-per-time
      data and no lower bound, the temporal width shrinks to a few frames' worth, and the model overfits the
      training views without learning the motion (experiments/dynamic)
"""

import math

import torch

from .gaussians import NAMES, GaussianModel


class SpacetimeModel(GaussianModel):
    names = NAMES + ("t_center", "log_t_scale", "velocity")

    def __init__(self, sh_degree=3, init_t_scale=0.2, t_split_thresh=0.05, min_t_scale=0.0):
        super().__init__(sh_degree)
        self.init_t_scale = init_t_scale
        self.min_t_scale = min_t_scale
        # Do not split just above the lower bound
        # (it cannot shrink past the lower bound, so splitting would barely change the temporal width)
        self.t_split_thresh = max(t_split_thresh, 1.6 * min_t_scale)

    @property
    def t_scales(self):
        """Temporal width σ_t (N,1)"""
        return self.min_t_scale + torch.exp(self.params["log_t_scale"])

    def _set(self, tensors):
        if "t_center" not in tensors:   # initialization: times uniform on [0,1], velocity 0
            n, dev = len(tensors["means"]), tensors["means"].device
            tensors = dict(tensors, t_center=torch.rand(n, 1, device=dev),
                           log_t_scale=torch.full((n, 1), math.log(max(self.init_t_scale - self.min_t_scale, 1e-3)),
                                                  device=dev),
                           velocity=torch.zeros(n, 3, device=dev))
        super()._set(tensors)

    def setup_optimizer(self, spatial_scale, lr_t=5e-4, lr_t_scale=5e-3, lr_velocity=1.6e-3, **kw):
        """The velocity learning rate is made proportional to the scene size, like the position one (no decay)"""
        super().setup_optimizer(spatial_scale, extra_lrs={
            "t_center": lr_t, "log_t_scale": lr_t_scale, "velocity": lr_velocity * spatial_scale}, **kw)

    def delta_at(self, t):
        """Offsets at time t (the delta of gs/render.py)"""
        dt = float(t) - self.params["t_center"]                          # (N,1)
        w = torch.exp(-0.5 * (dt / self.t_scales) ** 2)
        return {"means": self.params["velocity"] * dt, "opacity": w.squeeze(-1)}

    # ---------------------------------------------------------------- densification
    def _long(self):
        return self.t_scales.squeeze(-1) > self.t_split_thresh

    def _is_big(self, size_thresh):
        return super()._is_big(size_thresh) | self._long()

    def _split_children(self, split_mask, size_thresh, ns=2):
        """Split spatially large ones in space and ones with long temporal width in time (both if both apply)"""
        split = super()._split_children(split_mask, size_thresh, ns)
        rep = lambda v: v[split_mask].repeat(ns, *[1] * (v.dim() - 1))
        big = rep(GaussianModel._is_big(self, size_thresh))
        for k in ("means", "log_scales"):          # for those that are not spatially large, keep position and scale as they were
            split[k] = torch.where(big[:, None], split[k], rep(self.params[k]))
        long = rep(self._long())[:, None]
        sigma_t = torch.exp(split["log_t_scale"])                    # the part above the lower bound. The split shrinks this part
        dt = torch.where(long, torch.randn_like(sigma_t) * (self.min_t_scale + sigma_t), torch.zeros_like(sigma_t))
        split["t_center"] = split["t_center"] + dt
        split["means"] = split["means"] + split["velocity"] * dt     # at the new time, place it on the same trajectory as the original
        split["log_t_scale"] = torch.where(long, torch.log(sigma_t / (0.8 * ns)), split["log_t_scale"])
        return split

    def state_dict(self):
        return dict(super().state_dict(), init_t_scale=self.init_t_scale, t_split_thresh=self.t_split_thresh,
                    min_t_scale=self.min_t_scale)

    @classmethod
    def from_state_dict(cls, sd, device="cuda"):
        m = cls(sd["max_sh_degree"], sd["init_t_scale"], sd["t_split_thresh"], sd.get("min_t_scale", 0.0))
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m
