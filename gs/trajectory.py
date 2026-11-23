# -*- coding: utf-8 -*-
"""
Gaussians that directly hold per-time-knot positions
(a simplified version of Dynamic 3D Gaussians, Luiten et al. 3DV 2024).

The number, color, opacity, scale and orientation of the Gaussians do not depend on time. Only the position changes
with time. Each Gaussian has an offset for each time knot (n_knots of them, evenly spaced on [0,1]), and the offsets
are linearly interpolated between knots:

    offsets (N, n_knots, 3)      position at time t = means + interp(offsets, t)

Unlike gs/deform.py (an MLP offsets the canonical shape), the motion is free per Gaussian, and the canonical shape
does not need to roughly represent all times. Unlike gs/spacetime.py (appear and disappear), the same Gaussian
exists from start to end, so it has no choice but to keep following the same point of the object.

Training proceeds in time order (train_dynamic.py --trajectory): at first it trains only on the frames up to the
first few knots, and every fixed number of iterations it starts using the next knot. A new knot is initialized at
the position extrapolated from the previous 2 knots (= tracking starts from the position at the previous time).
If all times are trained at once from the start, at times not yet trained the Gaussians are left behind in places
where there is no object, and there is no gradient that pulls them to the object.

Differences from the paper:
    - The paper is multi-view (about 30 cameras): it builds the shape at the first time and then optimizes one time
      at a time. Here there is one view per time, so the frames of the first few knots are used together, and the
      earlier times keep being trained after knots are added
    - Rotation does not depend on time (per time in the paper). Losses such as local rigidity are not included
      (a loss that aligns the velocities of nearby Gaussians can be added with --lambda-smooth of train_dynamic.py)
"""

import torch

from .gaussians import NAMES, GaussianModel


class TrajectoryModel(GaussianModel):
    names = NAMES + ("offsets",)

    def __init__(self, sh_degree=3, n_knots=21, n_active=3):
        super().__init__(sh_degree)
        self.n_knots = n_knots
        self.n_active = min(n_active, n_knots)   # number of knots in use so far

    @property
    def t_active(self):
        """Upper limit of the times that may be used for training (the time of the most recently activated knot)"""
        return (self.n_active - 1) / (self.n_knots - 1)

    def _set(self, tensors):
        if "offsets" not in tensors:
            n, dev = len(tensors["means"]), tensors["means"].device
            tensors = dict(tensors, offsets=torch.zeros(n, self.n_knots, 3, device=dev))
        super()._set(tensors)

    def setup_optimizer(self, spatial_scale, lr_offsets=3.2e-4, **kw):
        """The learning rate of the knot offsets is made proportional to the scene size
        (no decay, because some knots only start being used later)"""
        super().setup_optimizer(spatial_scale, extra_lrs={"offsets": lr_offsets * spatial_scale}, **kw)

    def _segment(self, t):
        u = min(max(float(t), 0.0), 1.0) * (self.n_knots - 1)
        i = min(int(u), self.n_knots - 2)
        return i, u - i

    def delta_at(self, t):
        """Offsets at time t (the delta of gs/render.py)"""
        i, w = self._segment(t)
        o = self.params["offsets"]
        return {"means": (1 - w) * o[:, i] + w * o[:, i + 1]}

    def velocity_at(self, t):
        """Velocity at time t (N,3) (constant between knots)"""
        i, _ = self._segment(t)
        o = self.params["offsets"]
        return (o[:, i + 1] - o[:, i]) * (self.n_knots - 1)

    @torch.no_grad()
    def activate_next(self):
        """Start using the next knot. It is initialized at the position extrapolated from the previous 2 knots,
        and all later knots are set to the same position"""
        k = self.n_active
        if k >= self.n_knots:
            return False
        o = self.params["offsets"]
        o[:, k:] = (2 * o[:, k - 1] - o[:, k - 2])[:, None]
        self.n_active += 1
        return True

    def state_dict(self):
        return dict(super().state_dict(), n_knots=self.n_knots, n_active=self.n_active)

    @classmethod
    def from_state_dict(cls, sd, device="cuda"):
        m = cls(sd["max_sh_degree"], sd["n_knots"], sd["n_active"])
        m._set({k: v.to(device) for k, v in sd["params"].items()})
        return m
