# -*- coding: utf-8 -*-
"""
Colour spaces for training (colour experiment, stage 1; experiments/color/README.md).

The data space of a dataset is given by "color_space" in its transforms_*.json:
    absent / "display-srgb"   display values of an sRGB screen in [0,1] (8-bit PNG; the default of every older dataset)
    "linear-rec2020"          linear radiance, Rec.2020 primaries, D65 white (GoLD_rayt spectral build, --gamut rec2020, .pfm)

The colour mode (train.py --color-mode) says what the Gaussians' colours mean:
    display               display sRGB values, colour clamped at 0 (the original 3DGS). Linear data is first mapped to
                          display values exactly as GoLD_rayt writes a PNG (negative channels clipped, Reinhard
                          x / (1 + x), sRGB curve)
    linear-srgb           linear sRGB, colour clamped at 0 (colours outside the sRGB gamut cannot be produced)
    linear-srgb-noclamp   linear sRGB, no clamp
    linear-rec2020        linear Rec.2020, colour clamped at 0

The loss is taken on encode(x) = sign(x) * D(|x|), D(x) = sRGB curve of x / (1 + x), computed on linear sRGB for every
linear mode. For colours inside the sRGB gamut this equals the display values, so all modes weight errors the same way
and differ only in the space the Gaussians live in and in the clamp. Colours a mode cannot produce, it cannot be scored
on either: that is the point.

The knee k (train.py --loss-knee, linear modes only) generalizes D to the sRGB curve of x / (1 + x / k). k = 1 is the
encoding above; a larger k compresses highlights less (k = inf: the sRGB curve alone), while dark values stay nearly
the same (x / (1 + x / k) ~ x for x << k). PSNR and saved images always use k = 1 (ColorMode.view), so that runs with
different knees are scored alike.
"""

import json
from pathlib import Path

import torch

MODES = ("display", "linear-srgb", "linear-srgb-noclamp", "linear-rec2020")

# linear sRGB -> linear Rec.2020 (ITU-R BT.2087; the same matrix as GoLD_rayt's Color/Gamut.hpp)
SRGB_TO_2020 = torch.tensor([[0.6274040, 0.3292820, 0.0433136],
                             [0.0690970, 0.9195400, 0.0113612],
                             [0.0163916, 0.0880132, 0.8955950]])
REC2020_TO_SRGB = torch.linalg.inv(SRGB_TO_2020)


def data_color_space(datadir):
    meta = json.loads((Path(datadir) / "transforms_train.json").read_text(encoding="utf-8"))
    return meta.get("color_space", "display-srgb")


def _mat(x, m):
    return x @ m.to(x.device, x.dtype).T


def srgb_curve(t):
    return torch.where(t <= 0.0031308, 12.92 * t, 1.055 * t.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def srgb_curve_inv(v):
    return torch.where(v <= 0.04045, v / 12.92, ((v.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


def encode(x, knee=1.0):
    """sign(x) * sRGB(|x| / (1 + |x| / knee)). knee = 1: the display transform of GoLD_rayt for x >= 0, continued oddly
    below 0. knee = inf: the sRGB curve alone (above 1 it continues as 1.055 x^(1/2.4) - 0.055)"""
    a = x.abs()
    t = a if knee == float("inf") else a / (1 + a / knee)
    return torch.sign(x) * srgb_curve(t)


def display_to_linear(v, t_max=0.999):
    """Inverse of the display transform (for evaluation). Display 1 cannot be inverted; it is capped at t_max"""
    t = srgb_curve_inv(v.clamp(0, 1)).clamp(max=t_max)
    return t / (1 - t)


class ColorMode:
    def __init__(self, mode, data_space, loss_knee=1.0):
        assert mode in MODES, mode
        if mode == "display" and loss_knee != 1.0:
            raise ValueError("--loss-knee applies to the linear modes only")
        assert data_space in ("display-srgb", "linear-rec2020"), data_space
        if mode != "display" and data_space == "display-srgb":
            raise ValueError(f"--color-mode {mode} needs linear data; {data_space} is display values")
        self.mode, self.data_space = mode, data_space
        self.clamp = mode != "linear-srgb-noclamp"
        self.knee = loss_knee

    def image_fn(self, img):
        """data -> the space of the Gaussians (applied once when the images are loaded)"""
        if self.data_space == "display-srgb":
            return img
        if self.mode == "linear-rec2020":
            return img
        lin = _mat(img, REC2020_TO_SRGB)
        if self.mode == "display":
            return encode(lin.clamp(min=0))
        return lin

    def to_linear_srgb(self, x):
        """space of the Gaussians -> linear sRGB (for evaluation; negative values kept)"""
        if self.mode == "display":
            return display_to_linear(x)
        if self.mode == "linear-rec2020":
            return _mat(x, REC2020_TO_SRGB)
        return x

    def loss_view(self, x):
        """what the L1 / SSIM are computed on"""
        if self.mode == "display":
            return x
        return encode(self.to_linear_srgb(x), self.knee)

    def view(self, x):
        """what PSNR and saved images use: loss_view with knee 1"""
        if self.mode == "display":
            return x
        return encode(self.to_linear_srgb(x))
