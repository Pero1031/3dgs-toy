# -*- coding: utf-8 -*-
"""
A visible-difference loss between a render and the truth of the same view: the achromatic, still-image pathway of
ColorVideoVDP (Mantiuk et al., SIGGRAPH 2024, arXiv 2401.11485; github.com/gfxdisp/ColorVideoVDP, MIT), rewritten for
PyTorch 1.12 / Python 3.7 (the package itself needs Python 3.9 and PyTorch 1.13).

Why this metric: render and truth share shape, light and view, and for such pairs a visible-difference predictor of
this family (HDR-VDP-3) predicted how discriminable two glosses are (Cheeseman et al., see literature.md). L1 and SSIM
hardly punish the blurring of small sharp highlights; here it appears as a difference of contrast in the high bands,
weighted by the eye's sensitivity at the viewing distance.

Pathway (equation numbers of the paper; parameters from its cvvdp_parameters.json, sensitivity from its table
csf_lut_weber_fixed_size.json, both read from --data, default D:/assets/cvvdp):

    display        display values -> luminance in cd/m2 (sRGB curve, peak 200, contrast 1000, ambient 250 lx reflected
                   with k = 0.005), Eq. 2-3
    bands          Laplacian pyramid (kernel 0.05 0.25 0.4 0.25 0.05), band b at 0.3228 2^-(b-1) ppd/2 cpd (the first at
                   ppd/2), down to 0.2 cpd, plus the base band
    contrast       (G_b - expand(G_b+1)) / expand(G_b+1), each image over its own background, Eq. 6
    sensitivity    castleCSF at the band's frequency and the truth's background luminance, Eq. 7-8
    masking        mutual masking min(|T'|, |R'|) blurred (sigma 3 px), D = |T' - R'|^p / (1 + M), soft clamp, Eq. 9-13
    pooling        root mean square over space per band, 4-norm over bands, still-image constant, JOD, Eq. 14-17

Differences from the package:
    - achromatic channel only (the stimuli are grey: the chromatic contrasts are about zero)
    - `mask` (the object's coverage): the spatial mean is taken over the object instead of the whole image, so the black
      background does not dilute the highlights
    - replicate padding everywhere (the package's symmetric padding makes border artefacts on bright-on-dark images,
      its issue 29)
    - bands: those above 0.2 cpd and one base band (the package's count differs by one at some ppd, its issue 34)
So the JOD here is close to, but not, the package's JOD.

Viewing: 24 inch full HD at 0.6 m = 37.8 pixels per degree (the package's standard_fhd).

    python experiments/gloss_motion/vdp_loss.py bunny__coat_r0.05__sweep60__tex "label=render dir" ...   # CPU
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LUM = (0.2126729, 0.7151522, 0.0721750)
K5 = (0.05, 0.25, 0.4, 0.25, 0.05)


def srgb_to_linear(v):
    return torch.where(v <= 0.04045, v / 12.92, ((v.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


class VDPLoss(torch.nn.Module):
    def __init__(self, data="D:/assets/cvvdp", ppd=37.8, y_peak=200.0, contrast=1000.0, e_ambient=250.0, k_refl=0.005,
                 size=256):
        super().__init__()
        data = Path(data)
        p = json.loads((data / "cvvdp_parameters.json").read_text())
        lut = json.loads((data / "csf_lut_weber_fixed_size.json").read_text())
        self.y_black = y_peak / contrast + e_ambient / math.pi * k_refl
        self.y_range = y_peak - y_peak / contrast
        self.p = p["mask_p"]
        self.q = p["mask_q"][0]
        self.mask_gain = 10.0 ** p["mask_c"]
        self.xcm = 2.0 ** p["xcm_weights"][0]
        self.d_max = 10.0 ** p["d_max"]
        self.base_w = p["baseband_weight"][0]
        self.image_int = p["image_int"]
        self.jod_a, self.jod_exp = p["jod_a"], p["jod_exp"]
        self.beta_sch = float(p["beta_sch"])
        sens = 10.0 ** (p["sensitivity_correction"] / 20.0)

        # band frequencies (cpd): Laplacian bands above 0.2 cpd, then the base band (its sensitivity is taken at 0.1)
        freqs = [ppd / 2.0]
        while True:
            f = 0.3228 * 2.0 ** -(len(freqs) - 1) * ppd / 2.0
            if f <= 0.2 or size // 2 ** len(freqs) < 4:
                break
            freqs.append(f)
        self.freqs = freqs
        # sensitivity per band as a function of log10 background luminance (the table is log10 S [luminance][frequency])
        tab = np.asarray(lut["o0_c1"], np.float64)
        log_l, log_r = np.log10(lut["L_bkg"]), np.log10(lut["rho"])
        curves = [[np.interp(np.log10(f), log_r, row) for row in tab] for f in freqs + [0.1]]
        self.register_buffer("log_s", torch.tensor(np.asarray(curves) + np.log10(sens), dtype=torch.float32))  # bands+1, 32
        self.l0, self.dl = float(log_l[0]), float(log_l[1] - log_l[0])
        k = torch.tensor(K5, dtype=torch.float32)
        self.register_buffer("k2", (k[:, None] * k[None, :])[None, None])
        g = torch.exp(-0.5 * (torch.arange(-6, 7, dtype=torch.float32) / 3.0) ** 2)
        g = g / g.sum()
        self.register_buffer("g13", g)

    # ---- pyramid
    def reduce(self, x):
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), self.k2, stride=2)

    def expand(self, x, hw):
        h, w = x.shape[-2:]
        y = F.conv_transpose2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 4.0 * self.k2, stride=2)
        return y[..., 4:4 + hw[0], 4:4 + hw[1]]

    def blur(self, x):
        x = F.conv2d(F.pad(x, (6, 6, 0, 0), mode="replicate"), self.g13[None, None, None, :])
        return F.conv2d(F.pad(x, (0, 0, 6, 6), mode="replicate"), self.g13[None, None, :, None])

    def luminance(self, v):
        v = v.clamp(0.0, 1.0)
        y = srgb_to_linear(v)
        if y.shape[1] == 3:
            y = LUM[0] * y[:, 0:1] + LUM[1] * y[:, 1:2] + LUM[2] * y[:, 2:3]
        return self.y_range * y + self.y_black

    def sensitivity(self, band, l_bkg):
        x = ((torch.log10(l_bkg.clamp(min=1e-3)) - self.l0) / self.dl).clamp(0.0, self.log_s.shape[1] - 1.001)
        i = x.floor().long()
        t = x - i
        c = self.log_s[band]
        return 10.0 ** (c[i] * (1 - t) + c[i + 1] * t)

    def bands(self, test, ref, mask=None):
        """per band: the clamped visible difference D (B,1,h,w) and the weights of its pixels (the object's coverage)"""
        gt, gr = [self.luminance(test)], [self.luminance(ref)]
        gm = [torch.ones_like(gr[0]) if mask is None else mask.to(gr[0].dtype)]
        for _ in range(len(self.freqs)):
            gt.append(self.reduce(gt[-1]))
            gr.append(self.reduce(gr[-1]))
            gm.append(self.reduce(gm[-1]))
        out = []
        for b in range(len(self.freqs)):
            hw = gt[b].shape[-2:]
            et, er = self.expand(gt[b + 1], hw), self.expand(gr[b + 1], hw)
            ct = (gt[b] - et) / et.clamp(min=0.01)
            cr = (gr[b] - er) / er.clamp(min=0.01)
            gain = 1.0 if b == 0 else 2.0
            s = self.sensitivity(b, er.detach())
            tp, rp = ct.clamp(-1000, 1000) * gain * s, cr.clamp(-1000, 1000) * gain * s
            m = torch.min(tp.abs(), rp.abs())
            if min(hw) > 6:
                m = self.blur(m)
            m = self.xcm * (self.mask_gain * m) ** self.q
            d = ((tp - rp).abs() + 1e-5) ** self.p - 1e-5 ** self.p
            d = d / (1.0 + m)
            out.append((self.d_max * d / (self.d_max + d), gm[b]))
        b = len(self.freqs)                                   # base band: no masking
        mt = gr[b].clamp(min=0.01).mean(dim=(-2, -1), keepdim=True)
        s = self.sensitivity(b, gr[b].detach())
        d = ((gt[b] - gr[b]) / mt).abs() * s
        out.append((self.d_max * d / (self.d_max + d), gm[b]))
        return out

    def forward(self, test, ref, mask=None, return_bands=False):
        """test, ref: display values (B,1|3,H,W) in [0,1]; mask: the object's coverage (B,1,H,W). Returns 10 - JOD per
        image pair averaged over the batch (0: no visible difference)"""
        qs = []
        for i, (d, w) in enumerate(self.bands(test, ref, mask)):
            q = torch.sqrt((w * d * d).sum(dim=(1, 2, 3)) / w.sum(dim=(1, 2, 3)).clamp(min=1e-6) + 1e-12)
            qs.append(q * (self.base_w if i == len(self.freqs) else 1.0))
        qb = torch.stack(qs, dim=1)                           # B, bands
        q = (qb ** self.beta_sch).sum(dim=1) ** (1.0 / self.beta_sch) * self.image_int
        loss = (self.jod_a * (q + 1e-8) ** self.jod_exp).mean()
        return (loss, qb) if return_bands else loss


def main():
    import argparse
    import sys

    import imageio.v2 as imageio
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("renders", nargs="*", help="label=dir of 00000.png ... in the order of the test views")
    ap.add_argument("--png-root", default="data/gloss_motion_png")
    ap.add_argument("--blur", nargs="*", type=float, default=[0.5, 1.0, 2.0], help="check: the truth blurred (sigma, px)")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = root / a.png_root / a.name
    frames = json.loads((d / "transforms_test.json").read_text())["frames"]

    def load_rgba(p):
        x = torch.from_numpy(np.asarray(imageio.imread(p), np.float32) / 255.0)
        return x.permute(2, 0, 1)[None]

    truth, mask, seed2 = [], [], []
    for f in frames:
        x = load_rgba(d / (f["file_path"] + ".png"))
        mask.append(x[:, 3:4])
        truth.append(x[:, :3] * x[:, 3:4])
        p2 = d / (f["file_path"].replace("test/", "test_seed2/") + ".png")
        if p2.exists():
            y = load_rgba(p2)
            seed2.append(y[:, :3] * y[:, 3:4])
    truth, mask = torch.cat(truth), torch.cat(mask)
    vdp = VDPLoss(size=truth.shape[-1])
    print(f"{a.name}: {len(truth)} test views, bands at " + " ".join(f"{f:.2f}" for f in vdp.freqs) + " cpd + base")
    rows = {}

    def score(label, img):
        with torch.no_grad():
            per = [vdp(img[i:i + 1], truth[i:i + 1], mask[i:i + 1], return_bands=True) for i in range(len(truth))]
        loss = float(np.mean([float(p[0]) for p in per]))
        qb = torch.cat([p[1] for p in per]).mean(0).numpy()
        l1 = float(((img - truth).abs() * mask).sum() / (mask.sum() * 3))
        rows[label] = {"jod": 10 - loss, "l1": l1, "bands": qb.tolist()}
        print(f"{label:28s} JOD {10 - loss:6.3f}  L1 {l1:.4f}  per band " + " ".join(f"{v:6.2f}" for v in qb))

    if seed2:
        score("truth, other noise seed", torch.cat(seed2))
    for s in a.blur:
        r = int(3 * s + 0.5)
        g = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / s) ** 2)
        g = g / g.sum()
        x = truth.reshape(-1, 1, *truth.shape[-2:])
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g[None, None, None, :])
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g[None, None, :, None])
        score(f"truth blurred, sigma {s:g} px", x.reshape(truth.shape))
    for spec in a.renders:
        label, path = spec.split("=", 1)
        imgs = torch.cat([load_rgba(Path(path) / f"{i:05d}.png")[:, :3] for i in range(len(truth))])
        score(label, imgs)
    if a.out:
        Path(a.out).write_text(json.dumps({"name": a.name, "freqs_cpd": vdp.freqs, "rows": rows}, indent=1))
        print("->", a.out)


if __name__ == "__main__":
    main()
