# -*- coding: utf-8 -*-
"""
A perceptual gloss loss for 3DGS: the CLIP readout of glossiness (clip_readout.py) of the render, matched to the
readout of the training image of the same view.

The render goes through the same steps as the readout's inputs (export_display.py --background grey --crop 0.8):
linear luminance x EXPOSURE, clipped at 1, sRGB curve, composited onto grey 0.5 with the truth's coverage of each pixel,
a square crop in which the object spans 80% of the side, then CLIP's preprocessing (bicubic to 224, normalisation).
Everything is differentiable (the clip at 1 passes no gradient above the display's white).

Runs in the environment of image-to-3d-diffusion (it holds CLIP's weights and transformers), with
HF_HOME=D:\\project\\image-to-3d-diffusion\\data\\hf:

    python experiments/gloss_percept/gloss_loss.py export   # CPU: refit the readout on the cached embeddings, save it
    python experiments/gloss_percept/gloss_loss.py trial    # GPU, minutes: memory and time of CLIP's forward and
                                                            # backward next to a 3DGS render
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
READOUT = ROOT / "data" / "gloss_percept" / "clip" / "readout_glossiness.npz"
EXPOSURE = 3.0
BG = 0.5
CROP = 0.8
LUM = (0.2126729, 0.7151522, 0.0721750)


def export():
    import clip_readout as cr
    res = json.loads(cr.OUT.read_text(encoding="utf-8"))
    paths, y, groups, geoms = cr.serrano()
    z = np.load(cr.CACHE / f"serrano_sub{cr.PER_CELL}_s{cr.SEED}.npz", allow_pickle=True)
    assert list(z["paths"]) == [str(cr.SERRANO / "images" / p) for p in paths]
    m = cr.ridge(res["alpha"]).fit(z["emb"], y)
    sc, rg = m.named_steps["standardscaler"], m.named_steps["ridge"]
    np.savez(READOUT, mean=sc.mean_, scale=sc.scale_, coef=rg.coef_, intercept=rg.intercept_, alpha=res["alpha"])
    print(f"readout (alpha {res['alpha']:g}, {len(y)} images) -> {READOUT}")


class GlossReadout(torch.nn.Module):
    """glossiness (Serrano's 1-7 scale) of a batch of display images (B,3,H,W) in [0,1]"""

    def __init__(self, device="cuda"):
        super().__init__()
        from huggingface_hub import snapshot_download
        from transformers import CLIPVisionModelWithProjection
        repo = snapshot_download("bennyguo/zero123-xl-diffusers", local_files_only=True)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(repo, subfolder="image_encoder", variant="fp16",
                                                                  dtype=torch.float32).to(device).eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        z = np.load(READOUT)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.mean, self.scale, self.coef, self.b = t(z["mean"]), t(z["scale"]), t(z["coef"]), float(z["intercept"])
        self.nmean = t([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        self.nstd = t([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def embed(self, x):
        c = F.interpolate(x, size=224, mode="bicubic", antialias=True).clamp(0, 1)
        return self.clip((c - self.nmean) / self.nstd).image_embeds

    def forward(self, x):
        return ((self.embed(x) - self.mean) / self.scale) @ self.coef + self.b


def crop_box(cov):
    """(y0, x0, side) of the square crop in which the object (coverage > 0.5) spans CROP of the side"""
    ys, xs = torch.nonzero(cov > 0.5, as_tuple=True)
    cy, cx = (ys.min() + ys.max()).item() / 2, (xs.min() + xs.max()).item() / 2
    side = int(round(max((ys.max() - ys.min()).item(), (xs.max() - xs.min()).item()) / CROP))
    return int(round(cy - side / 2)), int(round(cx - side / 2)), side


def to_input(lin, cov, box):
    """lin: (H,W,3) linear radiance; cov: (H,W) the truth's coverage; box from crop_box -> (1,3,side,side) in [0,1]"""
    Y = (lin * torch.tensor(LUM, device=lin.device)).sum(-1)
    Y = (Y * EXPOSURE).clamp(0, 1)
    D = torch.where(Y <= 0.0031308, 12.92 * Y, 1.055 * Y.clamp_min(0.0031308) ** (1 / 2.4) - 0.055)
    D = cov * D + (1 - cov) * BG
    y0, x0, side = box
    H, W = D.shape
    pad = max(0, -y0, -x0, y0 + side - H, x0 + side - W)
    D = F.pad(D[None, None], (pad, pad, pad, pad), value=BG)[0, 0]
    D = D[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return D[None, None].expand(1, 3, side, side)


class PerceptLoss:
    """A perceptual loss on the training views, for train.py --percept gloss|embed.
    gloss: (readout(render) - readout(truth))^2, the glossiness readout only
    embed: 1 - cos(CLIP embedding of the render, of the truth), the whole embedding (a control: is it the gloss
           direction, or CLIP's features in general?)
    The truth's readout / embedding of every training view is computed once. Needs raw/train/*_mask.pfm (the truth's
    coverage, for the background and the crop) next to the data, as make_stimuli.py views writes it."""

    def __init__(self, kind, datadir, cams, cm, device="cuda"):
        from gs.dataset import read_pfm
        self.kind, self.cm = kind, cm
        self.readout = GlossReadout(device)
        meta = json.loads((Path(datadir) / "transforms_train.json").read_text(encoding="utf-8"))
        names = [Path(f["file_path"]).name for f in meta["frames"]]
        assert len(names) == len(cams)
        self.cov, self.box, self.target = [], [], []
        with torch.no_grad():
            for n, cam in zip(names, cams):
                cov = np.clip(read_pfm(str(Path(datadir) / "raw" / "train" / n.replace(".pfm", "_mask.pfm")))[..., 0], 0, 1)
                cov = torch.from_numpy(cov).to(device)
                box = crop_box(cov)
                x = to_input(cm.to_linear_srgb(cam.image), cov, box)
                t = self.readout(x) if kind == "gloss" else F.normalize(self.readout.embed(x), dim=-1)
                self.cov.append(cov)
                self.box.append(box)
                self.target.append(t)
        if kind == "gloss":
            v = torch.cat(self.target)
            print(f"percept loss 'gloss': truth readout of the {len(v)} training views {v.mean():.2f} "
                  f"(range {v.min():.2f}-{v.max():.2f})")

    def __call__(self, i, image):
        """i: index of the training view; image: the render (H,W,3) in the model's colour space"""
        x = to_input(self.cm.to_linear_srgb(image), self.cov[i], self.box[i])
        if self.kind == "gloss":
            return (self.readout(x) - self.target[i]).pow(2).mean()
        return 1 - (F.normalize(self.readout.embed(x), dim=-1) * self.target[i]).sum()

    def value(self, i, image):
        """the readout of the render (for logging)"""
        with torch.no_grad():
            return self.readout(to_input(self.cm.to_linear_srgb(image), self.cov[i], self.box[i])).item()


def trial(a):
    from gs.color import ColorMode, data_color_space
    from gs.dataset import load_cameras, read_pfm
    from gs.gaussians import GaussianModel
    from gs.render import render
    dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = ck["args"]
    cm = ColorMode(args["color_mode"], data_color_space(args["datadir"]))
    cams = load_cameras(args["datadir"], "train", False, ck["bg"].tolist(), dev, cm.image_fn)
    meta = json.loads((Path(args["datadir"]) / "transforms_train.json").read_text(encoding="utf-8"))
    names = [Path(f["file_path"]).name for f in meta["frames"]]
    covs = [torch.from_numpy(np.clip(read_pfm(str(Path(args["datadir"]) / "raw" / "train" / n.replace(".pfm", "_mask.pfm")))[..., 0], 0, 1)).to(dev)
            for n in names[:a.views]]
    model = GaussianModel.from_state_dict(ck["model"])
    bg = ck["bg"].to(dev)
    torch.cuda.reset_peak_memory_stats()
    gr = GlossReadout(dev)
    base_mem = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"{len(model)} Gaussians; CLIP + readout loaded: {base_mem:.2f} GB")
    t_render, t_gloss, vals = [], [], []
    for i in range(a.views):
        cam, cov = cams[i], covs[i]
        box = crop_box(cov)
        torch.cuda.synchronize()
        t0 = time.time()
        out = render(model, cam, bg, ck["sh_degree"], clamp_color=cm.clamp)
        img = cm.to_linear_srgb(out["image"])
        l1 = (cm.loss_view(out["image"]) - cm.loss_view(cam.image)).abs().mean()
        l1.backward(retain_graph=True)
        torch.cuda.synchronize()
        t1 = time.time()
        gn_l1 = {k: p.grad.norm().item() for k, p in model.params.items() if p.grad is not None}
        for p in model.params.values():
            p.grad = None
        with torch.no_grad():
            g_t = gr(to_input(cm.to_linear_srgb(cam.image), cov, box))
        g_r = gr(to_input(img, cov, box))
        loss = (g_r - g_t).pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.time()
        gn_g = {k: p.grad.norm().item() for k, p in model.params.items() if p.grad is not None}
        vals.append((g_t.item(), g_r.item(), {k: gn_g[k] / max(gn_l1.get(k, 0), 1e-12) for k in gn_g}))
        t_render.append(t1 - t0)
        t_gloss.append(t2 - t1)
        for p in model.params.values():
            p.grad = None
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"render + L1 backward: {1000 * np.median(t_render):.0f} ms; CLIP readout of truth and render, forward + "
          f"backward: {1000 * np.median(t_gloss):.0f} ms (medians over {a.views} views); peak memory {peak:.2f} GB")
    for i, (gt_, gr_, ratio) in enumerate(vals):
        print(f"  view {names[i]}: readout truth {gt_:.2f}, 3DGS {gr_:.2f}; gradient norm, gloss loss / L1 loss: "
              + ", ".join(f"{k} {v:.0f}" for k, v in ratio.items()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    t = sub.add_parser("trial")
    t.add_argument("--ckpt", default="logs/gp_blob1_r0.05/ckpt_007000.pt")
    t.add_argument("--views", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    {"export": lambda a: export(), "trial": trial}[a.cmd](a)


if __name__ == "__main__":
    main()
