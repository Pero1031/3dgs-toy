# -*- coding: utf-8 -*-
"""
Gloss test: render trained models on the training views and the test views, and output PSNR / SSIM / LPIPS and
comparison images.

To tell apart whether 3DGS breaks down on glossy surfaces because "the trained views were memorized but it does not
generalize to new views" or because "it cannot be represented in the first place", both the training views and the
test views are measured. Each metric is reported both as the value averaged over the statue region (mask) only and
as the full-frame value (the background is a uniform grey and easy to reproduce, so the full-frame values are
lenient).

    python experiments/gloss/evaluate.py logs/gloss_gold/ckpt_007000.pt logs/gloss_matte/ckpt_007000.pt \
        --masks data/gloss_matte

--masks: a training dataset that has the statue masks (*_mask.png). The masks are determined by geometry and cameras
         only, so they are common to all conditions.
Output: experiments/gloss/figures/{test_views,train_views,zoom}.png, experiments/gloss/results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from gs.dataset import load_cameras  # noqa: E402
from gs.gaussians import GaussianModel  # noqa: E402
from gs.loss import ssim_map  # noqa: E402
from gs.render import render  # noqa: E402

SHOW = {"test": (0, 3, 6), "train": (0, 40, 75)}  # views shown in the figures
ZOOM_VIEW, ZOOM_SIZE, ZOOM_SCALE = 0, 56, 4        # zoom figure: a 56 px square of test view 0, magnified 4×


def load_masks(root, split):
    frames = json.loads((Path(root) / f"transforms_{split}.json").read_text(encoding="utf-8"))["frames"]
    return [torch.from_numpy(imageio.imread(Path(root) / (f["file_path"].replace("_denoised", "") + "_mask.png"))
                             .astype(np.float32) / 255.0).cuda() for f in frames]


def metrics(img, gt, m, lp):
    """PSNR / SSIM / LPIPS over the statue region (weighted by the mask) and over the full frame"""
    se = ((img - gt) ** 2).mean(-1)
    s = ssim_map(img, gt).mean(1)[0]
    l = lp(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], normalize=True)[0, 0]  # per-pixel LPIPS
    wavg = lambda x: ((x * m).sum() / m.sum()).item()  # noqa: E731
    return {"psnr": -10 * math.log10(wavg(se)), "ssim": wavg(s), "lpips": wavg(l),
            "psnr_full": -10 * math.log10(se.mean().item()), "ssim_full": s.mean().item(), "lpips_full": l.mean().item()}


def crop_box(m, pad=6):
    ys, xs = np.nonzero(m > 0)
    return max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1


@torch.no_grad()
def evaluate(ckpt_path, masks, lp):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    model = GaussianModel.from_state_dict(ckpt["model"])
    bg = ckpt["bg"].cuda()
    res, panels, full = {}, {"train": [], "test": []}, {}
    for split in ("train", "test"):
        cams = load_cameras(a["datadir"], split, a["half_res"], ckpt["bg"].tolist())
        res[split] = []
        for k, (cam, m) in enumerate(zip(cams, masks[split])):
            img = render(model, cam, bg, ckpt["sh_degree"], tile=a["tile"])["image"].clamp(0, 1)
            res[split].append(metrics(img, cam.image, m, lp))
            if k in SHOW[split]:
                y0, y1, x0, x1 = crop_box(m.cpu().numpy())
                err = ((img - cam.image).abs().mean(-1, keepdim=True).expand(-1, -1, 3) * 4).clamp(0, 1)
                panels[split].append([t[y0:y1, x0:x1].cpu().numpy() for t in (cam.image, img, err)])
            if split == "test" and k == ZOOM_VIEW:
                full = {"gt": cam.image.cpu().numpy(), "img": img.cpu().numpy(), "mask": m.cpu().numpy()}
    return res, panels, full, len(model)


def sheet(rows, labels, path, note):
    """rows[r] = [[gt, render, err] × views]. Panels are padded to the same size and tiled, with a heading on each row"""
    H = max(p.shape[0] for row in rows for v in row for p in v)
    W = max(p.shape[1] for row in rows for v in row for p in v)

    def fit(p):
        c = np.ones((H, W, 3), np.float32)  # margins are white (to distinguish them from the black of zero error)
        y, x = (H - p.shape[0]) // 2, (W - p.shape[1]) // 2
        c[y:y + p.shape[0], x:x + p.shape[1]] = p
        return c

    gap = np.ones((H, 8, 3), np.float32)
    head = 18
    lines = []
    for row in rows:
        cells = []
        for v in row:
            cells += [fit(p) for p in v] + [gap]
        lines.append(np.pad(np.concatenate(cells[:-1], 1), ((head, 4), (0, 0), (0, 0)), constant_values=1))
    img = Image.fromarray((np.concatenate(lines, 0) * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(img)
    for r, lab in enumerate(labels):
        d.text((4, r * (H + head + 4) + 3), f"{lab}   ({note})", fill=(0, 0, 0))
    img.save(path)


def zoom_figure(fulls, names, path):
    """Pick the place where the error is most concentrated for gold, and show the ground truth and the render of
    each condition side by side, magnified at the same position"""
    g = fulls[0]
    err = torch.from_numpy(np.abs(g["img"] - g["gt"]).mean(-1) * g["mask"])[None, None]
    dens = F.avg_pool2d(err, ZOOM_SIZE, stride=4)[0, 0]
    iy, ix = np.unravel_index(int(dens.argmax()), dens.shape)
    y0, x0 = iy * 4, ix * 4
    up = lambda x: np.kron(x[y0:y0 + ZOOM_SIZE, x0:x0 + ZOOM_SIZE], np.ones((ZOOM_SCALE, ZOOM_SCALE, 1)))  # noqa: E731
    gap = np.ones((ZOOM_SIZE * ZOOM_SCALE, 8, 3))
    cells = []
    for f in fulls:
        cells += [up(f["gt"]), gap, up(f["img"]), gap, gap, gap]
    img = Image.fromarray((np.pad(np.concatenate(cells[:-3], 1), ((18, 0), (0, 0), (0, 0)), constant_values=1)
                           * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(img)
    w = ZOOM_SIZE * ZOOM_SCALE
    for i, n in enumerate(names):
        x = i * (2 * w + 40)
        d.text((x + 4, 3), f"{n}: truth", fill=(0, 0, 0))
        d.text((x + w + 12, 3), f"{n}: 3DGS", fill=(0, 0, 0))
    img.save(path)
    return [int(y0), int(x0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    masks = {s: load_masks(args.masks, s) for s in ("train", "test")}
    lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda()

    names, rows, fulls, summary = [], {"train": [], "test": []}, [], {}
    for c in args.ckpts:
        name = Path(c).parent.name.replace("gloss_", "")
        res, panels, full, n = evaluate(c, masks, lp)
        mean = {s: {k: float(np.mean([v[k] for v in res[s]])) for k in res[s][0]} for s in res}
        summary[name] = {"ckpt": Path(c).as_posix(), "n_gauss": n, "mean": mean, "per_view": res}
        names.append(name)
        fulls.append(full)
        for s in rows:
            rows[s].append(panels[s])

    for region, suf in (("statue region", ""), ("full frame", "_full")):
        print(f"\n[{region}]  {'cond':<6} {'views':<5} {'PSNR':>6} {'SSIM':>7} {'LPIPS':>7}")
        for n in names:
            for s in ("train", "test"):
                m = summary[n]["mean"][s]
                print(f"{'':10}{n:<8} {s:<5} {m['psnr' + suf]:6.2f} {m['ssim' + suf]:7.4f} {m['lpips' + suf]:7.4f}")

    note = "each view: truth | 3DGS | error x4"
    for s in ("test", "train"):
        labels = [f"{n} ({s} views): PSNR {summary[n]['mean'][s]['psnr']:.1f} dB, SSIM {summary[n]['mean'][s]['ssim']:.3f}, "
                  f"LPIPS {summary[n]['mean'][s]['lpips']:.3f}" for n in names]
        sheet(rows[s], labels, out / "figures" / f"{s}_views.png", note)
    summary["_zoom"] = {"test_view": ZOOM_VIEW, "top_left_yx": zoom_figure(fulls, names, out / "figures" / "zoom.png"),
                        "size": ZOOM_SIZE, "scale": ZOOM_SCALE}
    (out / "results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n-> {out / 'figures'}, {out / 'results.json'}")


if __name__ == "__main__":
    main()
