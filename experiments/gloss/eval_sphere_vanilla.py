# -*- coding: utf-8 -*-
"""
Baseline for comparison: evaluate vanilla 3DGS (colour of SH degree 3. Trained with train.py) on the test views of
the bumpy sphere, by PSNR over the sphere region at 512, 256, 128 px, in the same way as train_deferred.py.

    python train.py --datadir data/gloss_sphere/s256 --expname sphere_vanilla --iters 7000 --bg 0.396 0.396 0.396 --init-extent 0.25
    python experiments/gloss/eval_sphere_vanilla.py logs/sphere_vanilla/ckpt_007000.pt

Output: logs/sphere_vanilla/eval.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import train_deferred as td  # noqa: E402
from gs.dataset import load_cameras  # noqa: E402
from gs.gaussians import GaussianModel  # noqa: E402
from gs.render import render  # noqa: E402


@torch.no_grad()
def main():
    ckpt_path = Path(sys.argv[1])
    ck = torch.load(ckpt_path, map_location="cuda")
    model = GaussianModel.from_state_dict(ck["model"])
    bg = torch.tensor([td.BG] * 3, device="cuda")
    res = {"n_gauss": len(model)}
    d256 = ROOT / "data" / "gloss_sphere" / "s256"
    tr = load_cameras(d256, "train", bg=(td.BG,) * 3)[::8]
    trm = td.load_masks(d256, "train", "cuda")[::8]
    res["train_psnr_sphere"] = float(np.mean([td.masked_psnr(render(model, c, bg, ck["sh_degree"])["image"], c.image, m)
                                              for c, m in zip(tr, trm)]))
    print(f"PSNR over the sphere region on the training views (12 images) {res['train_psnr_sphere']:.2f} dB")
    res["eval"] = {}
    for scale in (512, 256, 128):
        d = ROOT / "data" / "gloss_sphere" / f"s{scale}"
        cams, masks = load_cameras(d, "test", bg=(td.BG,) * 3), td.load_masks(d, "test", "cuda")
        ps = [td.masked_psnr(render(model, c, bg, ck["sh_degree"])["image"], c.image, m) for c, m in zip(cams, masks)]
        res["eval"][str(scale)] = {"psnr_sphere": float(np.mean(ps))}
        print(f"  eval {scale} px: PSNR over the sphere region {np.mean(ps):.2f} dB")
    (ckpt_path.parent / "eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
