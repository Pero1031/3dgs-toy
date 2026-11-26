# -*- coding: utf-8 -*-
"""
CLIP glossiness readout of 3DGS renders and of the truth on the test views, prepared exactly as for the loss
(gloss_loss.to_input: display values, grey background, crop to 80%). Runs in image-to-3d-diffusion's environment:

    set HF_HOME=D:\\project\\image-to-3d-diffusion\\data\\hf
    python experiments/gloss_percept/score_clip.py logs/gp_blob1_r0.05/ckpt_007000.pt ...
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from gloss_loss import GlossReadout, crop_box, to_input  # noqa: E402
from gs.color import ColorMode, data_color_space  # noqa: E402
from gs.dataset import load_cameras, read_pfm  # noqa: E402
from gs.gaussians import GaussianModel  # noqa: E402
from gs.render import render  # noqa: E402


@torch.no_grad()
def main():
    dev = "cuda"
    gr = GlossReadout(dev)
    truth_done = set()
    for c in sys.argv[1:]:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        a = ck["args"]
        d = Path(a["datadir"])
        cm = ColorMode(a["color_mode"], data_color_space(a["datadir"]))
        cams = load_cameras(a["datadir"], "test", False, ck["bg"].tolist(), dev, cm.image_fn)
        names = [Path(f["file_path"]).name for f in json.loads((d / "transforms_test.json").read_text(encoding="utf-8"))["frames"]]
        covs = [torch.from_numpy(np.clip(read_pfm(str(d / "raw" / "test" / n.replace(".pfm", "_mask.pfm")))[..., 0], 0, 1)).to(dev) for n in names]
        model = GaussianModel.from_state_dict(ck["model"])
        bg = ck["bg"].to(dev)
        r, t = [], []
        for cam, cov in zip(cams, covs):
            box = crop_box(cov)
            img = render(model, cam, bg, ck["sh_degree"], clamp_color=cm.clamp)["image"]
            r.append(gr(to_input(cm.to_linear_srgb(img), cov, box)).item())
            t.append(gr(to_input(cm.to_linear_srgb(cam.image), cov, box)).item())
        if str(d) not in truth_done:
            print(f"{d.name}: truth {np.mean(t):.2f} (sd {np.std(t):.2f})")
            truth_done.add(str(d))
        print(f"  {Path(c).parent.name:28} 3DGS {np.mean(r):.2f} (sd {np.std(r):.2f}), diff {np.mean(r) - np.mean(t):+.2f}")


if __name__ == "__main__":
    main()
