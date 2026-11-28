# -*- coding: utf-8 -*-
"""
Where does a 4DGS-DR model put the highlights? Its image is

    image = (1 - r) * base + r * reflection

with base the Gaussians' own colour (spherical harmonics of degree 3: it can change with the view direction),
reflection the environment map looked up along the reflected ray, and r the reflection strength. The base is split
further by rendering it again with the higher SH coefficients set to zero:

    reflection part        r * reflection                      slides over the surface as the object turns
    view-dependent colour  (1 - r) * (base - base_dc)          a colour per Gaussian and view direction: it can memorise
                                                               the highlights of the training views
    constant colour        (1 - r) * base_dc

For the test views and for as many training views, the script prints the mean of each part inside the truth's
highlights (the brightest 5% of the object in each view) and on the rest of the object, and the visible difference
(vdp_loss.py) of the whole render, so that training and test views are compared under the same conditions.

Runs inside the 4DGS-DR copy (GPU, under a minute):

    D:\\3dgsdr_work\\run_deform.bat D:\\project\\3dgs-toy\\experiments\\gloss_motion\\render_parts.py ^
        --model_path D:\\3dgsdr_work\\output\\gm_bunny_r0.05_tex_dr2 --iteration 67000 ^
        --out D:\\project\\3dgs-toy\\experiments\\gloss_motion\\results\\parts_bunny_tex_dr2
"""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render  # noqa: E402
from scene import Scene  # noqa: E402
from scene.deform_model import DeformModel  # noqa: E402
from vdp_loss import VDPLoss  # noqa: E402

PARTS = ["truth", "render", "reflection", "view_dep_colour", "constant_colour"]


@torch.no_grad()
def parts_of(view, gaussians, pipe, bg, deform):
    full = render(view, gaussians, pipe, bg, deform=deform)
    rest = gaussians._features_rest.data.clone()
    gaussians._features_rest.data.zero_()
    dc = render(view, gaussians, pipe, bg, deform=deform)
    gaussians._features_rest.data.copy_(rest)
    r = full["refl_strength_map"]
    return {"truth": view.original_image[:3].cuda(),
            "render": full["render"],
            "reflection": r * full["refl_color_map"],
            "view_dep_colour": (1 - r) * (full["base_color_map"] - dc["base_color_map"]),
            "constant_colour": (1 - r) * dc["base_color_map"]}


def main():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out", required=True, help="prefix of the outputs (.json, _frames.png)")
    parser.add_argument("--frames", nargs="*", type=int, default=[0, 10, 18, 29])
    args = get_combined_args(parser)
    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.zeros(3, device="cuda")
    deform = DeformModel(dataset.is_blender, False)
    deform.use_scale = not getattr(dataset, "no_deform_scale", False)
    deform.load_weights(dataset.model_path, scene.loaded_iter)
    vdp = VDPLoss().cuda()

    test = scene.getTestCameras()
    train = scene.getTrainCameras()
    train = [train[i] for i in np.linspace(0, len(train) - 1, len(test)).round().astype(int)]
    res, shown = {}, []
    for name, views in (("test", test), ("train", train)):
        hl = {k: [] for k in PARTS}
        body = {k: [] for k in PARTS}
        jod, l1 = [], []
        for i, view in enumerate(views):
            p = parts_of(view, gaussians, pipe, bg, deform)
            m = view.gt_alpha_mask.cuda()
            obj = m[0] > 0.99
            y = p["truth"].mean(0)
            thr = torch.quantile(y[obj], 0.95)
            h = obj & (y >= thr)
            b = obj & (y < thr)
            for k in PARTS:
                v = p[k].mean(0)
                hl[k].append(v[h].mean().item())
                body[k].append(v[b].mean().item())
            jod.append(10 - vdp(p["render"][None], p["truth"][None], m[None]).item())
            l1.append(((p["render"] - p["truth"]).abs() * m).sum().item() / (3 * m.sum().item()))
            if name == "test" and i in args.frames:
                shown.append(p)
        res[name] = {"views": len(views), "jod": float(np.mean(jod)), "l1": float(np.mean(l1)),
                     "highlights": {k: float(np.mean(v)) for k, v in hl.items()},
                     "rest_of_object": {k: float(np.mean(v)) for k, v in body.items()}}
        print("\n{} views ({}): JOD {:.3f}  L1 {:.4f}".format(name, len(views), res[name]["jod"], res[name]["l1"]))
        print("  mean display value        " + "".join("{:>18s}".format(k) for k in PARTS))
        for label, d in (("truth's highlights (5%)", res[name]["highlights"]), ("rest of the object", res[name]["rest_of_object"])):
            print("  {:26s}".format(label) + "".join("{:18.4f}".format(d[k]) for k in PARTS))

    # rows: truth, render, reflection, view-dependent colour (x4, its negative part is clipped), constant colour
    rows = []
    for k in PARTS:
        gain = 4.0 if k == "view_dep_colour" else 1.0
        rows += [(gain * p[k]).clamp(0, 1) for p in shown]
    torchvision.utils.save_image(torch.stack(rows), args.out + "_frames.png", nrow=len(shown), padding=2)
    with open(args.out + ".json", "w") as f:
        json.dump({"model": dataset.model_path, "iteration": scene.loaded_iter, "rows_of_frames_png": PARTS, **res}, f, indent=1)
    print("->", args.out + ".json", args.out + "_frames.png")


if __name__ == "__main__":
    main()
