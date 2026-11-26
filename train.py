# -*- coding: utf-8 -*-
"""
Training of 3D Gaussian Splatting.

    python train.py --datadir data/toy_sphere --expname toy --iters 7000

Outputs val images (left: ground truth / right: render), log.csv, checkpoints and
a point_cloud.ply in the official format to logs/<expname>/.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import torch

from gs.color import MODES, ColorMode, data_color_space
from gs.dataset import load_cameras
from gs.gaussians import GaussianModel
from gs.loss import psnr, ssim
from gs.render import render


def scene_extent(cams):
    """Spread of the camera centers (same as getNerfppNorm of the official implementation).
    It is the reference for learning rates and size thresholds."""
    c = torch.stack([cam.center for cam in cams])
    return 1.1 * (c - c.mean(0)).norm(dim=1).max().item()


def to_u8(img):
    return (img.clamp(0, 1) * 255).round().byte().cpu().numpy()


@torch.no_grad()
def evaluate(model, cams, bg, sh_degree, args, cm, save_path=None):
    """PSNR on cm.view (display values, or the same encoding of linear colour; gs/color.py)"""
    vals, times = [], []
    for k, cam in enumerate(cams):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = render(model, cam, bg, sh_degree, tile=args.tile, max_entries=args.max_entries, clamp_color=cm.clamp)["image"]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        vals.append(psnr(cm.view(img), cm.view(cam.image)).item())
        if k == 0 and save_path is not None:
            imageio.imwrite(save_path, to_u8(cm.view(torch.cat([cam.image, img], 1))))
    return sum(vals) / len(vals), sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--expname", default="exp")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--half-res", action="store_true")
    ap.add_argument("--black-bkgd", action="store_true")
    ap.add_argument("--bg", type=float, nargs=3, default=None, metavar=("R", "G", "B"),
                    help="background color [0,1]. Default is white (black with --black-bkgd)")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--color-mode", choices=MODES, default="display",
                    help="what the colours of the Gaussians mean (gs/color.py). --bg is given in this space")
    ap.add_argument("--loss-knee", type=float, default=1.0,
                    help="linear modes: the loss encoding compresses x as x / (1 + x / k); larger k = highlights weigh "
                         "more, inf = sRGB curve alone (gs/color.py). PSNR always uses k = 1")
    # Initialization (for synthetic data: random points inside a cube)
    ap.add_argument("--init-points", type=int, default=20000)
    ap.add_argument("--init-extent", type=float, default=2.0, help="half side length of the cube in which the initial points are placed")
    ap.add_argument("--init-center", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    # densification (defaults are the same as the official implementation.
    # Only densify_until defaults to iters/2)
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--densify-until", type=int, default=None)
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--opacity-reset", type=int, default=3000)
    ap.add_argument("--grad-thresh", type=float, default=0.0002)
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--max-screen-size", type=float, default=20, help="prune those that appear larger than this [px]. 0 disables")
    ap.add_argument("--lambda-ssim", type=float, default=0.2)
    # Perceptual loss through CLIP (experiments/gloss_percept/gloss_loss.py; needs transformers and CLIP's weights)
    ap.add_argument("--percept", choices=("none", "gloss", "embed"), default="none",
                    help="gloss: match the CLIP glossiness readout of the render to the training image's; embed: match "
                         "the whole CLIP embedding (cosine)")
    ap.add_argument("--percept-ratio", type=float, default=1.0,
                    help="weight of the perceptual loss, set at --percept-start so that its gradient on the means is "
                         "this many times the gradient of the image loss (median over --percept-calib views)")
    ap.add_argument("--percept-start", type=int, default=2000)
    ap.add_argument("--percept-every", type=int, default=2, help="add the perceptual loss every this many iterations")
    ap.add_argument("--percept-calib", type=int, default=8)
    # Rasterizer
    ap.add_argument("--tile", type=int, default=8, help="side of a tile [px] (the official one uses 16. 8 is faster for the PyTorch version)")
    ap.add_argument("--max-entries", type=int, default=2 ** 25, help="lower this if VRAM runs out")
    # Logging
    ap.add_argument("--i-print", type=int, default=100)
    ap.add_argument("--i-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    bg_rgb = args.bg if args.bg else ([0.0] * 3 if args.black_bkgd else [1.0] * 3)
    cm = ColorMode(args.color_mode, data_color_space(args.datadir), args.loss_knee)
    train_cams = load_cameras(args.datadir, "train", args.half_res, bg_rgb, dev, cm.image_fn)
    test_cams = load_cameras(args.datadir, "test", args.half_res, bg_rgb, dev, cm.image_fn)
    val_cams = load_cameras(args.datadir, "val", args.half_res, bg_rgb, dev, cm.image_fn) or test_cams or train_cams[:8]
    bg = torch.tensor(bg_rgb, device=dev)
    extent = scene_extent(train_cams)
    densify_until = args.iters // 2 if args.densify_until is None else args.densify_until
    H, W = train_cams[0].H, train_cams[0].W
    print(f"train {len(train_cams)} images ({W}x{H}), val {len(val_cams)} images, extent={extent:.2f}")

    model = GaussianModel(args.sh_degree)
    model.init_random(args.init_points, args.init_extent, dev, center=args.init_center)
    model.setup_optimizer(extent)

    logdir = Path("logs") / args.expname
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_f = open(logdir / "log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "train_psnr", "n_gauss", "val_psnr", "sec"])

    percept, lam, p_log = None, 0.0, []
    if args.percept != "none":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments" / "gloss_percept"))
        from gloss_loss import PerceptLoss
        percept = PerceptLoss(args.percept, args.datadir, train_cams, cm, dev)

    def image_loss(out, cam):
        img, gt = cm.loss_view(out["image"]), cm.loss_view(cam.image)
        return (1 - args.lambda_ssim) * (img - gt).abs().mean() + args.lambda_ssim * (1 - ssim(img, gt))

    sh_degree, order, t_start = 0, [], time.time()
    for it in range(1, args.iters + 1):
        model.update_lr(it, args.iters)
        if it % 1000 == 0:  # raise the SH degree every 1000 iters (fit the diffuse color first, then view dependence)
            sh_degree = min(sh_degree + 1, args.sh_degree)
        if not order:
            order = torch.randperm(len(train_cams)).tolist()
        idx = order.pop()
        cam = train_cams[idx]

        if percept is not None and it >= args.percept_start and lam == 0.0 and (it - args.percept_start) % 100 == 0:
            # weight: gradient ratio on the means. Views where the perceptual loss passes (almost) no gradient (a
            # render clipped to white on the display, early in training) are left out; with fewer than half of the
            # views usable, the calibration is tried again 100 iterations later
            ratios = []
            for k in torch.randperm(len(train_cams))[:args.percept_calib].tolist():
                norms = []
                for f in (lambda o: image_loss(o, train_cams[k]), lambda o: percept(k, o["image"])):
                    o = render(model, train_cams[k], bg, sh_degree, tile=args.tile, max_entries=args.max_entries,
                               clamp_color=cm.clamp)
                    f(o).backward()
                    norms.append(model.params["means"].grad.norm().item())
                    model.optimizer.zero_grad(set_to_none=True)
                if norms[1] > 1e-9 * norms[0]:
                    ratios.append(norms[0] / norms[1])
            if 2 * len(ratios) >= args.percept_calib:
                lam = args.percept_ratio * float(torch.tensor(ratios).median())
                print(f"[{it:6d}] percept loss '{args.percept}' starts: weight {lam:.3e} "
                      f"(gradient ratio x {args.percept_ratio}, {len(ratios)} of {args.percept_calib} views)")
            else:
                print(f"[{it:6d}] percept loss: calibration postponed ({len(ratios)} usable views)")

        out = render(model, cam, bg, sh_degree, tile=args.tile, max_entries=args.max_entries, clamp_color=cm.clamp)
        loss = image_loss(out, cam)
        if percept is not None and lam > 0 and it % args.percept_every == 0:
            p = percept(idx, out["image"])
            loss = loss + lam * p
            p_log.append(p.item())
        loss.backward()

        with torch.no_grad():
            if it < densify_until and out["means2d"].grad is not None:
                vis = out["on_screen"]
                model.add_densification_stats(out["means2d"].grad[vis], out["ids"][vis],
                                              out["radius"][vis], cam.W, cam.H)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)

            if it < densify_until:
                if it > args.densify_from and it % args.densify_interval == 0:
                    big = args.max_screen_size if (args.max_screen_size and it > args.opacity_reset) else None
                    model.densify_and_prune(args.grad_thresh, args.min_opacity, extent, big)
                if it % args.opacity_reset == 0:
                    model.reset_opacity()

            if it % args.i_print == 0:
                train_psnr = psnr(cm.view(out["image"]), cm.view(cam.image)).item()
                val_psnr = ""
                if it % args.i_val == 0 or it == args.iters:
                    val_psnr, ms = evaluate(model, val_cams, bg, sh_degree, args, cm, logdir / f"val_{it:06d}.png")
                    val_psnr = f"{val_psnr:.2f}"
                el = time.time() - t_start
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                ptxt = f"  percept {sum(p_log) / len(p_log):.4f}" if p_log else ""
                p_log = []
                print(f"[{it:6d}] loss {loss.item():.4f}  psnr {train_psnr:5.2f}  N {len(model):7d}  "
                      f"val {val_psnr or '-':>5}  {el / it * 1000:5.0f} ms/it  peak {mem:.1f} GB{ptxt}")
                log.writerow([it, f"{loss.item():.5f}", f"{train_psnr:.3f}", len(model), val_psnr, f"{el:.1f}"])
                log_f.flush()

    log_f.close()
    torch.save({"model": model.state_dict(), "sh_degree": sh_degree, "bg": bg.cpu(),
                "camera": {"W": W, "H": H, "focal": train_cams[0].fx}, "args": vars(args), "data_space": cm.data_space},
               logdir / f"ckpt_{args.iters:06d}.pt")
    model.save_ply(logdir / "point_cloud.ply")
    print(f"training time {(time.time() - t_start) / 60:.1f} min, Gaussians {len(model)}")
    if test_cams:
        p, sec = evaluate(model, test_cams, bg, sh_degree, args, cm, logdir / "test_000.png")
        print(f"test PSNR {p:.2f} dB, render {sec * 1000:.1f} ms/image")


if __name__ == "__main__":
    main()
