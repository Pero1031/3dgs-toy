# -*- coding: utf-8 -*-
"""
3D Gaussian Splatting の学習。

    python train.py --datadir data/toy_sphere --expname toy --iters 7000

logs/<expname>/ に val 画像（左: 正解 / 右: 描画）、log.csv、チェックポイント、
公式形式の point_cloud.ply を出力する。
"""

import argparse
import csv
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import torch

from gs.dataset import load_cameras
from gs.gaussians import GaussianModel
from gs.loss import psnr, ssim
from gs.render import render


def scene_extent(cams):
    """カメラ中心群の広がり（公式実装の getNerfppNorm と同じ）。学習率や大きさの閾値の基準になる。"""
    c = torch.stack([cam.center for cam in cams])
    return 1.1 * (c - c.mean(0)).norm(dim=1).max().item()


def to_u8(img):
    return (img.clamp(0, 1) * 255).round().byte().cpu().numpy()


@torch.no_grad()
def evaluate(model, cams, bg, sh_degree, args, save_path=None):
    vals, times = [], []
    for k, cam in enumerate(cams):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = render(model, cam, bg, sh_degree, tile=args.tile, max_entries=args.max_entries)["image"]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        vals.append(psnr(img, cam.image).item())
        if k == 0 and save_path is not None:
            imageio.imwrite(save_path, to_u8(torch.cat([cam.image, img], 1)))
    return sum(vals) / len(vals), sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--expname", default="exp")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--half-res", action="store_true")
    ap.add_argument("--black-bkgd", action="store_true")
    ap.add_argument("--sh-degree", type=int, default=3)
    # 初期化（合成データ用：立方体内のランダム点）
    ap.add_argument("--init-points", type=int, default=20000)
    ap.add_argument("--init-extent", type=float, default=2.0, help="初期点を置く立方体の半辺長")
    # densification（既定値は公式実装と同じ。densify_until だけ既定で iters/2 にしている）
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--densify-until", type=int, default=None)
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--opacity-reset", type=int, default=3000)
    ap.add_argument("--grad-thresh", type=float, default=0.0002)
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--max-screen-size", type=float, default=20, help="これより大きく映るものを prune [px]。0 で無効")
    ap.add_argument("--lambda-ssim", type=float, default=0.2)
    # ラスタライザ
    ap.add_argument("--tile", type=int, default=8, help="タイルの一辺 [px]（公式は 16。PyTorch 版は 8 の方が速い）")
    ap.add_argument("--max-entries", type=int, default=2 ** 25, help="VRAM 不足なら下げる")
    # ログ
    ap.add_argument("--i-print", type=int, default=100)
    ap.add_argument("--i-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    white = not args.black_bkgd
    train_cams = load_cameras(args.datadir, "train", args.half_res, white, dev)
    val_cams = load_cameras(args.datadir, "val", args.half_res, white, dev) or train_cams[:8]
    test_cams = load_cameras(args.datadir, "test", args.half_res, white, dev)
    bg = torch.tensor([1.0, 1.0, 1.0] if white else [0.0, 0.0, 0.0], device=dev)
    extent = scene_extent(train_cams)
    densify_until = args.iters // 2 if args.densify_until is None else args.densify_until
    H, W = train_cams[0].H, train_cams[0].W
    print(f"train {len(train_cams)} 枚 ({W}x{H}), val {len(val_cams)} 枚, extent={extent:.2f}")

    model = GaussianModel(args.sh_degree)
    model.init_random(args.init_points, args.init_extent, dev)
    model.setup_optimizer(extent)

    logdir = Path("logs") / args.expname
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_f = open(logdir / "log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "train_psnr", "n_gauss", "val_psnr", "sec"])

    sh_degree, order, t_start = 0, [], time.time()
    for it in range(1, args.iters + 1):
        model.update_lr(it, args.iters)
        if it % 1000 == 0:  # SH の次数を 1000 iter ごとに上げる（まず拡散色を合わせてから視点依存へ）
            sh_degree = min(sh_degree + 1, args.sh_degree)
        if not order:
            order = torch.randperm(len(train_cams)).tolist()
        cam = train_cams[order.pop()]

        out = render(model, cam, bg, sh_degree, tile=args.tile, max_entries=args.max_entries)
        img = out["image"]
        l1 = (img - cam.image).abs().mean()
        loss = (1 - args.lambda_ssim) * l1 + args.lambda_ssim * (1 - ssim(img, cam.image))
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
                train_psnr = psnr(img, cam.image).item()
                val_psnr = ""
                if it % args.i_val == 0 or it == args.iters:
                    val_psnr, ms = evaluate(model, val_cams, bg, sh_degree, args, logdir / f"val_{it:06d}.png")
                    val_psnr = f"{val_psnr:.2f}"
                el = time.time() - t_start
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"[{it:6d}] loss {loss.item():.4f}  psnr {train_psnr:5.2f}  N {len(model):7d}  "
                      f"val {val_psnr or '-':>5}  {el / it * 1000:5.0f} ms/it  peak {mem:.1f} GB")
                log.writerow([it, f"{loss.item():.5f}", f"{train_psnr:.3f}", len(model), val_psnr, f"{el:.1f}"])
                log_f.flush()

    log_f.close()
    torch.save({"model": model.state_dict(), "sh_degree": sh_degree, "bg": bg.cpu(),
                "camera": {"W": W, "H": H, "focal": train_cams[0].fx}, "args": vars(args)},
               logdir / f"ckpt_{args.iters:06d}.pt")
    model.save_ply(logdir / "point_cloud.ply")
    print(f"学習時間 {(time.time() - t_start) / 60:.1f} 分, ガウシアン数 {len(model)}")
    if test_cams:
        p, sec = evaluate(model, test_cams, bg, sh_degree, args, logdir / "test_000.png")
        print(f"test PSNR {p:.2f} dB, 描画 {sec * 1000:.1f} ms/枚")


if __name__ == "__main__":
    main()
