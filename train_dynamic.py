# -*- coding: utf-8 -*-
"""
動くシーンの 3D Gaussian Splatting（4DGS の最小版）。表現は 2 通り:

    既定         基準の形のガウシアン 1 組と、時刻ごとにそれをずらす MLP（gs/deform.py）を同時に学習する
                 （Deformable 3D Gaussians の簡略版）
    --spacetime  各ガウシアンが自分の時刻・時間幅・速度を持つ（gs/spacetime.py。Spacetime Gaussians の簡略版）

データは D-NeRF 形式（transforms_*.json の各フレームに "time"）。

    python make_toy_dynamic.py
    python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy --iters 5000 --warm-up 1000
    python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy_static --iters 5000 --static   # 比較用

    python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy_st --iters 5000 --spacetime

    --static   変形なし（時刻を無視するふつうの 3DGS）。同じ手順・同じ評価で比べるため
    --warm-up  はじめの何回かは変形なしで学習し、おおまかな形ができてから MLP を動かす（論文は 3000 / 40000）

logs/<expname>/ に test 画像（上: 正解 / 下: 描画）、時刻を動かした GIF、log.csv、eval.json、チェックポイントを出力する。
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from gs.camera import Camera
from gs.dataset import load_cameras
from gs.deform import DeformNet
from gs.gaussians import GaussianModel
from gs.loss import psnr, ssim
from gs.poses import look_at_origin
from gs.render import render
from gs.spacetime import SpacetimeModel
from train import scene_extent, to_u8


def render_at(model, net, cam, t, bg, sh_degree, args):
    """時刻 t の形で描画する（時刻を持つガウシアンは自分でずれを出す。それ以外で net が None なら変形なし）"""
    if isinstance(model, SpacetimeModel):
        delta = model.delta_at(t)
    else:
        delta = None if net is None else net(model.params["means"].detach(), t)
    out = render(model, cam, bg, sh_degree, tile=args.tile, max_entries=args.max_entries, delta=delta)
    out["delta"] = delta
    return out


def velocity_smoothness(model, net, out, t, n_anchor=1024, k=8, tau=0.02, eps=0.02):
    """近くのガウシアンの速度をそろえる損失: 速度の差の 2 乗 / 距離の 2 乗（＝速度場の勾配の 2 乗）の平均。

    一様な色の面の上では、面に沿って滑っても画像が変わらないので、画像の誤差だけでは動きが決まらない。
    模様や縁のある所で決まった動きを、まわりへ広げる（オプティカルフローの Horn–Schunck の滑らかさの項と
    同じ考え。距離を保つ ARAP と違い、一様に膨らむ・縮む動きもほとんど罰しない）。
    その時刻に描いたガウシアンから n_anchor 個を選び、それぞれの近傍 k 個と比べる。
    """
    ids, delta = out["ids"], out["delta"]
    if len(ids) <= k:
        return torch.zeros((), device=ids.device)
    x = (model.params["means"] + delta["means"])[ids].detach()          # 時刻 t の位置
    a = torch.randint(len(ids), (min(n_anchor, len(ids)),), device=x.device)
    d, nb = torch.cdist(x[a], x).topk(k + 1, dim=1, largest=False)
    d, nb = d[:, 1:], nb[:, 1:]                                          # 先頭は自分自身
    if isinstance(model, SpacetimeModel):
        v = model.params["velocity"][ids]
        va, vn = v[a], v[nb]
    else:   # 変形場: τ だけ離れた時刻との位置の差を速度とみなす
        tau = tau if t + tau <= 1.0 else -tau
        pts = ids[torch.cat([a, nb.flatten()])]
        v = (net(model.params["means"][pts].detach(), t + tau)["means"] - delta["means"][pts]) / tau
        va, vn = v[: len(a)], v[len(a):].view(len(a), k, 3)
    return ((va[:, None] - vn).pow(2).sum(-1) / (d.pow(2) + eps ** 2)).mean()


@torch.no_grad()
def evaluate(model, net, cams, bg, sh_degree, args, save_path=None, n_show=4):
    ps, ss, times, shown = [], [], [], []
    show = set(np.linspace(0, len(cams) - 1, n_show).round().astype(int).tolist())
    for k, cam in enumerate(cams):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = render_at(model, net, cam, cam.time, bg, sh_degree, args)["image"]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        ps.append(psnr(img, cam.image).item())
        ss.append(ssim(img, cam.image).item())
        if k in show:
            shown.append(torch.cat([cam.image, img.clamp(0, 1)], 0))
    if save_path is not None:
        imageio.imwrite(save_path, to_u8(torch.cat(shown, 1)))
    return float(np.mean(ps)), float(np.mean(ss)), float(np.median(times))


@torch.no_grad()
def time_sweep(model, net, ref_cam, radius, bg, sh_degree, args, path, n=40):
    """視点を少しずつ回しながら時刻を 0 -> 1 に動かした GIF"""
    frames = []
    for k in range(n):
        c2w = look_at_origin(30.0 + 90.0 * k / (n - 1), 30.0, radius)
        cam = Camera.from_blender(c2w, ref_cam.W, ref_cam.H, ref_cam.fx)
        img = render_at(model, net, cam, k / (n - 1), bg, sh_degree, args)["image"]
        frames.append(to_u8(img))
    imageio.mimsave(path, frames, duration=1000 / 20, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--expname", default="dyn")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--half-res", action="store_true")
    ap.add_argument("--black-bkgd", action="store_true")
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--init-points", type=int, default=20000)
    ap.add_argument("--init-extent", type=float, default=1.5, help="初期点を置く立方体の半辺長")
    # 変形場
    ap.add_argument("--static", action="store_true", help="変形なし（ふつうの 3DGS）で学習する")
    ap.add_argument("--spacetime", action="store_true", help="時刻を持つガウシアン（gs/spacetime.py）で学習する")
    ap.add_argument("--init-t-scale", type=float, default=0.2, help="--spacetime: 時間幅 σ_t の初期値")
    ap.add_argument("--t-split-thresh", type=float, default=0.05, help="--spacetime: σ_t がこれより長いものは時間方向に分割する")
    ap.add_argument("--lr-velocity", type=float, default=1.6e-3, help="--spacetime: 速度の学習率（シーンの大きさを掛ける）")
    ap.add_argument("--min-t-scale", type=float, default=0.0, help="--spacetime: σ_t の下限（これより短くならない）")
    ap.add_argument("--lambda-smooth", type=float, default=0.0,
                    help="近くのガウシアンの速度をそろえる損失の重み（0 で使わない。変形場でも --spacetime でも使える）")
    ap.add_argument("--warm-up", type=int, default=3000, help="この回数までは変形なしで学習する")
    ap.add_argument("--deform-depth", type=int, default=8)
    ap.add_argument("--deform-width", type=int, default=256)
    ap.add_argument("--lr-deform", type=float, default=8e-4, help="MLP の学習率（指数減衰で 1/500 まで下げる）")
    # densification（train.py と同じ。--max-gauss で数の上限を決められる）
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--densify-until", type=int, default=None)
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--opacity-reset", type=int, default=3000)
    ap.add_argument("--grad-thresh", type=float, default=0.0002)
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--max-screen-size", type=float, default=20)
    ap.add_argument("--max-gauss", type=int, default=0, help="この数を超えたら増やさない（削除だけ続ける）。0 で無制限")
    ap.add_argument("--lambda-ssim", type=float, default=0.2)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--max-entries", type=int, default=2 ** 25)
    ap.add_argument("--i-print", type=int, default=100)
    ap.add_argument("--i-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    bg_rgb = [0.0] * 3 if args.black_bkgd else [1.0] * 3
    train_cams = load_cameras(args.datadir, "train", args.half_res, bg_rgb, dev)
    test_cams = load_cameras(args.datadir, "test", args.half_res, bg_rgb, dev)
    val_cams = load_cameras(args.datadir, "val", args.half_res, bg_rgb, dev) or test_cams
    assert all(c.time is not None for c in train_cams + test_cams), "各フレームに time が必要（D-NeRF 形式）"
    bg = torch.tensor(bg_rgb, device=dev)
    extent = scene_extent(train_cams)
    densify_until = args.iters // 2 if args.densify_until is None else args.densify_until
    H, W = train_cams[0].H, train_cams[0].W
    print(f"train {len(train_cams)} 枚 ({W}x{H}), test {len(test_cams)} 枚, extent={extent:.2f}")

    assert not (args.static and args.spacetime)
    if args.spacetime:
        model = SpacetimeModel(args.sh_degree, args.init_t_scale, args.t_split_thresh, args.min_t_scale)
        model.init_random(args.init_points, args.init_extent, dev)
        model.setup_optimizer(extent, lr_velocity=args.lr_velocity)
    else:
        model = GaussianModel(args.sh_degree)
        model.init_random(args.init_points, args.init_extent, dev)
        model.setup_optimizer(extent)
    net, opt_net = None, None
    if not (args.static or args.spacetime):
        net = DeformNet(args.deform_depth, args.deform_width).to(dev)
        opt_net = torch.optim.Adam(net.parameters(), lr=args.lr_deform, eps=1e-15)

    logdir = Path("logs") / args.expname
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_f = open(logdir / "log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "train_psnr", "n_gauss", "val_psnr", "sec"])

    sh_degree, order, t_start = 0, [], time.time()
    for it in range(1, args.iters + 1):
        model.update_lr(it, args.iters)
        deforming = net is not None and it > args.warm_up
        if deforming:   # MLP の学習率も位置と同じく指数減衰（変形を始めてから終わりまでで 1/500 に）
            s = (it - args.warm_up) / max(args.iters - args.warm_up, 1)
            for g in opt_net.param_groups:
                g["lr"] = args.lr_deform * math.exp(s * math.log(1 / 500))
        if it % 1000 == 0:
            sh_degree = min(sh_degree + 1, args.sh_degree)
        if not order:
            order = torch.randperm(len(train_cams)).tolist()
        cam = train_cams[order.pop()]

        out = render_at(model, net if deforming else None, cam, cam.time, bg, sh_degree, args)
        img = out["image"]
        l1 = (img - cam.image).abs().mean()
        loss = (1 - args.lambda_ssim) * l1 + args.lambda_ssim * (1 - ssim(img, cam.image))
        if args.lambda_smooth > 0 and out["delta"] is not None:
            loss = loss + args.lambda_smooth * velocity_smoothness(model, net, out, cam.time)
        loss.backward()

        with torch.no_grad():
            if it < densify_until and out["means2d"].grad is not None:
                vis = out["on_screen"]
                model.add_densification_stats(out["means2d"].grad[vis], out["ids"][vis],
                                              out["radius"][vis], cam.W, cam.H)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            if deforming:
                opt_net.step()
                opt_net.zero_grad(set_to_none=True)

            if it < densify_until:
                if it > args.densify_from and it % args.densify_interval == 0:
                    big = args.max_screen_size if (args.max_screen_size and it > args.opacity_reset) else None
                    full = args.max_gauss and len(model) >= args.max_gauss
                    model.densify_and_prune(math.inf if full else args.grad_thresh, args.min_opacity, extent, big)
                if it % args.opacity_reset == 0:
                    model.reset_opacity()

            if it % args.i_print == 0:
                train_psnr = psnr(img, cam.image).item()
                val_psnr = ""
                if it % args.i_val == 0 or it == args.iters:
                    v = evaluate(model, net if deforming else None, val_cams, bg, sh_degree, args,
                                 logdir / f"val_{it:06d}.png")
                    val_psnr = f"{v[0]:.2f}"
                el = time.time() - t_start
                mem = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"[{it:6d}] loss {loss.item():.4f}  psnr {train_psnr:5.2f}  N {len(model):7d}  "
                      f"val {val_psnr or '-':>5}  {el / it * 1000:5.0f} ms/it  peak {mem:.1f} GB", flush=True)
                log.writerow([it, f"{loss.item():.5f}", f"{train_psnr:.3f}", len(model), val_psnr, f"{el:.1f}"])
                log_f.flush()

    log_f.close()
    train_min = (time.time() - t_start) / 60
    torch.save({"model": model.state_dict(), "deform": None if net is None else net.state_dict(),
                "sh_degree": sh_degree, "bg": bg.cpu(), "args": vars(args),
                "camera": {"W": W, "H": H, "focal": train_cams[0].fx}},
               logdir / f"ckpt_{args.iters:06d}.pt")
    p, s, sec = evaluate(model, net, test_cams, bg, sh_degree, args, logdir / "test.png")
    radius = float(torch.stack([c.center for c in train_cams]).norm(dim=1).mean())
    time_sweep(model, net, test_cams[0], radius, bg, sh_degree, args, logdir / "time_sweep.gif")
    res = {"test_psnr": p, "test_ssim": s, "render_ms": sec * 1000, "n_gauss": len(model),
           "train_min": train_min, "ms_per_iter": train_min * 60000 / args.iters}
    (logdir / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"学習 {train_min:.1f} 分, ガウシアン {len(model)}, test PSNR {p:.2f} dB, SSIM {s:.3f}, "
          f"描画 {sec * 1000:.0f} ms/枚")


if __name__ == "__main__":
    main()
