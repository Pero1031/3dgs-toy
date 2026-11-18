# -*- coding: utf-8 -*-
"""
段階 3: 遅延シェーディングの 3DGS（deferred.py）を凹凸の球で学習し、粗さの決め方（mode）を比べる。

    python experiments/gloss/train_deferred.py --mode toksvig --expname dr_toksvig
    （--mode は learned / toksvig / both。データは make_sphere.py post で作る data/gloss_sphere/s256）

学習は 256 px。最後に、テスト視点を 512・256・128 px で描いて（学習していない画素の大きさも含む）、
球の領域の PSNR と、区画ごとの |n̄|・換算のぼけ・学習した粗さ・累積不透明度を測る。

設定の注意:
    - 初期値は基準の球面に置いた円盤（形状の大まかな初期値は分かっているとする。3DGS の SfM の点に当たる）。
    - 不透明度の一斉リセットは既定で切る。半透明になると、球の裏側のガウシアンの法線が合成に混ざって |n̄| が
      短くなり、なめらかな所でも換算のぼけが出てしまうため（初期の不透明度 0.5 で試すと、なめらかな球でも 18°）。

出力: logs/<expname>/ に log.csv、ckpt.pt、eval.json、test_*.png（左から 正解 / 描画 / 使ったぼけ）
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import deferred as dg  # noqa: E402
import make_sphere as ms  # noqa: E402
import sh_oracle as so  # noqa: E402
import sphere_rough as sr  # noqa: E402
from gs.dataset import load_cameras  # noqa: E402
from gs.loss import ssim  # noqa: E402

BG = 0.396


def scene_extent(cams):
    c = torch.stack([cam.center for cam in cams])
    return 1.1 * (c - c.mean(0)).norm(dim=1).max().item()


def load_masks(datadir, split, dev):
    meta = json.loads((Path(datadir) / f"transforms_{split}.json").read_text(encoding="utf-8"))
    out = []
    for fr in meta["frames"]:
        p = Path(datadir) / "masks" / split / (Path(fr["file_path"]).name + ".png")
        m = torch.from_numpy(imageio.imread(p).astype(np.float32) / 255.0).to(dev)
        out.append(-F.max_pool2d(-m[None, None], 3, 1, 1)[0, 0])   # 3x3 で収縮（縁の画素を除く）
    return out


def masked_psnr(img, gt, m):
    sel = m > 0.999
    return -10 * math.log10(((img - gt) ** 2)[sel].mean().item())


@torch.no_grad()
def sector_pixels(frm, scale):
    """画素ごとの区画（使う範囲の外は -1）。sphere_rough と同じ範囲（緯度 50° 以内、境目から 7° 以上、斜めすぎない）"""
    c = sr.Cam(frm, scale)
    j, i = torch.meshgrid(torch.arange(scale, device=sr.DEV), torch.arange(scale, device=sr.DEV), indexing="ij")
    u, v = i.reshape(-1).float() + 0.5, j.reshape(-1).float() + 0.5
    n0 = c.hit_sphere(u, v)
    cosv = (F.normalize(c.o - ms.R0 * n0, dim=-1) * n0).sum(-1).cpu().numpy()
    lon, lat = ms.lon_lat(n0.cpu().numpy().astype(np.float64))
    off = lon % ms.SECTOR_DEG
    sec = (lon // ms.SECTOR_DEG).astype(int)
    ok = (np.abs(lat) <= sr.LAT_MAX) & (off >= sr.BORDER_MARGIN) & (off <= ms.SECTOR_DEG - sr.BORDER_MARGIN) & (cosv > so.COS_MIN)
    return np.where(ok, sec, -1).reshape(scale, scale)


@torch.no_grad()
def evaluate(model, env, args, bg_rad, logdir, scales):
    res = {}
    frm = [f for sp, n, f in ms.views() if sp == "test"]
    T = env.textures()
    for scale in scales:
        d = ROOT / "data" / "gloss_sphere" / f"s{scale}"
        cams = load_cameras(d, "test", bg=(BG,) * 3)
        masks = load_masks(d, "test", "cuda")
        ps, acc = [], {}
        for k, (cam, m) in enumerate(zip(cams, masks)):
            out = dg.render_deferred(model, env, cam, args.mode, bg_rad, tile=args.tile, T=T)
            ps.append(masked_psnr(out["image"], cam.image, m))
            sec = sector_pixels(frm[k], scale)
            fg = (m > 0.999).cpu().numpy()
            for key in ("len_n", "sig_geo", "sig_mat", "sigma", "alpha"):
                x = out[key].cpu().numpy()
                for kk, (name, _, _) in enumerate(ms.SECTORS):
                    sel = (sec == kk) & fg
                    acc.setdefault(key, {}).setdefault(name, []).append(x[sel])
            if k < 2:
                sig = (out["sigma"] / math.radians(30)).clamp(0, 1)[..., None].expand(-1, -1, 3)
                imageio.imwrite(logdir / f"test_{scale}_{k:03d}.png",
                                (torch.cat([cam.image, out["image"], sig], 1).clamp(0, 1) * 255).round().byte().cpu().numpy())
        sectors = {}
        for name in [s[0] for s in ms.SECTORS]:
            row = {}
            for key, dd in acc.items():
                x = np.concatenate(dd[name])
                if len(x) == 0:
                    continue
                row[key] = float(np.degrees(x).mean()) if key.startswith("sig") else float(x.mean())
            row["n_pixels"] = int(sum(len(a) for a in acc["alpha"][name]))
            sectors[name] = row
        res[str(scale)] = {"psnr_sphere": float(np.mean(ps)), "sectors": sectors}
        print(f"  評価 {scale} px: 球の領域の PSNR {np.mean(ps):.2f} dB")
        for name, row in sectors.items():
            print(f"    {name:<9} |n̄| {row.get('len_n', float('nan')):.4f}  換算 {row.get('sig_geo', float('nan')):5.1f}°  "
                  f"学習 {row.get('sig_mat', float('nan')):5.1f}°  使った {row.get('sigma', float('nan')):5.1f}°  "
                  f"累積不透明度 {row.get('alpha', float('nan')):.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=str(ROOT / "data" / "gloss_sphere" / "s256"))
    ap.add_argument("--mode", choices=dg.MODES, required=True)
    ap.add_argument("--expname", required=True)
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--init-points", type=int, default=30000)
    ap.add_argument("--init-opacity", type=float, default=0.9)
    ap.add_argument("--lr-env", type=float, default=5e-3)
    ap.add_argument("--lr-base", type=float, default=2.5e-3)
    ap.add_argument("--lr-rough", type=float, default=5e-3)
    ap.add_argument("--lambda-env", type=float, default=1e-6, help="照明のなめらかさの罰（deferred.EnvSH.smoothness）")
    ap.add_argument("--lambda-flat", type=float, default=0.1, help="円盤からずれる罰（最短軸 / 最長軸 の平均）")
    ap.add_argument("--lambda-ssim", type=float, default=0.2)
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--densify-until", type=int, default=None)
    ap.add_argument("--densify-interval", type=int, default=100)
    ap.add_argument("--opacity-reset", type=int, default=0, help="0 で一斉リセットしない（既定）")
    ap.add_argument("--grad-thresh", type=float, default=0.0002)
    ap.add_argument("--min-opacity", type=float, default=0.005)
    ap.add_argument("--max-screen-size", type=float, default=20)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--max-entries", type=int, default=2 ** 24)
    ap.add_argument("--eval-scales", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--i-print", type=int, default=100)
    ap.add_argument("--i-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    train_cams = load_cameras(args.datadir, "train", bg=(BG,) * 3, device=dev)
    test_cams = load_cameras(args.datadir, "test", bg=(BG,) * 3, device=dev)
    test_masks = load_masks(args.datadir, "test", dev)
    bg_rad = float(dg.display_to_radiance(torch.tensor(BG)))
    extent = scene_extent(train_cams)
    densify_until = args.iters // 2 if args.densify_until is None else args.densify_until

    model = dg.DeferredModel()
    model.init_on_sphere(args.init_points, ms.R0, device=dev)
    with torch.no_grad():
        model.params["opacity_logits"].fill_(math.log(args.init_opacity / (1 - args.init_opacity)))
    model.setup_optimizer(extent, extra_lrs={"base": args.lr_base, "rough": args.lr_rough})
    env = dg.EnvSH(device=dev)
    opt_env = torch.optim.Adam([env.coef], lr=args.lr_env, eps=1e-15)

    logdir = ROOT / "logs" / args.expname
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_f = open(logdir / "log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["iter", "loss", "n_gauss", "val_psnr_sphere", "sec"])
    print(f"train {len(train_cams)} 枚 ({train_cams[0].W} px), mode={args.mode}, extent={extent:.2f}")

    order, t0 = [], time.time()
    for it in range(1, args.iters + 1):
        model.update_lr(it, args.iters)
        if not order:
            order = torch.randperm(len(train_cams)).tolist()
        cam = train_cams[order.pop()]
        out = dg.render_deferred(model, env, cam, args.mode, bg_rad, tile=args.tile, max_entries=args.max_entries)
        img = out["image"]
        l1 = (img - cam.image).abs().mean()
        loss = (1 - args.lambda_ssim) * l1 + args.lambda_ssim * (1 - ssim(img, cam.image))
        s = model.scales
        loss = loss + args.lambda_flat * (s.min(1).values / s.max(1).values).mean() + args.lambda_env * env.smoothness()
        loss.backward()

        with torch.no_grad():
            if it < densify_until and out["means2d"].grad is not None:
                vis = out["on_screen"]
                model.add_densification_stats(out["means2d"].grad[vis], out["ids"][vis], out["radius"][vis], cam.W, cam.H)
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
            opt_env.step()
            opt_env.zero_grad(set_to_none=True)
            if it < densify_until:
                if it > args.densify_from and it % args.densify_interval == 0:
                    big = args.max_screen_size if (args.max_screen_size and args.opacity_reset and it > args.opacity_reset) else None
                    model.densify_and_prune(args.grad_thresh, args.min_opacity, extent, big)
                if args.opacity_reset and it % args.opacity_reset == 0:
                    model.reset_opacity()
            if it % args.i_print == 0:
                val = ""
                if it % args.i_val == 0 or it == args.iters:
                    T = env.textures()
                    ps = [masked_psnr(dg.render_deferred(model, env, c, args.mode, bg_rad, tile=args.tile, T=T)["image"],
                                      c.image, m) for c, m in zip(test_cams[::3], test_masks[::3])]
                    val = f"{np.mean(ps):.2f}"
                el = time.time() - t0
                print(f"[{it:5d}] loss {loss.item():.4f}  N {len(model):7d}  val(球) {val or '-':>5}  "
                      f"{el / it * 1000:5.0f} ms/it  peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
                log.writerow([it, f"{loss.item():.5f}", len(model), val, f"{el:.1f}"])
                log_f.flush()
    log_f.close()
    torch.save({"model": model.state_dict(), "env": env.coef.detach().cpu(), "args": vars(args)}, logdir / "ckpt.pt")
    print(f"学習時間 {(time.time() - t0) / 60:.1f} 分、ガウシアン {len(model)}")

    result = {"args": vars(args), "n_gauss": len(model), "train_minutes": (time.time() - t0) / 60}
    with torch.no_grad():
        T = env.textures()
        result["train_psnr_sphere"] = float(np.mean([
            masked_psnr(dg.render_deferred(model, env, c, args.mode, bg_rad, tile=args.tile, T=T)["image"], c.image, m)
            for c, m in zip(train_cams[::8], load_masks(args.datadir, "train", dev)[::8])]))
    print(f"  学習視点（12 枚）の球の領域の PSNR {result['train_psnr_sphere']:.2f} dB")
    result["eval"] = evaluate(model, env, args, bg_rad, logdir, args.eval_scales)
    (logdir / "eval.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {logdir / 'eval.json'}")


if __name__ == "__main__":
    main()
