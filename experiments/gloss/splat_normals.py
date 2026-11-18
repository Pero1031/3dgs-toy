# -*- coding: utf-8 -*-
"""
段階 3 の準備: ガウシアンで表した面は、画素の中の法線のばらつきを持てるか（学習なしの検証）。

段階 2（sphere_rough.py）では、画素の足跡の中で平均した法線の長さ |n̄| から決めたぼけ σ = 2√(1 − |n̄|)
（Toksvig の換算）が、点ごとに最適化した粗さとよく合った。3DGS で遅延シェーディングをするなら、画素に重なる
ガウシアンの法線を α ブレンドで合成し、その長さを |n̄| として使うのが素直な実装になる。

ただし α ブレンドは「足跡の中の平均」ではなく「手前から順に覆っていく合成」なので、不透明なガウシアンが
重なると手前の数個の法線しか残らない。そこで、凹凸の球（make_sphere.py）の正解のメッシュの頂点に、面に
沿った小さな円盤状のガウシアンを置き（凹凸の 1 波長に約 10 個）、3dgs-toy のラスタライザで法線を合成して、
画素ごとの |n̄| を正解（画素の箱型の足跡の中で解析的な法線を平均したもの）と比べる。

    敷き詰め   面に沿った標準偏差 S_TAN、法線方向 S_NRM のガウシアン。不透明度は OPACITIES のそれぞれ
    合成       3dgs-toy の rasterize（色の代わりに法線を合成）。|n̄| は合成した法線を累積不透明度で割ってから測る
    超解像     同じガウシアンを SS 倍の解像度で描き、SS×SS 画素ごとに合成した法線を足し合わせてから |n̄| を測る
               （画面の上で足跡を平均する。法線マップのミップマップと同じ考え方）

    python experiments/gloss/splat_normals.py

出力: experiments/gloss/splat_normals.json（図は plot_deferred.py がテスト視点で描き直す）
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import make_sphere as ms  # noqa: E402
import sh_oracle as so  # noqa: E402
import sphere_rough as sr  # noqa: E402
from gs.camera import Camera  # noqa: E402
from gs.rasterize import conic_and_radius, project_gaussians, rasterize  # noqa: E402

DEV = sr.DEV
S_TAN = 0.3e-3            # 面に沿った標準偏差 [m]（頂点の間隔は約 0.41 mm）
S_NRM = 0.01e-3           # 法線方向の標準偏差 [m]（ほぼ平らな円盤）
OPACITIES = [0.99, 0.3]   # 不透明（3DGS が面を表すときの典型）と、半透明
VIEWS = [f"view_1_{i:03d}" for i in range(0, 24, 4)]  # 赤道の 6 視点（経度 4〜304°）
SCALES = {512: 2, 256: 4, 128: 4}   # 画素の大きさ -> 超解像の倍率
TRUTH_SAMPLES = 64        # 正解: 画素の箱型の中のサンプル数
OUT_JSON = HERE / "splat_normals.json"


def read_ply(path):
    """make_sphere.write_ply の形式（float xyz + nxnynz、リトルエンディアン）から頂点と法線を読む"""
    with open(path, "rb") as fh:
        n = None
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        v = np.frombuffer(fh.read(n * 24), dtype="<f4").reshape(n, 6)
    return v[:, :3].copy(), v[:, 3:].copy()


def to_camera(frm, scale):
    """sphere_rough.Cam（OpenCV 流儀の camera-to-world）から 3dgs-toy の Camera（world-to-camera）を作る"""
    c = sr.Cam(frm, scale)
    R = c.Rc2w.T.contiguous()
    return Camera(R=R, t=-(R @ c.o), fx=c.f, fy=c.f, cx=c.c, cy=c.c, W=scale, H=scale), c


@torch.no_grad()
def splat_normals(P, N, cam, opacity):
    """面に沿った円盤状のガウシアンで法線を合成 -> (H, W, 3) Σ w n、(H, W) 累積不透明度"""
    face = ((cam.center - P) * N).sum(-1) > 0          # 裏を向いたもの（球の向こう側）は除く
    P, N = P[face], N[face]
    cov3d = S_TAN ** 2 * torch.eye(3, device=DEV) + (S_NRM ** 2 - S_TAN ** 2) * N[:, :, None] * N[:, None, :]
    means2d, cov2d, depth = project_gaussians(P, cov3d, cam)
    op = torch.full((len(P),), opacity, device=DEV)
    conic, _, tile_radius = conic_and_radius(cov2d, op)
    img, alpha, _ = rasterize(means2d, conic, op, N, tile_radius, depth, cam.W, cam.H,
                              torch.zeros(3, device=DEV), tile=8, max_entries=2 ** 25)
    return img, alpha


def pool_normals(img, alpha, k):
    """k×k 画素ごとに Σ w n と累積不透明度を足す（画面の上で足跡を平均する）"""
    if k == 1:
        return img, alpha
    s = F.avg_pool2d(img.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0)
    a = F.avg_pool2d(alpha[None, None], k)[0, 0]
    return s, a


def sigma_from(sum_n, alpha):
    ln = (sum_n.norm(dim=-1) / alpha.clamp(min=1e-6)).clamp(max=1)
    return 2 * (1 - ln).sqrt()


@torch.no_grad()
def truth_and_select(c, scale, offs):
    """画素の中心が球に当たり、使う範囲（凹凸を弱める範囲・区画の境目・斜めすぎる所を除く）にある画素の
    番号と区画と、正解（画素の箱型の足跡の中で解析的な法線を平均）のぼけ σ"""
    j, i = torch.meshgrid(torch.arange(scale, device=DEV), torch.arange(scale, device=DEV), indexing="ij")
    u, v = i.reshape(-1).float() + 0.5, j.reshape(-1).float() + 0.5
    n0 = c.hit_sphere(u, v)
    X = ms.R0 * n0
    cosv = (F.normalize(c.o - X, dim=-1) * n0).sum(-1)
    q = n0.cpu().numpy().astype(np.float64)
    lon, lat = ms.lon_lat(q)
    off = lon % ms.SECTOR_DEG
    sec = (lon // ms.SECTOR_DEG).astype(int)
    ok = ((np.abs(lat) <= sr.LAT_MAX) & (off >= sr.BORDER_MARGIN) & (off <= ms.SECTOR_DEG - sr.BORDER_MARGIN)
          & np.isin(sec, sr.GEOM) & (cosv.cpu().numpy() > so.COS_MIN))
    idx = torch.tensor(np.nonzero(ok)[0], device=DEV)
    amp = torch.tensor(ms.amplitude(ms.R0 * q[ok]), dtype=torch.float32, device=DEV)
    sig = torch.empty(len(idx), device=DEV)
    for b in range(0, len(idx), 2048):
        jj = idx[b:b + 2048]
        n0s = c.hit_sphere(u[jj, None] + offs[None, :, 0], v[jj, None] + offs[None, :, 1])
        sig[b:b + 2048] = sr.mean_normal(n0s, amp[b:b + 2048])[1]
    return idx, sec[ok], sig


def main():
    t0 = time.time()
    P_np, N_np = read_ply(ms.DATA / "mesh" / "au.ply")
    P = torch.tensor(P_np, device=DEV)
    N = F.normalize(torch.tensor(N_np, device=DEV), dim=-1)
    print(f"ガウシアン {len(P)} 個（凹凸の球の au 区画の頂点）", flush=True)
    s = torch.quasirandom.SobolEngine(2, scramble=True, seed=sr.SEED).draw(TRUTH_SAMPLES).to(DEV) - 0.5
    frm = {n: f for sp, n, f in ms.views() if sp == "train"}
    rows = {}   # (scale, 方式) -> 区画 -> σ のリスト
    for scale, ss in SCALES.items():
        acc = {}
        for name in VIEWS:
            cam, c = to_camera(frm[name], scale)
            idx, sec, sig_true = truth_and_select(c, scale, s)
            res = {"truth": sig_true}
            for op in OPACITIES:
                img, alpha = splat_normals(P, N, cam, op)
                a = alpha.reshape(-1)[idx]
                res[f"splat_o{op}"] = sigma_from(img.reshape(-1, 3)[idx], a)
                res[f"alpha_o{op}"] = a
                cam_ss, _ = to_camera(frm[name], scale * ss)
                img2, alpha2 = pool_normals(*splat_normals(P, N, cam_ss, op), ss)
                res[f"ss{ss}_o{op}"] = sigma_from(img2.reshape(-1, 3)[idx], alpha2.reshape(-1)[idx])
            for k, v in res.items():
                for kk in sr.GEOM:
                    m = torch.tensor(sec == kk, device=DEV)
                    acc.setdefault(k, {}).setdefault(ms.SECTORS[kk][0], []).append(v[m].cpu().numpy())
            print(f"  {scale} px {name}: 画素 {len(idx)}（{time.time() - t0:.0f} 秒）", flush=True)
        out = {}
        for k, d in acc.items():
            out[k] = {}
            for sname, lst in d.items():
                x = np.concatenate(lst)
                if k.startswith("alpha"):
                    out[k][sname] = {"median": float(np.median(x)), "p10": float(np.percentile(x, 10))}
                else:
                    x = np.degrees(x)
                    out[k][sname] = {"mean_deg": float(x.mean()), "median_deg": float(np.median(x)), "n": int(len(x))}
        rows[str(scale)] = {"supersample": ss, "sectors": out}
        print(f"[{scale} px] 区画ごとのぼけの平均 [度]（正解 / 合成 o=0.99 / 超解像 o=0.99 / 合成 o=0.3 / 超解像 o=0.3）")
        for kk in sr.GEOM:
            sname = ms.SECTORS[kk][0]
            vals = [out[k][sname]["mean_deg"] for k in
                    ("truth", "splat_o0.99", f"ss{ss}_o0.99", "splat_o0.3", f"ss{ss}_o0.3")]
            print(f"    {sname:<9}" + "".join(f" {x:6.2f}" for x in vals)
                  + f"   累積不透明度の中央値 {out['alpha_o0.99'][sname]['median']:.3f} / {out['alpha_o0.3'][sname]['median']:.3f}")
    OUT_JSON.write_text(json.dumps({"settings": {"s_tan_mm": S_TAN * 1e3, "s_nrm_mm": S_NRM * 1e3,
                                                 "opacities": OPACITIES, "views": VIEWS,
                                                 "truth_samples": TRUTH_SAMPLES, "n_gaussians": len(P)},
                                    "scales": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT_JSON}（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
