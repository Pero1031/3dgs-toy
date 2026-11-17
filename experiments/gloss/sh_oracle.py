# -*- coding: utf-8 -*-
"""
3DGS を使わない「1 点ずつの SH 当てはめ」で、視線方向の SH という表し方そのものの限界を測る。

真の形状（Erato のメッシュ）が分かっているので、表面上の点ごとに
    - その点が見えている学習視点での観測色（画像の値）と視線方向を集め、
    - 次数 L の実 SH を最小二乗（ごく弱い ridge 付き）で当てはめ、
    - テスト視点では、当てはめた SH で色を予測して観測と比べる。
3DGS の最適化・densification・投影の影響を除いた、「点ごとに視線方向の SH で色を持つ」という
表し方の上限になる（ガウシアン 1 個が表面の 1 点を受け持つ理想的な場合）。

    python experiments/gloss/sh_oracle.py

出力: experiments/gloss/sh_oracle.json, figures/sh_oracle.png, figures/sh_oracle_point.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from geom_eval import load_mesh, sample_surface  # noqa: E402
from gs.dataset import load_cameras  # noqa: E402
from gs.sh import sh_basis  # noqa: E402

DEV = "cuda"
BG = [0.396] * 3
COS_MIN = math.cos(math.radians(75))  # これより斜めに見える観測は使わない
LAMBDA = 1e-3                         # ridge（係数が観測より多いときも解が決まるように）
MIN_TRAIN_OBS = 10                    # 学習視点でこれ未満しか見えていない点は使わない
SMOOTH_FIG = 0.03                     # 図 1 の右に載せる、なめらかさの罰の強さ
EXAMPLES = [(3, 0.0), (12, 0.0), (12, SMOOTH_FIG)]  # 図 2 に載せる (次数, なめらかさの罰)
LUMA = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float64)

# 図の色（dataviz の参照パレット: カテゴリ 1, 2 と文字・面の色）
C_SERIES = {"gold": "#2a78d6", "matte": "#eb6834"}
C_TEXT, C_TEXT2, C_GRID, C_SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"gold": "金", "matte": "拡散面"}


def load_masks(root, split):
    """像のマスクを 3x3 で収縮したもの（縁の画素は背景と混ざっているので使わない）"""
    frames = json.loads((Path(root) / f"transforms_{split}.json").read_text(encoding="utf-8"))["frames"]
    out = []
    for f in frames:
        m = torch.from_numpy(imageio.imread(Path(root) / (f["file_path"].replace("_denoised", "") + "_mask.png"))
                             .astype(np.float32) / 255.0).to(DEV)
        out.append(-F.max_pool2d(-m[None, None], 3, 1, 1)[0, 0])
    return out


@torch.no_grad()
def observe(cams, masks, P, N, D):
    """各視点で、点 P (n,3)（法線 N）が見えているか・観測色・視線方向（カメラ → 点）を返す。

    見えているかの判定: 表面を密に覆う点 D で画素ごとの最小深度（z バッファ）を作り、
    点の深度がそれ以下（斜めの面のぶんの許容幅つき）で、正面寄りから見ていて、像の内側の画素なら可視とする。
    """
    vis, col, dirs = [], [], []
    for cam, m in zip(cams, masks):
        def proj(X):
            pc = X @ cam.R.T + cam.t
            return cam.fx * pc[:, 0] / pc[:, 2] + cam.cx, cam.fy * pc[:, 1] / pc[:, 2] + cam.cy, pc[:, 2]

        ud, vd, zd = proj(D)
        ok = (ud >= 0) & (ud < cam.W) & (vd >= 0) & (vd < cam.H) & (zd > 0)
        zbuf = torch.full((cam.H * cam.W,), float("inf"), device=DEV).scatter_reduce(
            0, vd[ok].long() * cam.W + ud[ok].long(), zd[ok], "amin")

        u, v, z = proj(P)
        inb = (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        pix = v.clamp(0, cam.H - 1).long() * cam.W + u.clamp(0, cam.W - 1).long()
        to_cam = F.normalize(cam.center - P, dim=-1)
        cos = (N * to_cam).sum(-1)
        tan = torch.sqrt((1 - cos * cos).clamp(min=0)) / cos.clamp(min=COS_MIN)
        tol = 0.002 + 1.5 * (z / cam.fx) * tan  # 1 画素の中でも斜めの面は深度がこれくらい変わる
        visible = inb & (cos > COS_MIN) & (z <= zbuf[pix] + tol) & (m.reshape(-1)[pix] > 0.999)

        grid = torch.stack([2 * u / cam.W - 1, 2 * v / cam.H - 1], -1)[None, None]
        rgb = F.grid_sample(cam.image.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T
        vis.append(visible)
        col.append(rgb)
        dirs.append(-to_cam)
    return torch.stack(vis), torch.stack(col).double(), torch.stack(dirs)


def chunk_size(K):
    return max(200, min(2000, int(4e8 / (K * K * 8))))


def fit(deg, vis, col, dirs, smooth=0.0):
    """点ごとに min Σ_v w_v |B_v c - y_v|² + Σ_k (λ + s·l_k(l_k+1)) c_k² を解く -> 係数 (n, K, 3)

    λ = LAMBDA はごく弱い ridge（係数が観測より多いときも解が決まるように）。
    s はなめらかさの罰: 次数 l の係数に l(l+1) を掛けて罰する（球面上の勾配の大きさの二乗和に当たる）。
    定数項（l=0）は罰さないので、平均の色は偏らない。
    """
    V, n = vis.shape
    K = (deg + 1) ** 2
    coef = torch.zeros(n, K, 3, dtype=torch.float64, device=DEV)
    l = torch.tensor([math.isqrt(k) for k in range(K)], dtype=torch.float64, device=DEV)
    eye = torch.diag(LAMBDA + smooth * l * (l + 1))
    for i in range(0, n, chunk_size(K)):
        sl = slice(i, i + chunk_size(K))
        B = sh_basis(deg, dirs[:, sl].reshape(-1, 3)).double().reshape(V, -1, K)
        Bw = B * vis[:, sl, None]
        A = torch.einsum("vnk,vnj->nkj", Bw, B) + eye
        coef[sl] = torch.linalg.solve(A, torch.einsum("vnk,vnc->nkc", Bw, col[:, sl]))
    return coef


def predict(deg, coef, dirs):
    V, n = dirs.shape[:2]
    K = coef.shape[1]
    out = torch.empty(V, n, 3, dtype=torch.float64, device=DEV)
    for i in range(0, n, chunk_size(K)):
        sl = slice(i, i + chunk_size(K))
        B = sh_basis(deg, dirs[:, sl].reshape(-1, 3)).double().reshape(V, -1, K)
        out[:, sl] = torch.einsum("vnk,nkc->vnc", B, coef[sl])
    return out.clamp(0, 1)


def psnr(pred, col, vis):
    se = ((pred - col) ** 2).mean(-1)
    return -10 * math.log10(((se * vis).sum() / vis.sum()).item())


def to_cam_grid(n_az=361, n_el=101, center=0.0):
    az = torch.linspace(-180, 180, n_az, dtype=torch.float64)
    el = torch.linspace(-10, 90, n_el, dtype=torch.float64)
    E, A = torch.meshgrid(torch.deg2rad(el), torch.deg2rad(az + center), indexing="ij")
    return az, el, torch.stack([torch.cos(E) * torch.sin(A), torch.sin(E), torch.cos(E) * torch.cos(A)], -1)


def az_el(to_cam):
    t = to_cam.cpu().double()
    return torch.rad2deg(torch.atan2(t[:, 0], t[:, 2])).numpy(), torch.rad2deg(torch.asin(t[:, 1].clamp(-1, 1))).numpy()


def plot_curves(res, n_obs_median, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9,
                         "axes.edgecolor": C_TEXT2, "axes.labelcolor": C_TEXT, "xtick.color": C_TEXT2,
                         "ytick.color": C_TEXT2, "text.color": C_TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=200, facecolor=C_SURF, sharey=True)
    degs = res["degrees"]
    L_over = math.sqrt(n_obs_median) - 1  # 係数の数 (L+1)^2 が、1 点あたりの学習観測数の中央値を超える次数
    panels = [("(a) ほぼ最小二乗（ごく弱い ridge のみ）", lambda r: r),
              (f"(b) なめらかさの罰あり（s = {SMOOTH_FIG:g}）", lambda r: r["smooth_check"][str(SMOOTH_FIG)])]
    for ax, (title, get) in zip(axes, panels):
        ax.set_facecolor(C_SURF)
        ax.axvspan(L_over, degs[-1] + 0.5, color=C_GRID, alpha=0.55, lw=0)
        ax.text(L_over + 0.15, 0.03, f"係数の数 (L+1)² が\n1 点の学習観測数の中央値（{n_obs_median:.0f}）を超える",
                transform=ax.get_xaxis_transform(), fontsize=7, color=C_TEXT2, va="bottom")
        ax.axvline(3, color=C_TEXT2, lw=0.8, ls=":")
        ax.text(3.15, 0.97, "3DGS の既定（L=3）", transform=ax.get_xaxis_transform(), fontsize=7, color=C_TEXT2,
                va="top")
        ends = []
        for cond in res["conditions"]:
            r = get(res["conditions"][cond])
            for split, ls in (("train", (0, (4, 2))), ("test", "-")):
                y = r[f"{split}_psnr"]
                ax.plot(degs, y, color=C_SERIES[cond], lw=2, ls=ls, marker="o" if split == "test" else "s", ms=4.5,
                        mec=C_SURF, mew=1.2, label=f"{LABEL[cond]}・{'学習視点' if split == 'train' else 'テスト視点'}")
                ends.append([y[-1], f"{LABEL[cond]}・{'学習' if split == 'train' else 'テスト'}"])
        # 線の右端のラベルが重ならないよう、縦位置を最低 GAP dB 離す（上から順に押し下げる）
        GAP = 3.5
        ends.sort(key=lambda e: -e[0])
        pos = [ends[0][0]]
        for y, _ in ends[1:]:
            pos.append(min(y, pos[-1] - GAP))
        for (y, text), yl in zip(ends, pos):
            ax.annotate(text, (degs[-1], y), xytext=(degs[-1] + 0.45, yl), textcoords="data", va="center",
                        fontsize=7, color=C_TEXT)
        ax.set_xticks(degs)
        ax.set_xticklabels([f"{L}\n({(L + 1) ** 2})" for L in degs], fontsize=7)
        ax.set_xlabel("SH の次数 L（括弧内は 1 チャンネルあたりの係数の数）")
        ax.set_xlim(-0.5, degs[-1] + 3.2)
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_ylabel("観測色の PSNR [dB]")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=4, frameon=False, fontsize=7.5,
                   handlelength=2.6, columnspacing=1.2, borderaxespad=0.2)
    fig.suptitle("表面の 1 点ずつに視線方向の SH を当てはめたときの再現度（3DGS を使わない上限）", fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=C_SURF)
    plt.close(fig)


def plot_point(ex, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "MS Gothic"], "font.size": 9})
    c = ex["az_center"]
    az, el, grid = to_cam_grid(center=c)  # 方位はこの点の観測の中心からの相対値（-180°の折り返しを避ける）
    rel = lambda a: (np.asarray(a) - c + 180) % 360 - 180  # noqa: E731
    fig, axes = plt.subplots(1, len(EXAMPLES), figsize=(4.0 * len(EXAMPLES), 3.1), dpi=200, facecolor=C_SURF,
                             sharey=True, constrained_layout=True)
    for ax, (deg, s), coef in zip(axes, EXAMPLES, ex["coef"]):
        coef = torch.tensor(coef, dtype=torch.float64)
        B = sh_basis(deg, -grid.reshape(-1, 3))  # SH はカメラ → 点の向きで評価している
        lum = ((B @ coef).clamp(0, 1) @ LUMA).reshape(grid.shape[:2]).numpy()
        im = ax.imshow(lum, extent=[-180, 180, el[0], el[-1]], origin="lower", cmap="gray", vmin=0, vmax=1,
                       aspect="auto", interpolation="bilinear")
        for split, mk, ec, lab in (("train", "o", C_SERIES["gold"], "学習視点の観測"),
                                   ("test", "^", C_SERIES["matte"], "テスト視点の観測")):
            ax.scatter(rel(ex[split]["az"]), ex[split]["el"], c=ex[split]["lum"], cmap="gray", vmin=0, vmax=1,
                       marker=mk, s=60, edgecolors=ec, linewidths=1.6, label=lab, zorder=3)
        ax.set_title(f"L={deg}（{(deg + 1) ** 2} 係数）" + (f"、なめらかさの罰 s={s:g}" if s else "、罰なし"),
                     fontsize=9, loc="left")
        ax.set_xlabel("カメラの方位（観測の中心からの相対値、度）")
        ax.set_xlim(-110, 110)
        ax.set_ylim(-10, 90)
    axes[0].set_ylabel("カメラの仰角（度）")
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.85)
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cb.set_label("輝度（画像の値）", fontsize=8)
    fig.suptitle("金の像の 1 点: 視線方向による見え方（点 = 実際の観測、背景 = SH による補間）", fontsize=9, x=0.01,
                 ha="left")
    fig.savefig(path, facecolor=C_SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16])
    ap.add_argument("--n-points", type=int, default=10000)
    ap.add_argument("--check-smooth", type=float, nargs="*", default=[0.003, 0.01, SMOOTH_FIG, 0.1])
    args = ap.parse_args()

    v, f = load_mesh()
    P, N = sample_surface(v, f, args.n_points, seed=1)
    D, _ = sample_surface(v, f, 2_000_000, seed=2)
    P, N, D = (torch.tensor(x, dtype=torch.float32, device=DEV) for x in (P, N, D))
    masks = {s: load_masks(ROOT / "data" / "gloss_matte", s) for s in ("train", "test")}

    res = {"degrees": args.degrees, "lambda": LAMBDA, "n_points_sampled": args.n_points, "conditions": {}}
    example = None
    for cond in ("gold", "matte"):
        obs = {}
        for split in ("train", "test"):
            cams = load_cameras(ROOT / "data" / f"gloss_{cond}", split, False, BG, DEV)
            obs[split] = observe(cams, masks[split], P, N, D)
        keep = obs["train"][0].sum(0) >= MIN_TRAIN_OBS
        tr = [x[:, keep] for x in obs["train"]]
        te = [x[:, keep] for x in obs["test"]]
        n_tr = tr[0].sum(0).float()
        r = {"n_points": int(keep.sum()), "train_obs_per_point_median": float(n_tr.median()),
             "train_obs_per_point_mean": float(n_tr.mean()), "test_obs_total": int(te[0].sum()),
             "train_psnr": [], "test_psnr": []}
        for deg in args.degrees:
            coef = fit(deg, *tr)
            r["train_psnr"].append(psnr(predict(deg, coef, tr[2]), tr[1], tr[0]))
            r["test_psnr"].append(psnr(predict(deg, coef, te[2]), te[1], te[0]))
            print(f"{cond:<6} L={deg:2d} K={(deg + 1) ** 2:3d}  train {r['train_psnr'][-1]:6.2f} dB  "
                  f"test {r['test_psnr'][-1]:6.2f} dB", flush=True)
        res["conditions"][cond] = r
        print(f"{cond}: 点 {r['n_points']}、1 点あたりの学習観測 中央値 {r['train_obs_per_point_median']:.0f}"
              f"（平均 {r['train_obs_per_point_mean']:.1f}）、テスト観測 {r['test_obs_total']}")

        if cond == "gold":  # 例として見せる点: 学習・テストともよく見えていて、見え方の変化が最も大きい点
            lum_tr = (tr[1] @ LUMA.to(DEV))
            w = tr[0].double()
            mean = (lum_tr * w).sum(0) / w.sum(0)
            var = (((lum_tr - mean) ** 2) * w).sum(0) / w.sum(0)
            var[(tr[0].sum(0) < 30) | (te[0].sum(0) < 3)] = -1
            j = int(var.argmax())
            one = [x[:, j:j + 1] for x in tr]
            ex = {"examples": EXAMPLES, "coef": [fit(d, *one, smooth=s)[0].cpu().tolist() for d, s in EXAMPLES]}
            for split, o in (("train", tr), ("test", te)):
                sel = o[0][:, j]
                a, e = az_el(-o[2][sel, j])
                ex[split] = {"az": a.tolist(), "el": e.tolist(), "lum": (o[1][sel, j] @ LUMA.to(DEV)).cpu().tolist()}
            allaz = np.radians(np.array(ex["train"]["az"] + ex["test"]["az"]))
            ex["az_center"] = math.degrees(math.atan2(np.sin(allaz).mean(), np.cos(allaz).mean()))
            example = ex

        # 正則化への依存: なめらかさの罰を入れれば、高い次数でもテスト視点が良くなるか
        r["smooth_check"] = {}
        for s in args.check_smooth:
            fits = [(d, fit(d, *tr, smooth=s)) for d in args.degrees]
            tr_ps = [psnr(predict(d, c, tr[2]), tr[1], tr[0]) for d, c in fits]
            te_ps = [psnr(predict(d, c, te[2]), te[1], te[0]) for d, c in fits]
            k = int(np.argmax(te_ps))
            r["smooth_check"][str(s)] = {"train_psnr": tr_ps, "test_psnr": te_ps, "best_degree": args.degrees[k],
                                         "best_test_psnr": te_ps[k]}
            print(f"{cond:<6} s={s:g}: テスト視点の最高 {te_ps[k]:.2f} dB（L={args.degrees[k]}）、"
                  f"L={args.degrees[-1]} で学習 {tr_ps[-1]:.2f} / テスト {te_ps[-1]:.2f} dB", flush=True)

    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "sh_oracle.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    plot_curves(res, res["conditions"]["gold"]["train_obs_per_point_median"], HERE / "figures" / "sh_oracle.png")
    plot_point(example, HERE / "figures" / "sh_oracle_point.png")
    print(f"-> {HERE / 'sh_oracle.json'}, figures/sh_oracle.png, figures/sh_oracle_point.png")


if __name__ == "__main__":
    main()
