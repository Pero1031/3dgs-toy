# -*- coding: utf-8 -*-
"""
env_rough.py の「点ごとに最適化した粗さ」と「測った法線のばらつき」の相関が弱かった（Spearman ρ ≈ 0.14）ことの切り分け。

    1. 折半信頼性: 学習視点を偶数番目と奇数番目に分けて、それぞれで点ごとの粗さを求め直し、両者の順位相関をとる。
       最適化した粗さ自体がどれだけ再現するか（測定の雑音の大きさ）が分かり、ρ の上限の目安になる
       （Spearman–Brown の式で全視点の信頼性に直し、相関の希薄化を補正する）。
    2. 回収テスト: 「測ったばらつきから換算したぼけ」で作った合成の観測（同じ視点・同じ点・当てはめた照明）に、
       同じ手順を当てて、ぼけの幅を取り戻せるかを見る。手順そのものに、点ごとの粗さを見分ける力があるかが分かる。
       雑音なしと、実データの当てはめ残差と同じ大きさの独立な雑音ありの 2 通り。

    python experiments/gloss/env_rough_check.py

出力: experiments/gloss/env_rough_check.json
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import env_rough as er  # noqa: E402

DEV = er.DEV
LP, S = 0, 0.1  # env_rough.py で粗さの比較に使った設定（点ごとの定数、照明のなめらかさの罰 0.1）


def identifiable(loss):
    rmse = loss.sqrt()
    return ((rmse.max(1).values - rmse.min(1).values) > er.ID_CONTRAST).cpu().numpy()


def rho(a, b, m):
    return float(spearmanr(a[m], b[m])[0])


def main():
    t0 = time.time()
    grid = torch.tensor(np.radians(er.SIGMA_GRID_DEG), dtype=torch.float32, device=DEV)
    obs, Np, sel, radii, ln_smooth, ln_face, sig_meas = er.prepare(t0)
    meas = sig_meas[er.R_NOMINAL_MM / 1000]
    meas_deg = np.degrees(meas.cpu().numpy())
    o_tr, o_te = er.Obs(*obs["train"], Np, "radiance"), er.Obs(*obs["test"], Np, "radiance")
    out = {"local_degree": LP, "env_smooth": S, "radius_mm": er.R_NOMINAL_MM}

    # 全視点（env_rough.py と同じ）
    sig, _, _, _, loss, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
    full_deg, id_full = np.degrees(sig.cpu().numpy()), identifiable(loss)
    out["full"] = {"rho_measured": rho(meas_deg, full_deg, id_full), "n_identifiable": int(id_full.sum()),
                   "median_fitted_deg": float(np.median(full_deg[id_full]))}
    print(f"全視点: ρ（測ったばらつき, 最適化した粗さ）= {out['full']['rho_measured']:.3f}（{time.time() - t0:.0f} 秒）",
          flush=True)

    # 1. 折半信頼性
    halves = {}
    for name, start in (("even", 0), ("odd", 1)):
        o_h = er.Obs(*[x[start::2] for x in obs["train"]], Np, "radiance")
        s_h, _, _, _, l_h, _ = er.run_fitted(o_h, o_te, LP, S, grid)
        halves[name] = (np.degrees(s_h.cpu().numpy()), identifiable(l_h))
        del o_h
        torch.cuda.empty_cache()
        print(f"  半分（{name}）を当てはめた（{time.time() - t0:.0f} 秒）", flush=True)
    (se, ie), (so_, io) = halves["even"], halves["odd"]
    both = ie & io
    r_half = rho(se, so_, both)
    r_full = 2 * r_half / (1 + r_half)  # Spearman–Brown: 観測を 2 倍にしたときの信頼性
    out["split_half"] = {
        "n_both_identifiable": int(both.sum()), "rho_even_odd": r_half, "reliability_full_spearman_brown": r_full,
        "rho_measured_even": rho(meas_deg, se, both), "rho_measured_odd": rho(meas_deg, so_, both),
        "rho_measured_full_corrected": out["full"]["rho_measured"] / math.sqrt(max(r_full, 1e-6))}
    print(f"折半: ρ（偶数, 奇数）= {r_half:.3f} → 全視点の信頼性 {r_full:.3f}、"
          f"希薄化を補正した ρ（測ったばらつき, 最適化した粗さ）= {out['split_half']['rho_measured_full_corrected']:.3f}",
          flush=True)

    # 2. 回収テスト: 測ったばらつきのぼけで作った合成の観測から、ぼけを取り戻せるか
    _, ptr, _, _ = er.run_fixed(o_tr, o_te, meas, LP, S)
    clean = er.radiance_to_display(ptr)
    noise = math.sqrt(er.mse(o_tr, ptr))  # 実データでの当てはめ残差（表示値の RMSE）
    real = o_tr.ydisp
    gen = torch.Generator(device=DEV).manual_seed(0)
    out["recovery"] = {"residual_rmse_real": noise}
    rec = {}
    for name, sd in (("no_noise", 0.0), ("noise_as_real_residual", noise)):
        o_tr.ydisp = (clean + sd * torch.randn(clean.shape, generator=gen, device=DEV, dtype=clean.dtype)).clamp(0, 1)
        s_r, _, _, _, l_r, _ = er.run_fitted(o_tr, o_te, LP, S, grid)
        rec_deg, id_r = np.degrees(s_r.cpu().numpy()), identifiable(l_r)
        out["recovery"][name] = {"noise_rmse": sd, "n_identifiable": int(id_r.sum()),
                                 "rho_true_recovered": rho(meas_deg, rec_deg, id_r),
                                 "median_true_deg": float(np.median(meas_deg[id_r])),
                                 "median_recovered_deg": float(np.median(rec_deg[id_r]))}
        r = out["recovery"][name]
        rec[name] = (rec_deg, id_r)
        print(f"回収（{name}、雑音 {sd:.3f}）: ρ（真のぼけ, 取り戻したぼけ）= {r['rho_true_recovered']:.3f}、"
              f"中央値 真 {r['median_true_deg']:.1f}° / 取り戻した {r['median_recovered_deg']:.1f}°"
              f"（{time.time() - t0:.0f} 秒）", flush=True)
    o_tr.ydisp = real

    # 3. 最適化した粗さは何と相関するか（点ごとの候補との順位相関、粗さが決まる点だけ）
    v, f = er.so.load_mesh()
    Ps = er.so.sample_surface(v, f, 10000, seed=1)[0][sel]
    center = np.array([0.0, 0.25, 0.0])  # 像の中心（make_views.py の TARGET）
    lights = [np.asarray(c, dtype=float) for c in er.ed.SIDE] + [np.array([0.0, 1.9, 0.0])]

    def angle(a, b):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        b = b / np.linalg.norm(b, axis=-1, keepdims=True)
        return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))

    # 視差: 光源を像の中心から見た向きと、その点から見た向きの差（遠方の照明とみなしたときのずれ）。5 つの光源の最大
    parallax = np.max([angle(L - Ps, L - center) for L in lights], axis=0)
    vis, _, dirs = obs["train"]
    r = er.eo.reflect(dirs, Np)
    Pb = torch.tensor(Ps, dtype=torch.float32, device=DEV)[None].expand(*vis.shape, 3)
    n_obs = vis.sum(0).double()
    hit = (er.ed.light_hit(Pb, r) & vis).sum(0).double() / n_obs
    cosv = ((-dirs * Np[None]).sum(-1) * vis).sum(0).double() / n_obs
    cands = {"測った法線のばらつき（半径 2.72 mm）": meas_deg,
             "光源の視差（像の中心から見た向きとの差、最大）": parallax,
             "像の中心からの距離": np.linalg.norm(Ps - center, axis=1),
             "高さ": Ps[:, 1],
             "反射光線が光源に当たる観測の割合": hit.cpu().numpy(),
             "学習視点の観測数": n_obs.cpu().numpy(),
             "視線と法線の cos の平均": cosv.cpu().numpy()}
    out["correlates"] = {k: rho(full_deg, x, id_full) for k, x in cands.items()}
    out["parallax_deg"] = {"median": float(np.median(parallax)), "p90": float(np.percentile(parallax, 90))}
    print(f"光源の視差: 中央値 {np.median(parallax):.1f}°、90% 点 {np.percentile(parallax, 90):.1f}°")
    for k, x in out["correlates"].items():
        print(f"  ρ（最適化した粗さ, {k}）= {x:+.3f}")

    (HERE / "env_rough_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    # 図: 実データと、仮説どおりに作った合成データ（雑音あり）で、同じ手順の結果を並べる
    plt = er.setup_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.7), dpi=200, facecolor=er.so.C_SURF, sharey=True)
    rr = out["recovery"]["noise_as_real_residual"]
    rec_deg, id_r = rec["noise_as_real_residual"]
    er.scatter_panel(ax1, meas_deg[id_full], full_deg[id_full],
                     f"(a) 実際の画像（{int(id_full.sum())} 点、ρ = {out['full']['rho_measured']:.2f}、"
                     f"折半信頼性 {r_full:.2f}）")
    er.scatter_panel(ax2, meas_deg[id_r], rec_deg[id_r],
                     f"(b) 横軸どおりにぼかして作った合成の観測 + 雑音 {noise:.3f}（{rr['n_identifiable']} 点、"
                     f"ρ = {rr['rho_true_recovered']:.2f}）", ylabel="")
    fig.suptitle("同じ手順で点ごとのぼけを当てはめた結果: 手順は換算どおりのぼけを取り戻せるが、実際の画像のぼけは換算に従わない",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "env_rough_check.png", facecolor=er.so.C_SURF)
    plt.close(fig)
    print(f"-> {HERE / 'env_rough_check.json'}, figures/env_rough_check.png（{time.time() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
