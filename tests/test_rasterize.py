# -*- coding: utf-8 -*-
"""
タイル化ラスタライザ (rasterize) を、全ガウシアン×全画素の素朴な実装と突き合わせる検算。
画像と、全パラメータへの勾配が一致することを確かめる。

    python tests/test_rasterize.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gs.camera import Camera  # noqa: E402
from gs.poses import look_at_origin  # noqa: E402
from gs.rasterize import (SegmentedExclusiveCumsum, conic_and_radius, covariance_3d,  # noqa: E402
                          project_gaussians, rasterize, rasterize_dense_reference)


def run_segment_gradcheck():
    """手書きの逆伝播を数値微分と比べる（torch.autograd.gradcheck）"""
    counts = torch.tensor([3, 0, 1, 4, 2], device="cuda")
    starts = torch.cumsum(counts, 0) - counts
    tl = torch.repeat_interleave(torch.arange(len(counts), device="cuda"), counts)
    x = torch.randn(3, int(counts.sum()), dtype=torch.float64, device="cuda", requires_grad=True)
    ok = torch.autograd.gradcheck(SegmentedExclusiveCumsum.apply,
                                  (x, starts[tl], (starts + counts - 1)[tl]))
    print(f"SegmentedExclusiveCumsum gradcheck: {ok}")
    return ok


def run(max_entries, tile, N=400, W=75, H=53, seed=0):
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    leaf = {
        "means": (torch.rand(N, 3, generator=g) * 2 - 1) * 1.2,
        "log_scales": torch.log(torch.rand(N, 3, generator=g) * 0.15 + 0.01),
        "quats": torch.randn(N, 4, generator=g),
        "opacity_logits": torch.randn(N, generator=g),
        "color": torch.rand(N, 3, generator=g),
    }
    leaf = {k: v.to(dev).requires_grad_(True) for k, v in leaf.items()}
    cam = Camera.from_blender(look_at_origin(30, 25, 4.0), W, H, focal=70.0, device=dev)
    bg = torch.tensor([0.2, 0.5, 0.9], device=dev)
    weight = torch.rand(H, W, 3, generator=g).to(dev)  # 適当な重みで損失を作り勾配を比べる

    def forward(fn, **kw):
        cov = covariance_3d(leaf["log_scales"].exp(), F.normalize(leaf["quats"], dim=-1))
        m2, cov2, depth = project_gaussians(leaf["means"], cov, cam)
        opacity = torch.sigmoid(leaf["opacity_logits"])
        conic, _, radius = conic_and_radius(cov2, opacity)
        img, alpha = fn(m2, conic, opacity, leaf["color"], radius, depth, W, H, bg, tile=tile, **kw)[:2]
        grads = torch.autograd.grad((img * weight).sum() + alpha.sum(), list(leaf.values()))
        return img.detach(), dict(zip(leaf, grads))

    img_t, g_t = forward(rasterize, max_entries=max_entries)
    img_r, g_r = forward(rasterize_dense_reference)

    print(f"tile={tile:2d} max_entries={max_entries:8d}  image max|diff| = {(img_t - img_r).abs().max():.2e}")
    ok = torch.allclose(img_t, img_r, atol=1e-5)
    for k in leaf:
        err = (g_t[k] - g_r[k]).norm() / g_r[k].norm()
        print(f"    grad {k:15s} rel.err = {err:.2e}")
        ok &= err.item() < 1e-4
    return ok


def run_radius_equivalence(N=400, W=75, H=53, seed=1):
    """不透明度で打ち切った半径でタイルを割り当てても、3σ で割り当てた場合と同じ画像・勾配になるか"""
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    means = ((torch.rand(N, 3, generator=g) * 2 - 1) * 1.2).to(dev)
    scales = (torch.rand(N, 3, generator=g) * 0.3 + 0.01).to(dev)
    quats = F.normalize(torch.randn(N, 4, generator=g), dim=-1).to(dev)
    logits = (torch.randn(N, generator=g) * 3 - 2).to(dev).requires_grad_(True)  # 低い不透明度を多めに
    color = torch.rand(N, 3, generator=g).to(dev)
    cam = Camera.from_blender(look_at_origin(30, 25, 4.0), W, H, focal=70.0, device=dev)
    bg = torch.ones(3, device=dev)
    m2, cov2, depth = project_gaussians(means, covariance_3d(scales, quats), cam)
    opacity = torch.sigmoid(logits)
    conic, r3, r_op = conic_and_radius(cov2, opacity)
    img3 = rasterize(m2, conic, opacity, color, r3, depth, W, H, bg)[0]
    imgo = rasterize(m2, conic, opacity, color, r_op, depth, W, H, bg)[0]
    g3, = torch.autograd.grad(img3.sum(), logits, retain_graph=True)
    go, = torch.autograd.grad(imgo.sum(), logits)
    print(f"radius equivalence: 3σ 半径の平均 {r3.mean():.1f}px -> 不透明度で打ち切り {r_op.mean():.1f}px, "
          f"image max|diff| = {(img3 - imgo).abs().max():.2e}, grad max|diff| = {(g3 - go).abs().max():.2e}")
    return torch.allclose(img3, imgo, atol=1e-6) and torch.allclose(g3, go, atol=1e-6)


if __name__ == "__main__":
    ok = run_segment_gradcheck()
    ok &= run_radius_equivalence()
    ok &= run(max_entries=2 ** 24, tile=16)  # チャンク 1 つ
    ok &= run(max_entries=3000, tile=16)     # 複数チャンク + gradient checkpointing
    ok &= run(max_entries=2 ** 24, tile=8)
    print("OK" if ok else "NG")
    sys.exit(0 if ok else 1)
