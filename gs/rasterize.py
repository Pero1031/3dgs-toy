# -*- coding: utf-8 -*-
"""
3D Gaussian Splatting の微分可能ラスタライザ（PyTorch のテンソル演算のみ）。

公式実装（diff-gaussian-rasterization）は前向き・逆向きとも CUDA で手書きしているが、
ここでは前向き計算だけを PyTorch で書き、勾配は autograd に任せる。処理の流れは公式と同じ:

  1. project_gaussians : 3D ガウシアンを画面に投影し、2D ガウシアン（中心 μ'・共分散 Σ'）にする
  2. bin_tiles         : 画面を tile×tile 画素のタイルに分け、各ガウシアンが覆うタイルを列挙して
                         (タイル番号, 深度) をキーにソートする
  3. composite         : タイルごとに手前から順に α ブレンディングする
                             C = Σ_i c_i α_i T_i ,   T_i = Π_{j<i} (1 - α_j)
"""

import functools

import torch
from torch.utils.checkpoint import checkpoint

ALPHA_MIN = 1.0 / 255.0  # これ未満の α は寄与なしとして捨てる（公式実装と同じ）
ALPHA_MAX = 0.99         # α=1 だと 1-α=0 で後ろへの勾配が消えるので上限を置く（公式実装と同じ）


# ============================================================ 1. 3D -> 2D 投影

def quat_to_rotmat(q):
    """正規化済みクォータニオン (N,4) [w,x,y,z] -> 回転行列 (N,3,3)"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


def covariance_3d(scales, quats):
    """3D 共分散 Σ = R S Sᵀ Rᵀ（論文 式(6)）。

    Σ の 6 成分を直接最適化すると半正定値性が崩れうるので、
    軸ごとの標準偏差 s（S = diag(s)）と回転 q に分解して持つ。
    """
    M = quat_to_rotmat(quats) * scales[:, None, :]  # R @ diag(s)：R の各列に s を掛ける
    return M @ M.transpose(1, 2)


def project_gaussians(means, cov3d, cam):
    """3D ガウシアンを画面へ投影する（EWA splatting、論文 式(5)）。

    透視投影 (x,y,z) -> (fx x/z + cx, fy y/z + cy) は非線形なので、ガウシアン中心のまわりで
    一次近似（ヤコビアン J）すると、ガウシアンはまたガウシアンに写る:
        Σ' = J W Σ Wᵀ Jᵀ     （W：ワールド→カメラの回転）

    Returns
        means2d (N,2) 画素座標, cov2d (N,2,2), depth (N,) カメラ座標の z
    """
    p = means @ cam.R.T + cam.t
    x, y, z = p.unbind(-1)

    # 画面外に大きくはみ出したガウシアンで J が極端にならないよう、
    # 視野の 1.3 倍の範囲にクランプしてから J を作る（公式実装と同じ）
    lim_x = 1.3 * 0.5 * cam.W / cam.fx
    lim_y = 1.3 * 0.5 * cam.H / cam.fy
    tx = (x / z).clamp(-lim_x, lim_x) * z
    ty = (y / z).clamp(-lim_y, lim_y) * z

    zero = torch.zeros_like(z)
    J = torch.stack([
        cam.fx / z, zero, -cam.fx * tx / (z * z),
        zero, cam.fy / z, -cam.fy * ty / (z * z),
    ], -1).reshape(-1, 2, 3)
    JW = J @ cam.R
    cov2d = JW @ cov3d @ JW.transpose(1, 2)
    # 1 画素より細くならないようにする簡易ローパス（公式実装と同じ +0.3）
    cov2d = cov2d + 0.3 * torch.eye(2, device=cov2d.device)

    means2d = torch.stack([cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy], -1)
    return means2d, cov2d, z


def conic_and_radius(cov2d, opacity=None):
    """2D 共分散の逆行列（conic）と、画面上の影響半径 [px] を求める。

    conic は対称 2x2 行列 Σ'^{-1} の (00, 01, 11) 成分。
    半径は Σ' の大きい方の固有値 λ から 3√λ（公式実装と同じ。タイル割り当てにだけ使うので勾配不要）。

    opacity を渡すと、半径を「α = o·exp(-½ r²/λ) が ALPHA_MIN を下回る距離」でも打ち切った
    tile_radius も返す。それより外側の画素では α < 1/255 で結局捨てられるので、描画結果は
    3√λ で割り当てた場合と厳密に同じまま、計算する (交差 × 画素) の数だけが減る。
    （不透明度が低いうちは大きな差になる。opacity reset 直後の o=0.01 なら半径は約 1/2）
    """
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)
    with torch.no_grad():
        mid = 0.5 * (a + c)
        sigma = torch.sqrt(mid + torch.sqrt((mid * mid - det).clamp(min=0.1)))
        radius = torch.ceil(3.0 * sigma)
        if opacity is None:
            return conic, radius
        k = torch.sqrt(2.0 * torch.log(opacity.detach() / ALPHA_MIN).clamp(min=0.0))
        tile_radius = torch.ceil(torch.minimum(k, torch.full_like(k, 3.0)) * sigma)
    return conic, radius, tile_radius


# ============================================================ 2. タイル割り当てとソート

def _tile_rect(means2d, radius, tiles_x, tiles_y, tile):
    """各ガウシアンの 3σ 外接矩形が覆うタイル範囲 [x0, x1) × [y0, y1)"""
    u, v = means2d[:, 0], means2d[:, 1]
    x0 = ((u - radius) / tile).floor().clamp(0, tiles_x).long()
    x1 = ((u + radius) / tile).floor().add(1).clamp(0, tiles_x).long()
    y0 = ((v - radius) / tile).floor().clamp(0, tiles_y).long()
    y1 = ((v + radius) / tile).floor().add(1).clamp(0, tiles_y).long()
    return x0, x1, y0, y1


@torch.no_grad()
def bin_tiles(means2d, radius, depth, W, H, tile):
    """(ガウシアン, タイル) の交差を列挙し、タイル番号→深度の順に並べる。

    公式実装はキー = [タイル番号 | 深度] の 64bit 整数を基数ソートする。ここでも同じく
    キー = タイル番号 * N + 深度の順位 を作って 1 回ソートする。

    Returns
        gid      (M,) 各交差のガウシアン番号（タイル内では手前から順）
        tile_id  (M,) 各交差のタイル番号（昇順）
        on_screen (N,) 1 タイル以上を覆うか
        tiles_x, tiles_y
    """
    tiles_x = (W + tile - 1) // tile
    tiles_y = (H + tile - 1) // tile
    x0, x1, y0, y1 = _tile_rect(means2d.detach(), radius, tiles_x, tiles_y, tile)
    n = (x1 - x0) * (y1 - y0) * (radius > 0)  # 覆うタイル数（画面外・半径 0 なら 0）

    N = means2d.shape[0]
    ar = torch.arange(N, device=means2d.device)
    gid = torch.repeat_interleave(ar, n)                # ガウシアン i を n_i 回繰り返す
    k = torch.arange(gid.numel(), device=gid.device) - (torch.cumsum(n, 0) - n)[gid]
    w = (x1 - x0)[gid]                                  # 矩形の幅（タイル数）
    tile_id = (y0[gid] + k // w) * tiles_x + (x0[gid] + k % w)

    rank = torch.empty_like(ar)
    rank[torch.argsort(depth.detach())] = ar            # 深度の順位（0 が最も手前）
    order = torch.argsort(tile_id * N + rank[gid])
    return gid[order], tile_id[order], n > 0, tiles_x, tiles_y


# ============================================================ 3. α ブレンディング

class SegmentedExclusiveCumsum(torch.autograd.Function):
    """区間ごとにリセットされる「自分を含まない」累積和。

    x (P, M) を行優先で 1 列に並べたものを区間に区切り（区間は行をまたがない）、
        y[p, m] = Σ_{start(m) <= j < m} x[p, j]
    を返す。start(m), end(m) は m が属する区間の先頭・末尾の列番号。

    前向き: 全体の累積和を 1 回とり（1 次元なので GPU の並列スキャンが使える）、区間先頭の値を引く。
    逆向き: この線形写像の転置（随伴）は「区間ごとの逆向き累積和」
        ∂L/∂x[p, j] = Σ_{j < m <= end(j)} ∂L/∂y[p, m]
    で、前向きと同じやり方で計算できる。
    autograd に任せると「区間先頭の値を引く」部分の逆伝播が float64 の atomicAdd になり、
    それを持たない古い GPU（Maxwell 等）ではとても遅いので、逆伝播を手で書いている。
    全体の累積和は大きな値になるので、桁落ちを避けて float64 で計算する。
    """

    @staticmethod
    def forward(ctx, x, seg_start, seg_end):
        ctx.save_for_backward(seg_start, seg_end)
        P, M = x.shape
        excl = torch.cumsum(x.double().reshape(-1), 0).reshape(P, M) - x   # 先頭から自分の手前までの和
        return (excl - excl.index_select(1, seg_start)).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        seg_start, seg_end = ctx.saved_tensors
        P, M = g.shape
        rev = torch.cumsum(g.double().reshape(-1).flip(0), 0).flip(0).reshape(P, M) - g  # 自分の後ろから末尾までの和
        return (rev - rev.index_select(1, seg_end)).to(g.dtype), None, None


@functools.lru_cache(maxsize=None)
def _pixel_offsets(tile, device):
    """タイル左上から見た各画素中心の座標 (tile*tile, 2)。並びは行優先 (ly, lx)。"""
    ly, lx = torch.meshgrid(torch.arange(tile), torch.arange(tile), indexing="ij")
    return (torch.stack([lx, ly], -1).reshape(-1, 2).float() + 0.5).to(device)


def composite(means2d, conic, opacity, color, tile_xy, tile_local, counts, tile):
    """連続したタイル群（チャンク）の α ブレンディング。

    入力はすべて「交差」単位（M 個）で、タイル番号→深度の順に並んでいる。
        means2d (M,2), conic (M,3), opacity (M,), color (M,3)
        tile_xy (M,2) その交差のタイル左上の画素座標
        tile_local (M,) チャンク内のタイル番号 0..n_tiles-1（昇順）
        counts (n_tiles,) 各タイルの交差数
    タイル内の全画素 (P = tile*tile 個) と交差 m の組を (P, M) のテンソルとして一度に計算する。

    Returns
        rgb (n_tiles, P, 3)   Σ c_i α_i T_i
        T_final (n_tiles, P)  全ガウシアン通過後の透過率（背景色の重み）
    """
    P = tile * tile
    offs = _pixel_offsets(tile, means2d.device)
    dx = offs[:, 0:1] + (tile_xy[:, 0] - means2d[:, 0])[None]   # (P, M) 画素 - 中心
    dy = offs[:, 1:2] + (tile_xy[:, 1] - means2d[:, 1])[None]

    # α = o · exp(-½ dᵀ Σ'^{-1} d)   （論文 式(2)(3) の 2D 版）
    A, B, C = conic[:, 0], conic[:, 1], conic[:, 2]
    power = -0.5 * (A * dx * dx + C * dy * dy) - B * dx * dy
    alpha = (opacity * torch.exp(power)).clamp(max=ALPHA_MAX)
    alpha = torch.where(alpha >= ALPHA_MIN, alpha, 0.0)

    # 透過率 T_i = Π_{j<i}(1-α_j) を「画素ごと・タイル内での」累積積として求める。
    # (P, M) の各行（= 画素）で、同じタイルの交差は手前から順に連続して並んでいるので、
    # log 空間で「タイルの区間ごとにリセットされる累積和」をとれば log T_i になる。
    log1m = torch.log1p(-alpha)                                       # log(1-α) <= 0
    starts = torch.cumsum(counts, 0) - counts
    T = torch.exp(SegmentedExclusiveCumsum.apply(
        log1m, starts[tile_local], (starts + counts - 1)[tile_local]))  # (P, M)

    n_tiles = counts.shape[0]
    weight = alpha * T                                                # 各ガウシアンの寄与 α_i T_i
    rgb = torch.zeros(P, n_tiles, 3, device=weight.device).index_add(
        1, tile_local, weight[..., None] * color[None])
    log_T = torch.zeros(P, n_tiles, device=weight.device).index_add(1, tile_local, log1m)
    return rgb.transpose(0, 1), torch.exp(log_T).transpose(0, 1)


def _chunks(counts, cap):
    """タイルを、交差数の合計が cap 以下になる連続区間に分ける -> [(t0, t1), ...]"""
    bounds, acc = [0], 0
    for t, c in enumerate(counts):
        if acc > 0 and acc + c > cap:
            bounds.append(t)
            acc = 0
        acc += c
    bounds.append(len(counts))
    return list(zip(bounds[:-1], bounds[1:]))


def rasterize(means2d, conic, opacity, color, radius, depth, W, H, bg,
              tile=16, max_entries=2 ** 25):
    """2D ガウシアン群を画像にする。radius はタイル割り当てに使う半径 [px]。

    max_entries: 一度に作る (画素 × 交差) テンソルの要素数の上限。超える場合はタイルを
    いくつかのチャンクに分けて順に処理し、各チャンクは gradient checkpointing で
    逆伝播時に再計算する（中間テンソルを全部覚えておくと VRAM が足りないため）。
    1 要素あたり、逆伝播用の保存分と一時領域を合わせて 60〜70 バイト程度使う。

    Returns
        image (H,W,3), alpha (H,W) 累積不透明度, on_screen (N,)
    """
    gid, tile_id, on_screen, tiles_x, tiles_y = bin_tiles(means2d, radius, depth, W, H, tile)
    n_tiles = tiles_x * tiles_y
    P = tile * tile

    # 交差ごとに属性を集める（ここは微分可能：勾配は元のガウシアンに足し込まれる）
    m2, cn, op, col = means2d[gid], conic[gid], opacity[gid], color[gid]
    tile_xy = torch.stack([tile_id % tiles_x, tile_id // tiles_x], -1).float() * tile

    counts = torch.bincount(tile_id, minlength=n_tiles)
    offsets = [0] + torch.cumsum(counts, 0).tolist()
    chunks = _chunks(counts.tolist(), max(max_entries // P, 1))
    use_ckpt = len(chunks) > 1 and torch.is_grad_enabled()

    rgbs, Ts = [], []
    for t0, t1 in chunks:
        i0, i1 = offsets[t0], offsets[t1]
        fn = functools.partial(composite, counts=counts[t0:t1], tile=tile)
        args = (m2[i0:i1], cn[i0:i1], op[i0:i1], col[i0:i1], tile_xy[i0:i1], tile_id[i0:i1] - t0)
        rgb, T = checkpoint(fn, *args, use_reentrant=False) if use_ckpt else fn(*args)
        rgbs.append(rgb)
        Ts.append(T)
    rgb = torch.cat(rgbs)
    T = torch.cat(Ts)

    rgb = rgb + T[..., None] * bg  # 残った透過率のぶん背景が見える

    def to_image(x):  # (tiles_y*tiles_x, P, C) -> (H, W, C)
        C = x.shape[-1]
        x = x.reshape(tiles_y, tiles_x, tile, tile, C).permute(0, 2, 1, 3, 4)
        return x.reshape(tiles_y * tile, tiles_x * tile, C)[:H, :W]

    image = to_image(rgb)
    alpha = 1.0 - to_image(T[..., None])[..., 0]
    return image, alpha, on_screen


# ============================================================ テスト用の素朴な実装

def rasterize_dense_reference(means2d, conic, opacity, color, radius, depth, W, H, bg, tile=16):
    """全ガウシアン × 全画素を愚直に計算して手前から合成する（rasterize の検算用）。

    タイル化・ソート・log 空間の累積和を使わず、教科書どおり cumprod で T_i を求める。
    ただし「3σ 矩形が覆うタイルの画素にだけ寄与する」という打ち切り規則は rasterize と揃える。
    """
    order = torch.argsort(depth.detach())
    means2d, conic, opacity, color, radius = (x[order] for x in (means2d, conic, opacity, color, radius))

    ys, xs = torch.meshgrid(torch.arange(H, device=means2d.device),
                            torch.arange(W, device=means2d.device), indexing="ij")
    pix = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5         # (P, 2)
    d = pix[None] - means2d[:, None, :]                                  # (N, P, 2)
    dx, dy = d[..., 0], d[..., 1]
    power = -0.5 * (conic[:, 0:1] * dx * dx + conic[:, 2:3] * dy * dy) - conic[:, 1:2] * dx * dy
    alpha = (opacity[:, None] * torch.exp(power)).clamp(max=ALPHA_MAX)
    alpha = alpha * (alpha >= ALPHA_MIN)

    tiles_x, tiles_y = (W + tile - 1) // tile, (H + tile - 1) // tile
    x0, x1, y0, y1 = _tile_rect(means2d.detach(), radius, tiles_x, tiles_y, tile)
    ptx, pty = (pix[:, 0] // tile).long(), (pix[:, 1] // tile).long()
    inside = ((ptx[None] >= x0[:, None]) & (ptx[None] < x1[:, None])
              & (pty[None] >= y0[:, None]) & (pty[None] < y1[:, None]) & (radius > 0)[:, None])
    alpha = alpha * inside

    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:1]), 1 - alpha], 0), 0)  # (N+1, P)
    rgb = ((alpha * T[:-1])[..., None] * color[:, None, :]).sum(0) + T[-1][:, None] * bg
    return rgb.reshape(H, W, 3), (1 - T[-1]).reshape(H, W)
