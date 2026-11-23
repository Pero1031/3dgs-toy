# -*- coding: utf-8 -*-
"""
Differentiable rasterizer for 3D Gaussian Splatting (PyTorch tensor operations only).

The official implementation (diff-gaussian-rasterization) hand-writes both the forward and backward passes in CUDA.
Here only the forward computation is written in PyTorch, and gradients are left to autograd. The flow is the same
as the official one:

  1. project_gaussians : project the 3D Gaussians onto the screen as 2D Gaussians (center μ', covariance Σ')
  2. bin_tiles         : divide the screen into tiles of tile×tile pixels, enumerate the tiles each Gaussian covers,
                         and sort with (tile index, depth) as the key
  3. composite         : α-blend front to back within each tile
                             C = Σ_i c_i α_i T_i ,   T_i = Π_{j<i} (1 - α_j)
"""

import functools

import torch
from torch.utils.checkpoint import checkpoint

ALPHA_MIN = 1.0 / 255.0  # α below this is discarded as no contribution (same as the official implementation)
ALPHA_MAX = 0.99         # with α=1, 1-α=0 and the gradient to what lies behind vanishes, so set an upper limit (same as official)


# ============================================================ 1. 3D -> 2D projection

def quat_to_rotmat(q):
    """Normalized quaternions (N,4) [w,x,y,z] -> rotation matrices (N,3,3)"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


def covariance_3d(scales, quats):
    """3D covariance Σ = R S Sᵀ Rᵀ (paper Eq. (6)).

    Optimizing the 6 components of Σ directly can break positive semi-definiteness, so it is stored
    factored into the per-axis standard deviations s (S = diag(s)) and a rotation q.
    """
    M = quat_to_rotmat(quats) * scales[:, None, :]  # R @ diag(s): multiply each column of R by s
    return M @ M.transpose(1, 2)


def project_gaussians(means, cov3d, cam):
    """Project 3D Gaussians onto the screen (EWA splatting, paper Eq. (5)).

    The perspective projection (x,y,z) -> (fx x/z + cx, fy y/z + cy) is nonlinear, so it is approximated to first
    order around the Gaussian center (Jacobian J). A Gaussian then maps to a Gaussian again:
        Σ' = J W Σ Wᵀ Jᵀ     (W: world-to-camera rotation)

    Returns
        means2d (N,2) pixel coordinates, cov2d (N,2,2), depth (N,) z in camera coordinates
    """
    p = means @ cam.R.T + cam.t
    x, y, z = p.unbind(-1)

    # So that J does not become extreme for Gaussians that extend far off screen,
    # clamp to 1.3 times the field of view before building J (same as the official implementation)
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
    # Simple low-pass so that nothing gets thinner than 1 pixel (same +0.3 as the official implementation)
    cov2d = cov2d + 0.3 * torch.eye(2, device=cov2d.device)

    means2d = torch.stack([cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy], -1)
    return means2d, cov2d, z


def conic_and_radius(cov2d, opacity=None):
    """Compute the inverse of the 2D covariance (conic) and the screen-space radius of influence [px].

    conic is the (00, 01, 11) components of the symmetric 2x2 matrix Σ'^{-1}.
    The radius is 3√λ from the larger eigenvalue λ of Σ' (same as the official implementation. It is used only
    for tile assignment, so no gradient is needed).

    If opacity is given, also returns tile_radius, where the radius is additionally cut off at "the distance
    at which α = o·exp(-½ r²/λ) falls below ALPHA_MIN". Pixels farther out have α < 1/255 and are discarded
    anyway, so the rendered result stays exactly the same as when assigning with 3√λ, and only the number of
    (intersection × pixel) pairs computed goes down.
    (The difference is large while opacity is low. With o=0.01 right after an opacity reset, the radius is about 1/2.)
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


# ============================================================ 2. tile assignment and sorting

def _tile_rect(means2d, radius, tiles_x, tiles_y, tile):
    """Tile range [x0, x1) × [y0, y1) covered by the 3σ bounding rectangle of each Gaussian"""
    u, v = means2d[:, 0], means2d[:, 1]
    x0 = ((u - radius) / tile).floor().clamp(0, tiles_x).long()
    x1 = ((u + radius) / tile).floor().add(1).clamp(0, tiles_x).long()
    y0 = ((v - radius) / tile).floor().clamp(0, tiles_y).long()
    y1 = ((v + radius) / tile).floor().add(1).clamp(0, tiles_y).long()
    return x0, x1, y0, y1


@torch.no_grad()
def bin_tiles(means2d, radius, depth, W, H, tile):
    """Enumerate the (Gaussian, tile) intersections and order them by tile index, then depth.

    The official implementation radix-sorts 64-bit integer keys = [tile index | depth]. Likewise, here we build
    key = tile index * N + depth rank and sort once.

    Returns
        gid      (M,) Gaussian index of each intersection (front to back within a tile)
        tile_id  (M,) tile index of each intersection (ascending)
        on_screen (N,) whether it covers at least 1 tile
        tiles_x, tiles_y
    """
    tiles_x = (W + tile - 1) // tile
    tiles_y = (H + tile - 1) // tile
    x0, x1, y0, y1 = _tile_rect(means2d.detach(), radius, tiles_x, tiles_y, tile)
    n = (x1 - x0) * (y1 - y0) * (radius > 0)  # number of tiles covered (0 if off screen or radius 0)

    N = means2d.shape[0]
    ar = torch.arange(N, device=means2d.device)
    gid = torch.repeat_interleave(ar, n)                # repeat Gaussian i n_i times
    k = torch.arange(gid.numel(), device=gid.device) - (torch.cumsum(n, 0) - n)[gid]
    w = (x1 - x0)[gid]                                  # width of the rectangle (in tiles)
    tile_id = (y0[gid] + k // w) * tiles_x + (x0[gid] + k % w)

    rank = torch.empty_like(ar)
    rank[torch.argsort(depth.detach())] = ar            # depth rank (0 is the nearest)
    order = torch.argsort(tile_id * N + rank[gid])
    return gid[order], tile_id[order], n > 0, tiles_x, tiles_y


# ============================================================ 3. α blending

class SegmentedExclusiveCumsum(torch.autograd.Function):
    """Cumulative sum "not including itself" that is reset per segment.

    x (P, M) is laid out in row-major order as one sequence and divided into segments (a segment does not span
    rows). Returns
        y[p, m] = Σ_{start(m) <= j < m} x[p, j]
    where start(m), end(m) are the first and last column indices of the segment that m belongs to.

    Forward: take one cumulative sum over everything (it is 1-D, so the GPU's parallel scan can be used), then
    subtract the value at the segment start.
    Backward: the transpose (adjoint) of this linear map is a "per-segment reverse cumulative sum"
        ∂L/∂x[p, j] = Σ_{j < m <= end(j)} ∂L/∂y[p, m]
    which can be computed in the same way as the forward pass.
    If left to autograd, the backward of the "subtract the value at the segment start" part becomes a float64
    atomicAdd, which is very slow on old GPUs that lack it (Maxwell etc.), so the backward pass is written by hand.
    The global cumulative sum becomes large, so it is computed in float64 to avoid loss of significance.
    """

    @staticmethod
    def forward(ctx, x, seg_start, seg_end):
        ctx.save_for_backward(seg_start, seg_end)
        P, M = x.shape
        excl = torch.cumsum(x.double().reshape(-1), 0).reshape(P, M) - x   # sum from the start up to just before itself
        return (excl - excl.index_select(1, seg_start)).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        seg_start, seg_end = ctx.saved_tensors
        P, M = g.shape
        rev = torch.cumsum(g.double().reshape(-1).flip(0), 0).flip(0).reshape(P, M) - g  # sum from just after itself to the end
        return (rev - rev.index_select(1, seg_end)).to(g.dtype), None, None


@functools.lru_cache(maxsize=None)
def _pixel_offsets(tile, device):
    """Coordinates of each pixel center seen from the tile's top-left (tile*tile, 2). Order is row-major (ly, lx)."""
    ly, lx = torch.meshgrid(torch.arange(tile), torch.arange(tile), indexing="ij")
    return (torch.stack([lx, ly], -1).reshape(-1, 2).float() + 0.5).to(device)


def composite(means2d, conic, opacity, color, tile_xy, tile_local, counts, tile):
    """α-blending of a run of consecutive tiles (a chunk).

    All inputs are per "intersection" (M of them), ordered by tile index, then depth.
        means2d (M,2), conic (M,3), opacity (M,), color (M,C)  C is 3 for color (features such as normals can be
                                                               composited in the same way)
        tile_xy (M,2) pixel coordinates of the top-left of that intersection's tile
        tile_local (M,) tile index within the chunk 0..n_tiles-1 (ascending)
        counts (n_tiles,) number of intersections of each tile
    The pairs of all pixels in a tile (P = tile*tile of them) and intersection m are computed at once
    as a (P, M) tensor.

    Returns
        rgb (n_tiles, P, C)   Σ c_i α_i T_i
        T_final (n_tiles, P)  transmittance after passing all Gaussians (weight of the background color)
    """
    P = tile * tile
    offs = _pixel_offsets(tile, means2d.device)
    dx = offs[:, 0:1] + (tile_xy[:, 0] - means2d[:, 0])[None]   # (P, M) pixel - center
    dy = offs[:, 1:2] + (tile_xy[:, 1] - means2d[:, 1])[None]

    # α = o · exp(-½ dᵀ Σ'^{-1} d)   (2D version of paper Eqs. (2)(3))
    A, B, C = conic[:, 0], conic[:, 1], conic[:, 2]
    power = -0.5 * (A * dx * dx + C * dy * dy) - B * dx * dy
    alpha = (opacity * torch.exp(power)).clamp(max=ALPHA_MAX)
    alpha = torch.where(alpha >= ALPHA_MIN, alpha, 0.0)

    # Compute the transmittance T_i = Π_{j<i}(1-α_j) as a cumulative product "per pixel, within a tile".
    # In each row (= pixel) of (P, M), the intersections of the same tile are contiguous and ordered front to back,
    # so taking a "cumulative sum that is reset per tile segment" in log space gives log T_i.
    log1m = torch.log1p(-alpha)                                       # log(1-α) <= 0
    starts = torch.cumsum(counts, 0) - counts
    T = torch.exp(SegmentedExclusiveCumsum.apply(
        log1m, starts[tile_local], (starts + counts - 1)[tile_local]))  # (P, M)

    n_tiles = counts.shape[0]
    weight = alpha * T                                                # contribution of each Gaussian α_i T_i
    rgb = torch.zeros(P, n_tiles, color.shape[-1], device=weight.device).index_add(
        1, tile_local, weight[..., None] * color[None])
    log_T = torch.zeros(P, n_tiles, device=weight.device).index_add(1, tile_local, log1m)
    return rgb.transpose(0, 1), torch.exp(log_T).transpose(0, 1)


def _chunks(counts, cap):
    """Split the tiles into contiguous ranges whose total intersection count is at most cap -> [(t0, t1), ...]"""
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
    """Turn a set of 2D Gaussians into an image. radius is the radius [px] used for tile assignment.

    max_entries: upper limit on the number of elements of the (pixel × intersection) tensor built at once. If it is
    exceeded, the tiles are split into several chunks and processed in turn, and each chunk is recomputed during
    backpropagation with gradient checkpointing (keeping all intermediate tensors would not fit in VRAM).
    Each element uses about 60-70 bytes, counting what is saved for backpropagation and temporary storage.

    color is (N,C). bg has length C (or is a scalar).

    Returns
        image (H,W,C), alpha (H,W) accumulated opacity, on_screen (N,)
    """
    gid, tile_id, on_screen, tiles_x, tiles_y = bin_tiles(means2d, radius, depth, W, H, tile)
    n_tiles = tiles_x * tiles_y
    P = tile * tile

    # Gather the attributes per intersection
    # (this is differentiable: gradients are accumulated into the original Gaussians)
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

    rgb = rgb + T[..., None] * bg  # the background shows through by the amount of remaining transmittance

    def to_image(x):  # (tiles_y*tiles_x, P, C) -> (H, W, C)
        C = x.shape[-1]
        x = x.reshape(tiles_y, tiles_x, tile, tile, C).permute(0, 2, 1, 3, 4)
        return x.reshape(tiles_y * tile, tiles_x * tile, C)[:H, :W]

    image = to_image(rgb)
    alpha = 1.0 - to_image(T[..., None])[..., 0]
    return image, alpha, on_screen


# ============================================================ naive implementation for testing

def rasterize_dense_reference(means2d, conic, opacity, color, radius, depth, W, H, bg, tile=16):
    """Compute all Gaussians × all pixels by brute force and composite front to back (for checking rasterize).

    Does not use tiling, sorting or the log-space cumulative sum, and computes T_i with cumprod as in the textbook.
    However, the cutoff rule "contributes only to the pixels of tiles covered by the 3σ rectangle" is kept the same
    as in rasterize.
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
    return rgb.reshape(H, W, -1), (1 - T[-1]).reshape(H, W)
