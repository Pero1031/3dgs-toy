# -*- coding: utf-8 -*-
"""
Checks of the arbitrary-degree SH basis (sh_basis):
  1. Up to degree 3 it agrees with eval_sh, which uses the same formulas as the official implementation
  2. Orthonormal (integrating over the sphere gives the identity matrix)

    python tests/test_sh.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gs.sh import eval_sh, sh_basis  # noqa: E402


def main():
    g = torch.Generator().manual_seed(0)
    dirs = torch.nn.functional.normalize(torch.randn(2000, 3, generator=g, dtype=torch.float64), dim=-1)
    ok = True
    for deg in range(4):
        sh = torch.randn(2000, (deg + 1) ** 2, 3, generator=g, dtype=torch.float64)
        a = eval_sh(deg, sh, dirs)
        b = (sh_basis(deg, dirs)[:, :, None] * sh).sum(1)
        err = (a - b).abs().max().item()
        print(f"deg {deg}: |eval_sh - sh_basis| max = {err:.2e}")
        ok &= err < 1e-10

    # Orthonormality: averaging Y Yᵀ over uniform points on the sphere and multiplying by 4π gives the identity matrix
    deg = 12
    u = torch.nn.functional.normalize(torch.randn(400000, 3, generator=g, dtype=torch.float64), dim=-1)
    Y = sh_basis(deg, u)
    G = 4 * math.pi * Y.T @ Y / len(u)
    err = (G - torch.eye(G.shape[0], dtype=torch.float64)).abs().max().item()
    print(f"deg {deg}: orthonormality error (Monte Carlo integration) max = {err:.3f}")
    ok &= err < 0.05
    print("OK" if ok else "NG")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
