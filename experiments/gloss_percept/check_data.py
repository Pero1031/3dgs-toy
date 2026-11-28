# -*- coding: utf-8 -*-
"""
Sanity checks of one condition's multi-view data (make_stimuli.py views), to catch rendering problems before training.

    python experiments/gloss_percept/check_data.py data/gloss_percept/sphere__coat_r0.05

Prints per split: non-finite and negative pixels, the object's coverage (mask) per view, the object's mean and 99.9th
percentile luminance per view (outlier views are flagged), and isolated very bright pixels ("fireflies": a pixel more
than 20x the median of its 3x3 neighbours and above the 99.9th percentile). Writes check.png: thumbnails of 8 training
and all test views (one exposure, sRGB curve, clipped at 1).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gs.dataset import read_pfm  # noqa: E402

LUM = np.array([0.2126729, 0.7151522, 0.0721750])


def fireflies(Y, p999):
    pad = np.pad(Y, 1, mode="edge")
    nb = np.stack([pad[i:i + Y.shape[0], j:j + Y.shape[1]] for i in range(3) for j in range(3) if (i, j) != (1, 1)])
    return int(((Y > 20 * np.median(nb, 0)) & (Y > p999)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datadir")
    ap.add_argument("--pattern", default="view_*[0-9].pfm", help="raw renders to check (t_*[0-9].pfm for gloss_motion)")
    a = ap.parse_args()
    d = Path(a.datadir)
    import imageio.v2 as imageio
    thumbs = []
    for split in ("train", "test", "test_seed2"):
        files = sorted((d / "raw" / split).glob(a.pattern))
        if not files:
            continue
        cov, mean, p999, ff, nonfin, neg = [], [], [], 0, 0, 0
        for f in files:
            x = read_pfm(str(f)).astype(np.float64)
            m = read_pfm(str(f).replace(".pfm", "_mask.pfm"))[..., 0]
            Y = x @ LUM
            nonfin += int((~np.isfinite(x)).any(-1).sum())
            neg += int((x < -1e-4).any(-1).sum())
            obj = m > 0.99
            cov.append(obj.mean())
            mean.append(Y[obj].mean())
            p999.append(np.percentile(Y[obj], 99.9))
            ff += fireflies(np.where(obj, Y, 0), np.percentile(Y[obj], 99.9))
            if split == "test" or (split == "train" and len(thumbs) < 8 and f.name.endswith(("000.pfm", "015.pfm"))):
                thumbs.append(read_pfm(str(d / split / f.name)).astype(np.float64))
        cov, mean, p999 = map(np.array, (cov, mean, p999))
        z = np.abs(mean - np.median(mean)) / (1.4826 * np.median(np.abs(mean - np.median(mean))) + 1e-12)
        print(f"{split:10} {len(files):3} views | non-finite px {nonfin}, negative px {neg} | coverage "
              f"{cov.min():.3f}-{cov.max():.3f} | object mean Y {mean.min():.4f}-{mean.max():.4f} (median {np.median(mean):.4f}) "
              f"| p99.9 Y {p999.min():.2f}-{p999.max():.2f} | fireflies {ff} ({ff / len(files):.1f}/view)")
        for i in np.where(z > 5)[0]:
            print(f"   outlier view {files[i].name}: mean Y {mean[i]:.4f} (robust z {z[i]:.1f})")
    if thumbs:
        k = 0.18 / np.median(np.concatenate([t[..., 1].ravel() for t in thumbs]))
        rows = [np.concatenate(thumbs[i:i + 6], 1) for i in range(0, len(thumbs) - len(thumbs) % 6, 6)]
        img = np.concatenate(rows, 0) * k
        disp = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.clip(img, 0.0031308, None) ** (1 / 2.4) - 0.055)
        imageio.imwrite(d / "check.png", (np.clip(disp, 0, 1) * 255 + 0.5).astype(np.uint8))
        print(f"thumbnails -> {d / 'check.png'}")


if __name__ == "__main__":
    main()
