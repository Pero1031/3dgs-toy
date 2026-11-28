# -*- coding: utf-8 -*-
"""
Predicted glossiness of the truth and of 3DGS renders, with the cue model of D:\\project\\gloss-perception-lab.

Runs in the lab's environment (it imports the lab's code):

    D:\\project\\gloss-perception-lab\\.venv\\Scripts\\python.exe experiments/gloss_percept/lab_cues.py

The model is the lab's best one under shape hold-out ("cues: trio + B", notes/handoff_stereo_highlight.md):
highlight cues on the region where the senior's gloss map P exceeds the threshold of runs/highlight_cues/cues_all9_q95.json
(coverage, contrast, sharp_grad, sharp_scale, has_hl) plus threshold-free quantiles of P and of luminance over the object,
in a ridge regression with balanced weights, fitted here on all Serrano images of the nine geometries (glossiness,
rescaled 1-7 -> 0-1). The object mask is the truth's silhouette eroded by 10 px, as the lab's geometry masks are.
Background cues are not used (our background is a uniform grey).

Input: data/gloss_percept/lab_input/ (export_display.py). Output: lab_cues.csv next to it (cues and the prediction per
image) and a table per set: mean predicted glossiness on the 1-7 scale.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(r"D:\project\gloss-perception-lab")
sys.path.insert(0, str(LAB))
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.cue_analysis import PQ, YR  # noqa: E402
from scripts.cue_model import TRIO, fill, fit_predict, load  # noqa: E402
from scripts.highlight_cues import cues_for_batch, erode  # noqa: E402
from src.models.features import GlossFeatures  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "gloss_percept" / "lab_input"
FEATS = TRIO + PQ + YR
ERODE = 10


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(LAB / "runs" / "highlight_cues" / "cues_all9_q95.json"))
    feat = GlossFeatures(meta["sigma_max"], sigma_step=1).to(dev)
    with open(IN / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ours = {k: [] for k in FEATS}
    with torch.no_grad():
        for r in rows:
            x = torch.from_numpy(np.asarray(Image.open(IN / r["path"]).convert("RGB"), np.float32) / 255).permute(2, 0, 1)
            m = torch.from_numpy(np.asarray(Image.open(IN / r["mask"])) > 127)
            obj = erode(m[None], ERODE)[0].to(dev)
            c, _, _ = cues_for_batch(x[None].to(dev), feat, obj, meta["threshold"], None)
            for k in FEATS:
                ours[k].append(float(c[k][0]))
    # Serrano: all nine geometries, glossiness
    paths, y, sfeats = load(str(LAB / "data" / "serrano" / "manifest_all9.csv"), str(LAB / "runs" / "highlight_cues" / "cues_all9_q95.csv"))
    sfeats = fill({k: sfeats[k] for k in FEATS})
    o = {k: np.array(v) for k, v in ours.items()}
    o["sharp_scale"] = np.where(np.isnan(o["sharp_scale"]), np.nanmax(sfeats["sharp_scale"]), o["sharp_scale"])
    o = {k: np.nan_to_num(v, nan=0.0) for k, v in o.items()}
    Xs = np.stack([sfeats[k] for k in FEATS], 1)
    Xo = np.stack([o[k] for k in FEATS], 1)
    pred = 1 + 6 * fit_predict("ridge", Xs, y["glossiness"], Xo)
    with open(IN / "lab_cues.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0]) + FEATS + ["pred_glossiness"])
        for i, r in enumerate(rows):
            w.writerow(list(r.values()) + [f"{o[k][i]:.6g}" for k in FEATS] + [f"{pred[i]:.4f}"])
    # summary per set
    sets = {}
    for i, r in enumerate(rows):
        sets.setdefault((r["shape"], r["material"], r["kind"]), []).append(i)
    print(f"{'shape':8}{'material':12}{'kind':13}{'n':>3}{'pred gloss':>11}{'sd':>6}{'has_hl':>8}{'coverage':>9}{'contrast':>9}")
    for (s, m, k), idx in sorted(sets.items()):
        p = pred[idx]
        print(f"{s:8}{m:12}{k:13}{len(idx):3}{p.mean():11.2f}{p.std():6.2f}{o['has_hl'][idx].mean():8.2f}"
              f"{o['coverage'][idx].mean():9.4f}{o['contrast'][idx].mean():9.3f}")
    print(f"-> {IN / 'lab_cues.csv'}")


if __name__ == "__main__":
    main()
