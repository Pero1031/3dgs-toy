# -*- coding: utf-8 -*-
"""
Predicted glossiness from a linear readout of CLIP image embeddings, trained on Serrano et al. 2021's ratings.

The image encoder is the CLIP ViT-L/14 of Zero-1-to-3 XL (image_embeds, 768 values), with the preprocessing of
D:\\project\\image-to-3d-diffusion (translucency readout: 256 px, bicubic to 224, CLIP normalisation). The readout is a
ridge regression on standardised embeddings, its strength chosen by 5-fold cross-validation over material groups
(gloss-perception-lab's material_groups: an edited MERL material stays with its source). It is scored by that
cross-validation and by leaving one of the nine geometries out, then refitted on the whole subset. The subset holds
the same number of images in every geometry x illumination cell (PER_CELL x 81 = 5,022 of the 42,120): the learning
curve flattens by about 2,000 images.

Before reading the 3DGS renders, the readout must order the truth's ladder of gloss levels (matte < coat 0.5 < 0.3 <
0.15 < 0.05) on each shape and background; if it does not, the 3DGS numbers are not interpreted.

Runs in the environment of image-to-3d-diffusion, which holds the weights:

    set HF_HOME=D:\\project\\image-to-3d-diffusion\\data\\hf
    D:\\project\\image-to-3d-diffusion\\.venv\\Scripts\\python.exe experiments/gloss_percept/clip_readout.py

Input: data/gloss_percept/clip_input/{grey,env}/ (export_display.py --background grey|env --crop 0.8).
-> data/gloss_percept/clip/ (embeddings), experiments/gloss_percept/results/clip_readout.json
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(r"D:\project\gloss-perception-lab")
SERRANO = LAB / "data" / "serrano"
IN = ROOT / "data" / "gloss_percept" / "clip_input"
CACHE = ROOT / "data" / "gloss_percept" / "clip"
OUT = ROOT / "experiments" / "gloss_percept" / "results" / "clip_readout.json"
ALPHAS = [1, 10, 100, 1e3, 1e4, 1e5]
PER_CELL = 62               # images per geometry x illumination cell: 62 x 81 = 5,022 of the 42,120
SEED = 0
CURVE = [250, 500, 1000, 2000, 3500]   # training-set sizes of the learning curve (within the subset)
LADDER = ["matte", "coat_r0.5", "coat_r0.3", "coat_r0.15", "coat_r0.05"]
dev = "cuda"


class Encoder:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from transformers import CLIPVisionModelWithProjection
        repo = snapshot_download("bennyguo/zero123-xl-diffusers", local_files_only=True)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(repo, subfolder="image_encoder", variant="fp16",
                                                                  torch_dtype=torch.float32).to(dev).eval()
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, paths, batch=48, log_every=50):
        out = []
        for b in range(0, len(paths), batch):
            ims = [np.asarray(Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)) for p in paths[b:b + batch]]
            x = torch.from_numpy(np.stack(ims)).float().div(255).permute(0, 3, 1, 2).to(dev)
            c = F.interpolate(x, size=224, mode="bicubic", antialias=True).clamp(0, 1)
            out.append(self.clip((c - self.mean) / self.std).image_embeds.float().cpu().numpy())
            if (b // batch) % log_every == 0:
                print(f"  {b + len(ims)}/{len(paths)}", flush=True)
        return np.concatenate(out)


def embed(enc, name, paths):
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["paths"]) == [str(p) for p in paths]:
            return z["emb"]
    CACHE.mkdir(parents=True, exist_ok=True)
    emb = enc(paths)
    np.savez(f, emb=emb, paths=np.array([str(p) for p in paths]))
    return emb


def serrano():
    """a random subset, the same number of images in every geometry x illumination cell (PER_CELL x 81)"""
    sys.path.insert(0, str(LAB))
    from src.data.datasets import material_groups, material_of
    with open(SERRANO / "manifest_all9.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cells = {}
    for r in rows:
        cells.setdefault(re.match(r"\[(.+?)\]\[(.+?)\]\[", r["path"]).groups(), []).append(r)
    rng = np.random.default_rng(SEED)
    rows = [r for k in sorted(cells) for r in rng.permutation(cells[k])[:PER_CELL]]
    paths = [r["path"] for r in rows]
    y = np.array([float(r["glossiness"]) for r in rows])
    mats = [material_of(p) for p in paths]
    gmap = material_groups(sorted(set(mats)))
    groups = np.array([gmap[m] for m in mats])
    geoms = np.array([re.match(r"\[.+?\]\[(.+?)\]\[", p).group(1) for p in paths])
    return paths, y, groups, geoms


def ridge(alpha):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def main():
    from sklearn.model_selection import GroupKFold
    paths, y, groups, geoms = serrano()
    enc = Encoder()
    print(f"Serrano: {len(paths)} images ({PER_CELL} per geometry x illumination, seed {SEED})", flush=True)
    X = embed(enc, f"serrano_sub{PER_CELL}_s{SEED}", [SERRANO / "images" / p for p in paths])
    # strength by material-group cross-validation
    cv = {}
    for a in ALPHAS:
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            pred[te] = ridge(a).fit(X[tr], y[tr]).predict(X[te])
        cv[a] = (float(np.corrcoef(pred, y)[0, 1]), pred)
        print(f"alpha {a:g}: material-group CV r = {cv[a][0]:.3f}", flush=True)
    alpha = max(cv, key=lambda a: cv[a][0])
    # leave one geometry out
    loso = {}
    for g in sorted(set(geoms)):
        te = geoms == g
        p = ridge(alpha).fit(X[~te], y[~te]).predict(X[te])
        loso[g] = float(np.corrcoef(p, y[te])[0, 1])
    print(f"chosen alpha {alpha:g}; leave-one-geometry-out r: mean {np.mean(list(loso.values())):.3f} "
          + ", ".join(f"{g} {r:.2f}" for g, r in loso.items()), flush=True)
    # learning curve: material-group folds; in each, train on a random subset of n training images (3 repeats)
    rng = np.random.default_rng(1)
    curve = {}
    for n in CURVE:
        rs = []
        for rep in range(3):
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, groups):
                sub = rng.choice(tr, size=min(n, len(tr)), replace=False)
                pred[te] = ridge(alpha).fit(X[sub], y[sub]).predict(X[te])
            rs.append(float(np.corrcoef(pred, y)[0, 1]))
        curve[n] = (float(np.mean(rs)), float(np.std(rs)))
    curve[len(y) * 4 // 5] = (cv[alpha][0], 0.0)
    print("learning curve (training images: material-group CV r): "
          + ", ".join(f"{n}: {m:.3f}" for n, (m, s) in curve.items()), flush=True)
    model = ridge(alpha).fit(X, y)
    res = {"n_images": len(y), "alpha": alpha, "cv_r": cv[alpha][0], "loso_r": loso,
           "curve": {str(k): v for k, v in curve.items()}, "sets": {}}
    for bg in ("grey", "env"):
        with open(IN / bg / "manifest.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        E = embed(enc, f"ours_{bg}", [IN / bg / r["path"] for r in rows])
        pred = model.predict(E)
        sets = {}
        for r, p in zip(rows, pred):
            sets.setdefault((r["shape"], r["material"], r["kind"]), []).append(float(p))
        res["sets"][bg] = {"|".join(k): {"mean": float(np.mean(v)), "sd": float(np.std(v)), "n": len(v)} for k, v in sets.items()}
        print(f"\nbackground: {bg}")
        for shape in sorted({k[0] for k in sets}):
            lad = [np.mean(sets[(shape, m, "truth")]) for m in LADDER]
            ok = all(b > a for a, b in zip(lad, lad[1:]))
            print(f"  {shape:7} truth ladder " + "  ".join(f"{m} {v:.2f}" for m, v in zip(LADDER, lad))
                  + f"   -> {'ordered' if ok else 'NOT ordered'}")
            for m in ("coat_r0.05", "coat_r0.15"):
                t, s2 = np.mean(sets[(shape, m, "truth")]), np.mean(sets[(shape, m, "truth_seed2")])
                g = sets.get((shape, m, "3dgs"))
                if g:
                    print(f"     {m:11} truth {t:.2f}  seed2 {s2:.2f}  3DGS {np.mean(g):.2f} (sd {np.std(g):.2f})  "
                          f"diff {np.mean(g) - t:+.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
