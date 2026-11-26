# 3DGS and 4DGS from scratch, and where they fail

3D Gaussian Splatting (Kerbl et al. 2023) and three dynamic (4DGS) representations, written in PyTorch with no
3DGS library. The differentiable tile rasterizer, which the official code writes in CUDA, is tensor code
differentiated by autograd.

On top of it are controlled experiments with known ground truth. Each one finds where Gaussian splatting fails,
tests the cause, and where possible tries a fix. Everything runs on one Quadro M4000 (8 GB, Maxwell).

| | Result |
|---|---|
| **3DGS from scratch** | The rasterizer matches a naive per-pixel renderer to 1e-6, image and gradients. 37.9 dB on the toy scene |
| **4DGS, three representations** | Deformation field, spacetime Gaussians and per-time trajectories in one code base. D-NeRF `jumpingjacks`: 34.7 dB, SSIM 0.987 |
| **Motion** | High PSNR with wrong motion: the deformation field slides along the surface (error 0.74 of the true displacement), and PSNR does not show it |
| **Gloss** | SH cannot draw sharp, moving highlights: 3DGS keeps 11% of them. A CLIP perceptual loss only fools CLIP. A deferred-reflection model (3DGS-DR) brings back 91% |
| **Colour** | On wide-gamut scenes, the clamp of colour at 0 costs exactly what clipping the truth costs (Δu'v' 0.022). PSNR does not show it |
| **Translucency** | A relightable method (R3DG) relights wax as if it were opaque: 2–3× the true shading contrast |

![D-NeRF jumpingjacks: ground truth, render, and the canonical Gaussians](experiments/dynamic/figures/dnerf_jumpingjacks.png)

*Deformation-field 4DGS on D-NeRF `jumpingjacks`, test views at unseen times. Bottom row: the canonical Gaussians
before deformation.*

## Experiments

### [Dynamic scenes: three representations against the true motion](experiments/dynamic/README.md)

The toy motion is known in closed form, so the motion of the Gaussians is scored, not only the images.

- Strengths are opposite. The deformation field leads by 2 dB on small motion, and spacetime Gaussians lead by 8 dB
  on large motion.
- On a uniformly coloured surface, sliding does not change the image (the aperture problem). The deformation field
  gets the tangential motion mostly wrong (0.74), while PSNR looks fine.
- Spacetime Gaussians barely learn motion: their temporal width shrinks to about two frames, and they fit the
  training views by appearing and disappearing.
- Neither aligning neighbouring velocities (0.49) nor an error-free 2D tracking loss (0.51) removes the sliding.

### [Gloss perception: perceptual loss or a reflection model?](experiments/gloss_percept/README.md)

Achromatic objects under an environment map, with a ladder of coat roughness. Gloss is scored with image cues from
psychophysics and with a linear readout of CLIP fitted to human gloss ratings (r = 0.81).

- On a blob with a sharp coat, 3DGS keeps 11% of the highlights, and the skew of the luminance histogram turns
  negative.
- Matching the CLIP readout during training lifts the readout from 2.33 to 3.95 (truth 4.44), but the highlights do
  not come back: the image changes by less than one 8-bit step. The loss is met by fooling CLIP, because SH cannot draw
  the highlights.
- 3DGS-DR (official code, built for sm_52) brings back 91% of the highlights in 36 min. A compiled rasterizer is 18×
  faster than this repository's PyTorch one. What remains is highlight quality: half the sharpness, dimmer peaks.

![Truth, vanilla 3DGS, 3DGS-DR](experiments/gloss_percept/results/png_blob1_r0.05.png)

*Left to right: truth, vanilla 3DGS, 3DGS-DR.*

### [Does 3DGS break on glossy surfaces?](experiments/gloss/README.md)

A gold statue and a diffuse one of the same colour, identical in everything else.

- Gold is 16–17 dB worse, on the training views as well: a lack of capacity, not of generalization.
- A higher SH degree does not help. Per-point fits saturate at 24–27 dB on test views, set by how coarsely each
  point is observed.
- Shared lighting through the reflection direction recovers the light correctly, but loses on this statue:
  38% of the reflections see the statue itself.

### [Colour: does 3DGS get the chromaticity right?](experiments/color/README.md)

Scene-linear ground truth with colours outside sRGB, scored in CIE terms (ΔE00, Δu'v').

- On the gold statue under candle light, most of the chromaticity error of out-of-gamut colours is lost highlight
  energy (the bright, saturated pixels get 24–42% of their energy), not colour handling.
- On a matte scene the model fits, the clamp at 0 costs exactly the clip floor, and removing it brings the error to
  the noise level. Test PSNR (44.2 dB) does not show the difference.

### [Relighting a translucent object with Relightable 3D Gaussians](experiments/relight/README.md)

- Under the training light, novel views keep the look (35–37 dB). Relit, the wax scores 22–30 dB.
- The relit wax has 2–3× the shading contrast of the truth under every light, a cue to opacity. Training five times
  longer does not change it: the material model has no scattering.
- PSNR ranks the conditions differently from the image cues.

A [literature survey of glossy 3DGS methods](experiments/gloss/reports/glossy_3dgs_literature_survey.md) is
organised around the failure modes above.

## Implementation

| File | Contents |
|---|---|
| `gs/rasterize.py` | Projection (EWA, Eq. 5), tile binning and depth sorting with one sort, alpha blending |
| `gs/gaussians.py` | Parameters, densification (clone / split / prune) with the Adam state rebuilt, `.ply` export |
| `gs/render.py`, `gs/sh.py` | Activation → projection → SH colour → rasterization |
| `gs/deform.py`, `gs/spacetime.py`, `gs/trajectory.py` | The three dynamic representations |
| `gs/color.py` | Colour modes: display values, or linear radiance in sRGB (clamped or not) or Rec.2020 |
| `train.py`, `train_dynamic.py`, `render_video.py` | Training and novel-view rendering |
| `tests/` | Rasterizer against a naive per-pixel renderer (image and gradients), `gradcheck` of the hand-written backward |

Differences from the official code:

- Only the forward pass is written. The exception is the transmittance cumulative sum: autograd's backward needs a
  float64 `atomicAdd`, which Maxwell lacks, so its adjoint is written by hand.
- All (pixel, Gaussian) pairs of a tile are one tensor, with no early termination. Tiles are 8 px (official 16 px),
  and above `--max-entries` they are processed in chunks with gradient checkpointing.
- The binning radius is cut where α falls below 1/255. The image is unchanged (tested); only the work shrinks.

| Dynamic flag | Representation | Simplified from |
|---|---|---|
| (default) | One canonical set; an MLP maps position and time to (δx, δq, δs) | Deformable 3D Gaussians (Yang et al., CVPR 2024) |
| `--spacetime` | Each Gaussian has a time, a temporal width and a velocity | Spacetime Gaussians (Li et al., CVPR 2024) |
| `--trajectory` | Fixed Gaussians with a position per time knot | Dynamic 3D Gaussians (Luiten et al., 3DV 2024) |

A pitfall found on the way: the first toy ground truth was aliased (one ray per pixel), and PSNR stalled at 30 dB.
The cause was the scoring, not the model: the rasterizer's 0.3 px² filter cannot draw staircase edges. Scored on
antialiased ground truth, the same model gains 4.7 dB.

## D-NeRF `jumpingjacks`

Deformation field, 400 px, 200 training frames, 20 test frames at unseen times.

| Iterations | Test PSNR / SSIM | At t = 0 and 1 (arms raised) | Gaussians | Time |
|---|---|---|---|---|
| 7,000 | 29.7 dB / 0.973 | 22.0 / 19.2 dB | 30,133 | 35 min |
| 20,000 | 34.7 dB / 0.987 | 29.3 / 28.3 dB | 55,824 | 141 min |

At 7,000 iterations the deformation MLP's learning rate decays before the arms have moved. The canonical arms become
a haze, and Gaussians are shrunk away per time step instead of being moved. At 20,000 iterations they are moved. The
hands, thinnest and fastest, are still broken. One scene at half resolution: a check that the implementation works on
a benchmark, not a reproduction of published numbers.

## Usage

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118   # sm_52 needs CUDA 11.8
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python tests/test_rasterize.py                                   # rasterizer check
.venv/Scripts/python make_toy_dataset.py --out data/toy_sphere                 # static toy scene
.venv/Scripts/python train.py --datadir data/toy_sphere --expname toy --iters 7000
.venv/Scripts/python train.py --datadir data/nerf_synthetic/lego --expname lego --half-res --init-points 50000
.venv/Scripts/python make_toy_dynamic.py                                       # dynamic toy scene
.venv/Scripts/python train_dynamic.py --datadir data/toy_dynamic --expname dyn_toy --iters 7000 --warm-up 1000 --densify-until 5000 --init-extent 2.0
.venv/Scripts/python train_dynamic.py --datadir data/dnerf/data/jumpingjacks --expname dnerf_jj --half-res --iters 20000 --warm-up 600 --densify-until 10000
```

Input is the Blender / D-NeRF format (`transforms_*.json` + images). `point_cloud.ply` uses the official format
and opens in viewers such as SuperSplat.

## License

MIT
