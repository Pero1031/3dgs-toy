# Dynamic scenes: three representations against the true motion

Three dynamic-scene representations were implemented and compared on toy data whose motion is known in
closed form (`make_toy_dynamic.py`).

| | Representation | Implementation |
|---|---|---|
| Deformation field | One canonical set of Gaussians, shifted per time step by an MLP (simplified Deformable 3D Gaussians) | `gs/deform.py` |
| Spacetime | Each Gaussian has a time μ_t, temporal width σ_t, and velocity v. It appears only around that time and moves in a straight line (simplified Spacetime Gaussians) | `gs/spacetime.py` (`train_dynamic.py --spacetime`) |
| Trajectory | Count, colour, opacity, and size are fixed. Only the position is stored per time knot (simplified Dynamic 3D Gaussians). Covered in the [addition below](#addition-the-trajectory-representation) | `gs/trajectory.py` (`train_dynamic.py --trajectory`) |

## Evaluation (`motion_eval.py`)

The true displacement Φ of a point attached to an object is known (for the sphere, the change of centre and
radius; for the cube, a translation). For Gaussians near a surface, the position after Δ is compared with Φ.

- **ratio** = motion error / true displacement (ratio of opacity-weighted means). 0 means exactly right, 1 means the same as not moving
- **normal / tangential**: the ratio with error and displacement split into the surface normal and tangential directions
- **alive**: the fraction of the opacity present at time t that still remains after Δ (always 1 for the deformation field)
- **cover**: mean distance from points on the true surface to the nearest Gaussian. Large when the object has disappeared

```bash
.venv/Scripts/python experiments/dynamic/motion_eval.py --ckpts logs/dyn_toy/ckpt_007000.pt logs/dyn_toy_st/ckpt_007000.pt logs/dyn_toy_static/ckpt_007000.pt
.venv/Scripts/python experiments/dynamic/motion_eval.py --orbit --ckpts logs/dyn_orbit/ckpt_005000.pt logs/dyn_orbit_st/ckpt_005000.pt logs/dyn_orbit_static/ckpt_005000.pt
```

## Results

Training settings are the same as in the "dynamic" section of the top-level README (256 px, 100 training
frames with one view per time step, 20 test frames at times not in the training set).
Numbers are in `motion_toy.json` / `motion_orbit.json`. The ratio is for Δ=0.1 (one tenth of the whole duration).
One training run each (seed 0).

**Note: every number on this page was measured on data whose ground-truth images had aliased edges**
(`make_toy_dynamic.py --supersample 1`; the default is now 4). Absolute PSNR values come out low.
See "Addition: antialiased ground-truth images" at the end.

### Small motion (the sphere bounces and pulses, the cube swings ±15°; 7000 iterations)

| | test PSNR | Gaussians | Sphere ratio (normal / tangential) | Cube ratio (normal / tangential) | alive |
|---|---|---|---|---|---|
| Deformation field | **25.7 dB** | 39,946 | 0.46 (0.18 / 0.74) | 0.13 (0.27 / 0.13) | 1.00 |
| Spacetime | 23.7 dB | 243,811 | 0.78 (0.49 / 1.08) | 0.63 (1.03 / 0.64) | 0.24 |
| Static | 14.1 dB | 29,332 | 1.00 | 1.00 | 1.00 |

### Large motion (`--orbit`, the cube travels half a circle around the sphere; 5000 iterations)

| | test PSNR | Gaussians | Sphere ratio (normal / tangential) | Cube ratio (normal / tangential) | Cube cover |
|---|---|---|---|---|---|
| Deformation field (warm-up 1000) | 15.1 dB | 25,802 | 0.50 (0.25 / 0.74) | 1.04 (1.25 / 1.06) | 0.464 |
| Spacetime | **23.0 dB** | 129,856 | 0.81 (0.52 / 1.11) | 0.26 (0.41 / 0.26) | 0.046 |
| Static | 11.8 dB | 19,599 | 1.00 | 1.00 | 0.387 |

## Findings

1. **Strengths are opposite.** On small motion the deformation field is 2 dB ahead; on large motion spacetime
   is 8 dB ahead. The deformation field loses the cube on `--orbit` (cover 0.46 is worse than static).
   Spacetime needs no canonical shape, so it stays around 23 dB on both datasets and places the cube
   correctly (cover 0.05).
2. **Even with the deformation field, about 70% of the tangential motion on the sphere is wrong.** The ratio is
   0.18 in the normal direction but 0.74 in the tangential direction. Inside a uniformly coloured stripe,
   a Gaussian sliding along the surface does not change the image, so the image does not determine the motion
   (the aperture problem). This failure is invisible in the 25.7 dB PSNR. On the cube, where edges and
   corners determine the motion, the ratio goes down to 0.13.
3. **Spacetime Gaussians barely learn motion.** The median temporal width σ_t shrinks by itself to 0.022
   (about two frames at the training frame interval of 0.01), and only a quarter of the opacity remains
   after Δ=0.1. Only a few frames see each Gaussian, so "appear and disappear where those few frames need it"
   is learned before "move with the object". The sphere ratio is 0.78, close to not moving. Training PSNR
   is 32 dB and test PSNR is 23.7 dB, a clear overfit to the training views. The count also grows to
   6 times that of the deformation field.
4. The cube in the large-motion data is the exception: spacetime reaches a ratio of 0.26 there. The likely
   reason is that the displacement between frames (about 0.05) is not negligible against the cube size (0.6),
   so even a few frames cannot be fitted without the right velocity.

## Addition: a lower bound on the temporal width, and velocity regularization

Two remedies were tried for the failures above (spacetime overfitting, tangential sliding in the deformation
field). Small-motion data, 7000 iterations.

- **Lower bound on the temporal width** (`--min-t-scale`, spacetime only): σ_t = bound + exp(·), so it cannot get shorter than the bound
- **Velocity regularization** (`--lambda-smooth`, either representation): align the velocity with nearby
  Gaussians (8 neighbours). The loss is the mean of squared velocity difference / squared distance
  (the squared gradient of the velocity field; the same idea as the smoothness term of Horn–Schunck optical flow).
  ARAP, which preserves distances, was not used: on this sphere, whose radius pulses, it would penalize the true motion

| | test PSNR | Gaussians | Sphere ratio (normal / tangential) | Cube ratio (normal / tangential) | Median σ_t |
|---|---|---|---|---|---|
| Deformation field | 25.7 dB | 39,946 | 0.46 (0.18 / 0.74) | 0.13 (0.27 / 0.13) | |
| Deformation field + regularization 1e-4 | **26.3 dB** | 33,296 | 0.38 (0.15 / 0.61) | 0.09 (0.14 / 0.09) | |
| Deformation field + regularization 1e-3 | 25.2 dB | 34,489 | **0.34** (0.20 / 0.49) | **0.08** (0.13 / 0.08) | |
| Spacetime | 23.7 dB | 243,811 | 0.78 (0.49 / 1.08) | 0.63 (1.03 / 0.64) | 0.022 |
| Spacetime + bound 0.05 | 23.9 dB | 139,788 | 0.73 (0.49 / 1.00) | 0.59 (0.84 / 0.63) | 0.055 |
| Spacetime + bound 0.1 | 22.7 dB | 96,135 | 0.90 (0.76 / 1.07) | 0.66 (0.85 / 0.69) | 0.104 |
| Spacetime + bound 0.05 + regularization 1e-3 | 21.8 dB | 123,663 | 0.90 (0.94 / 0.90) | 0.95 (1.02 / 0.95) | 0.053 |

5. **Velocity regularization works for the deformation field.** The tangential ratio on the sphere goes
   0.74 → 0.61 → 0.49 as the weight increases. At weight 1e-4, PSNR also rises by 0.5 dB. At 1e-3 the motion
   fits better still, but PSNR drops by 0.5 dB and the normal direction gets slightly worse (over-smoothing).
   Even so, half of the true displacement remains as tangential error.
6. **A lower bound on the temporal width alone does not make spacetime learn motion.** Lifetimes stick to the
   bound (median 0.055) and the count drops by 40%, but the ratio is nearly unchanged (0.78 → 0.73).
   With a bound of 0.1, straight-line motion can no longer represent the bouncing sphere, and both PSNR
   and ratio get worse.
7. **Adding velocity regularization to spacetime makes it worse.** The median speed fell from 2.1 to 0.5.
   For Gaussians that were not following the object in the first place, the cheapest way to "agree with
   the neighbours" is for everyone to stop, so the result is motionless Gaussians that appear and disappear
   (ratio 0.9, PSNR 21.8 dB). Before aligning velocities, something has to make the same Gaussian keep
   following the same point on the object (fixing count and colour and tracking the position per time step
   as in Dynamic 3D Gaussians, or a loss that uses 2D tracking results).

## Addition: the trajectory representation

Following finding 7, a representation in which the same Gaussian keeps following the same point (`--trajectory`) was added.

- **Representation**: each Gaussian has a position offset for each of 21 time knots, with linear interpolation
  between knots. Count, colour, opacity, size, and orientation do not depend on time
- **Training order** (the experiments in this section; `--knots-init 3`): go in time order. Start with only
  the frames up to the first 3 knots, then from iteration 1000 start using the next knot every 150 iterations.
  A new knot is initialized at the position extrapolated from the previous two. The reason is that when all
  times are trained together from the start, there is no gradient pulling the Gaussians of a not-yet-fitted
  time towards the object
- The original paper uses 27 synchronized cameras and rigidity regularization. Here there is one view per
  time step, and the only regularization is the velocity regularization above

### Small motion (7000 iterations)

| | test PSNR | Gaussians | Sphere ratio (normal / tangential) | Cube ratio (normal / tangential) |
|---|---|---|---|---|
| Trajectory | 13.2 dB | 75,975 | 3.51 (2.79 / 4.61) | 2.66 (4.45 / 2.63) |
| Trajectory + regularization 1e-4 | 19.3 dB | 54,058 | 0.79 (0.42 / 1.18) | 0.57 (0.75 / 0.65) |
| Trajectory + regularization 1e-3 | 18.8 dB | 54,063 | 0.85 (0.51 / 1.18) | 0.65 (1.03 / 0.67) |
| Trajectory + regularization 1e-4, 11 knots | 17.0 dB | 64,667 | 0.72 (0.47 / 1.03) | 0.48 (0.86 / 0.45) |
| (repeated) Deformation field | **25.7 dB** | 39,946 | 0.46 (0.18 / 0.74) | 0.13 (0.27 / 0.13) |
| (repeated) Static | 14.1 dB | 29,332 | 1.00 | 1.00 |

### Large motion (`--orbit`)

| | iter | test PSNR | Gaussians | Sphere ratio (normal / tangential) | Cube ratio (normal / tangential) | Cube cover |
|---|---|---|---|---|---|---|
| Trajectory | 5000 | 9.1 dB | 65,150 | 3.72 (2.70 / 5.22) | 1.49 (1.87 / 1.45) | 0.129 |
| Trajectory + regularization 1e-4 | 5000 | 9.6 dB | 51,951 | 1.07 (0.87 / 1.40) | 0.51 (0.52 / 0.55) | 0.103 |
| Trajectory + regularization 1e-4 | 7000 | 17.6 dB | 51,930 | 0.81 (0.49 / 1.13) | 0.39 (0.68 / 0.35) | 0.054 |
| (repeated) Deformation field | 5000 | 15.1 dB | 25,802 | 0.50 (0.25 / 0.74) | 1.04 (1.25 / 1.06) | 0.464 |
| (repeated) Spacetime | 5000 | **23.0 dB** | 129,856 | 0.81 (0.52 / 1.11) | 0.26 (0.41 / 0.26) | 0.046 |

The 7000-iteration row continues densification until iteration 5000 (as for the small-motion data).
The two repeated rows are still at 5000 iterations, so training lengths do not match.

8. **Without regularization it collapses.** Test PSNR is lower than static, and the ratio is around 3
   (moving several times the true distance at random). Training PSNR rises to about 24 dB, so this is
   overfitting to the training views. Positions can move freely at each knot, yet only about 10 frames
   constrain each knot.
9. **Velocity regularization works for this representation.** PSNR goes 13.2 → 19.3 dB and the sphere ratio
   3.51 → 0.79. The same loss that backfired for spacetime works for a representation that keeps tracking
   (as finding 7 predicted). But a 10× larger weight does not help, and it does not reach the deformation
   field (25.7 dB, 0.46). The tangential ratio on the sphere is 1.18, worse than not moving.
10. **It does not lose the cube even under large motion.** On `--orbit` the cube cover is 0.054 (deformation
    field 0.464, spacetime 0.046) and the ratio is 0.39. PSNR 17.6 dB is above the deformation field's
    15.1 dB but below spacetime's 23.0 dB.
11. **Gaussians added while training in time order stay unless cleaned up later.** With the 5000-iteration
    setting, densification stops at iteration 2400, when only 60% of the times have entered training.
    The opacity reset at iteration 3000 does not apply either, and the test PSNR stays at 9.6 dB. When
    densification continues until iteration 5000, the validation PSNR jumps from 10 to 16 dB right after
    a reset cuts the count from 73,558 to around 30,000. The small-motion data shows the same jump.
12. Reducing the knots to 11 fits the motion slightly better but lowers PSNR by 2.3 dB. "Few frames per knot"
    (finding 8) alone does not explain this, and the cause was not isolated (→ finding 13 below shows that
    the time-ordered procedure was the cause).

## Addition: error-free 2D tracking, and revisiting time-ordered training

Real monocular methods add the output of a pretrained point tracker to the loss. Since the toy motion is
known in closed form, the upper bound for perfectly correct tracking can be measured (`--lambda-track`).
For Gaussians that are near a surface at time t and on the side visible from the camera, their position in
another frame within a time difference of 0.1 is projected onto that frame's image and compared with the
projection of the true destination (image-plane position only; displacement along the view direction is
not penalized). The weight is 1 (gradients of about the same size as the image loss).

Small-motion data, 7000 iterations. "Floating" is the fraction of opacity located 0.1 or more away from the true surface.

| | test PSNR | Gaussians | Sphere ratio (normal / tangential) | Cube ratio (normal / tangential) | Floating |
|---|---|---|---|---|---|
| Deformation field | 25.7 dB | 39,946 | 0.46 (0.18 / 0.74) | 0.13 (0.27 / 0.13) | 0.14 |
| Deformation field + tracking | 25.9 dB | 49,660 | 0.33 (0.14 / 0.51) | 0.08 (0.18 / 0.08) | 0.12 |
| Trajectory (time order) + tracking | 12.1 dB | 77,082 | 2.83 (2.08 / 3.84) | 2.02 (3.87 / 1.96) | 0.89 |
| Trajectory (time order) + regularization 1e-4 | 19.3 dB | 54,058 | 0.79 (0.42 / 1.18) | 0.57 (0.75 / 0.65) | 0.61 |
| Trajectory (all times from the start) + regularization 1e-4 | 24.3 dB | 65,924 | 0.49 (0.28 / 0.70) | 0.36 (0.54 / 0.38) | 0.41 |
| Trajectory (all times from the start) + regularization 1e-4 + tracking | 24.3 dB | 64,543 | 0.45 (0.26 / 0.67) | 0.36 (0.57 / 0.38) | |

Large-motion data (`--orbit`, 7000 iterations, regularization 1e-4):

| | test PSNR | Gaussians | Sphere ratio | Cube ratio | Cube cover |
|---|---|---|---|---|---|
| Trajectory (time order) | 17.6 dB | 51,930 | 0.81 | 0.39 | 0.054 |
| Trajectory (all times from the start) | 20.4 dB | 57,743 | 0.49 | 0.72 | 0.106 |

13. **The main reason the trajectory representation did badly was the time-ordered training procedure.**
    In time order, 60–90% of the opacity sits away from the true surface (0.14 for the deformation field).
    The shape is built and densified from the first 3 knots only (about 11 views), which produces floaters
    that fit just those views, and the picture is matched by rearranging them per time step. Using all
    times from the start gives PSNR 19.3 → 24.3 dB and sphere ratio 0.79 → 0.49, close to the deformation
    field (25.7 dB, 0.46). The original paper can go in time order because it has 27 views at each time.
    Floating is still 0.41, and the cube ratio (0.36) does not reach the deformation field (0.13).
14. **The advantage of time-ordered training remains only for tracking an object that moves a lot.**
    On `--orbit`, using all times from the start gives higher PSNR (20.4 vs 17.6 dB), but the cube ratio is
    0.72, so the cube is hardly tracked (0.39 in time order). The default of `train_dynamic.py` was changed
    to all times from the start.
15. **Even error-free 2D tracking does not remove the sliding.** With the deformation field, the tangential
    ratio on the sphere goes 0.74 → 0.51. That is about the same as velocity regularization 1e-3 (0.49),
    and PSNR hardly changes. For the trajectory representation it does not work in time order (it does not
    apply to Gaussians that are not near a surface), and with all times from the start it only goes
    0.49 → 0.45. Only one weight (1) was tried, so it is not known whether the weight is too small or the
    per-Gaussian form of the loss is weak (real methods compare positions rendered per pixel).

Training times cannot be compared because the runs were executed in parallel (run one at a time, spacetime
takes about 30 minutes and trajectory 36 minutes).

## Addition: antialiased ground-truth images

The data up to this point was rendered with one ray per pixel, so object edges and pattern boundaries were
staircase-shaped (an oversight). The Gaussian rasterizer adds blur, so it cannot reproduce staircase edges.
Only the deformation field was retrained on data rendered with 4×4 rays per pixel and averaged
(`--supersample 4`) (small-motion data, 7000 iterations, seed 0).

| Ground-truth images | test PSNR | SSIM | Gaussians | Sphere ratio (normal / tangential) | Cube ratio (normal / tangential) |
|---|---|---|---|---|---|
| Aliased (so far) | 25.7 dB | 0.952 | 39,946 | 0.46 (0.18 / 0.74) | 0.13 (0.27 / 0.13) |
| Antialiased | 28.3 dB | 0.966 | 41,800 | 0.49 (0.19 / 0.75) | 0.12 (0.27 / 0.12) |

16. **The image-quality number rises by 2.6 dB, but the motion accuracy does not change.** The sliding along
    the sphere surface remains as it was. The conclusions of this page (ranking of representations, sliding,
    floaters) rest on the motion evaluation, so they are believed to hold as they are. Only the deformation
    field was re-measured, though.
17. **How much of the 2.6 dB is due to scoring was not measured.** In the static scene, 4.7 dB of a 7.5 dB gain
    was "the same model re-scored against antialiased ground truth" (see the top-level README).
    The static scene reaches 37.9 dB on antialiased data, so a gap of about 10 dB to the dynamic scene remains.
    That gap comes from the motion representation and the one-view-per-time-step condition, and the quality
    of the ground-truth images does not close it.

## What to check next

- Tracking loss: raise the weight; compare positions rendered per pixel (finding 15)
- Trajectory: the remaining floaters (0.41) and the gap in the cube ratio. Combine "build the shape from all times, then track in time order"
- Deformation field: how the remaining tangential sliding changes with the fineness of the pattern, and with a method that carries motion on control points (SC-GS)
- How things break when the ratio between camera speed and object speed changes (monocular ambiguity)
