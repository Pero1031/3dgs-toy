# What gloss perception says a "same material in motion" loss must keep

2026-11-26. Notes for designing a perceptual loss whose goal is that observers see the reconstruction as the same
material as the truth while the object moves, not that it matches the truth's pixels. Collected from reviews and the
papers they point to; the primary studies were read through the reviews unless marked. Statements marked
"(inference)" are ours.

Sources read:

- Chadwick & Kentridge (2015, Vision Research 109), "The perception of gloss: a review". Read in the version of
  Chadwick's 2016 Durham PhD thesis (chapter 1.5), since the journal PDF was not accessible.
- Fleming (2017, Annual Review of Vision Science 3), "Material perception", gloss sections.
- Later work: Storrs, Anderson & Fleming (2021, Nature Human Behaviour); Prokott, Tamura & Fleming (2021, JOV);
  Cheeseman, Ferwerda, Morimoto & Fleming (2025, JOV); Morimoto et al. (2026, Nature Human Behaviour; preprint read).

## Image statistics are not the cue

- Histogram or sub-band skewness (Motoyoshi et al. 2007) is treated as refuted. Glossy images with negative skew exist,
  and skew discards the position and orientation of highlights that gloss depends on (Anderson & Kim 2009). The
  adaptation aftereffect did not replicate (Kim & Anderson 2010). Skew explains neither the effect of relief (Wijntjes
  & Pont 2010) nor the interaction of shape and illumination (Olkkonen & Brainard 2010). Fleming (2017): neither
  necessary nor sufficient.
- Later data agree: pixel statistics matched human errors far worse than texture statistics or CNNs (Prokott et al.
  2021); skew did not predict gloss discrimination (Cheeseman et al. 2025).
- The one role left for skew is in the illumination: natural light has a few small bright sources. Under light
  without them, gloss constancy fails and reflections look like texture (Fleming, Dror & Adelson 2003).

## Highlights must fit the shape

- Highlights must sit where the shape and the shading put them and run along the direction of minimum curvature
  (Beck & Prazdny 1981; Fleming, Torralba & Adelson 2004).
- Rotating highlights, or moving them away from the shading peak, lowers perceived gloss; rotation hurts more (Kim,
  Marlow & Anderson 2011; Marlow, Kim & Anderson 2011). Dark "lowlights" in congruent positions also produce gloss
  (Kim, Marlow & Anderson 2012).
- The same intensity gradient looks glossy or matte depending on the 3D shape implied by stereo, motion, texture or
  outline (Marlow & Anderson 2015, 2016).
- Gloss is local: ratings fall with distance from a highlight (Berzhanskaya et al. 2005).

## What predicts the level of gloss

- Perceived coverage, contrast and sharpness of the highlights predict gloss and its failures across relief and
  illumination (Marlow, Kim & Anderson 2012), for a limited set of conditions (Fleming 2014).
- Shape alone changed perceived gloss by up to 80% under one illumination (Marlow et al. 2012).
- Gloss is multidimensional and judged as a whole; observers weight cues differently and lean on whichever cue varies
  most in the stimulus set (Marlow & Anderson 2013).
- With shape, light and view matched within a pair, an image-difference metric (HDR-VDP-3 on the object) predicted
  how discriminable two glosses are (r about 0.87); among specular measures only contrast mattered (Cheeseman et al.
  2025). Fleming (2017): under matched conditions, matching reduces to low-level cues such as highlight size and
  intensity.

## Learned models of gloss are static and blind to congruence

- An unsupervised PixelVAE predicted ratings (R² about 0.84) and constancy failures, but translating the highlights,
  which breaks congruence, did not change its prediction (Storrs et al. 2021).
- Networks of 3–5 layers matched human errors best; rotating highlights barely affected them (Prokott et al. 2021).
- Shallow networks, down to a single filter, beat networks trained on ground truth at predicting ratings (Morimoto et
  al. 2026).
- All of these used still, monocular images.
- (inference) The CLIP readout of gloss_percept is of the same kind: a static predictor of a gloss level. That it was
  fooled by a change that is not a highlight fits this.

## Motion

- On a rotating object, highlights stay roughly fixed relative to the viewer while the surface turns under them
  (Hurlbert, Cumming & Parker 1991).
- Reflections painted onto a rotating object look shiny when still and matte, like a pattern, once it moves (Hartung &
  Kersten 2002; Doerschner et al. 2011). In Doerschner et al., observers picked the sliding version as shinier in
  2AFC with motion and were at chance with still frames.
- Specular flow differs from the flow of texture: it is compressed and hard to track at high curvature (patchy
  coverage), it diverges and converges over regions, and it does not follow epipolar geometry. Highlights cling to
  curved regions and rush across flat ones (Koenderink & van Doorn 1980). A classifier on these cues predicted both
  successes and failures of observers, and motion overrode static cues (Doerschner et al. 2011).
- Motion improves gloss constancy (Wendt et al. 2010). Image change that follows head motion strengthens gloss (Sakano
  & Ando 2010); image motion alone does too (Tani et al. 2013). Two viewpoints without the transition do not help
  (Ho, Maloney & Landy 2007).
- When the object turns under fixed light, the specularities and the diffuse shading both shift over the surface;
  only texture moves with it (Fleming 2017, Table 1).

Details from the primary papers (read in full: Doerschner et al. 2011, Dövencioğlu et al. 2017, Toscani et al. 2019,
Tani et al. 2013, Wendt & Faul 2018; Tamura et al. 2018 through a summary; others from abstracts):

- The three flow measures of Doerschner et al. (2011): coverage (the share of pixels whose motion can be tracked, and
  its drop from 2 to 3 frames), divergence (the share of pixels with strong expansion or contraction), and the fit of
  the flow to one rigid 3D motion. Each alone sorted the training stimuli; combined they matched observers, including
  two illusions. Coverage had the largest effect.
- Flow alone is not enough: dots moving with specular flow statistics do not look shiny (Roth et al. 2003, cited in
  Doerschner et al. 2011).
- Too little variety in the reflection motion also looks matte: a specular ellipsoid, or an environment whose elongated
  features run parallel to the motion (Doerschner et al. 2011).
- The reverse illusion: non-rigidly deforming matte objects look somewhat specular, and rigidly rotating mirrors can
  look non-rigid (Doerschner et al. 2011; Yildiran et al. 2023, preprint). (inference) A deformation field that
  wobbles may add a look of its own.
- When judging gloss, observers look more at regions of high divergence (Toscani, Yücel & Doerschner 2019).
- Glass and mirror differ in motion: reflections moving against the rotation mark glass (Tamura, Higashi & Nakauchi
  2018). Frame-shuffled clips were judged like still ones.
- Size of the effects: motion decides shiny versus matte (near ceiling with motion, chance without; Doerschner et al.
  2011). It raises rated gloss moderately (d about 1.4–1.5 for rendered viewpoint motion; Tani et al. 2013). Rotation
  at 60°/s raised gloss but did not undo the underestimation caused by overlapping highlights (Wendt & Faul 2018).
- Stimulus parameters that worked: 15° back and forth about six axes (Doerschner et al. 2011); ±10° at 12°/s
  (Dövencioğlu et al. 2017); 30° at 30°/s over 1 s (Tamura et al. 2018); 4 and 8°/s (Toscani et al. 2019); continuous
  60°/s (Wendt & Faul 2018). No study measured thresholds of amplitude or speed.
- Static frames and frame-shuffled clips of the same sequences separate errors of static appearance from errors of
  motion (Doerschner et al. 2011; Tamura et al. 2018).

## Stereo

- Highlights have a disparity different from the surface's, and any detectable offset reads as gloss, even with the
  wrong sign (Blake & Bülthoff 1990; Kerrigan & Adams 2013; Muryy et al. 2013, 2016). Not needed for monocular viewing.

## Consequences for the loss (inference)

Must be kept, in order of how directly the literature supports it:

1. Highlights that slide over the surface as it turns. Specular radiance baked into colour attached to the Gaussians
   makes the object look matte in motion. This is a property of the representation, not of a loss.
2. Highlights congruent with the shape and shading, and the specular flow that follows from curvature. Both depend on
   the normals and their curvature, so geometry is part of the gloss.
3. Highlight contrast (in linear luminance), coverage and sharpness.
4. An environment with natural statistics: a few small bright sources and a wide dynamic range.

Probably free to differ:

- The details of the reflected environment, as long as 4 holds.
- Small highlight displacements that stay consistent with the shape. No study measured how large they may be.

What the literature does not say:

- No threshold for errors in highlight position or motion in a moving scene, for material identity. The literature
  tests the level of gloss and constancy, not whether two moving objects look like the same material. Side-by-side
  comparisons under matched conditions are predicted by image differences (Cheeseman et al. 2025), so the task design
  decides whether observers judge the material or the image.
- No study of dynamic reconstructions (3DGS / NeRF) judged by material. The dynamic specular methods (NeRF-DS,
  SpectroMotion and later ones) report PSNR, SSIM and LPIPS only.

## Dynamic reconstruction of glossy objects (what is already shown)

| Work | What it shows or does | Evaluation |
|---|---|---|
| NeRF-DS (Yan et al., CVPR 2023) | Dynamic NeRFs give a point the same colour for the same canonical position and view direction, even after it turns (their Fig. 4). Conditions colour on observation-space position and normal | PSNR, MS-SSIM, LPIPS |
| SpectroMotion (Fan et al., CVPR 2025) | Deformable 3DGS plus split-sum shading with a learned cube map; corrects normals after deformation | PSNR, SSIM, LPIPS |
| Hirai et al. (arXiv 2603.26665, 2026) | Closest setup: a rigid object moved under static light. Per-Gaussian SH burns the lighting in and corrupts geometry; specular is a shared SH of the reflection direction | Geometry and image metrics |
| DR-GS (arXiv 2606.29379, 2026) | Static PBR reconstruction, then deformed by simulation (not reconstructed from video). User study of plausibility and temporal coherence, not of material | PSNR, SSIM, LPIPS, ratings |

- That reflections attached to the surface look matte in motion is established (Hartung & Kersten 2002; Doerschner et
  al. 2011). An observer experiment that only shows this for a vanilla 4DGS would repeat it.
- Not found: an analysis of the deformation-field case where the surface has no texture, so that nothing in the
  interior anchors the true rotation, and of what a reconstruction does with the highlights then.

## Normals and rigid motion of deforming Gaussians (2026-11-28)

Read from the papers' method sections and code through three searches; details and the use made of them are in the
[README](README.md) ("The textured bunny").

- Normal transport: SpectroMotion deforms sizes too and corrects the normal with a residual carried by the frame of the
  shortest and longest axes; it states that rotation-only transport fails when the relative axis lengths change.
  NeRF-DS rotates one canonical normal with the deformation's rotation. ReflFlow uses 2DGS surfels with depth-normal
  and monocular-normal losses.
- Sizes constant over time: Dynamic 3D Gaussians, Shape of Motion, MoSca, SC-GS (paper; its default code outputs a
  size change).
- Rotation tied to the neighbours' motion: Dynamic 3D Gaussians' local rigidity, rotation and isometry losses
  (weights 4, 4, 2; multi-view data only); SC-GS's control points with skinning and an ARAP loss (works on monocular
  D-NeRF). The Jacobian of the deformation (PhysGaussian, DIGS) is not used for monocular RGB.
- Well-defined normals: PGSR and NeuSG shrink the shortest size (weight 100, scene-scale dependent); Gaussian Surfels
  fix the normal axis with zero thickness; 2DGS adds normal consistency and depth distortion; Ref-Gaussian and Ref-GS
  combine 2DGS with deferred reflection.
- Schedules: SpectroMotion 0-3k static, 3-6k deformation, 6-9k normal loss, 9-15k shading with geometry and
  deformation frozen, then joint. ReflFlow: diffuse to 9k, specular 9-15k, then joint.
- Visible-difference metrics as losses: ColorVideoVDP has a `loss()` (10 - JOD) and its authors advise using it with
  L1 or after L1 has nearly converged; no use in 3DGS or NeRF training was found. HDR-VDP-3 is MATLAB only. The JOV
  paper of Cheeseman et al. cited above could not be re-opened on 2026-11-28; the abstract of Cheeseman's thesis
  (Giessen 2024) states the same result.
