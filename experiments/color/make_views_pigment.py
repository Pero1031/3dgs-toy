# -*- coding: utf-8 -*-
"""
Colour experiment, stage 2: a matte still life whose colours lie outside the sRGB gamut because of the materials
themselves, under ordinary D65 light. No gloss, low dynamic range: a scene 3DGS fits well, so that the cost of the
clamp at 0 is not buried under other error (stage 1).

Every surface is diffuse with a measured spectral reflectance from GoLD_rayt's assets/Fading_measured (pigments and
dyed papers, round 1; measured 440-680 nm, the renderer holds the end values beyond):
    statue   Erato statue of the gloss experiment, pigment green    (outside sRGB and outside Rec.2020 under D65)
    6 balls  around it, partly in front of it from most views (colour edges):
             pigment blue, dyed-paper blue, pigment yellow           (outside sRGB, inside Rec.2020)
             pigment vermilion, dyed-paper yellow-green, dyed-origami purple   (inside sRGB: controls)
Spectra whose reflectance exceeds 1 somewhere (dyed-paper red, purple, orange) are left out.
Lighting, cameras, views and seeds as in the gloss experiment (softbox above, 4 sphere lights, uniform ambient; all D65).

    python experiments/color/make_views_pigment.py --render

-> data/color_pigment/ (layout as make_views_spectral.py; the mask covers the statue and the balls)
"""

import math

import make_views_spectral as spectral

gloss = spectral.gloss
NAME = "color_pigment"
FADING = "assets/Fading_measured"
STATUE = f"{FADING}/pigment/r1/green.csv"
BALLS = [   # (material name, reflectance, height of the centre)
    ("pig_blue", f"{FADING}/pigment/r1/blue.csv", 0.08),
    ("pig_vermilion", f"{FADING}/pigment/r1/syuiro.csv", 0.30),
    ("paper_blue", f"{FADING}/dye_paper/r1/blue.csv", 0.16),
    ("paper_yellowgreen", f"{FADING}/dye_paper/r1/kimidori.csv", 0.40),
    ("pig_yellow", f"{FADING}/pigment/r1/yellow.csv", 0.12),
    ("origami_purple", f"{FADING}/dye_origami/r1/purple.csv", 0.24),
]
BALL_RADIUS = 0.055
BALL_RING = 0.17      # distance of the ball centres from the statue's axis


def scene(frm, size, spp, seed, out_pfm):
    sc = gloss.scene("matte", frm, size, spp, seed, out_pfm, True)
    sc["materials"]["statue"] = {"type": "diffuse", "reflectance": STATUE}
    for k, (name, refl, y) in enumerate(BALLS):
        az = math.radians(60.0 * k + 30.0)
        sc["materials"][name] = {"type": "diffuse", "reflectance": refl}
        sc["objects"].append({"shape": "sphere", "center": [BALL_RING * math.sin(az), y, BALL_RING * math.cos(az)],
                              "radius": BALL_RADIUS, "material": name})
    sc["options"]["mask"] = ",".join(["statue"] + [b[0] for b in BALLS])
    return sc


if __name__ == "__main__":
    spectral.main(NAME, scene)
