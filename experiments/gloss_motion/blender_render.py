# -*- coding: utf-8 -*-
"""
Render gloss stimuli with Blender Cycles, matched to the GoLD_rayt scenes of experiments/gloss_percept/make_stimuli.py.
Runs inside Blender (4.5):

    blender -b --factory-startup --python experiments/gloss_motion/blender_render.py -- JOBS.json

JOBS.json (written by make_motion.py):
    {"env": hdr, "env_scale": 2.0, "mesh": ply, "material": {...as in make_stimuli.MATERIALS},
     "device": "OPTIX" | "CUDA" | "CPU", "size": 256, "fov": 30,
     "jobs": [{"c2w": 4x4 camera-to-world in the y-up world of GoLD_rayt (OpenGL convention), "rotate_deg": angle about
               the vertical axis, "spp": 512, "seed": 1, "out": ".../name.pfm"}, ...]}

Each job writes <out> (linear radiance of the object, premultiplied by its coverage: the environment is not in the
background) and <out without .pfm>_mask.pfm (the coverage, 3 channels), both as PFM.

How the GoLD_rayt scene is matched:
    world       GoLD_rayt is y-up, Blender z-up: (x, y, z) -> (x, -z, y), a rotation of +90 degrees about x. With it the
                equirectangular lookups agree (GoLD_rayt: u = (atan2(z, x) + pi) / 2 pi, v = 1/2 + asin(y) / pi), so the
                environment is not turned
    camera      the same camera-to-world matrix moved by that rotation (both look along -z with y up on screen), the
                same field of view
    object      the mesh moved by that rotation; the turn about GoLD_rayt's y is a turn about Blender's z (right-handed
                in both)
    material    "diffuse": a Diffuse BSDF (Lambert). "layered": a Principled BSDF with the base's dielectric specular
                switched off (Specular IOR Level 0), coat weight 1, the coat's IOR and roughness (alpha = roughness^2 in
                both renderers)
    sampling    a 1-pixel box filter (GoLD_rayt jitters uniformly inside the pixel), no clamping, no glossy filter,
                no adaptive sampling, no denoising, up to 16 bounces
"""

import json
import math
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix

CONV = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))   # GoLD_rayt world -> Blender world


def write_pfm(path, rgb_bottom_up):
    """rgb_bottom_up: (H,W,3) float32, bottom row first (PFM's own order and Blender's)"""
    h, w = rgb_bottom_up.shape[:2]
    with open(path, "wb") as f:
        f.write(f"PF\n{w} {h}\n-1.0\n".encode("ascii"))
        f.write(np.ascontiguousarray(rgb_bottom_up, "<f4").tobytes())


def setup(cfg):
    sc = bpy.context.scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    if cfg["device"] == "CPU":
        cy.device = "CPU"
    else:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = cfg["device"]
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == cfg["device"]
        cy.device = "GPU"
    cy.use_adaptive_sampling = False
    cy.use_denoising = False
    cy.sample_clamp_direct = 0.0
    cy.sample_clamp_indirect = 0.0
    cy.blur_glossy = 0.0
    cy.max_bounces = cy.diffuse_bounces = cy.glossy_bounces = cy.transmission_bounces = 16
    cy.caustics_reflective = cy.caustics_refractive = True
    cy.pixel_filter_type = "BOX"
    cy.filter_width = 1.0
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = cfg["size"]
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0

    # environment
    world = bpy.data.worlds.new("env")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexEnvironment")
    tex.image = bpy.data.images.load(cfg["env"])
    tex.interpolation = "Linear"
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = cfg["env_scale"]
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # mesh
    bpy.ops.wm.ply_import(filepath=cfg["mesh"])
    obj = bpy.context.selected_objects[0]
    obj.data.transform(CONV)
    obj.data.shade_smooth()
    obj.rotation_mode = "XYZ"

    # material
    m = cfg["material"]
    mat = bpy.data.materials.new("obj")
    mat.use_nodes = True
    mt = mat.node_tree
    mt.nodes.clear()
    mout = mt.nodes.new("ShaderNodeOutputMaterial")
    if m["type"] == "diffuse":
        b = mt.nodes.new("ShaderNodeBsdfDiffuse")
        b.inputs["Color"].default_value = (*m["albedo"], 1.0)
        b.inputs["Roughness"].default_value = 0.0
    elif m["type"] == "layered":
        b = mt.nodes.new("ShaderNodeBsdfPrincipled")
        # GoLD_rayt sums the walks between coat and base exactly (pbrt-v4's LayeredBxDF), which darkens the base under
        # the coat; Principled only scales the base by the coat's albedo. "blender_base_albedo" replaces the base colour
        # by one calibrated against GoLD_rayt (make_motion.py bcalib)
        b.inputs["Base Color"].default_value = (*m.get("blender_base_albedo", m["base"]["albedo"]), 1.0)
        b.inputs["Metallic"].default_value = 0.0
        b.inputs["Specular IOR Level"].default_value = 0.0
        b.inputs["Coat Weight"].default_value = 1.0
        b.inputs["Coat Roughness"].default_value = m["coat"]["roughness"]
        b.inputs["Coat IOR"].default_value = m["coat"]["ior"]
        b.inputs["Coat Tint"].default_value = (1.0, 1.0, 1.0, 1.0)
    else:
        raise SystemExit(f"material type {m['type']} is not matched in Blender")
    if m.get("texture"):
        # a grey pattern fixed to the object: noise in object coordinates (metres), base x [1 - c, 1 + c]
        tx = m["texture"]
        tc = mt.nodes.new("ShaderNodeTexCoord")
        nz = mt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = tx["scale"]
        nz.inputs["Detail"].default_value = tx["detail"]
        nz.inputs["Roughness"].default_value = tx["roughness"]
        mr = mt.nodes.new("ShaderNodeMapRange")
        mr.clamp = True
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = tx["fac_range"]
        mr.inputs["To Min"].default_value = 1.0 - tx["contrast"]
        mr.inputs["To Max"].default_value = 1.0 + tx["contrast"]
        key = "Base Color" if m["type"] == "layered" else "Color"
        grey = b.inputs[key].default_value
        assert grey[0] == grey[1] == grey[2], "the pattern scales a grey base"
        mul = mt.nodes.new("ShaderNodeMath")            # a float into a colour socket becomes grey
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = grey[0]
        mt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        mt.links.new(nz.outputs["Fac"], mr.inputs["Value"])
        mt.links.new(mr.outputs["Result"], mul.inputs[0])
        mt.links.new(mul.outputs["Value"], b.inputs[key])
    mt.links.new(b.outputs[0], mout.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    # camera
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "AUTO"
    cam_data.angle = math.radians(cfg["fov"])
    cam_data.clip_start = 0.01
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return sc, obj, cam


def main():
    cfg = json.loads(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8").read())
    sc, obj, cam = setup(cfg)
    tmp = cfg.get("tmp_exr", cfg["jobs"][0]["out"].replace(".pfm", "_tmp.exr"))
    t_all = time.time()
    for i, job in enumerate(cfg["jobs"]):
        t0 = time.time()
        cam.matrix_world = CONV @ Matrix(job["c2w"])
        obj.rotation_euler = (0.0, 0.0, math.radians(job["rotate_deg"]))
        sc.cycles.samples = job["spp"]
        sc.cycles.seed = job["seed"]
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(tmp, check_existing=False)
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        bpy.data.images.remove(img)
        px = px.reshape(h, w, 4)
        write_pfm(job["out"], px[..., :3])
        write_pfm(job["out"].replace(".pfm", "_mask.pfm"), np.repeat(px[..., 3:4], 3, axis=-1))
        print(f"[blender] {i + 1}/{len(cfg['jobs'])} {job['out']} {time.time() - t0:.2f}s", flush=True)
    print(f"[blender] {len(cfg['jobs'])} renders in {time.time() - t_all:.1f} s", flush=True)


main()
