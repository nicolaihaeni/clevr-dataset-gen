# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
from utils import *

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Matrix, Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print(
            "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth"
        )
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument(
    "--base_scene_blendfile",
    default="data/base_scene.blend",
    help="Base blender file on which all scenes are based; includes "
    + "ground plane, lights, and camera.",
)
parser.add_argument(
    "--shape_dir",
    default="data/ycb/models/",
    help="Directory where .blend files for object models are stored",
)

# Settings for objects
parser.add_argument(
    "--min_objects",
    default=1,
    type=int,
    help="The minimum number of objects to place in each scene",
)
parser.add_argument(
    "--max_objects",
    default=4,
    type=int,
    help="The maximum number of objects to place in each scene",
)
parser.add_argument(
    "--min_dist",
    default=0.25,
    type=float,
    help="The minimum allowed distance between object centers",
)
parser.add_argument(
    "--margin",
    default=0.4,
    type=float,
    help="Along all cardinal directions (left, right, front, back), all "
    + "objects will be at least this distance apart. This makes resolving "
    + "spatial relationships slightly less ambiguous.",
)
parser.add_argument(
    "--min_pixels_per_object",
    default=200,
    type=int,
    help="All objects will have at least this many visible pixels in the "
    + "final rendered images; this ensures that no objects are fully "
    + "occluded by other objects.",
)
parser.add_argument(
    "--max_retries",
    default=50,
    type=int,
    help="The number of times to try placing an object before giving up and "
    + "re-placing all objects in the scene.",
)

# Output settings
parser.add_argument(
    "--start_idx",
    default=0,
    type=int,
    help="The index at which to start for numbering rendered images. Setting "
    + "this to non-zero values allows you to distribute rendering across "
    + "multiple machines and recombine the results later.",
)
parser.add_argument(
    "--num_images", default=5, type=int, help="The number of images to render"
)
parser.add_argument(
    "--filename_prefix",
    default="CLEVR",
    help="This prefix will be prepended to the rendered images and JSON scenes",
)
parser.add_argument(
    "--split",
    default="new",
    help="Name of the split for which we are rendering. This will be added to "
    + "the names of rendered images, and will also be stored in the JSON "
    + "scene structure for each image.",
)
parser.add_argument(
    "--output_dir",
    default="../output/",
    help="The directory where output images will be stored. It will be "
    + "created if it does not exist.",
)
parser.add_argument(
    "--output_scene_dir",
    default="../output/",
    help="The directory where output JSON scene structures will be stored. "
    + "It will be created if it does not exist.",
)
parser.add_argument(
    "--output_scene_file",
    default="../output/CLEVR_scenes.json",
    help="Path to write a single JSON file containing all scene information",
)
parser.add_argument(
    "--output_blend_dir",
    default="output/blendfiles",
    help="The directory where blender scene files will be stored, if the "
    + "user requested that these files be saved using the "
    + "--save_blendfiles flag; in this case it will be created if it does "
    + "not already exist.",
)
parser.add_argument(
    "--save_blendfiles",
    type=int,
    default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for "
    + "each generated image to be stored in the directory specified by "
    + "the --output_blend_dir flag. These files are not saved by default "
    + "because they take up ~5-10MB each.",
)
parser.add_argument(
    "--version",
    default="1.0",
    help='String to store in the "version" field of the generated JSON file',
)
parser.add_argument(
    "--license",
    default="Creative Commons Attribution (CC-BY 4.0)",
    help='String to store in the "license" field of the generated JSON file',
)
parser.add_argument(
    "--date",
    default=dt.today().strftime("%m/%d/%Y"),
    help='String to store in the "date" field of the generated JSON file; '
    + "defaults to today's date",
)

# Rendering options
parser.add_argument(
    "--use_gpu",
    default=0,
    type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. "
    + "You must have an NVIDIA GPU with the CUDA toolkit installed for "
    + "to work.",
)
parser.add_argument(
    "--width",
    default=320,
    type=int,
    help="The width (in pixels) for the rendered images",
)
parser.add_argument(
    "--height",
    default=240,
    type=int,
    help="The height (in pixels) for the rendered images",
)
parser.add_argument(
    "--key_light_jitter",
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the key light position.",
)
parser.add_argument(
    "--fill_light_jitter",
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the fill light position.",
)
parser.add_argument(
    "--back_light_jitter",
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the back light position.",
)
parser.add_argument(
    "--camera_jitter",
    default=0.5,
    type=float,
    help="The magnitude of random jitter to add to the camera position",
)
parser.add_argument(
    "--render_num_samples",
    default=512,
    type=int,
    help="The number of samples to use when rendering. Larger values will "
    + "result in nicer images but will cause rendering to take longer.",
)
parser.add_argument(
    "--render_min_bounces",
    default=8,
    type=int,
    help="The minimum number of bounces to use for rendering.",
)
parser.add_argument(
    "--render_max_bounces",
    default=8,
    type=int,
    help="The maximum number of bounces to use for rendering.",
)
parser.add_argument(
    "--render_tile_size",
    default=256,
    type=int,
    help="The tile size to use for rendering. This should not affect the "
    + "quality of the rendered image but may affect the speed; CPU-based "
    + "rendering may achieve better performance using smaller tile sizes "
    + "while larger tile sizes may be optimal for GPU-based rendering.",
)


def main(args):
    num_digits = 6
    prefix = "%s_" % (args.filename_prefix)
    img_template = "%s%%0%dd" % (prefix, num_digits)
    scene_template = "%s%%0%dd.json" % (prefix, num_digits)
    blend_template = "%s%%0%dd.blend" % (prefix, num_digits)
    output_path = os.path.join(args.output_dir, args.split, "images")
    img_template = os.path.join(args.output_dir, args.split, "images", img_template)
    scene_template = os.path.join(
        args.output_scene_dir, args.split, "scenes", scene_template
    )
    blend_template = os.path.join(args.output_blend_dir, args.split, blend_template)

    if not os.path.isdir(os.path.join(args.output_dir, args.split, "images")):
        os.makedirs(os.path.join(args.output_dir, args.split, "images"))
    if not os.path.isdir(os.path.join(args.output_dir, args.split, "scenes")):
        os.makedirs(os.path.join(args.output_dir, args.split, "scenes"))
    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        os.makedirs(os.path.join(args.output_blend_dir, args.split))

    all_scene_paths = []
    for i in range(args.start_idx, args.num_images):
        img_path = img_template % (i)
        scene_path = scene_template % (i)
        all_scene_paths.append(scene_path)
        blend_path = None
        if args.save_blendfiles == 1:
            blend_path = blend_template % (i)
        num_objects = random.randint(args.min_objects, args.max_objects)
        render_scene(
            args,
            num_objects=num_objects,
            output_index=(i),
            output_split=args.split,
            output_dir=output_path,
            output_image=img_path,
            output_scene=scene_path,
            output_blendfile=blend_path,
        )

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, "r") as f:
            all_scenes.append(json.load(f))
    output = {
        "info": {
            "date": args.date,
            "version": args.version,
            "split": args.split,
            "license": args.license,
        },
        "scenes": all_scenes,
    }
    with open(args.output_scene_file, "w") as f:
        json.dump(output, f)


def render_scene(
    args,
    num_objects=5,
    output_index=0,
    output_split="none",
    output_dir="render.png",
    output_image="render.png",
    output_scene="render_json",
    output_blendfile=None,
):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = "CUDA"
            bpy.context.user_preferences.system.compute_device = "CUDA_0"
        else:
            cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
            cycles_prefs.compute_device_type = "CUDA"

    # Some CYCLES-specific stuff
    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    # Depth/Normal/segmentation images
    scene = bpy.context.scene
    scene.use_nodes = True
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_object_index = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new("CompositorNodeRLayers")

    # Create depth output nodes
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = "Depth Output"
    depth_file_output.base_path = ""
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = "OPEN_EXR"
    depth_file_output.format.color_depth = "16"
    links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = "MULTIPLY"
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = "ADD"
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    normal_file_output.base_path = ""
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = "OPEN_EXR"
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])

    # Create id map output nodes
    id_file_output = nodes.new(type="CompositorNodeOutputFile")
    id_file_output.label = "ID Output"
    id_file_output.base_path = ""
    id_file_output.file_slots[0].use_node_format = True
    id_file_output.format.file_format = "OPEN_EXR"
    id_file_output.format.color_depth = "16"
    links.new(render_layers.outputs["IndexOB"], id_file_output.inputs[0])

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        "split": output_split,
        "image_index": output_index,
        "image_filename": os.path.basename(output_image),
        "objects": [],
        "directions": {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(size=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects["Camera"].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects["Camera"]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct["directions"]["behind"] = tuple(plane_behind)
    scene_struct["directions"]["front"] = tuple(-plane_behind)
    scene_struct["directions"]["left"] = tuple(plane_left)
    scene_struct["directions"]["right"] = tuple(-plane_left)
    scene_struct["directions"]["above"] = tuple(plane_up)
    scene_struct["directions"]["below"] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Key"].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Back"].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Fill"].location[i] += rand(args.fill_light_jitter)

    # Now make some random objects
    objects, blender_objects = add_random_objects(
        scene_struct, num_objects, args, camera
    )

    # Set object IDs
    for ii, obj in enumerate(bpy.data.objects):
        obj.pass_index = ii + 1

    # Render the scene and dump the scene data structure
    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)
    K = get_calibration_matrix_K_from_blender(camera.data)
    scene_struct["intrinsics"] = [list(row) for row in K]
    scene_struct["extrinsics"] = []

    step_size = 360.0 / 4
    rotation_mode = "XYZ"

    original_cam_matrix = camera.matrix_world
    while True:
        try:
            for ii in range(0, 4):
                render_file_path = output_image + "_{0:03d}".format(int(ii * step_size))
                bpy.context.scene.render.filepath = render_file_path
                depth_file_output.file_slots[0].path = render_file_path + "_depth"
                normal_file_output.file_slots[0].path = render_file_path + "_normal"
                id_file_output.file_slots[0].path = render_file_path + "_id"

                rot = Matrix.Rotation(math.radians(ii * step_size), 4, "Z")
                camera.matrix_world = rot @ original_cam_matrix

                bpy.ops.render.render(write_still=True)
                Rt = get_world2cam_from_blender_cam(camera)
                cam2world = Rt.inverted()
                scene_struct["extrinsics"].append([list(row) for row in cam2world])
            break
        except Exception as e:
            print(e)

    with open(output_scene, "w") as f:
        json.dump(scene_struct, f, indent=2)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def add_random_objects(scene_struct, num_objects, args, camera):
    """
    Add random objects to the current blender scene
    """
    # Load object library
    base_path = os.path.dirname(__file__)
    dirname = os.path.join(base_path, args.shape_dir)
    shapes = sorted(os.listdir(dirname))

    positions = []
    objects = []
    blender_objects = []
    for i in range(num_objects):
        r = random.uniform(0.7, 1.0)

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(scene_struct, num_objects, args, camera)
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ["left", "right", "front", "behind"]:
                    direction_vec = scene_struct["directions"][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        print(margin, args.margin, direction_name)
                        print("BROKEN MARGIN!")
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        shape_name = random.choice(shapes)
        obj_file_name = os.path.join(dirname, shape_name, "google_16k/textured.obj")

        # Choose random orientation for the object.
        theta = 360.0 * random.random()

        # Actually add the object to the scene
        utils.add_ycb_object(obj_file_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append(
            {
                "shape": shape_name,
                "3d_coords": tuple(obj.location),
                "rotation": theta,
                "pixel_coords": pixel_coords,
                "shape": r,
            }
        )

    # # Check that all objects are at least partially visible in the rendered image
    # all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
    # if not all_visible:
    # # If any of the objects are fully occluded then start over; delete all
    # # objects from the scene and place them all again.
    # print("Some objects are occluded; replacing objects")
    # for obj in blender_objects:
    # utils.delete_object(obj)
    # return add_random_objects(scene_struct, num_objects, args, camera)

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct["directions"].items():
        if name == "above" or name == "below":
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related = set()
            for j, obj2 in enumerate(scene_struct["objects"]):
                if obj1 == obj2:
                    continue
                coords2 = obj2["3d_coords"]
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix=".png")
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter(
        (p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4)
    )
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path="flat.png"):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.simplify_gpencil_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.display.shading.light = "FLAT"
    bpy.context.scene.display.render_aa = "OFF"
    render_args.simplify_gpencil_antialiasing = False

    # # Move the lights and ground to layer 2 so they don't render
    # utils.set_layer(bpy.data.objects["Lamp_Key"], 2)
    # utils.set_layer(bpy.data.objects["Lamp_Fill"], 2)
    # utils.set_layer(bpy.data.objects["Lamp_Back"], 2)
    # utils.set_layer(bpy.data.objects["Ground"], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials["Material"]
        mat.name = "Material_%d" % i
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors:
                break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b, 1.0]
        mat.use_nodes = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # # Move the lights and ground back to layer 0
    # utils.set_layer(bpy.data.objects["Lamp_Key"], 0)
    # utils.set_layer(bpy.data.objects["Lamp_Fill"], 0)
    # utils.set_layer(bpy.data.objects["Lamp_Back"], 0)
    # utils.set_layer(bpy.data.objects["Ground"], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.simplify_gpencil_antialiasing = old_use_antialiasing

    return object_colors


if __name__ == "__main__":
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
    else:
        print("This script is intended to be called from blender like this:")
        print()
        print("blender --background --python render_images.py -- [args]")
        print()
        print("You can also run as a standalone python script to view all")
        print("arguments like this:")
        print()
        print("python render_images.py --help")
