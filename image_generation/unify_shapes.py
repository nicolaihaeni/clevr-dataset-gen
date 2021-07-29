import os
import bpy
import numpy as np
import math

base_path = "/home/nicolai/sra/code/clevr-dataset-gen/image_generation/data/ycb/models/"
shape_names = sorted(os.listdir(base_path))

# Delete the default cube (default selected)
bpy.ops.object.delete()

for shape_name in shape_names:
    shape_file_path = os.path.join(base_path, shape_name, "google_16k", "textured.obj")

    bpy.ops.import_scene.obj(filepath=str(shape_file_path), split_mode="OFF")
    obj = bpy.context.selected_objects[0]
    x = obj.dimensions.x
    y = obj.dimensions.y
    z = obj.dimensions.z
    max_dim = np.max(np.array((x, y, z)))

    print(obj.rotation_euler)
    obj.rotation_euler = (0, 0, 0)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)  # center the bounding box!
    scale = 2.0 / max_dim
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.export_scene.obj(filepath=shape_file_path)

    # Remember which meshes were just imported
    meshes_to_remove = []
    for ob in bpy.context.selected_objects:
        meshes_to_remove.append(ob.data)

    bpy.ops.object.delete()

    # Remove the meshes from memory too
    for mesh in meshes_to_remove:
        bpy.data.meshes.remove(mesh)
