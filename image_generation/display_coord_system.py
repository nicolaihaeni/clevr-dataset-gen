import os
import json
import numpy as np
import open3d as o3d


base_dir = "/home/nicolai/sra/code/clevr-dataset-gen/output/scenes"
json_files = sorted(os.listdir(base_dir))
extrinsics = []

for file in json_files:
    with open(os.path.join(base_dir, file), "r") as f:
        data = json.load(f)
        ext = np.array(data["cam_extrinsics"])
        extrinsics.append(ext)


mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
mesh_box.compute_vertex_normals()
mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
coordinate_frames = []

for ext in extrinsics:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frames.append(frame.transform(ext))
    o3d.visualization.draw_geometries([mesh_box, base_frame] + coordinate_frames)
