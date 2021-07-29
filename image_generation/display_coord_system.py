import os
import h5py
import numpy as np
import open3d as o3d
import cv2


def point_cloud_from_depth(depth, K):
    """Transform depth image pixels selected in mask into point cloud in camera frame
    depth: [N, H, W] depth image
    mask: [N, H, W] mask of pixels to transform into point cloud
    K: [N, 3, 3] Intrinsic camera matrix
    returns: [N, 3, K]
    """
    H, W = depth.shape
    # Create 3D grid data
    u_img_range = np.arange(0, W)
    v_img_range = np.arange(0, H)
    u_grid, v_grid = np.meshgrid(u_img_range, v_img_range)
    u_img, v_img, d = (
        u_grid.reshape(-1),
        v_grid.reshape(-1),
        depth.reshape(-1),
    )

    # homogenuous coordinates
    uv = np.stack((u_img, v_img, np.ones_like(u_img)), axis=0)

    # get the unscaled position for each of the points in the image frame
    unscaled_points = np.linalg.inv(K) @ uv

    # scale points by their depth value
    return d * unscaled_points


base_dir = "/home/nicolai/sra/code/clevr-dataset-gen/output/h5/"
h5_path = os.path.join(base_dir, "clevr_3_objs_val.h5")

with h5py.File(h5_path, "r") as h5_file:
    length = h5_file["image"].shape[0]
    n_views = h5_file["image"].shape[1]

    idx = 0
    imgs = h5_file["image"][idx].astype(np.uint8)
    depths = h5_file["depth"][idx]
    normals = h5_file["normal"][idx]
    ids = h5_file["id"][idx].astype(np.uint8)
    extrinsics = h5_file["extrinsics"][idx]
    intrinsics = h5_file["intrinsics"][idx]

view_idx = [0, 1]
imgs = imgs[view_idx]
depths = depths[view_idx]
normals = normals[view_idx]
ids = ids[view_idx]
extrinsics = extrinsics[view_idx]

base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
coordinate_frames = []
point_clouds = []

for ii, ext in enumerate(extrinsics):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coordinate_frames.append(frame.transform(ext))

    rgb = imgs[ii]
    depth = depths[ii]
    pcl_camera_frame = point_cloud_from_depth(depth, intrinsics)
    pcl_world_frame = ext @ np.concatenate(
        (pcl_camera_frame, np.ones_like(pcl_camera_frame[-1][None, :])), axis=0
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.transpose(pcl_world_frame)[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)
    point_clouds.append(pcd)

o3d.visualization.draw_geometries(point_clouds + [base_frame] + coordinate_frames)
