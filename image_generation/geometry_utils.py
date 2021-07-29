import torch
import numpy as np
from torchinterp1d import Interp1d


t_to_depth = lambda x, v_min, v_max: x * (v_max - v_min)


def unit_to_camera_coords(coords, H, W, K):
    coords[:, 0] = coords[:, 0] * W
    coords[:, 1] = coords[:, 1] * W
    d = t_to_depth(coords[:, 2], 7.0291489, 38.5342930)
    coords[:, 2] = torch.ones_like(coords[:, 2])
    uv_unscaled = torch.linalg.inv(K) @ coords.t()
    return (d * uv_unscaled).t()


def get_relative_trafo(w_T_c1, w_T_c2):
    """Computes the relative transformation matrix between two camera matrices
    c1_T_w: [N, 4, 4] World to camera matrix 1
    c2_T_w: [N, 4, 4] World to camera matrix 2
    returns: [N, 4, 4] relative transformation matrix between camera1 and camera 2
    """
    return torch.linalg.inv(w_T_c1) @ w_T_c2


def point_cloud_from_depth(depth, K):
    """Transform depth image pixels selected in mask into point cloud in camera frame
    depth: [N, H, W] depth image
    mask: [N, H, W] mask of pixels to transform into point cloud
    K: [N, 3, 3] Intrinsic camera matrix
    returns: [N, 3, K]
    """
    batch_size, H, W = depth.shape
    # Create 3D grid data
    u_img_range = torch.arange(0, W, device=depth.device)
    v_img_range = torch.arange(0, H, device=depth.device)
    u_grid, v_grid = torch.meshgrid(u_img_range, v_img_range)
    u_grid = u_grid.t()[None, :].repeat(batch_size, 1, 1)
    v_grid = v_grid.t()[None, :].repeat(batch_size, 1, 1)
    u_img, v_img, d = (
        u_grid.reshape(batch_size, -1),
        v_grid.reshape(batch_size, -1),
        depth.reshape(batch_size, -1),
    )

    # homogenuous coordinates
    uv = torch.stack((u_img, v_img, torch.ones_like(u_img)), dim=1).float()

    # get the unscaled position for each of the points in the image frame
    unscaled_points = torch.linalg.inv(K) @ uv

    # scale points by their depth value
    return d[:, None] * unscaled_points


def plane_fitting(xyz, mask):
    """Computes least squares fit of a plane normal vector from 3D points
    xyz: [batch_size, 3, N] contains N 3D points (xyz coordinates)
    returns: [batch_size, 6] estimated plane parameters as [point, normal]
    """
    batch_size = xyz.shape[0]
    centroid = torch.mean(xyz, dim=2)
    x, y, z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]
    idx = torch.stack(
        [
            mask[ii].reshape(-1).float().multinomial(100, replacement=False)
            for ii in range(batch_size)
        ]
    )

    x = torch.gather(x, 1, idx)
    y = torch.gather(y, 1, idx)
    z = torch.gather(z, 1, idx)

    A = torch.stack((x, y, torch.ones_like(x)), dim=2)
    B = z.unsqueeze(2)
    X = torch.linalg.pinv(A) @ B
    normal = torch.stack((X[:, 0], X[:, 1], -torch.ones_like(X[:, 0])), dim=1)
    normal = normal / torch.linalg.norm(normal)
    return torch.cat((centroid, normal.squeeze(-1)), dim=1)


def ellipsoid(params, t, u):
    centroids = params[:, :3]
    radii = params[:, 3:]
    coords = torch.stack(
        (
            radii[:, 0].view([-1] + [1] * (u.dim() - 1)) * torch.sin(u) * torch.cos(t),
            radii[:, 1].view([-1] + [1] * (u.dim() - 1)) * torch.sin(u) * torch.sin(t),
            radii[:, 2].view([-1] + [1] * (u.dim() - 1)) * torch.cos(u),
        ),
        dim=1,
    )
    return coords


def sample_points_on_surface(params, func, t, u, n_points=1000, shape="ellipsoid"):
    n_shapes = params.shape[0]
    coords = func(params, t, u)
    # Computer cummulative surface are at each coordinate
    delta_t_temp = torch.diff(coords, dim=3)
    delta_u_temp = torch.diff(coords, dim=2)

    # Pad with zeros so that small rand_S can still be interpd
    delta_t = torch.zeros_like(coords)
    delta_u = torch.zeros_like(coords)
    delta_t[:, :, :, 1 : coords.shape[3]] = delta_t_temp
    delta_u[:, :, 1 : coords.shape[2], :] = delta_u_temp

    # Area of each parallelogramg
    delta_S = torch.linalg.norm(torch.cross(delta_t, delta_u, 1), dim=1)
    cum_S_t = torch.cumsum(delta_S.sum(dim=1), dim=1)
    cum_S_u = torch.cumsum(delta_S.sum(dim=2), dim=1)

    # Random values
    rand_S_t = cum_S_t[:, -1].view(-1, 1) * torch.rand(
        (n_shapes, n_points), device=params.device
    )
    rand_S_u = cum_S_u[:, -1].view(-1, 1) * torch.rand(
        (n_shapes, n_points), device=params.device
    )

    # Find corresponding t-values by interpolation
    rand_S_t = Interp1d()(cum_S_t, t[:, 0, :], rand_S_t)
    rand_S_u = Interp1d()(cum_S_u, u[:, :, 0], rand_S_u)
    rand_coords = func(params, rand_S_t, rand_S_u)

    # Rotate points into camera coordinates
    if shape == "ellipsoid":
        centroids = params[:, :3]
        rot_mat = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        rand_coords = rot_mat.unsqueeze(0).to(rand_coords.device) @ rand_coords.float()
        rand_coords = rand_coords + centroids[:, :, None].repeat(1, 1, n_points)
    return rand_coords


def sample_points_on_plane(params, extents, W, K, n_points=10000):
    batch_size = params.shape[0]
    centroids = params[:, :3]
    normals = params[:, 3:]

    zz = (
        torch.rand((batch_size, n_points), device=params.device)
        * (extents[:, 0, 2] - extents[:, 1, 2])[:, None]
        + extents[:, 1, 2][:, None]
    )

    # Compute min/max coordinates of x in frustum
    K_inv = torch.linalg.inv(K)
    min_u, max_u = 0, W
    min_x = zz * (K_inv[:, 0, 0] * min_u + K_inv[:, 0, -1])[:, None]
    max_x = zz * (K_inv[:, 0, 0] * max_u + K_inv[:, 0, -1])[:, None]

    xx = torch.rand((batch_size, n_points), device=params.device)
    xx = (min_x - max_x) * xx + max_x

    d = (
        normals[:, 0] * centroids[:, 0]
        + normals[:, 1] * centroids[:, 1]
        + normals[:, 2] * centroids[:, 2]
    )

    yy = (
        (-normals[:, 0][:, None] * xx - normals[:, 2][:, None] * zz + d[:, None])
        * 1.0
        / normals[:, 1][:, None]
    )
    return torch.stack((xx, yy, zz), dim=1)


def sample_points(plane_params, ellipsoid_params, H, W, K, extents):
    b, _ = plane_params.shape
    points = sample_points_on_plane(plane_params, extents, W, K)

    for ii in range(b):
        params = ellipsoid_params[ii]
        params = params[params.sum(dim=1) != 0]
        params[:, :3] = unit_to_camera_coords(params[:, :3], H, W, K[ii])

        domain_t = [0, np.pi]
        domain_u = [0, np.pi]
        t, u = np.meshgrid(
            np.linspace(*domain_t, 25),
            np.linspace(*domain_u, 25),
        )
        t = torch.tensor(t, device=params.device)[None, :].repeat(params.shape[0], 1, 1)
        u = torch.tensor(u, device=params.device)[None, :].repeat(params.shape[0], 1, 1)

        pts = (
            sample_points_on_surface(params, ellipsoid, t, u)
            .transpose(0, 1)
            .reshape(3, -1)
        )

        points[ii][:, : pts.shape[1]] = pts
    return points
