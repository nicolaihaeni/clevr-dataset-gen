import os
import json
import argparse

import numpy as np
import h5py
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../output", help="data directory")
    parser.add_argument("--out_dir", default="../output/h5", help="data directory")
    parser.add_argument("--split", default="train", help="data split train|val|test")
    parser.add_argument(
        "--img_dir", default="images", help="subdir where images are stored"
    )
    parser.add_argument(
        "--scene_dir", default="scenes", help="subdir where scene files are stored"
    )
    parser.add_argument("--n_imgs", type=int, default=10000, help="number of scenes")
    parser.add_argument(
        "--n_views", type=int, default=4, help="number of views per scene"
    )
    parser.add_argument("--H", type=int, default=240, help="number of views per scene")
    parser.add_argument("--W", type=int, default=320, help="number of views per scene")
    return parser


def main():
    parser = get_args()
    args = parser.parse_args()
    print(args)

    img_dir = os.path.join(args.data_dir, args.split, args.img_dir)
    scene_dir = os.path.join(args.data_dir, args.split, args.scene_dir)

    # Create output directory for h5 files
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_name = f"clevr_{args.split}.h5"
    out_file = h5py.File(os.path.join(args.out_dir, out_name), "w")

    # Create the h5 file data structure
    H, W = args.H, args.W
    image_set = out_file.create_dataset(
        "image",
        (args.n_imgs, args.n_views, H, W, 3),
        dtype="i8",
        chunks=(1, 1, H, W, 3),
        compression="gzip",
    )
    depth_set = out_file.create_dataset(
        "depth",
        (args.n_imgs, args.n_views, H, W),
        dtype="f",
        chunks=(1, 1, H, W),
        compression="gzip",
    )
    normal_set = out_file.create_dataset(
        "normal",
        (args.n_imgs, args.n_views, H, W),
        dtype="f",
        chunks=(1, 1, H, W),
        compression="gzip",
    )
    id_set = out_file.create_dataset(
        "id",
        (args.n_imgs, args.n_views, H, W),
        dtype="i8",
        chunks=(1, 1, H, W),
        compression="gzip",
    )
    extrinsics_set = out_file.create_dataset(
        "extrinsics",
        (args.n_imgs, args.n_views, 4, 4),
        dtype="f",
        chunks=(1, 1, 4, 4),
        compression="gzip",
    )
    intrinsics_set = out_file.create_dataset(
        "intrinsics",
        (args.n_imgs, 3, 3),
        dtype="f",
        chunks=(1, 3, 3),
        compression="gzip",
    )

    for ii in range(args.n_imgs):
        if ii % 1000 == 0:
            print(f"Writting data point {ii} of {args.n_imgs}")

        with open(
            os.path.join(scene_dir, f"CLEVR_{str(ii).zfill(6)}.json"),
            "r",
        ) as json_file:
            data = json.load(json_file)

        for jj in range(0, args.n_views):
            angle = jj * 90
            rgb = cv2.imread(
                os.path.join(
                    img_dir, f"CLEVR_{str(ii).zfill(6)}_{str(angle).zfill(3)}.png"
                )
            )
            image_set[ii, jj] = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            depth = cv2.imread(
                os.path.join(
                    img_dir,
                    f"CLEVR_{str(ii).zfill(6)}_{str(angle).zfill(3)}_depth0001.exr",
                ),
                cv2.IMREAD_ANYDEPTH,
            )
            depth_set[ii, jj] = depth  # depth in m

            normal = cv2.imread(
                os.path.join(
                    img_dir,
                    f"CLEVR_{str(ii).zfill(6)}_{str(angle).zfill(3)}_normal0001.exr",
                ),
                cv2.IMREAD_ANYDEPTH,
            )
            normal_set[ii, jj] = normal

            id = cv2.imread(
                os.path.join(
                    img_dir,
                    f"CLEVR_{str(ii).zfill(6)}_{str(angle).zfill(3)}_id0001.exr",
                ),
                cv2.IMREAD_ANYDEPTH,
            ).astype(np.uint8)
            # Rescale values
            unique = np.unique(id)
            new_range = np.arange(0, len(unique))

            for kk, u in enumerate(unique):
                id[:, :] = np.where(id == u, new_range[kk], id[:, :])
            id_set[ii, jj] = id

            extr = np.array(data["extrinsics"])
            extrinsics_set[ii, jj] = extr[jj]

            # Only load intrinsics once
            if jj == 0:
                intr = np.array(data["intrinsics"])
                intrinsics_set[ii] = intr


if __name__ == "__main__":
    main()
