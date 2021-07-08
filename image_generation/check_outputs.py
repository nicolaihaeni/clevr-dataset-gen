import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

split = "val"
base_path = f"/home/nicolai/sra/code/clevr-dataset-gen/output/{split}/images/"

for index in range(100):
    # fig, ax = plt.subplots(3, 4)
    for ii in range(3):
        rgb = cv2.imread(
            os.path.join(base_path, f"CLEVR_{split}_{str(index).zfill(6)}_{ii}.png")
        )
        depth = cv2.imread(
            os.path.join(
                base_path, f"CLEVR_{split}_{str(index).zfill(6)}_{ii}_depth0001.exr"
            ),
            cv2.IMREAD_ANYDEPTH,
        )
        if depth is None or np.max(depth) > 100000:
            print(index)

        normal = cv2.imread(
            os.path.join(
                base_path, f"CLEVR_{split}_{str(index).zfill(6)}_{ii}_normal0001.exr"
            ),
            cv2.IMREAD_ANYDEPTH,
        )
        ids = cv2.imread(
            os.path.join(
                base_path, f"CLEVR_{split}_{str(index).zfill(6)}_{ii}_id0001.exr"
            ),
            cv2.IMREAD_ANYDEPTH,
        )

        # ax[ii, 0].imshow(rgb)
        # ax[0, 0].set_title("RGB")
        # ax[ii, 1].imshow(depth, cmap="plasma")
        # ax[0, 1].set_title("Depth [mm]")
        # ax[ii, 2].imshow(normal)
        # ax[0, 2].set_title("Normal map")
        # ax[ii, 3].imshow(ids, cmap="plasma")
        # ax[0, 3].set_title("Object Ids")
    # plt.show()
