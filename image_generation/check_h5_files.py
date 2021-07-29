import os
import numpy as np
import cv2
import h5py
from matplotlib import pyplot as plt


split = "test"
file_path = (
    f"/home/nicolai/sra/code/clevr-dataset-gen/output/h5/clevr_3_objs_{split}.h5"
)
h5_file = h5py.File(file_path, "r")

start, stop = 0, 2
for index in range(start, stop):
    fig, ax = plt.subplots(4, 4)
    for ii in range(4):
        rgb = h5_file["image"][index][ii]
        depth = h5_file["depth"][index][ii]
        normal = h5_file["normal"][index][ii]
        ids = h5_file["id"][index][ii]

        ax[ii, 0].imshow(rgb)
        ax[0, 0].set_title("RGB")
        ax[ii, 1].imshow(depth, cmap="plasma")
        ax[0, 1].set_title("Depth [mm]")
        ax[ii, 2].imshow(normal)
        ax[0, 2].set_title("Normal map")
        ax[ii, 3].imshow(ids, cmap="plasma")
        ax[0, 3].set_title("Object Ids")
    plt.show()
