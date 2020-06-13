"""
Loading and sampling from Toy space
"""

from pytraits.image.io import *
from pytraits.image.base import *

import numpy as np
from torch.utils.data import TensorDataset
from torch import FloatTensor, LongTensor


def load_toy_image(fpath):
    """
    Load a grayscale image and return a mapped variant with pixel values
    from [1,2,...,n], where n is the number of unique values from the source 
    """
    img = read_image(fpath)
    if img.shape != (256, 256):
        raise Exception("Image must be grayscale with size=(256,256)")

    # The values of inv's elements are indices of u. u is a unique
    # sorted 1D array, so the indexes come from the consecutive
    # vector [0,1,2,...,len(u)-1], which maps 1<->1 to
    # [u_min,u_1,u_2,...,u_max], preserving the order. Indices
    # of inv's values also correspondent to respective u_i
    # coordinates in u

    # This will map img to a new one with pixel values from
    # [1,...,len(u)]
    u, inv = np.unique(img, return_inverse=True)
    inv += 1  # (we need the zero later)
    return inv.reshape(img.shape).astype(np.uint8), len(u)


def convert_to_dataset(images, labels, device) -> TensorDataset:
    """
    Converts numpy-based data sets to device-bound torch dataset
    """
    if labels is not None:
        return TensorDataset(
            FloatTensor(images).to(device), LongTensor(labels).to(device)
        )
    else:
        return TensorDataset(FloatTensor(images).to(device))


def maximize_image(img):
    result = img.copy()
    # preserve zero for background
    return cv2.normalize(img, result, 1, 255, cv2.NORM_MINMAX)


def write_color_image(img, fname):
    # https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    mplot_map = "PuBuGn"
    write_image(
        fname, colormapped_image(img, mplot_map),
    )


def write_sample_image(shape, max_value, samples, fname):
    samples_img = np.zeros(shape, dtype=np.uint8)
    for p in samples:
        samples_img[p[1], p[2]] = np.uint8(255.0 * p[0] / max_value)

    write_color_image(samples_img, fname)
