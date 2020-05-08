"""
Loading and sampling from Toy space
"""

from pytraits.image.io import *
import numpy as np
from torch.utils.data import TensorDataset
from torch import FloatTensor, LongTensor


def loadToyImage(fpath):
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


def convert2Dataset(images, labels, device) -> TensorDataset:
    """
    Converts numpy-based data sets to device-bound torch dataset
    """
    if labels is not None:
        return TensorDataset(
            FloatTensor(images).to(device), LongTensor(labels).to(device)
        )
    else:
        return TensorDataset(FloatTensor(images).to(device))
