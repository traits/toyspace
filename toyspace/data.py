"""
Loading and sampling from Toy space
"""

from pytraits.image.io import *
import numpy as np
from torch.utils.data import TensorDataset
from torch import FloatTensor


def loadToyImage(fpath):
    img = read_image(fpath)
    return img


def selectSample(img, sampler, *args):
    samples = sampler(img, *args)
    # return np.multiply(samples, 1.0 / 255.0), coords
    return samples


def convert2Dataset(images, device) -> TensorDataset:
    """
    Convert numpy-based data sets to device-bound torch dataset
    """
    return TensorDataset(FloatTensor(images).to(device))
