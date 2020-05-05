"""
Loading and sampling from Toy space
"""

from pytraits.image.io import *
import numpy as np
from torch.utils.data import TensorDataset
from torch import FloatTensor, LongTensor


def loadToyImage(fpath):
    img = read_image(fpath)
    return img


def selectSample(img, sampler, *args):
    samples = sampler(img, *args)
    # return np.multiply(samples, 1.0 / 255.0), coords
    return samples


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


def sampler_ROI(img, rect):
    """
    Returns complete content of rect. area as array of (pixel_value, y, x) samples 
    
    Parameters:
        :img: input gray image
        :rect: rect. ROI as list [x0,y0,x1,y1]
        :return: 1D-array of (pixel_value, y, x) samples
    """

    x0, y0, x1, y1 = rect

    roi = img[y0:y1, x0:x1]
    samples = roi.copy()
    y, x = np.indices(roi.shape)
    y += y0
    x += x0
    samples = np.dstack((samples, y, x))
    return samples.reshape(-1, samples.shape[2])


def sampler_random(img, number):
    """
    Samples pixel from random coordinates
    
    Parameters:
        :img: input gray image
        :number: number of pixel drawn (w/o replacement)
        :return: 1D-array of (pixel_value, y, x) samples
    """
    y, x = np.indices(img.shape)
    samples = np.dstack((img.copy(), y, x))
    samples = samples.reshape(
        -1, samples.shape[2]
    )  # create 1D array of (p,y,x) 'points'
    random_indices = np.arange(0, samples.shape[0])  # array of all indices
    np.random.shuffle(random_indices)
    return samples[random_indices[:number]]  # get N samples without replacement
