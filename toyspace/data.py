"""
Loading and sampling from Toy space
"""

from pytraits.image.io import *
import numpy as np
from torch.utils.data import TensorDataset
from torch import FloatTensor


def applyColormap(img, matplot_map_name):
    """
    Applies matplotlib colormap to opencv grayscale image
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcol
    import cv2

    cmap = plt.get_cmap(matplot_map_name)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # replace 1st entry by black
    cmaplist[0] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )  # clip this later, if using 1.0 values for color components
    # colormap = mcol.LinearSegmentedColormap.from_list("pytraits", cmaplist, cmap.N)
    colormap = mcol.ListedColormap(cmaplist, "pytraits", cmap.N)
    cmap = colormap(img) * 2 ** 16
    # np.clip(cmap, 0, 2 ** 16 - 1, out=cmap)  # avoid overflows (see above)
    result = cmap.astype(np.uint16)[:, :, :3]
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def loadToyImage(fpath):
    img = read_image(fpath)
    return img


def selectSample(img, sampler, *args):
    samples = sampler(img, *args)
    # return np.multiply(samples, 1.0 / 255.0), coords
    return samples


def convert2Dataset(images, device) -> TensorDataset:
    """
    Converts numpy-based data sets to device-bound torch dataset
    """
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
