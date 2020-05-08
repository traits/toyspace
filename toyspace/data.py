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


def selectSample(img, sampler, *args):
    samples = sampler(img, *args)
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


def ROI_sampler(img, rect):
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


def random_sampler(img, number):
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


def random_sampler2(img, contour, number):
    """
    Samples pixel from random coordinates in contour
    
    Parameters:
        :img: input gray image
        :contour: ROI as OpenCV contour
        :number: number of pixel drawn (w/o replacement)
        :return: 1D-array of (pixel_value, y, x) samples
    """

    # calculate rect. hull
    rhull = cv2.boundingRect(contour)
    # calc. area quotient
    ra = rhull[2] * rhull[3]
    ca = cv2.contourArea(contour)
    q = ra / ca
    # increase 'number' accordingly
    number = int(np.round(number * q))
    # sample in rect. hull

    x, y, w, h = rhull

    roi = img[y : y + h, x : x + w]
    Y, X = np.indices(roi.shape)
    Y += y
    X += x
    samples = np.dstack((roi.copy(), Y, X))
    samples = samples.reshape(
        -1, samples.shape[2]
    )  # create 1D array of (p,Y,X) 'points'
    random_indices = np.arange(0, samples.shape[0])  # array of all indices
    np.random.shuffle(random_indices)

    samples = samples[random_indices[:number]]

    # mask with contour
    samples = [
        s for s in samples if 0 < cv2.pointPolygonTest(contour, (s[2], s[1]), False)
    ]
    return samples  # get N samples without replacement

    # write output


def partition_sampler(img):
    """
    Creates partition of the whole input image. Every sub-array contains all elements 
    with the same pixel value and associated coordinates (pixel_value, y, x).
    
    Parameters:
        :img: input gray image
        :return: partition of the img
    """
    y, x = np.indices(img.shape)
    samples = np.dstack((img.copy(), y, x))
    samples = samples.reshape(
        -1, samples.shape[2]
    )  # create 1D array of (p,y,x) 'points'

    samples = samples[samples[:, 0].argsort()]  # sort, using pixel value
    first = samples[:, 0]  # sorted 1D array of pixel values
    # calculate indices, where pixel values (class) change
    _, indices = np.unique(first, return_index=True)
    # split the original array at these indices
    return np.split(samples, indices)


def grid_sampler(img, steps, rect=None):
    """
    Returns values at grid coordinates for required coordinate density.
    Coordinates will be rounded, so they are not quite equidistant (up to 2 pixel)
    
    Parameters:
        :img: input gray image
        :steps: number of equidistant sampling points per coordinate: [xpoints, ypoints]
        :rect:  ROI: [x0,y0,x1,y1] or None (whole img)
        :return: 1D-array of (pixel_value, y, x) samples
    """

    x_steps, y_steps = steps

    if not rect:
        rect = [0, 0, img.shape[1] - 1, img.shape[0] - 1]

    x0 = rect[0]
    y0 = rect[1]
    x1 = rect[2]
    y1 = rect[3]

    X = np.around(np.linspace(x0, x1, x_steps)).astype(np.uint8)
    Y = np.around(np.linspace(y0, y1, y_steps)).astype(np.uint8)

    return np.asarray([np.array([img[y, x], y, x]) for x in X for y in Y])
