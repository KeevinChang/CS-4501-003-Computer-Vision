# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#

import numpy as np
import cv2
import math
from scipy.ndimage import gaussian_laplace, maximum_filter


def detectBlobs(im, param=None):
    # Input:
    #   IM - input image
    #
    # Ouput:
    #   BLOBS - n x 4 array with blob in each row in (x, y, radius, score)
    #
    # Dummy - returns a blob at the center of the image

    if im.ndim == 3:
        im = (im * 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im = cv2.equalizeHist(im)

    im = im.astype(np.float32)/255

    starting_sigma = 2
    num_scales = 15
    k = 1.2
    threshold = 0.005

    h, w = im.shape[0], im.shape[1]

    scale_space = np.zeros((h, w, num_scales))
    sigmas = starting_sigma * np.power(k, np.arange(num_scales))

    for i, sigma in enumerate(sigmas):
        log_response = gaussian_laplace(im, sigma) * (sigma ** 2)
        scale_space[:, :, i] = log_response ** 2

    maxima = (scale_space == maximum_filter(scale_space, size=(3, 3, 3)))

    blobs = []
    for i in range(num_scales):
        y, x = np.where((scale_space[:, :, i] > threshold) & maxima[:, :, i])
        r = sigmas[i] * math.sqrt(2)
        scores = scale_space[y, x, i]
        blobs.extend(zip(x, y, [r] * len(x), scores))

    blobs = np.array(blobs)
    if blobs.size > 1000:
        blobs = blobs[np.argsort(-blobs[:, 3])]

    return blobs