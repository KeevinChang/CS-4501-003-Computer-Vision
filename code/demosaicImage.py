# This code is part of:
#
#   CS4501-00:  Computer Vision, Spring 2025
#   University of Virginia
#   Instructor: Zezhou Cheng

import numpy as np


def demosaicImage(image, method):
    """Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    """

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == "nn":
        return demosaicNN(image.copy())  # Implement this
    elif method.lower() == "linear":
        return demosaicLinear(image.copy())  # Implement this
    elif method.lower() == "adagrad":
        return demosaicAdagrad(image.copy())  # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    """Baseline demosaicing.

    Replaces missing values with the mean of each color channel.

    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline
        algorithm.
    """
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[1:image_height:2, 1:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 0] = img[
        1:image_height:2, 1:image_width:2
    ]

    blue_values = img[0:image_height:2, 0:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 2] = img[
        0:image_height:2, 0:image_width:2
    ]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    """Nearest neighbor demosaicing.

    Args:
        img: np.array of size NxM.
    """

    N, M = img.shape
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])

    # red onto blue
    for i in range(0, N-1, 2):
        for j in range(0, M-1, 2):
            # take from bottom right
            mos_img[i, j, 0] = mos_img[i + 1, j + 1, 0]

    # red onto green in blue rows
    for i in range(0, N-1, 2):
        for j in range(1, M-1, 2):
            # take from below unless bottom row, then take from top
            if i == N-1:
                mos_img[i, j, 0] = mos_img[i+1, j, 0]
            else:
                mos_img[i, j, 0] = mos_img[i-1, j, 0]

    # red onto green in red rows
    for i in range(1, N-1, 2):
        for j in range(0, M-1, 2):
            # take from right unless on rightmost column, then take from left
            if j == M-1:
                mos_img[i, j, 0] = mos_img[i, j-1, 0]
            else:
                mos_img[i, j, 0] = mos_img[i, j+1, 0]

    # blue onto red
    for i in range(1, N - 1, 2):
        for j in range(1, M - 1, 2):
            # take from top left
            mos_img[i, j, 2] = mos_img[i - 1, j - 1, 2]

    # blue onto green in blue rows
    for i in range(0, N - 1, 2):
        for j in range(1, M - 1, 2):
            # take from left
            mos_img[i, j, 2] = mos_img[i, j - 1, 2]

    # blue onto green in red rows
    for i in range(1, N - 1, 2):
        for j in range(0, M - 1, 2):
            # take from top
            mos_img[i, j, 2] = mos_img[i - 1, j, 2]

    # green channel
    for i in range(0, N):
        start = 0 if i % 2 == 0 else 1
        for j in range(start, M, 2):
            # take the one to the right unless in rightmost column, then take from left
            if j == M-1:
                mos_img[i, j, 1] = mos_img[i, j-1, 1]
            else:
                mos_img[i, j, 1] = mos_img[i, j+1, 1]

    return mos_img


def demosaicLinear(img):
    """Nearest neighbor demosaicing.

    Args:
        img: np.array of size NxM.
    """
    N, M = img.shape
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])

    # red onto blue
    for i in range(0, N - 1, 2):
        for j in range(0, M - 1, 2):
            mos_img[i, j, 0] = (mos_img[i+1, j+1, 0] + mos_img[i+1, j-1, 0] +
                                mos_img[i-1, j+1, 0] + mos_img[i-1, j-1, 0])/4

    # red onto green in blue rows
    for i in range(0, N - 1, 2):
        for j in range(1, M - 1, 2):
            mos_img[i, j, 0] = (mos_img[i+1, j, 0] + mos_img[i-1, j, 0])/2

    # red onto green in red rows
    for i in range(1, N - 1, 2):
        for j in range(0, M - 1, 2):
            mos_img[i, j, 0] = (mos_img[i, j+1, 0] + mos_img[i, j-1, 0])/2

    # blue onto red
    for i in range(1, N - 1, 2):
        for j in range(1, M - 1, 2):
            mos_img[i, j, 2] = (mos_img[i+1, j+1, 2] + mos_img[i+1, j-1, 2] +
                                mos_img[i-1, j+1, 2] + mos_img[i-1, j-1, 2])/4

    # blue onto green in blue rows
    for i in range(0, N - 1, 2):
        for j in range(1, M - 1, 2):
            mos_img[i, j, 2] = (mos_img[i, j-1, 2] + mos_img[i, j+1, 2])/2

    # blue onto green in red rows
    for i in range(1, N - 1, 2):
        for j in range(0, M - 1, 2):
            mos_img[i, j, 2] = (mos_img[i-1, j, 2] + mos_img[i+1, j, 2])/2

    # green channel
    for i in range(0, N-1):
        start = 0 if i % 2 == 0 else 1
        for j in range(start, M-1, 2):
            mos_img[i, j, 1] = (mos_img[i+1, j, 1] + mos_img[i, j+1, 1] +
                                mos_img[i-1, j, 1] + mos_img[i, j-1, 1])/4

    return mos_img


def demosaicAdagrad(img):
    """Nearest neighbor demosaicing.

    Args:
        img: np.array of size NxM.
    """
    N, M = img.shape
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])

    # red onto blue
    for i in range(0, N-1, 2):
        for j in range(0, M-1, 2):
            gt = mos_img[i+1, j+1, 0]
            gr = mos_img[i+1, j-1, 0]
            gl = mos_img[i-1, j+1, 0]
            gb = mos_img[i-1, j-1, 0]

            if abs(gt-gb) > abs(gl-gr):
                mos_img[i, j, 0] = (gl+gr)/2
            else:
                mos_img[i, j, 0] = (gt+gb)/2


    # red onto green in blue rows
    for i in range(0, N-1, 2):
        for j in range(1, M-1, 2):
            mos_img[i, j, 0] = (mos_img[i+1, j, 0] + mos_img[i-1, j, 0])/2

    # red onto green in red rows
    for i in range(1, N-1, 2):
        for j in range(0, M-1, 2):
            mos_img[i, j, 0] = (mos_img[i, j+1, 0] + mos_img[i, j-1, 0])/2

    # blue onto red
    for i in range(1, N-1, 2):
        for j in range(1, M-1, 2):
            gt = mos_img[i+1, j+1, 2]
            gr = mos_img[i+1, j-1, 2]
            gl = mos_img[i-1, j+1, 2]
            gb = mos_img[i-1, j-1, 2]

            if abs(gt-gb) > abs(gl-gr):
                mos_img[i, j, 2] = (gl+gr)/2
            else:
                mos_img[i, j, 2] = (gt+gb)/2

    # blue onto green in blue rows
    for i in range(0, N-1, 2):
        for j in range(1, M-1, 2):
            mos_img[i, j, 2] = (mos_img[i, j-1, 2] + mos_img[i, j+1, 2])/2

    # blue onto green in red rows
    for i in range(1, N-1, 2):
        for j in range(0, M-1, 2):
            mos_img[i, j, 2] = (mos_img[i-1, j, 2] + mos_img[i+1, j, 2])/2

    # green channel
    for i in range(0, N - 1):
        start = 0 if i % 2 == 0 else 1
        for j in range(start, M - 1, 2):
            gt = mos_img[i-1, j, 1]
            gb = mos_img[i+1, j, 1]
            gl = mos_img[i, j-1, 1]
            gr = mos_img[i, j+1, 1]

            if abs(gt-gb) > abs(gl-gr):
                mos_img[i, j, 1] = (gl+gr)/2
            else:
                mos_img[i, j, 1] = (gt+gb)/2

    return mos_img
