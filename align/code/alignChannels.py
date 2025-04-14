# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#
import numpy as np


def central_crop(img, crop):
    height, width = img.shape[0], img.shape[1]
    crop_height, crop_width = crop[0], crop[1]
    start_height = (height - crop_height)//2
    start_width = (width - crop_width)//2
    return img[start_height:start_height + crop_height, start_width:start_width + crop_width]


def sum_square_diff(img1, img2):
    return np.sum((img1-img2)**2)


def alignChannels(img, max_shift):

    # ensure img has the 3 channels
    assert img.shape[2] == 3

    reference_channel = img[:, :, 0]
    pred_shift = np.zeros((2, 2), dtype=int)

    # tune this
    crop_size = [19*reference_channel.shape[0]//20, 19*reference_channel.shape[1]//20]

    cropped_ref = central_crop(reference_channel, crop_size)

    for channel in [1, 2]:
        shift = (0, 0)
        best_score = float("inf")

        for shift_i in range(-max_shift[0], max_shift[0]+1):
            for shift_j in range(-max_shift[1], max_shift[1] + 1):
                cropped_test_channel = central_crop(np.roll(img[:, :, channel], [shift_i, shift_j], axis=[0, 1]),
                                                    crop_size)
                cur_ssd = sum_square_diff(cropped_ref, cropped_test_channel)

                if cur_ssd < best_score:
                    shift = (shift_i, shift_j)
                    best_score = cur_ssd
        pred_shift[channel-1] = shift
        img[:, :, channel] = np.roll(img[:, :, channel], [shift[0], shift[1]], axis=(0, 1))

    # assume max_shift is the most amount of boundary we need to crop out
    border = max(max_shift)
    img = img[border:-border, border:-border, :]
    return img, pred_shift
