import numpy as np
from scipy.spatial.distance import cdist

# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#

def computeMatches(f1, f2):
    """ Match two sets of SIFT features f1 and f2 """

    threshold = 0.8

    N = f1.shape[0]
    M = f2.shape[0]
    ret = -np.ones(N, dtype=int)

    if M == 1:
        ret[:] = 0
    else:
        distances = cdist(f1, f2, metric='euclidean')

        indices = np.argsort(distances, axis=1)
        best = indices[:, 0]
        second_best = indices[:, 1]

        lowest_dist = distances[np.arange(N), best]
        second_lowest = distances[np.arange(N), second_best]

        valid_match = (lowest_dist/second_lowest) < threshold
        valid_match = np.where(valid_match)[0]
        ret[valid_match] = best[valid_match]

    return ret
