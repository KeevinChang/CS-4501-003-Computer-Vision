import numpy as np
import cv2
import time

# This code is part of:
#
#   CS4501-003: Computer Vision
#   University of Virginia
#   Instructor: Zezhou Cheng
#

def inverse_affine(transf):
    m = transf[:, :2]
    t = transf[:, 2]

    m_inv = np.linalg.inv(m)
    t_inv = -m_inv @ t

    transf_inv = np.hstack((m_inv, t_inv.reshape(2, 1)))
    return transf_inv

def ransac(matches, blobs1, blobs2):
    valid_matches = []
    for i in range(len(matches)):
        if matches[i] != -1:
            valid_matches.append((i, matches[i]))

    iterations = 1000
    threshold = 5
    inliers = []
    transf = None

    for _ in range(iterations):
        random_points = np.random.choice(len(valid_matches), 3, replace=False)

        blob1_points = []
        blob2_points = []

        for i in random_points:
            b1, b2 = valid_matches[i]
            blob1_points.append(blobs1[b1][:2])
            blob2_points.append(blobs2[b2][:2])

        blob1_points = np.array(blob1_points, dtype=np.float32)
        blob2_points = np.array(blob2_points, dtype=np.float32)

        this_transf = cv2.getAffineTransform(blob1_points, blob2_points)
        this_inliers = []

        for b1, b2 in valid_matches:
            p1 = np.array([blobs1[b1][0], blobs1[b1][1], 1])
            transformed = this_transf.dot(p1)[:2]

            distance = np.linalg.norm(transformed - blobs2[b2][:2])

            if distance < threshold:
                this_inliers.append(b1)

        if len(this_inliers) > len(inliers):
            inliers = this_inliers
            transf = this_transf

    # necessary because my matrix is the inverse of the expected transformation
    transf = inverse_affine(transf)

    return inliers, transf
