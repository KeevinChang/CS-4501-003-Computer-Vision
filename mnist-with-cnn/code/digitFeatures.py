import numpy as np
from scipy.ndimage.filters import convolve
import time


# EXTRACTDIGITFEATURES extracts features from digit images
#   features = extractDigitFeatures(x, featureType) extracts FEATURES from images
#   images X of the provided FEATURETYPE. The images are assumed to the of
#   size [W H 1 N] where the first two dimensions are the width and height.
#   The output is of size [D N] where D is the size of each feature and N
#   is the number of images.
def getFeature(x, featureType):
    if featureType == "pixel":
        dim = np.shape(x)
        features = np.reshape(x, [dim[0] * dim[1], dim[2]])
        normalizeFeatures = True
        if normalizeFeatures:
            features = np.sign(features) * np.sqrt(np.fabs(features))
            normf = np.sqrt(np.sum(features**2, axis=0))
            features = np.matmul(features, np.diag(1 / normf))

    elif featureType == "hog":
        numOri = 8
        oriBinSize = 180 / numOri
        binSize = 4
        numImages = x.shape[2]
        fx = np.array([[1, 0, -1]])
        fy = np.transpose(fx)
        nx = int(np.ceil(x.shape[0] / binSize))
        ny = int(np.ceil(x.shape[1] / binSize))
        features = np.zeros((numOri, nx, ny, numImages))
        for i in range(numImages):
            im = x[:, :, i]
            gx = convolve(im, fx, mode="constant")
            gy = convolve(im, fy, mode="constant")
            mag = np.sqrt(gx**2 + gy**2)
            ori = np.degrees(np.arctan(gy / (gx + 1e-5)))  # -90 to 90 degrees

            oribin = np.minimum(np.floor((ori + 90) / oriBinSize), numOri)

            for xx in range(nx):
                for yy in range(ny):
                    for ori in range(numOri):
                        xx_ = np.arange(
                            xx * binSize, np.minimum(x.shape[1], (xx + 1) * binSize)
                        )
                        yy_ = np.arange(
                            yy * binSize, np.minimum(x.shape[0], (yy + 1) * binSize)
                        )
                        features[ori, yy, xx, i] = sum(
                            mag[yy_, xx_] * (oribin[yy_, xx_] == ori)
                        )

        features = np.reshape(features, [numOri * nx * ny, numImages])
        normalizeFeatures = True
        if normalizeFeatures:
            features = square_root_scale(features)
            # features = l2_norm(features)

    return features


def square_root_scale(features):
    return np.sign(features) * np.sqrt(np.abs(features))


def l2_norm(features):
    norm = np.sqrt(np.sum(features**2, axis=0))
    return features / norm


def zeroFeatures(x):
    return np.zeros((10, x.shape[2]))
