import numpy as np
import os
import errno
import matplotlib.pyplot as plt
import scipy.io as spio
from skimage.util import montage


# images (x) is the array e.g., data['x']
def montageDigits(x):
    num_images = x.shape[2]
    m = montage(x.transpose(2, 0, 1))
    plt.imshow(m, cmap="gray")
    plt.show()

    return np.mean(x, axis=2)


# Evaluate the accuracy of labels
def evaluateLabels(y, ypred, visualize=True):

    classLabels = np.unique(y)
    conf = np.zeros((len(classLabels), len(classLabels)))
    for tc in range(len(classLabels)):
        for pc in range(len(classLabels)):
            conf[tc, pc] = np.sum(
                np.logical_and(y == classLabels[tc], ypred == classLabels[pc]).astype(
                    float
                )
            )

    acc = np.sum(np.diag(conf)) / y.shape[0]

    if visualize:
        plt.figure()
        plt.imshow(conf, cmap="gray")
        plt.ylabel("true labels")
        plt.xlabel("predicted labels")
        plt.title("Confusion matrix (Accuracy={:.2f})".format(acc * 100))
        plt.show()

    return (acc, conf)


# Show matching between a pair of images
def showMatches(im1, im2, c1, c2, matches, title=""):
    """
    Visualizes matches between two images.

    Parameters:
      im1, im2 : np.array
          Input images (grayscale or RGB).
      c1, c2 : np.array
          Keypoint coordinates as 2 x N arrays. The original code swaps the rows, so we assume the first row is y and second row is x.
      matches : np.array
          Array of length N, where matches[i] is the index in c2 corresponding to keypoint i in c1, or -1 if no match.
      title : str
          Title for the plot.
    """
    # Build match pairs and filter out invalid matches.
    disp_matches = np.vstack([np.arange(matches.shape[0]), matches]).T.astype(int)
    valid_matches = np.where(matches >= 0)[0]
    disp_matches = disp_matches[valid_matches, :]  # shape: (M, 2)

    # Create concatenated image.
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    out_h = max(h1, h2)
    out_w = w1 + w2
    # Ensure images are 3-channel for color plotting.
    if im1.ndim == 2:
        im1 = np.stack([im1] * 3, axis=-1)
    if im2.ndim == 2:
        im2 = np.stack([im2] * 3, axis=-1)
    out_img = np.zeros((out_h, out_w, 3), dtype=im1.dtype)
    out_img[:h1, :w1, :] = im1
    out_img[:h2, w1 : w1 + w2, :] = im2

    # Convert keypoint coordinates.
    # The original code does: c1[[1, 0], :].astype(int).T so we assume that is desired.
    keypoints1 = c1[[1, 0], :].astype(int).T  # shape: (N, 2)
    keypoints2 = c2[[1, 0], :].astype(int).T  # shape: (N, 2)

    # Plot the results.
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(out_img)
    for match in disp_matches:
        idx1, idx2 = match
        pt1 = keypoints1[idx1]
        pt2 = keypoints2[idx2].copy()
        # Offset the x-coordinate of keypoints in im2.
        pt2[0] += w1
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], "r-", linewidth=1)
        ax.plot(pt1[0], pt1[1], "bo", markersize=3)
        ax.plot(pt2[0], pt2[1], "bo", markersize=3)
    ax.set_title(title)
    ax.axis("off")
    plt.show()


# Thanks to mergen from https://stackoverflow.com/questions/7008608
def todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict


# data.mat is a dict with the following structure:
#   {train:
#       {x: array with raw data (image height, image width, number of images)}
#       {y: array with corresponding labels}},
#   {test:
#       {x: array with raw data (image height, image width, number of images)}
#       {y: array with corresponding labels}}
def loadmat(path):
    return todict(spio.loadmat(path, struct_as_record=False, squeeze_me=True)["data"])


def imread(path):
    img = plt.imread(path).astype(float)
    # Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    # Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0

    return img


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        print("Directory {} already exists.".format(dirpath))


# Thanks to ali_m from https://stackoverflow.com/questions/17190649
def gaussian(hsize=3, sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    shape = (hsize, hsize)
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
