import numpy as np
from skimage import io
from skimage.color import rgb2gray
import random
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt


def synthRandomPatch(img, tileSize, numTiles, outSize):
    out = np.zeros((outSize, outSize), dtype=img.dtype)

    for i in range(0, outSize, tileSize):
        for j in range(0, outSize, tileSize):
            top = img.shape[0] - tileSize
            bot = img.shape[1] - tileSize
            y = np.random.randint(0, top+1)
            x = np.random.randint(0, bot+1)

            patch = img[y:y+tileSize, x:x+tileSize]
            out[i:i+tileSize, j:j+tileSize] = patch
    return out


def synthEfrosLeung(img, winsize, outSize):
    out = np.full((outSize, outSize), np.nan, dtype=float)

    h, w = img.shape
    mid = winsize // 2
    mask = np.zeros((outSize, outSize), dtype=bool)

    seed_y = np.random.randint(0, h - 3)
    seed_x = np.random.randint(0, w - 3)
    seed_patch = img[seed_y:seed_y + 3, seed_x:seed_x + 3]
    center_y = outSize // 2 - 1
    center_x = outSize // 2 - 1
    out[center_y:center_y + 3, center_x:center_x + 3] = seed_patch
    mask[center_y:center_y + 3, center_x:center_x + 3] = True

    while np.isnan(out).any():
        dilated = binary_dilation(mask)
        border = dilated & ~mask
        candidates = np.argwhere(border)

        pixel_list = []
        for x, y in candidates:
            top = max(0, x - 1)
            bottom = min(outSize, x + 2)
            left = max(0, y - 1)
            right = min(outSize, y + 2)
            count = np.sum(mask[top:bottom, left:right])
            pixel_list.append((count, (x, y)))
        pixel_list.sort(key=lambda p: p[0], reverse=True)

        filled = False

        for pixel in pixel_list:
            x, y = pixel[1]

            out_top = x - mid
            out_left = y - mid

            window = np.full((winsize, winsize), np.nan, dtype=float)
            window_mask = np.zeros((winsize, winsize), dtype=bool)

            for i in range(winsize):
                for j in range(winsize):
                    out_i = out_top + i
                    out_j = out_left + j
                    if 0 <= out_i < outSize and 0 <= out_j < outSize:
                        window[i, j] = out[out_i, out_j]
                        window_mask[i, j] = mask[out_i, out_j]

            if not window_mask.any():
                continue

            ssd_list = []
            min_ssd = float('inf')
            for i in range(mid, h - mid):
                for j in range(mid, w - mid):
                    img_window = img[i - mid:i + mid + 1, j - mid:j + mid + 1]
                    diff = (img_window - window)[window_mask]
                    ssd = np.sum(diff ** 2)
                    if ssd < min_ssd:
                        min_ssd = ssd
                    ssd_list.append((ssd, (i, j)))

            if len(ssd_list) == 0:
                continue

            err_threshold = 0.1
            best_matches = []
            for ssd, match in ssd_list:
                if ssd <= min_ssd * (1 + err_threshold):
                    best_matches.append(match)

            if len(best_matches) == 0:
                continue

            rand_pick = random.choice(best_matches)
            out[x, y] = img[rand_pick[0], rand_pick[1]]
            mask[x, y] = True
            filled = True

        if not filled:
            break

    return out


# Load images
#img = rgb2gray(io.imread('../data/texture/D20.png')[:,:,:3])
img = rgb2gray(io.imread('../data/texture/Texture2.bmp')[:,:,:3])
#img = rgb2gray(io.imread('../data/texture/english.jpg')[:,:,:3])

# Random patches
tileSize = 40 # specify block sizes
numTiles = 5
outSize = numTiles * tileSize # calculate output image size
# implement the following, save the random-patch output and record run-times
im_patch = synthRandomPatch(img, tileSize, numTiles, outSize)
plt.imshow(im_patch, cmap='gray')
plt.show()
plt.savefig('random_patch.png')


# Non-parametric Texture Synthesis using Efros & Leung algorithm  
winsize = 15 # specify window size (5, 7, 11, 15)
outSize = 70 # specify size of the output image to be synthesized (square for simplicity)
# implement the following, save the synthesized image and record the run-times
im_synth = synthEfrosLeung(img, winsize, outSize)
plt.imshow(im_synth, cmap='gray')
plt.show()
plt.savefig('synth_efros_leung.png')

