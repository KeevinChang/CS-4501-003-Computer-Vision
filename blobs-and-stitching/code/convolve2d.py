import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

img = Image.open('cat.jpg').convert('L')
img = np.array(img, dtype=float)

# Kernel for gradient in X direction
horizontal_prewitt = np.array([[-1, 0, 1],
                               [-1, 0, 1],
                               [-1, 0, 1]])

# Kernel for gradient in Y direction
vertical_prewitt = np.array([[1, 1, 1],
                             [0, 0, 0],
                             [-1, -1, -1]])

# 5x5 Kernel (5x5 of ones then divide by 25 to average)
average_grid = np.ones((5,5), dtype=float)/25

horizontal_grad = convolve2d(img, horizontal_prewitt, boundary='symm', mode='same')
vertical_grad = convolve2d(img, vertical_prewitt, boundary='symm', mode='same')
average_grad = convolve2d(img, average_grid, boundary='symm', mode='same')

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(horizontal_grad, cmap='gray')
plt.title('Gradient along X')

plt.subplot(1,3,2)
plt.imshow(vertical_grad, cmap='gray')
plt.title('Gradient along Y')

plt.subplot(1,3,3)
plt.imshow(average_grad, cmap='gray')
plt.title('5x5 Average Filter')

plt.tight_layout()
plt.show()
