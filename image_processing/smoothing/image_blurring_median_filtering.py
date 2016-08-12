import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/owl.jpg')

# Computes the median of all the pixeds under that kernel window and the central
# pixel is replaced with this median value.
# Effective in removing salt-and-pepper noise
# Note: for Gaussian and box filters, the filtered value may not exist in the Original
# image but for the median filtering, the central element will always be replaced
# with some pixel value in the image
# Param 1: Kernel size (positive and odd)
dst = cv2.medianBlur(img, 5)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img), plt.title('Median Blur')
plt.xticks([]), plt.yticks([])
plt.show()
