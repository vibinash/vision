import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/owl.jpg')

# Blurring is achieved by convolving the image with a low pass filter kernel.
# Useful for removing noise like high frequency content (like noise, edges) resulting
# in edges being blurred

# Averaging is done by convolving the image with a normalized box filter.
# Takes the average of all the pixels under the kernel area and replaces the centeral
# element with this average.

# use a kernel of 5x5
# alternativly use cv2.boxFilter()
blur = cv2.blur(img, (5,5))

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Avg Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
