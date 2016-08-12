import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/owl.jpg')

'''
 Bilateral filter is defined for highly effective at noise removal while perserving
 the edges. Uses the gaussian filter but uses one more (multiplicative) gaussian
 filter component which is a function of pixel intensity differences.
 The Gaussian function of space makes sure that only pixels are 'spatial neighbors'
 are considered for filtering, while the Gaussian component applied in the intensity
 domain (a Gaussian function of intensity differences) ensures that only those pixels
 with intensities similar to that of the central pixel ('intensity neighbors') are
 included to compute the blurred intensity value. As a result, this method preserves
 edges, since for pixels lying near edges, neighboring pixels placed on the other
 side of the edge, and therefore exhibiting large intensity variations when compared
 to the central pixel, will not be included for blurring.
'''
# Param 1: Kernel size (positive and odd)
dst = cv2.bilateralFilter(img,9,75,75)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img), plt.title('Bilateral Filter')
plt.xticks([]), plt.yticks([])
plt.show()
