import cv2
import numpy as np

img = cv2.imread('../../images/letter.jpg', 0)
ret, thresh = cv2.threshold(img, 127,255, cv2.THRESH_BINARY_INV)

# The size of the kernel affects the amount of erosion
kernel = np.ones((5,5), np.uint8)

# Gradient = difference between dilation and erosion
closing = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('Original', thresh)
cv2.imshow('Gradient', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
