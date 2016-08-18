import cv2
import numpy as np

img = cv2.imread('../../images/letter.jpg', 0)
ret, thresh = cv2.threshold(img, 127,255, cv2.THRESH_BINARY_INV)

# The size of the kernel affects the amount of erosion
kernel = np.ones((25,25), np.uint8)

# Erodes the boundary of the foreground object. (Keep the foreground in white)
# A pixel in the original image (1/0) will be considered 1 only if all the pixels
# under the kernel is 1, otherwise eroded (0)
erosion = cv2.erode(thresh, kernel, iterations = 1)

cv2.imshow('Original', thresh)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
