import cv2
import numpy as np

img = cv2.imread('../../images/letter_in.jpg', 0)
ret, thresh = cv2.threshold(img, 127,255, cv2.THRESH_BINARY_INV)

# The size of the kernel affects the amount of erosion
kernel = np.ones((5,5), np.uint8)

dilation = cv2.dilate(thresh, kernel, iterations=3)

# closing (reverse of opening) = Dilation followed by Erosion
# useful in removing holes inside foreground objects
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', thresh)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
