import cv2
import numpy as np

img = cv2.imread('../../images/letter_out.jpg', 0)
ret, thresh = cv2.threshold(img, 127,255, cv2.THRESH_BINARY_INV)

# The size of the kernel affects the amount of erosion
kernel = np.ones((25,25), np.uint8)

# opening  = Erosion followed by Dilation
# useful in removing noise
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow('Original', thresh)
cv2.imshow('Opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
