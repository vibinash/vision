import cv2
import numpy as np

img = cv2.imread('../../images/owl.jpg', 1)

# Resize with INTER_CUBIC (Typically slower)
res = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

# Resize with INTER_LINEAR
res1 = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_LINEAR)

# OR INTER_AREA
height, width = img.shape[:2]
res2 = cv2.resize(img, (2*width, 2*height), interpolation = cv2.INTER_AREA)


cv2.imshow('cubic', res)
cv2.imshow('linear', res1)
cv2.imshow('area', res2)

cv2.waitKey(0)
cv2.destroyAllWindows()
