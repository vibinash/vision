import cv2
import numpy as np,sys
from matplotlib import pyplot as plt

im = cv2.imread('../images/owl.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

# Find contours(source img, contour retreival mode, contour approx method)
image, contours, hierarchy = cv2.findContours(thresh,
        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw Contours(img, contour list, index/-1 for all, color, thickness)
img = cv2.drawContours(imgray, contours, -1, (0,255,0), 2)
cv2.imshow('owl', img)

# cv2.CHAIN_APPROX_NONE, all the boundary points are stored
# cv2.CHAIN_APPROX_SIMPLE, removes all the redundant points and compresses
# the contour, thus saving memory

## Contours NONE APPROX
# Create a black image
im1 = np.zeros((300,300,3), np.uint8)

# Draw a rectangle
im1 = cv2.rectangle(im1, (50,50), (200,200), (255,255,255), -1)

imgray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
ret1, thresh1 = cv2.threshold(imgray1, 127, 255, cv2.THRESH_BINARY)
image1, contours1, hierarchy = cv2.findContours(thresh1,
        cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img1 = cv2.drawContours(im1, contours1, -1, (0,0,255), 3)
cv2.imshow('contour simple method', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
