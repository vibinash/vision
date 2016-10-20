import cv2
import numpy as np

img = cv2.imread('../images/star.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im,contours,hierarchy = cv2.findContours(thresh, 1, 2)

print 'Hierarchy: ', hierarchy
