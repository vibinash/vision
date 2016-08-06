import cv2
import numpy as np

img = cv2.imread('../../images/owl.jpg',0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('original', img)
cv2.imshow('gray', gray)
cv2.imshow('hsv', hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()
