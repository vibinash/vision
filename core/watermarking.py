import cv2
import numpy as np

# load the images
img = cv2.imread('../images/jurassic_world.jpg')
logo = cv2.imread('../images/horse_logo_cropped.jpg')

# create an ROI
rows, cols, channels = logo.shape
roi = img[0:rows, 0:cols]

# Create a mask and inverse mask of logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Black out the area of the logo in ROI
img_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

# Take only region of logo from logo image
logo_fg = cv2.bitwise_and(logo, logo, mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img_bg, logo_fg)
img[0:rows, 0:cols] = dst

cv2.imshow('watermarked', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
