import cv2
import numpy as np

# CLAHE: contrast Limited Adaptive histogram equalization
# Sometimes global contrast of the image is not a good idea
# since certain parts of the image can face over-brightness

# Adaptive histogram equalization is where the image is divided into small
# blocks called 'tiles'. Contrast Limiting is used to prevent noise being
# amplified. If any bin is above the specified limit (default: 40), those
# pixels are clipped and distributed uniformly to other bins before applying
# equalization. Bilinear interopation is applied to remove artifacts in the
# tile borders

img = cv2.imread('../images/victoria.jpg', 0)

# Create a CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

cv2.imshow('clahe', cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()
