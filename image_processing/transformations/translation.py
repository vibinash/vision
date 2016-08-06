import cv2
import numpy as np

# Translation Matrix
# M = [ 1 0 tx]
#     [ 0 1 ty]

img = cv2.imread('../../images/jurassic_world.jpg', 1)
rows, cols, color = img.shape

# Shifting by (100,50)
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
