import cv2
import numpy as np

# Rotation Matrix = [ cos0 -sin0 ]
#                   [ sin0  cos0 ]

# OpenCV provides a scaled rotation with adjustable center of rotation
#  [  alpha  beta   (1 - alpha)*center*x - beta*center*y      ]
#  [  -beta  alpha  beta*center*x + (1 - alpha)*center*y ]

# alpha = scale * cos0
# beta = scale * sin0

img = cv2.imread('../../images/owl.jpg', 1)
rows, cols, color = img.shape

M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('rotated', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
