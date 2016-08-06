import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/receipt.jpg',1)
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# to find the transformation matrix, we need 3 points. One set from the Input
# image and the other for the output image

# will create a 2x3 matrix
M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
