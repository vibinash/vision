import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/receipt.jpg',1)
rows, cols, ch = img.shape

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# to find the transformation matrix, we need 4 points. One set from the Input
# image and the other for the output image

# will create a 3x3 matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (cols, rows))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
