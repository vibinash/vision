import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/receipt.jpg',0)

# Gradient or High Pass Filter
# Sobel - joint Gaussian smoothing plus differentiation opertations (more resistant to noise)
# direction of the derivatives (vertical - yorder, horizontal - xorder)
# ksize - size of the kernel
# ksize = -1, a 3x3 Scharr filter is used

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=3)

plt.subplot(2,2,1), plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,2), plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3), plt.imshow(sobelx, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,4), plt.imshow(sobely, cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
