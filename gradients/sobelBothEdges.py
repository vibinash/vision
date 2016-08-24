import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/box.jpg',0)

# with output datatype as cv2.CV_8U, black to white transition is a positive slope,
# Whereas a white to black transition is a negative slope
# thus when converting to np.uint8, all negative slopes will be made 0
# therefore losing that edge

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img, cv2.CV_8U,1,0, ksize=3)

# Output dtype = cv2.CV_64F.
sobelx64f = cv2.Sobel(img, cv2.CV_64F,1,0, ksize=3)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2), plt.imshow(sobelx8u, cmap='gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3), plt.imshow(sobel_8u, cmap='gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()
