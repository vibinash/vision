import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/owl.jpg')

# Instead of using a box filter consisting of equal filter coefficients, a
# Gaussian Blur is used.
# Param 2: specify the width and height of the kernel (should be postive & odd)
# Param 3: specify the standard deviation (sigmaX, sigmaY)
#          If sigmxX is specified, sigmaY is taken to equal signmaX
#          If both are zero, they are then calculated from the kernel size.
dst = cv2.GaussianBlur(img, (5,5), 0)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img), plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.show()
