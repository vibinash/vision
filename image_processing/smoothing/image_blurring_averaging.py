import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/owl.jpg')

# use a kernel of 5x5
blur = cv2.blur(img, (5,5))

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Avg Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
