import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../images/owl.jpg')

kernel = np.ones((5,5), np.float32)/25

# Images can be filtered with LPF, HPF etc
# LPF removes noise and blurs the image
# HPF helps finds edges in an image

# Filtering with kernel results in: for each pixel, a 5x5 window is centered on
# this pixel, all pixels falling within this window are summed up and the result
# is then divided by 25. Same as computing the average of the pixel values inside
# that window. This is performed for all pixels in the image
dst = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
