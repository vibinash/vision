import cv2
import numpy as np
from matplotlib import pyplot as plt


# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img].
# channels : It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively. eg: [0]
# mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask.
# histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
# ranges : this is the RANGE. Normally, it is [0,256].

img = cv2.imread('../images/jurassic_world.jpg',0)

# create a mask ()
mask = np.zeros(img.shape[:2], np.uint8)
# set a region of the mask as white/blank
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img, img, mask = mask)

# calculate histogram with mask and without mask
hist_full = cv2.calcHist([img], [0], None, [256], [0,256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(img, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()
