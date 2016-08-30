import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/jurassic_world.jpg',0)

# Canny Edge Detection - multi stage algorithm
# 1. Noise Reduction (using a 5x5 gaussian filter)
# 2. Finding intensity gradient of the image
#  - filtered with a Sobel kernel in both vertical and horizontal directions
#  - we find the edge gradient and direction for each pixel
# 3. Non-maximum suppression
#   - a full scan of the image to remove any unwanted pixels which may not
#     constitute an edge. Looks if the points forms a local maximum, if not
#     it would be suprresed to 0
# 4. Hysteresis Thresholding
#   - decides on which edges are valid and which are not
#   - maxVal and minVal are thresholds

# Canny(input_image, minVal, maxVal)
edges = cv2.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
