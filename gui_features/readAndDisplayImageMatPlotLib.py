import cv2
import numpy as np
from matplotlib import pyplot as plt

# Note: imread second arg flag:  1 color
#		 		 0 grayscale
#		 		 -1 unchanged (including alpha channel) 

# Read in grayscale (0)
img = cv2.imread('jurassic_world.jpg', 1)

# Convert (BGR) to (RBG)
img2 = img[:,:,::-1]

plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([])
plt.yticks([])
plt.show()

