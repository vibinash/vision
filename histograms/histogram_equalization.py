import cv2
import numpy as np

img = cv2.imread('../images/victoria.jpg',0)

# Does a global histogram equalization causing a global contrast of the img
equal = cv2.equalizeHist(img)

# Stack the images horizontally side-by-side
res = np.hstack((img, equal))

cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
