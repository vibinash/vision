import cv2

# Rectangular Kernel
cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
