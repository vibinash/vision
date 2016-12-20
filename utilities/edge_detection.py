import cv2
import numpy as np

# Image should be in grayscale (single channel pixel intensities)
def auto_canny(image, sigma=0.33):
    # Compute the median
    median = np.median(image)

    # find the upper and lower thresholds based on sigma
    # lower threshold include higher frequency noise
    lower = int(max(0, (1.0 - sigma) * median))
    # upper threshold includes lower frequency noise
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(image, lower, upper)

    return edges
