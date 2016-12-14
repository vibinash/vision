import cv2
import numpy as np

img = cv2.imread('../../images/puzzle.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

minLineLength = 100
maxLineGap = 10

# Probabilistic Hough Transform is an optimization of Hough Transform
# It takes a random subset of points that is sufficient for line detection
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength, maxLineGap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)

print 'number of lines found', len(lines)

cv2.imshow('lines found', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
