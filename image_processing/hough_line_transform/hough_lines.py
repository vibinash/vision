import cv2
import numpy as np

# Hough Transform is a popular technique to detect shape, if it can be
# represented in mathematical form

# line:
# Cartesian form: y = mx + b
# Parametric form: p = xcos(theta) + ysin(theta)
# p = perpendicular distance from orgin to line
# theta = angle formed by this perpendicular line and the horizontal axis

img = cv2.imread('../../images/puzzle.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# cv2.HoughLines(binary_img, p accuracies, theta accuracies, threshold -
# min vote to be considered a line)
# returns an array of (p, theta) values
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

for line in lines:
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 + 1000*(a))

        cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

print 'number of lines found', len(lines)

cv2.imshow('lines found', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
