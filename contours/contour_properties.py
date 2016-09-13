import cv2
import numpy as np

img = cv2.imread('../images/star.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]

# Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# Aspect Ratio:
# Ratio of width to hieght of bounding Rectangle
aspect_ratio = float(w)/h
print 'Aspect Ratio: ', aspect_ratio

# Extent
# Ratio of contour area to bounding rectangle area
area = cv2.contourArea(cnt)
bound_rect_area = w * h
extent = float(area)/bound_rect_area
print 'Extent: ', extent

# Solidity
# Ratio of contour area to its convex hull area
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print 'Solidity: ', solidity

# Equivalent Diameter
# Diameter of the circle whose area is the same as the contour area
equ_diameter = np.sqrt(4*area/np.pi)
print 'Equivalent Diameter: ', equ_diameter

# Orientation
# angle at which the object is directed. Also returns Major and Minor Axis length
(x,y), (Ma, ma), angle = cv2.fitEllipse(cnt)
print 'Orientation: '
print '     Angle on X: ', x
print '     Angle on Y: ', y
print '     length of Major Axis: ', Ma
print '     length of minor axis: ', ma

# Mask and Get Pixel Points
# Get all the points comprising of that object
mask = np.zeros(imgray.shape, np.uint8)
cv2.drawContours(mask, [cnt], 0,255,-1)
# Numpy function returns coordinates in (row, column)
pixelpoints = np.transpose(np.nonzero(mask))
# CV gives coordinates in (x,y) format
# Note: row = x, column = y
# pixelpoints = cv2.findNonZero(mask)

# Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('star', img)
# cv2.imshow('points', pixelpoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
