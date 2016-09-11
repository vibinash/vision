import cv2
import numpy as np

img = cv2.imread('../images/star.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im,contours,hierarchy = cv2.findContours(thresh, 1, 2)

# Image Moments
cnt = contours[0]
M = cv2.moments(cnt)
print 'Moments: ', M

# Centroid
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print 'Cx: ', cx
print 'Cy: ', cy

# Contour Area
area = cv2.contourArea(cnt)
print 'Area: ', area

# Contour Perimeter
perimeter = cv2.arcLength(cnt,True)
print 'Perimeter: ', perimeter

# Convex Hull
# converHull(contours, hull (AVOID), clockwise[True/False], returnPoints[True/False])
hull = cv2.convexHull(cnt)
print 'Hull: ', hull

# Check if Contour is Convex?
k = cv2.isContourConvex(cnt)
print 'Is Convex? : ', k

# Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('star', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
