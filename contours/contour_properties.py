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
print '     Length of mask of object (all points): ', len(pixelpoints)

# Max Value, Min Value, Max Value Location, Min Value Location
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray, mask = mask)
print 'Min Value: ', min_val
print 'Max Value: ', max_val
print 'Min Loc: ', min_loc
print 'Min Loc: ', max_loc

# Mean Color/ Mean Intensity
mean_val = cv2.mean(imgray, mask = mask)
print 'Mean Value: ', mean_val

# Get extreme points
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
print 'Left most point: ', leftmost
img = cv2.circle(img, leftmost, 5, (255,0,0), -1)
img = cv2.circle(img, rightmost, 5, (255,0,0), -1)
img = cv2.circle(img, topmost, 5, (255,0,0), -1)
img = cv2.circle(img, bottommost, 5, (255,0,0), -1)

# Convexity Defects
# Any deviation of this object from the convex hull can be
# considered as Convexity Defects
# adding parameter returnPoints = False, finds the convexity defects
# returns an array where each row is
# [start pt, end pt, farthest pt, approx dist to farthest pt]
dhull = cv2.convexHull(cnt, returnPoints = False)
defects = cv2.convexityDefects(cnt, dhull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img, start, end, [0,255,0],2)
    cv2.circle(img, far, 5, [0,0,255], -1)

print 'Dimensions of defects: ', defects.shape

# Point Polygon Test
# Finds the shortest distance between a point in the image and a
# contour. It returns the distance which is negative when the point
# is outside the contour, positive when the point is inside and zero
# if the point is on the contour
# if measureDist = False, it returns -1,0,+1
dist = cv2.pointPolygonTest(cnt, (50,50), measureDist = True)
print 'Point Polygon Test: ', dist

# Match Shapes
# compare two shapes or contours and returns a metric demonstrating
# the similarity. The lower the result, the better match it is
# calculate on hu-moment values
# ret = cv2.matchShapes(cnt1, cnt2, 1, 0, 0)

# Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('star', img)
# cv2.imshow('points', pixelpoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
