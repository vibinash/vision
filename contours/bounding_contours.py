import cv2
import numpy as np

img = cv2.imread('../images/bolt.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
im,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[len(contours)-1]

# Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
image_rect = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# Rotated Rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
image_rect = cv2.drawContours(image_rect,[box],0,(0,0,255),2)

cv2.imshow('bolt_rect', image_rect)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Minimum Encolsing circle
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
image_circle = cv2.circle(img,center,radius,(0,255,0),2)

# Fitting an Ellipse
ellipse = cv2.fitEllipse(cnt)
image_circle = cv2.ellipse(image_circle,ellipse,(0,255,0),2)

cv2.imshow('bolt_circle', image_circle)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Fitting a line
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
img_line = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

cv2.imshow('bolt_line', img_line)
cv2.waitKey(0)
cv2.destroyAllWindows()
