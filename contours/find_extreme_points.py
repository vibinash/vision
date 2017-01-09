import cv2

image = cv2.imread('../images/hand.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)

# threshold the image, perform erosions & dilations
# to remove small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]
c = max(cnts, key=cv2.contourArea)

# find extreme points along contour
ext_left = tuple(c[c[:, :, 0].argmin()][0])
ext_right = tuple(c[c[:, :, 0].argmax()][0])
ext_top = tuple(c[c[:, :, 1].argmin()][0])
ext_bot = tuple(c[c[:, :, 1].argmax()][0])

# draw the outline of the object, then draw the
# extreme points
cv2.drawContours(image, [c], -1, (0,255,255), 2)
cv2.circle(image, ext_left, 8, (0,0,255), -1)
cv2.circle(image, ext_right, 8, (0,255,0), -1)
cv2.circle(image, ext_top, 8, (255,0,0), -1)
cv2.circle(image, ext_left, 8, (255,255,0), -1)

# show the output image
cv2.imshow('image', image)
cv2.waitKey(0)
