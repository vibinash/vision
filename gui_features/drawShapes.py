import numpy as np
from cv2 import *

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Note: -1 on thickness will fill the closed shape

# Draw a diagonal blue line with thickness 5px
img = line(img, (0,0), (511, 511), (255, 0, 0), 5)

# Draw a rectangle
img = rectangle(img, (384,0), (510,120), (0,255, 0), 3)

# Draw an Ellipse [ Center location(x,y) | (major axis length, minor axis length) | angle | startAngle | endAngle | color | thickness ]
img = ellipse(img, (256,256), (100,50), 0,0,270,(0,255,0), 1)

# Draw a circle [img, center(x, y), radius, color(b,g,r), thickness]
img = circle(img, (477,63), 63, (0,0,255), -1)

# Draw a polygon
# ROWSx1x2
pts = np.array([ [10,5], [20,30], [70,20], [50,10]], np.int32)

imshow('drawing', img)
waitKey(0)
destroyAllWindows()
