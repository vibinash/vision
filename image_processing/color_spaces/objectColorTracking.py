import cv2
import numpy as np

mode = 'blue'

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# define range of green color in HSV
lower_green = np.array([50,100,100])
upper_green = np.array([70,255,255])

# define range of red color in HSV
lower_red = np.array([-10,100,100])
upper_red = np.array([10,255,255])

# global lower and upper hsv parameters (Default to blue)
lower = lower_blue
upper = upper_blue

cap = cv2.VideoCapture(0)

while(1):
	k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
		mode = 'red'
		lower = lower_red
		upper = upper_red
	elif k == ord('b'):
		mode = 'blue'
		lower = lower_blue
		upper = upper_blue
	elif k == ord('g'):
		mode = 'green'
		lower = lower_blue
		upper = upper_blue
	elif k == 27:
		break

	_, frame = cap.read()

	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# Threshold the HSV image to get the right color range
	mask = cv2.inRange(hsv, lower, upper)

	# Bitwise AND mask and original image
	res = cv2.bitwise_and(frame, frame, mask = mask)

	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	cv2.imshow('res', res)

cv2.destroyAllWindows()
