import cv2
import numpy as np

green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
hsv_green = hsv_green[0][0][0]
print "Green: " + str(hsv_green)
print "Green: Lower bound: [[[" + str(hsv_green-10) + ",100,100]]]"
print "Green: Upper bound: [[[" + str(hsv_green+10) + ",255,255]]]"

blue  = np.uint8([[[255,0,0 ]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
hsv_blue = hsv_blue[0][0][0]
print "Blue: " + str(hsv_blue)
print "Blue: Lower bound: [[[" + str(hsv_blue-10) + ",100,100]]]"
print "Blue: Upper bound: [[[" + str(hsv_blue+10) + ",255,255]]]"

red = np.uint8([[[0,0,255 ]]])
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
hsv_red = hsv_red[0][0][0]
print "Red: " + str(hsv_red)
print "Red: Lower bound: [[[" + str(hsv_red-10) + ",100,100]]]"
print "Red: Upper bound: [[[" + str(hsv_red+10) + ",255,255]]]"
