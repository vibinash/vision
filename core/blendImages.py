import cv2
import numpy as np
from time import sleep

img1 = cv2.imread('../images/jurassic_world_crop.jpg')
img2 = cv2.imread('../images/owl.jpg')

dst = None
weight = 0
destroyWindows = False
while weight <= 1 and destroyWindows == False:
	if cv2.waitKey(20) & 0xFF == 27:
		destroyWindows = True
	dst = cv2.addWeighted(img1, weight, img2, 1-weight, 0)
	weight += 0.1
	cv2.imshow('blended', dst) 
	sleep(0.25)

if destroyWindows or cv2.waitKey(0):
	cv2.destroyAllWindows()
