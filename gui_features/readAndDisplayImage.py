import cv2

# Note: imread second arg flag:  1 color
#		 		 0 grayscale
#		 		 -1 unchanged (including alpha channel) 

# Read in grayscale (0)
img = cv2.imread('../images/jurassic_world.jpg', 0)

# WINDOW_AUTOSIZE - you cannot resize
cv2.namedWindow('jurassic', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)

# wait for any key stroke to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
