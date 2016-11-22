import numpy as np
import cv2

OPENCV_PATH = '/Users/vibinash/Documents/workspace/opencv'

face_cascade = cv2.CascadeClassifier(OPENCV_PATH + '/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(OPENCV_PATH + '/data/haarcascades/haarcascade_eye.xml')

img = cv2.imread('../images/obama.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.5, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex+x, ey+y), (ex+ew+x, ey+eh+y), (0,255,0), 2)

print 'Number of faces detected: ', len(faces)
# print 'Number of eyes detected: ', len(eyes)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
