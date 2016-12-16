import numpy as np
import cv2

OPENCV_PATH = '/Users/vibinash/Documents/workspace/opencv'

face_cascade = cv2.CascadeClassifier(OPENCV_PATH + '/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(OPENCV_PATH + '/data/haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == 113:
        break

    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.5,
                minNeighbors=5,
                minSize=(30,30)
            )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex+x, ey+y), (ex+ew+x, ey+eh+y), (0,255,0), 2)
#    print 'Number of faces detected: ', len(faces)

    cv2.imshow('img_frame', frame)

cv2.destroyAllWindows()
