import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

import_file_path = os.getcwd().split('vision')[0] + 'vision/utilities'
sys.path.append(import_file_path)
from point_manipulation import transform_edge_points

roi_points = []
frame = None

def selectROI(event, x, y, flags, param):
    global frame, roi_points

    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x,y))
        cv2.circle(frame, (x,y), 4, (0,255,0),2)
        cv2.imshow('frame', frame)

def perspective_transformation(img):
    rows, cols, ch = img.shape

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    # to find the transformation matrix, we need 4 points. One set from the Input
    # image and the other for the output image

    # will create a 3x3 matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (cols, rows))

    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()

def main():
    global roi_points, frame

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', selectROI)
    warped = None
    key = None

    while key != ord('q') and key != 113:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        while len(roi_points) < 4 and key != ord('q') and key != 113:
            print 'Press any key when done adding points'
            key = cv2.waitKey(0)

        if warped is None and len(roi_points) >= 4:
            warped = transform_edge_points(frame, np.array(roi_points))
            cv2.imshow('warped', warped)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    frame = cv2.imread('../../images/receipt.jpg',1)
    main()
