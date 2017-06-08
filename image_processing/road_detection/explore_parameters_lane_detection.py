import cv2
import numpy as np
import argparse
from lane_detection import LaneDetection
import sys

original = None
image = None

def nothing(x):
    pass

def init():
    USAGE = "explore_parameters_lane_detection.py -i '<PATH_TO_IMAGE>'"
    # construct and parse args
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image file')
    args = vars(ap.parse_args())

    global original, image

    if args.get('image', False):
        original = cv2.imread(args['image'])
        gray = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)
        image = cv2.bilateralFilter(gray,9,75,75)

        cv2.namedWindow('img')

        # Create trackbars
        cv2.createTrackbar('apply mask', 'img', 0, 1, nothing)
        cv2.createTrackbar('canny high threshold', 'img', 0, 255, nothing)
        cv2.createTrackbar('canny low threshold', 'img', 0, 255, nothing)
        cv2.createTrackbar('hough threshold', 'img', 1, 255, nothing)
        cv2.createTrackbar('hough max line gap', 'img', 1, 255, nothing)
        cv2.createTrackbar('hough min line length', 'img', 1, 255, nothing)
    else:
        print USAGE
        sys.exit()

def main():
    global original, image
    tracker = {'apply mask': 0, 'canny high threshold': 0
                ,'canny low threshold' : 0, 'canny low threshold': 0
                , 'hough threshold': 1, 'hough max line gap': 1
                , 'hough min line length': 1}

    img = image
    img_g = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX

    ld = LaneDetection()

    while(True):
        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # Get current positions of the trackers
        apply_mask = cv2.getTrackbarPos('apply mask', 'img')
        canny_high_threshold = cv2.getTrackbarPos('canny high threshold', 'img')
        canny_low_threshold = cv2.getTrackbarPos('canny low threshold', 'img')
        hough_threshold = cv2.getTrackbarPos('hough threshold', 'img')
        hough_max_line_gap = cv2.getTrackbarPos('hough max line gap', 'img')
        hough_min_line_length = cv2.getTrackbarPos('hough min line length', 'img')

        if canny_high_threshold != tracker['canny high threshold'] or canny_low_threshold != tracker['canny low threshold']:
            img = ld.findCannyEdges(image.copy(), canny_low_threshold, canny_high_threshold)
            img_g = img
            tracker['canny high threshold'] = canny_high_threshold
            tracker['canny low threshold'] = canny_low_threshold
            '''
            print 'Canny_low: ', canny_low_threshold
            print 'Canny_high: ', canny_high_threshold
            '''

        if apply_mask == 0 and apply_mask != tracker['apply mask']:
            img = image.copy()
        elif apply_mask == 1:
            img = ld.getRoi(img)

        if hough_threshold != tracker['hough threshold'] or hough_max_line_gap != tracker['hough max line gap'] or hough_min_line_length != tracker['hough min line length']:
            lines = ld.findHoughLines(img_g, hough_threshold, hough_min_line_length, hough_max_line_gap)
            canvas = np.zeros(original.shape, np.uint8)
            canvas = ld.drawLines(canvas, lines, (0, 200, 0), 2)
            img_c = cv2.cvtColor(img_g.copy(), cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(canvas, img_c)
            tracker['hough threshold'] = hough_threshold
            tracker['hough max line gap'] = hough_max_line_gap
            tracker['hough min line length'] = hough_min_line_length
            '''
            print 'hough_threshold', hough_threshold
            print 'hough_max_line_gap', hough_max_line_gap
            print 'hough_min_line_length', hough_min_line_length
            '''

        # Put the text of tracker's parameters on the iamge
        cv2.putText(img, 'Canny high thresh: '+ str(canny_high_threshold), (10, 440), font, 0.5, (255,255,255),1, cv2.LINE_AA)
        cv2.putText(img, 'Canny low thresh: '+ str(canny_low_threshold), (10, 460), font, 0.5, (255,255,255),1, cv2.LINE_AA)
        cv2.putText(img, 'hough thresh: '+ str(hough_threshold), (10, 480), font, 0.5, (255,255,255),1, cv2.LINE_AA)
        cv2.putText(img, 'hough max line gap: '+ str(hough_max_line_gap), (10, 500), font, 0.5, (255,255,255),1, cv2.LINE_AA)
        cv2.putText(img, 'hough min line length: '+ str(hough_min_line_length), (10, 520), font, 0.5, (255,255,255),1, cv2.LINE_AA)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    init()
    main()
