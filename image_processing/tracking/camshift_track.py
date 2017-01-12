import numpy as np
import argparse
import cv2

# initialize the current frame of the video, list of
# ROI points along with the input mode
frame = None
roi_points = []
input_mode = False

def selectROI(event, x, y, flags, param):
    global frame, roi_points, input_mode

    if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x,y))
        cv2.circle(frame, (x,y), 4, (0,255,0), 2)
        cv2.imshow('frame', frame)

def main():
    # construct and parse args
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', help='\
        path to optional video file')
    args = vars(ap.parse_args())

    global frame, roi_points, input_mode

    if not args.get('video', False):
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(args['video'])

    # set mouse callback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', selectROI)

    # termination criteria for a maximum of ten iterations
    # or movement by a least 1 pixel along with the bounding
    # box of the ROI
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roi_box = None

    while True:
        (success, frame) = camera.read()

        if not success:
            break

        if roi_box is not None:
            # convert the frame to the HSV color space
            # perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)

            # apply cam shift to the back projection
            # convert the points to bounding box and draw them

            # cv2.CamShift( <output of the histogram back projection>,
            # <estimated bounding box of the object to be tracked>,
            # <termination criteria> )
            # RETURNS: <estimated position, size and orientation of the object>
            # RETURNS: <newly estimated position of the ROI> - fed into the next Camshift calls
            (r, roi_box) = cv2.CamShift(back_proj, roi_box, termination)
            pts = np.int0(cv2.boxPoints(r))
            cv2.polylines(frame, [pts], True, (0,255,0), 2)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # selection mode
        if key == ord('i') and len(roi_points) < 4:
            input_mode = True
            orig = frame.copy()

            while len(roi_points) < 4:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

            # determine the top-left and bottom-right points
            roi_points = np.array(roi_points)
            s = roi_points.sum(axis =1)
            tl = roi_points[np.argmin(s)]
            br = roi_points[np.argmax(s)]

            # grab the ROI for the bounding box and convert it to
            # HSV color space
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # compute a HSV histogram for ROI and store the
            # bounding box
            roi_hist = cv2.calcHist([roi], [0], None, [16], [0,180])
            roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            roi_box = (tl[0], tl[1], br[0], br[1])

        # if the 'q' key is pressed, stop the loop
        elif key == ord('q') or key == 113:
            break

    # cleanup
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
