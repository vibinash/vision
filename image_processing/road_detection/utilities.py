import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

class Utilities:

    class Undistort:
        def __init__(self, LOG_LEVEL='NONE'):
            if LOG_LEVEL == 'INFO':
                self.LOG_LEVEL = 1
            elif LOG_LEVEL == 'DEBUG':
                self.LOG_LEVEL = 2
            else: # No logging
                self.LOG_LEVEL = 0

            self.objpoints = None
            self.imgpoints = None
            self.shape = None

            try:
                self.objpoints = np.load('data/objpoints.npy')
                self.imgpoints = np.load('data/imgpoints.npy')
                self.shape = tuple(np.load('data/shape.npy'))
            except:
                pass

            if self.objpoints is None or self.imgpoints is None:
                print 'No data files found, calibrating camera...'
                self.find_corners()

            #print self.shape
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.shape, None, None)

        def find_corners(self):
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0)... (6,5,0)
            objp = np.zeros((6*9,3), np.float32)
            objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

            # Store object points and image points from all the images
            self.objpoints =[] # 3d points in the real world space
            self.imgpoints =[] # 2d points in image plane

            list_images = glob.glob('images/camera_cal/calibration*.jpg')

            for file_name in list_images:
                img = cv2.imread(file_name)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the cheshboard corners
                ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

                self.shape = gray.shape[::-1]

                # Once found, add object and image points
                if ret == True:
                    self.objpoints.append(objp)
                    self.imgpoints.append(corners)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                    if self.LOG_LEVEL >0:
                        cv2.imshow('chessboard', img)
                        cv2.waitKey(500)
            if self.LOG_LEVEL >0:
                cv2.destroyAllWindows()

            np.save('data/objpoints', self.objpoints)
            np.save('data/imgpoints', self.imgpoints)
            np.save('data/shape', self.shape)

        def undistort(self, img):
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    class Threshold:
        def __init__(self):
            self.SOBEL_KERNEL_SIZE = 15
            self.THRESH_DIR_MIN = 0.7
            self.THRESH_DIR_MAX = 1.2
            self.THRESH_MAG_MIN = 50
            self.THRESH_MAG_MAX = 255

        def dir_thresh(self, sobelx, sobely):
            abs_sobelx = np.abs(sobelx)
            abs_sobely = np.abs(sobely)
            scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)
            sxbinary = np.zeros_like(scaled_sobel)
            sxbinary[(scaled_sobel >= self.THRESH_DIR_MIN) & (scaled_sobel <= self.THRESH_DIR_MAX)] = 1
            return sxbinary

        def mag_thresh(self, sobelx, sobely):
            gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
            scale_factor = np.max(gradmag) / 255
            gradmag = (gradmag / scale_factor).astype(np.uint8)
            binary_output = np.zeros_like(gradmag)
            binary_output[(gradmag >= self.THRESH_MAG_MIN) & (gradmag <= self.THRESH_MAG_MAX)] = 1
            return binary_output

        def color_thresh(self, img):
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            yellow_min = np.array([50, 80, 0], np.uint8)
            yellow_max = np.array([255, 255, 255], np.uint8)
            yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

            white_min = np.array([0, 0, 200], np.uint8)
            white_max = np.array([255, 255, 255], np.uint8)
            white_mask = cv2.inRange(hsv, white_min, white_max)

            binary_output = np.zeros_like(img[:,:,0])
            binary_output[((yellow_mask != 0) | (white_mask !=0 ))] = 1

            hsv = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_or(yellow_mask, white_mask))
            final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            return binary_output

        def threshold(self, img):
            sobelx = cv2.Sobel(img[:,:,2], cv2.CV_64F, 1, 0, ksize=self.SOBEL_KERNEL_SIZE)
            sobely = cv2.Sobel(img[:,:,2], cv2.CV_64F, 0, 1, ksize=self.SOBEL_KERNEL_SIZE)

            direc = self.dir_thresh(sobelx, sobely)
            mag = self.mag_thresh(sobelx, sobely)
            color = self.color_thresh(img)

            result = np.zeros_like(direc)
            result[((color == 1) & ((mag == 1) | (direc == 1)))] = 1
            return result

    class Warper:
        def __init__(self):
            self.src = np.float32([
                [500, 460],
                [700, 460],
                [1040, 680],
                [260, 680]
            ])

            self.dst = np.float32([
                [260, 0],
                [1040, 0],
                [1040, 720],
                [260, 720]
            ])

            self.M = cv2.getPerspectiveTransform(self.src, self.dst)
            self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

        def warp(self, img):
            return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]),
                flags=cv2.INTER_LINEAR)

        def unwarp(self, img):
            return cv2.warpPerspective(img, self.Minv, (img.shape[1], img.shape[0]),
                flags=cv2.INTER_LINEAR)

    class PolyFitter:
        def __init__(self):
            self.left_fit = None
            self.right_fit = None
            self.leftx = None
            self.rightx = None

        def polyfit(self, img):
            return self.polyfit_sliding(img)

        def polyfit_sliding(self, img):
            histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
            out_img = np.dstack((img, img, img)) * 255
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nwindows = 9
            window_height = np.int(img.shape[0] / nwindows)
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            leftx_current = leftx_base
            rightx_current = rightx_base
            margin = 100
            minpix = 50
            left_lane_inds = []
            right_lane_inds = []

            for window in range(nwindows):
                win_y_low = img.shape[0] - (window + 1) * window_height
                win_y_high = img.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            self.leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            self.rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            self.left_fit = np.polyfit(lefty, self.leftx, 2)
            self.right_fit = np.polyfit(righty, self.rightx, 2)

            return self.left_fit, self.right_fit

    class PolyDrawer:
        def draw(self, img, left_fit, right_fit, Minv):
            color_warp = np.zeros_like(img).astype(np.uint8)

            fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
            left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
            right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, fity]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])
            pts = np.hstack((pts_left, pts_right))
            pts = np.array(pts, dtype=np.int32)

            cv2.fillPoly(color_warp, pts, (0, 255, 0))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
            # Combine the result with the original image
            result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

            return result
