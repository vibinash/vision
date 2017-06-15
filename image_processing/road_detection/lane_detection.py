import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
from utilities import Utilities

def showImage(name, img):
    cv2.namedWindow(name, flags=cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(name, 100, 100)
    cv2.moveWindow(name, 10,10)
    cv2.imshow(name, img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.destroyWindow(name)

class LaneDetection:

    # two levels of log_level = info, debug
    def __init__(self, LOG_LEVEL='INFO'):
        if LOG_LEVEL == 'INFO':
            self.LOG_LEVEL = 1
        elif LOG_LEVEL == 'DEBUG':
            self.LOG_LEVEL = 2
        else: # No logging
            self.LOG_LEVEL = 0
        self.IMAGE_NAME =  'flat_road_left.jpg'
        # IMAGE_NAME = 'long_road.jpg'
        # IMAGE_NAME = 'winding_road.jpg'
        self.detected_lanes = []

        # Advanced Lane detection
        self.undistorter = Utilities.Undistort()
        self.thresholder = Utilities.Threshold()
        self.warper = Utilities.Warper()
        self.polyfitter = Utilities.PolyFitter()
        self.polydrawer = Utilities.PolyDrawer()

    def gaussianBlur(self, img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def findHoughLines(self, img_edges, threshold=20, minLineLength=50, maxLineGap=2):
        # minLineLength =100 #120
        # maxLineGap = 10 #20
        # threshold = 20
        # HoughLinesP( img, rho (in pixels) eg: 1, theta (in degree = 1), threshold (200), minLineLength, maxLineGap )
        lines = cv2.HoughLinesP(img_edges, 1, np.pi/180, threshold, np.array([]), minLineLength, maxLineGap)

        if lines is None:
            print 'No lines found'
        else:
            if self.LOG_LEVEL >1:
                print 'Lines found: ', len(lines)
        return lines

    def findCannyEdges(self, img, min_val=50, max_val=150, aperture_size=3):
        # Canny(input_image, minVal, maxVal, apertureSize [Sobel kernel])
        canny_edges = cv2.Canny(img, min_val, max_val, apertureSize=aperture_size)
        if self.LOG_LEVEL >1:
            showImage('Canny', canny_edges)
        return canny_edges

    def splitImageIntoPart(self, img, y_axis, x_axis, part, total_part):
        x_start, x_end, y_start, y_end = (0, img.shape[1], 0, img.shape[0])
        img_shape = img.shape
        if x_axis:
            delta_x = img_shape[1]/total_part
            x_start = delta_x * part
            x_end = x_start + delta_x
        if y_axis:
            delta_y = img_shape[0]/total_part
            y_start = delta_y * part
            y_end = y_start + delta_y
        return x_start, x_end, y_start, y_end

    # Note: roi_type = None is also permitted
    # In that case, the image is not masked
    def getRoi(self, img, roi_type='polygon'):
        imshape = img.shape
        # ploygon_vertices = np.array([[(110,imshape[0]),(410, 310),(480, 310), (imshape[1],imshape[0])]], dtype=np.int32)
        triangle_vertices = np.array([[(0, imshape[0]),(imshape[1]/2, imshape[0]/2),(imshape[1],imshape[0])]], dtype=np.int32)
        polygon_vertices = np.array([[(0, imshape[0]),(imshape[1]/2-imshape[1]/8, imshape[0]/2+imshape[0]/8),(imshape[1]/2+imshape[1]/8, imshape[0]/2+imshape[0]/8),(imshape[1],imshape[0])]], dtype=np.int32)
        # noisy
        # polygon_vertices = np.array([[(0, imshape[0]),(0, 3*imshape[0]/4),(imshape[1]/4,
        #    imshape[0]/2),(3*imshape[1]/4,imshape[0]/2),(imshape[1],3*imshape[0]/4),(imshape[1], imshape[0])]], dtype=np.int32)
        mask = np.zeros_like(img)

        # Use a mask color of the same number of channels as the image
        if len(imshape) > 2:
            channel_count = imshape[2]
            ignore_mask_color = (255,)* channel_count
        else:
            ignore_mask_color = 255

        if roi_type == 'triangle':
            cv2.fillPoly(mask, triangle_vertices, ignore_mask_color)
        elif roi_type == 'polygon':
            cv2.fillPoly(mask, polygon_vertices, ignore_mask_color)

        #if self.LOG_LEVEL >1:
        #    showImage('roi_type_mask', mask)
        masked_image = cv2.bitwise_and(img, mask)
        if self.LOG_LEVEL >0:
            showImage('masked_image', masked_image)
        return masked_image

    def separateLines(self, lines):
        left = []
        right = []
        for line in lines:
            slope = self.findSlope(line)
            # since the y-axis is inverted, the slopes are inverted too
            # i.e: + slope = left line and - slope = right line
            if slope > 0.3:
                right.append(line)
            elif slope < -0.3: # Dont care of lines with slope of 0
                left.append(line)
        return np.array(left), np.array(right)

    def findExtremeLines(self, img, lines):
        if lines is not None:
            # find lines farthest and smallest x with a positive or negative slope
            leftmost_line, rightmost_line = -1, -1
            leftmost_x, rightmost_x = img.shape[1], 0
            left_bottommost_y = 0
            right_bottommost_y = 0
            # max_slope, min_slope = 0, 0
            sum_left_slope, sum_right_slope = 0, 0
            sum_left_b, sum_right_b = 0, 0
            index =0
            count_valid_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Since the y axis is inverted, the slope should be inverted
                slope = self.findSlope(line)

                if slope > 0:
                    sum_left_slope += slope
                    sum_left_b += y1 - slope * x1
                    count_valid_lines += 1
                elif slope < 0:
                    sum_right_slope += slope
                    sum_right_b += y1 - slope * x1
                    count_valid_lines += 1

                if leftmost_x > min(x1, x2) and left_bottommost_y < max(y1, y2) and slope != 0.0:
                    leftmost_line = index
                    # max_slope = slope
                    leftmost_x = min(x1,x2)
                    left_bottommost_y = max(y1, y2)
                elif rightmost_x < max(x1, x2) and right_bottommost_y < max(y1, y2) and slope != 0.0:
                    rightmost_line = index
                    # min_slope = slope
                    rightmost_x = max(x1,x2)
                    right_bottommost_y = max(y1, y2)
                index += 1
            if self.LOG_LEVEL >1:
                print 'left most line: ', lines[leftmost_line]
                print 'left slope: ', self.findSlope(lines[leftmost_line])
                print 'right most line: ', lines[rightmost_line]
                print 'right slope: ', self.findSlope(lines[rightmost_line])
                print 'Average left slope: ', sum_left_slope/count_valid_lines
                print 'Average right slope: ', sum_right_slope/count_valid_lines
                print 'Average left B: ', sum_left_b/count_valid_lines
                print 'Average right B: ', sum_right_b/count_valid_lines
            extremeLines = [lines[leftmost_line], lines[rightmost_line]]
            result = self.drawLines(img, extremeLines, (0,0,255), 2)
            result = self.extendExtremeLines(result, extremeLines)
            result = self.findIntersection(result, lines[leftmost_line], lines[rightmost_line])
            return result
        else:
            return img

    def average_lines(self, lines):
        # x1, y1, x2, y2 = np.mean(lines, axis=0)[0]
        # line = np.array([[x1, y1, x2, y2]])
        if lines.size >0:
            line = np.mean(lines, axis=0)
            return np.array([line], dtype=np.int32)
        else:
            return np.array([[]], dtype=np.int32)

    def extendLine(self, line, scale):
        # Extend lines using unit vectors
        # line = np.asarray(line, dtype=int)
        if line.size > 0:
            x1, y1, x2, y2 = line[0][0]
            line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            x_1 = int(x2 + (x2 - x1)/ line_len * -1 * scale)
            y_1 = int(y2 + (y2 - y1)/ line_len * -1 * scale)

            x_2 = int(x2 + (x2 - x1)/ line_len * scale)
            y_2 = int(y2 + (y2 - y1)/ line_len * scale)
            line = np.array([[x_1,y_1, x_2, y_2]])
        return np.array([line], dtype=np.int32)

    def extendExtremeLines(self, img, lines):
        max_y = img.shape[0]
        min_y = max_y/2
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Extend lines using angles
            angle = math.atan2((y1 - y2), (x2 - x1)) * 180/ math.pi

            p_y = int(img.shape[0]/2) - (img.shape[0] - min(y1,y2))
            p_y_orig = img.shape[0]/2
            y_max = img.shape[0]/2

            # Extend lines using unit vectors
            line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            x_bottom = int(x2 + (x2 - x1)/ line_len * -1000)
            y_bottom = int(y2 + (y2 - y1)/ line_len * -1000)

            x_top = int(x2 + (x2 - x1)/ line_len * 1000)
            y_top = int(y2 + (y2 - y1)/ line_len * 1000)

            img = cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), (0,255,255), 2)

            # Extend lines using Y-intercept and slope
            if y2 > y1:
                m = (y2 - y1)/(x2 - x1 * 1.0)
            else:
                m = (y1 - y2)/(x1 - x2 * 1.0)

            b =  -1 * m * x1 + y1

            if m != 0:
                p_x = int(math.fabs((y_max - b)/m))

                ## img = cv2.line(img, (x1, y1), (p_x, y_max), (0,0,255), 2)

            if angle < 0:
                p_x = int(p_y/math.tan((180 + angle) * math.pi/180)) + x1
            else:
                p_x = int(p_y/math.tan(angle * math.pi/180)) + x1

            if self.LOG_LEVEL >1:
                print 'angle: ', angle
                print 'Point x', p_x
                print 'Point y', p_y
                print 'Slope of new line ', self.findSlope([[x1, y1, p_x, p_y]])
            img = cv2.circle(img, (p_x, p_y_orig), 5, (255,0,0), -1)
            ##  img = cv2.line(img, (x1, y1), (p_x, p_y_orig), (255,0,0), 2)
        return img

    def findIntersection(self, img, left_line, right_line):
        l_x1, l_y1, l_x2, l_y2 = left_line[0]
        r_x1, r_y1, r_x2, r_y2 = right_line[0]

        l_m = self.findSlope(left_line)
        r_m = self.findSlope(right_line)
        l_b = l_y1 - l_m * l_x1
        r_b = r_y1 - r_m * r_x1

        try:
            x = int((-l_b + r_b)/ (l_m - r_m))
            y = int(l_m * l_x1 + l_b)

            if self.LOG_LEVEL >1:
                print 'X of Intercept: ',x
                print 'Y of Intercept: ',y
            img = cv2.circle(img, (x, y), 10, (255,0,255), -1)
        except:
            if self.LOG_LEVEL >1:
                print 'Unable to find intercept'
            pass
        return img

    def findSlope(self, line):
        x1, y1, x2, y2 = line[0]
        # Since the y axis is inverted, the slope should be inverted
        return (y2 - y1)/((x2 - x1) * 1.0)

    def getLength(self, line):
        x1, y1, x2, y2 = line[0]
        return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    def drawLines(self, img, lines, color, thickness):
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # Example: cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    img = cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            return img
        else:
            print 'Empty list of lines'
            return img

    def weightedImage(self, image_1, image_2, alpha=0.8, beta=1.0, lamda=0.):
        # image_1 * alpha + image * beta + lamda
        return cv2.addWeighted(image_1, alpha, image_2, beta, lamda)

    def fitLine(self, left_lines, right_lines):
        # Use Linear Regression to fit a line best for the points
        left_x =[]
        left_y =[]
        right_x = []
        right_y =[]

        for line in left_lines:
            for x1, y1, x2, y2 in line:
                left_x += [x1, x2]
                left_y += [y1, y2]

        for line in right_lines:
            for x1, y1, x2, y2 in line:
                right_x += [x1, x2]
                right_y += [y1, y2]

        left_poly_line = np.polynomial.Polynomial.fit(np.array(left_x), np.array(left_y), 1)
        right_poly_line = np.polynomial.Polynomial.fit(np.array(right_x), np.array(right_y), 1)

        l_x1 = (left_poly_line - img.shape[0]).roots()
        r_x1 = (right_poly_line - img.shape[0]).roots()

        l_x2 = (left_poly_line - imshape[0]/2).roots()
        r_x2 = (right_poly_line - img.shape[0]/2).roots()

        left_line = np.array([[l_x1, result.shape[0], l_x2, result.shape[0]/2+result.shape[0]/8]])
        right_line = np.array([[r_x1, result.shape[0], r_x2, result.shape[0]/2+result.shape[0]/8]])
        return left_line, right_line

    def low_pass_filter(self, left_line, right_line, weight):
        if len(self.detected_lanes) < 2:
            self.detected_lanes.append(left_line[0][0])
            self.detected_lanes.append(right_line[0][0])
            return left_line, right_line
        else:
            left_lane = self.smooth_points(left_line, weight, 0)
            right_lane = self.smooth_points(right_line, weight, 1)
            return np.array([left_lane], dtype=np.int32), np.array([right_lane], dtype=np.int32)

    def smooth_points(self, line, weight, index):
        if len(line[0]) > 0:
            x1, y1, x2, y2 = line[0][0]

            prev_x1, prev_y1, prev_x2, prev_y2 = self.detected_lanes[index]
            new_x1 = x1 * (1-weight) + prev_x1 * weight
            new_y1 = y1 * (1-weight) + prev_y1 * weight
            new_x2 = x2 * (1-weight) + prev_x2 * weight
            new_y2 = y2 * (1-weight) + prev_y2 * weight
            self.detected_lanes[index] = [new_x1, new_y1, new_x2, new_y2]
            return np.array([[new_x1, new_y1, new_x2, new_y2]])
        else:
            return line

    def filter_lane_colors(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        yellow_min = np.array([50, 80, 0], np.uint8)
        yellow_max = np.array([255, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsv, yellow_min, yellow_max)

        white_min = np.array([0, 0, 200], np.uint8)
        white_max = np.array([255, 255, 255], np.uint8)
        white_mask = cv2.inRange(hsv, white_min, white_max)

        hsv = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_or(yellow_mask, white_mask))
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if self.LOG_LEVEL >0:
            showImage('filteredColors', final)

        return final

    def find_lanes(self, img):
        emphasedLaneColors = self.filter_lane_colors(img)
        #showImage('emphasedLaneColors',emphasedLaneColors)
        gray = cv2.cvtColor(emphasedLaneColors, cv2.COLOR_BGR2GRAY)

        # Blur the image
        filtered = cv2.bilateralFilter(gray,9,75,75)

        # Perform a adaptive histogram equalization to deal with contrast changes
        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(filtered)

        imshape = gray.shape

        SPLIT_SIZE = 1

        for i in range(SPLIT_SIZE):
            blurred = self.gaussianBlur(equalized, 3)
            canny_edges = self.findCannyEdges(blurred) #, min_val=60, max_val=140)
            roi_image = self.getRoi(canny_edges)
            x_start, x_end, y_start, y_end = self.splitImageIntoPart(roi_image, True, False, i, SPLIT_SIZE)

            if self.LOG_LEVEL >1:
                print 'x_start', x_start
                print 'x_end', x_end
                print 'y_start', y_start
                print 'y_end', y_end
            splitted_roi = roi_image[y_start:y_end, x_start:x_end]
            # clone = img.copy()
            lines = self.findHoughLines(splitted_roi, 15, 20, 2) #, threshold=15, minLineLength=65, maxLineGap=2)
            result = None
            if lines is not None and len(lines) > 0:
                # draw the Hough Lines
                line_image = np.copy((img)*0)
                resultHoughLines = self.drawLines(line_image, lines, (0,255,0, 0.5), 2)
                if self.LOG_LEVEL >1:
                    showImage('resultHoughLines', resultHoughLines)
                    print splitted_roi.shape
                left_lines, right_lines = self.separateLines(lines)

                ## Average each set of lines to get the final detected lane
                left_lines = self.average_lines(left_lines)
                right_lines = self.average_lines(right_lines)

                left_lane, right_lane = self.low_pass_filter(left_lines,right_lines, 0.7)

                left_lane = self.extendLine(left_lane, 1000)
                right_lane = self.extendLine(right_lane, 1000)

                if left_lane.size > 0 and right_lane.size > 0:
                    lines = np.concatenate((left_lane, right_lane), axis=0)
                    result_lines = self.drawLines(line_image, lines, (0,0,255), 5)
                #else:
                #    lines = left_lane if right_lane.size == 0 else right_lane

                if left_lane.size == 0:
                    left_lane = np.array([[self.detected_lanes[0]]])
                    left_lane = self.extendLine(left_lane, 1000)
                    result_lines = self.drawLines(line_image, left_lane, (255,0,0), 5)
                if right_lane.size ==0:
                    right_lane = np.array([[self.detected_lanes[1]]])
                    right_lane = self.extendLine(right_lane, 1000)
                    result_lines = self.drawLines(line_image, right_lane, (255,0,0), 5)

                result = self.getRoi(result_lines)
                result = self.weightedImage(result, img)
                # result = self.findExtremeLines(resultHoughLines, lines)
            return result

    def find_lanes_advanced(self, img):
        # Undistort the image
        undistort = self.undistorter.undistort(img)

        # Use gradients, color transforms etc to create a thresholded binary image
        thresh = self.thresholder.threshold(undistort)

        # Apply a perspective transform to rectifiy binary image ('birds-eye view')
        if self.LOG_LEVEL >0:
            perspective_lines = np.copy(thresh)
            # pts = np.array([[500, 460],[700, 460],[1040, 680],[260, 680]], dtype=np.int32)
            pts = np.array(self.warper.src, dtype=np.int32)
            src_img = cv2.polylines(perspective_lines, [pts], True, (255,0,0), 3)
            showImage('src', src_img)

            pts = np.array(self.warper.dst, dtype=np.int32)
            dst_img = cv2.polylines(perspective_lines, [pts], True, (255,0,0), 3)
            showImage('dst', dst_img)

        warp = self.warper.warp(thresh)

        # Detect lane pixels to fit and find the lane boundary
        left_fit, right_fit = self.polyfitter.polyfit(warp)

        # Determine the curvature of the lane and vehichle position with respect to center
        # Warp the detected lane boundaries back to original image
        # Output visual display of the lane boundaries and numerical estimation
        lane_boundary = self.polydrawer.draw(undistort, left_fit, right_fit, self.warper.Minv)

        # of lane curvature and vehicle position

        return lane_boundary

if __name__ == "__main__":
    ld = LaneDetection(LOG_LEVEL = 'INFO')
    img = cv2.imread('../../images/'+ ld.IMAGE_NAME)
    result = ld.find_lanes(img)
    showImage('result', result)
