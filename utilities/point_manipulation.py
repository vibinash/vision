import numpy as np

def order_points(pts):
    # Initialize a rectangular result list  in this order
    # (top-left, top-right, bottom-right, bottom-left)
    result = np.zeros((4,2), dtype='float32')

    # find the top-left and bottom-right
    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]
    result[2] = pts[np.argmax(s)]

    # fdin
    d = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(d)]
    result[3] = pts[np.argmax(d)]

    return result

def transform_edge_points(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # calculate the max height of the new image
    heigthA = int(np.sqrt((tl[1] - bl[1])**2 + (tl[0] - bl[0])**2))
    heigthB = int(np.sqrt((tr[1] - br[1])**2 + (tr[0] - br[0])**2))
    max_height = max(heigthA, heigthB)

    # calculate the max width of the new image
    widthA = int(np.sqrt((tl[1] - tr[1])**2 + (tl[0] - tr[0])**2))
    widthB = int(np.sqrt((bl[1] - br[1])**2 + (bl[0] - br[0])**2))
    max_width = max(widthA, widthB)

    # construct the top-down view of the image
    result = np.array([
        [0,0],              # top-left
        [max_width -1, 0],  # top-right
        [max_width -1, max_height -1], # bottom-right
        [0, max_width -1]], dtype = 'float32'
    )

    # compute the persective transform matrix
    M = cv2.getPerspectiveTransform(rect, result)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped
