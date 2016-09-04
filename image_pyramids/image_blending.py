import cv2
import numpy as np,sys


# Image blending using Gaussian pyramids
# Steps:
# 1. Find the gaussian pyramids for both images (levels: 6)
# 2. From gaussian pyramids, find the Laplacian pyramids
# 3. Join the left half of image 1 and the right half of image 2
#    in each levels of Laplacian pyramids
# 4. From this joint image pyramids, reconstruct the original image

A = cv2.imread('../images/owl.jpg')
B = cv2.imread('../images/owl.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in xrange(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in xrange(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:cols/2],B[:,cols/2:]))

cv2.imshow('Pyramid_blending2.jpg',ls_)
cv2.imshow('Direct_blending.jpg',real)

cv2.waitKey(0)
cv2.destroyAllWindows()
