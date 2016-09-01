import cv2
import numpy as np,sys
from matplotlib import pyplot as plt

G = cv2.imread('../images/jurassic_world.jpg')

# generate Gaussian pyramid for A
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

counter =711
for j in gpA:
    plt.subplot(counter), plt.imshow(j)
    plt.title('Gaussian_'+str(counter), fontsize=8)
    plt.xticks([]), plt.yticks([])
    counter += 1
plt.show()

H = gpA[len(gpA)-1]
gpB = [H]
for k in xrange(6):
    H = cv2.pyrUp(H)
    gpB.append(H)

counter =711
for l in gpB:
    plt.subplot(counter), plt.imshow(l)
    plt.title('Gaussian_'+str(counter), fontsize=8)
    plt.xticks([]), plt.yticks([])
    counter += 1
plt.show()
