import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp

img = cv2.imread(' ',0)  #insert path-to-imagefile

def distance(point1, point2):
  return sqrt(((point1[0]-point2[0])**2)+((point1[1]-point2[1])**2))

def ideanLPF(D0, img_shape):
  base = np.zeros(img_shape[:2])
  rows, cols = img_shape[:2]
  center = (rows/2, cols/2)
  for x in range(cols):
    for y in range(rows):
      base[y,x] = np.exp((-distance(y,x))**2/D0**2)
     # if distance((y,x), center) < D0:
      #  base[y,x] = 1
  
  return base

LPF_image = ideanLPF(50,img.shape)
plt.imshow(LPF_image, cmap='gray')
plt.show()
