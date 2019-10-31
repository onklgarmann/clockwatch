import cv2
import numpy as np
import sys
import time



#n=1
image = cv2.imread('./input.bmp')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(image.shape)

map = image.copy()

height, width = map.shape
hsignal = np.zeros(width)
vsignal = np.zeros(height)
map[0:height, 0:width] = 0
with np.nditer(vsignal) as itv:
    with np.nditer(hsignal, op_flags=['readwrite']) as ith:
        for x in ith:
            if image[itv.iterindex][ith.iterindex]==0:
                x[...]=255
    
cv2.imwrite('./twopass.bmp', image)