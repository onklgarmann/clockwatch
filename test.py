import cv2
import numpy as np
import sys
import time
import math
from subprocess import call

import RPi.GPIO as GPIO

image = cv2.imread('./houghlines1.bmp')
image2 = cv2.imread('./houghlines1.bmp')
image2=~image2
gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 75, 150)
cv2.imwrite('houghlinesEdge.bmp',edges)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=250)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),2)
for x1,y1,x2,y2 in lines[1]:
    length1 = sqrt((x2-x1)^2+(y2-y1)^2)
    cv2.line(image2,(x1,y1),(x2,y2),(0,255,0),2)
print(lines[0], lines[1])
length1 = 
print("length: ", )
cv2.imwrite('houghlinesOut.bmp',image2)