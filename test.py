import cv2
import numpy as np
import sys
import time
from subprocess import call

n=1

hist = np.array([1,1,2,3,4,5,3,20,3,0,0,2,0,1,0])
signal = np.array(0)
for i in range(0, hist.size):
    signal.append(np.max(np.nonzero(hist[i])))
    print (signal[i])