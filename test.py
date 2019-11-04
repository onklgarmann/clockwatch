import cv2
import numpy as np
import sys
import time
from subprocess import call

n=1


signal = [1, 1, 1, 0, 1, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0]
print (np.argmax(signal))