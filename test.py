import cv2
import numpy as np
import sys
import time
from subprocess import call

n=1


call("./finnViser ./{}/output.bmp ./{}/finnViser.bmp".format(n,n), shell='true')