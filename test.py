import cv2
import numpy as np
import sys
import time
from subprocess import call

import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(18,GPIO.OUT)
print ("RESET ON")
GPIO.output(18,GPIO.HIGH)
time.sleep(10)
print ("RESET OFF")
GPIO.output(18,GPIO.LOW)
time.sleep(20)