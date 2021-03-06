import cv2
import numpy as np
import sys
import time
from picamera import PiCamera
from subprocess import call


def grayscale(n, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   #fjern fargedimensjonen
    cv2.imwrite('./static/grayscale.bmp'.format(n), image) #Skriv ut fil
    median = np.median(image)
    threshold = int(min(255, (1.0 + 0.33) * median)+20)
    return threshold, image

def cannyEdge(n, threshold, image):
    image = cv2.Canny(image, threshold/2, threshold)
    cv2.imwrite('./static/cannyEdge.bmp'.format(n), image) #Skriv ut fil

def houghCircle(n, threshold, image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1,750, param1=threshold,param2=30,minRadius=200, maxRadius=300)
    image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 3, y - 3), (x + 3, y + 3), (0, 0, 255), -1)
            cv2.imwrite('./static/houghCircle.bmp'.format(n), image) #Skriv ut fil
        if circles.size//3 is not 1:
            raise Exception('Several circles detected')
    else:
        raise Exception('No circles detected')
    return x,y,r, image

def cropClock(n, x=676, y=324, r=209, size=720):
    r=4*r//5
    image = cv2.imread('./static/grayscale.bmp'.format(n),0)[y-r:y+r, x-r:x+r]
    mask = np.zeros(image.shape, dtype = "uint8")
    cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
    image = cv2.bitwise_and(image, mask)
    mask = ~mask
    image = cv2.bitwise_xor(image, mask)
    image = cv2.resize(image,(size,size))
    median = np.median(image)
    mean = np.mean(image)
    print("median: ", median, " mean: ", mean)
    cv2.imwrite('./static/cropClock.bmp'.format(n), image) #Skriv ut fil
    return image

def clockDepolarize(n, image, size = 720):
    radius = size //2
    image = cv2.linearPolar(image, (radius, radius), radius, cv2.WARP_FILL_OUTLIERS)
    cv2.imwrite('./static/depolarizedClock.bmp'.format(n), image)
    return image


def gaussianBlur(n, image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    cv2.imwrite('./static/gblurredClock.bmp'.format(n), image)
    return image
    

def thresholdClock(n, image):
    binc = np.bincount(image.ravel())
    hist = binc[10:-10]
    retval,image = cv2.threshold(image,(np.argmax(hist)-15),255, cv2.THRESH_BINARY)
    print("threshold: ", retval)
    cv2.imwrite('./static/thresholdClock.bmp'.format(n), image)
    return image

def openClock(n, image):
    kernelerode = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kerneldilate = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    """for n in range (1):
        image = cv2.erode(image, kernelerode, iterations=1)
        image = cv2.dilate(image, kerneldilate, iterations=12)
    """
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernelerode, iterations =3)
    cv2.imwrite('./static/openClock.bmp'.format(n), image)
    return image

def dilateClock(n, image):
    kerneldilate = cv2.getStructuringElement(cv2.MORPH_RECT,(image.shape[0]//5,1))
    image = cv2.dilate(image, kerneldilate)
    cv2.imwrite('./static/dilateClock.bmp', image)
    return image

def getSignal(image):
    signal = np.empty(image.shape[0], dtype=int)
    for i in range(0, image.shape[0]):
        signal[i]=np.max(np.nonzero(image[i]))
    for i in range(0, image.shape[0]):
        for j in range (0, signal[i]):
            image[i][j] = 255
    return image, signal
        

finalOutput= None
#liste = [1, 2, 3, 8]

    try:
            
        image = cv2.imread('./static/input.bmp')
        
        threshold, image = grayscale(n, image)
        cannyEdge(n, threshold, image)
        try:
            x, y, r , image = houghCircle(n, threshold, image)
            image = cropClock(n, x, y, r)
        except Exception as e:
            print(e.args[0])
            print("Using default circle")
            image = cropClock(n)
        #image = cv2.equalizeHist(image)
        image = gaussianBlur(n, image)
        image = thresholdClock(n, image)
        
        image = openClock(n, image)
        
        ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        image = ~image
        cv2.imwrite('./static/negative.bmp', image)
        call("./finnViser ./static/negative.bmp ./static/finnViser.bmp", shell='true')
        image = cv2.imread('./static/finnViser.bmp'.format(n))
        image = clockDepolarize(n, image, image.shape[0])
        image = getSignal(image)
        
        
        
        
        image = cv2.imwrite('./static/output.bmp'.format(n), image)
        

    except Exception as e:
        print(e.args)


