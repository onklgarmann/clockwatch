import cv2
import numpy as np
import sys
import time
from picamera import PiCamera

'''
with PiCamera(resolution="HD") as camera:
    camera.rotation = 180
    
    
    for n in range (1,11):
        camera.start_preview()
        time.sleep(2)
        print(n)
        camera.capture('./{}/input.bmp'.format(n))
        
        time.sleep(30)
    camera.close()
'''


def grayscale(n, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   #fjern fargedimensjonen
    cv2.imwrite('./{}/grayscale.bmp'.format(n), image) #Skriv ut fil
    median = np.median(image)
    threshold = int(min(255, (1.0 + 0.33) * median)+20)
    return threshold, image

def cannyEdge(n, threshold, image):
    image = cv2.Canny(image, threshold/2, threshold)
    cv2.imwrite('./{}/cannyEdge.bmp'.format(n), image) #Skriv ut fil

def houghCircle(n, threshold, image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1,750, param1=threshold,param2=30,minRadius=200, maxRadius=300)
    image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 3, y - 3), (x + 3, y + 3), (0, 0, 255), -1)
            cv2.imwrite('./{}/houghCircle.bmp'.format(n), image) #Skriv ut fil
        if circles.size//3 is not 1:
            raise Exception('Several circles detected')
    else:
        raise Exception('No circles detected')
    return x,y,r, image

def cropClock(n, x=676, y=324, r=209, size=720):
    image = cv2.imread('./{}/grayscale.bmp'.format(n))[y-r:y+r, x-r:x+r]
    mask = np.zeros(image.shape, dtype = "uint8")
    cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
    image = cv2.bitwise_and(image, mask)
    image = cv2.resize(image,(size,size))
    median = np.median(image)
    mean = np.mean(image)
    print("median: ", median, " mean: ", mean)
    cv2.imwrite('./{}/cropClock.bmp'.format(n), image) #Skriv ut fil
    return image

def clockDepolarize(n, image, size = 720):
    radius = size //2
    image = cv2.linearPolar(image, (radius, radius), radius, cv2.WARP_FILL_OUTLIERS)
    cv2.imwrite('./{}/depolarizedClock.bmp'.format(n), image)
    return image


def gaussianBlur(n, image):
    image = cv2.GaussianBlur(image, (21, 21), 0)
    
    cv2.imwrite('./{}/gblurredClock.bmp'.format(n), image)
    return image
    

def thresholdClock(n, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    #image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    retval,image = cv2.threshold(image,(np.median(image)//2+np.median(image)//5),255, cv2.THRESH_BINARY)
    print("threshold: ", retval)
    cv2.imwrite('./{}/thresholdClock.bmp'.format(n), image)
    return image

def openClock(n, image):
    kernelerode = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    kerneldilate = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    for n in range (1):
        image = cv2.erode(image, kernelerode, iterations=2)
        image = cv2.dilate(image, kerneldilate, iterations=1)
    
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernelerode, iterations =3)
    cv2.imwrite('./{}/openClock.bmp'.format(n), image)
    return image

def dilateClock(n, image):
    kerneldilate = cv2.getStructuringElement(cv2.MORPH_RECT,(image.shape[0]//5,1))
    image = cv2.dilate(image, kerneldilate)
    cv2.imwrite('./{}/dilateClock.bmp'.format(n), image)
    return image

def findViser(n, image):
    map = image.copy()
    
    height, width = map.shape
    hsignal = np.zeros(width)
    vsignal = np.zeros(height)
    map[0:height, 0:width] = 0
    with np.nditer(vsignal) as itv:
        with np.nditer(hsignal) as ith:
            for x in ith:
                if image[itv.iterindex][ith.iterindex]==0:
                    x[...]=255
        
    return map



finalOutput= None
liste = [1, 2, 3, 8]
for n in liste:
    print(n)
    try:
            
        image = cv2.imread('./{}/cropClock.bmp'.format(n))
        '''
        threshold, image = grayscale(n, image)
        cannyEdge(n, threshold, image)
        try:
            x, y, r , image = houghCircle(n, threshold, image)
            image = cropClock(n, x, y, r)
        except Exception as e:
            print(e.args[0])
            print("Using default circle")
            image = cropClock(n)
        '''
        image = thresholdClock(n, image)
        print("threshold done")
        image = gaussianBlur(n, image)
        retval,image = cv2.threshold(image,np.median(image)//2+np.median(image)//5,255, cv2.THRESH_BINARY)
        image = ~image
        
        #image = findViser(n, image)

        #image = openClock(n, image)
        #image = clockDepolarize(n, image)
        #image = openClock(n, image)
        #image = gaussianBlur(n, image)
        #image = openClock(n, image)
        #image = image[0:image.shape[0], 0:image.shape[1]//6*5]
        #image = dilateClock(n, image)
        

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        gblur = cv2.imread('./{}/gblurredClock.bmp'.format(n))
        depolarizedImage = cv2.imread('./{}/depolarizedClock.bmp'.format(n))
        thresholdImage = cv2.imread('./{}/thresholdClock.bmp'.format(n))
        cropImage = cv2.imread('./{}/cropClock.bmp'.format(n))
        openImage = cv2.imread('./{}/openClock.bmp'.format(n))
        
        output = cv2.imwrite('./{}/output.bmp'.format(n), image)
        

    except Exception as e:
        print(e.args)

finalOutput = cv2.imread('./{}/output.bmp'.format(1))
for n in range (2,4):
    image = cv2.imread('./{}/output.bmp'.format(n))
    finalOutput = np.vstack([finalOutput, image])
image = cv2.imread('./{}/output.bmp'.format(8))
finalOutput = np.vstack([finalOutput, image])
finalOutput = cv2.cvtColor(finalOutput, cv2.COLOR_RGB2GRAY)
cv2.imwrite('./finalOutput.bmp', finalOutput)
sys.exit()
