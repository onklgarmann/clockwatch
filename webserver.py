from flask import Flask, render_template
from flask import url_for, jsonify, render_template, redirect, url_for
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import os
import random
from scipy.signal import find_peaks
from subprocess import call
app = Flask(__name__)

iterations = random.randint(1,99999999) 

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(18,GPIO.OUT)
    print ("RESETTING")
    GPIO.output(18,GPIO.HIGH)
    time.sleep(3)
    print ("RESET")
    GPIO.output(18,GPIO.LOW)
    return redirect('')

@app.route('/knips', methods=['GET', 'POST'])
def knips():
    for file in os.listdir("/home/pi/ClockWatch/static"):
	    if file.endswith(".bmp"):
		    os.remove(os.path.join("/home/pi/ClockWatch/static", file))
    global iterations
    iterations+=1
    cap = cv2.VideoCapture('http://localhost:8080/stream.mjpg')
    ret, frame = cap.read()
    
    cv2.imwrite("./static/input{}.bmp".format(iterations),frame)
    cap.release()
    return render_template('capturedImage.html', iterations = iterations)

@app.route('/behandle', methods=['GET', 'POST'])
def behandle():
    try:
            
        image = cv2.imread('./static/input{}.bmp'.format(iterations))
        
        threshold, image = grayscale(image)
        cannyEdge(threshold, image)
        try:
            x, y, r , image = houghCircle(threshold, image)
            image = cropClock(x, y, r)
        except Exception as e:
            print(e.args[0])
            print("Using default circle")
            image = cropClock()
        #image = cv2.equalizeHist(image)
        image = gaussianBlur(image)
        print("MEDIAN: ", np.median(image))
        image = thresholdClock(image)
        
        
        
        ret,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        image = ~image
        image = openClock(image)
        cv2.imwrite('./static/negative{}.bmp'.format(iterations), image)
        call("./finnViser ./static/negative{}.bmp ./static/finnViser{}.bmp".format(iterations,iterations), shell='true')
        image = cv2.imread('./static/finnViser{}.bmp'.format(iterations), 0)
        image = clockDepolarize(image, image.shape[0])
        
        
        
        image = image[0:image.shape[0], image.shape[1]//2:image.shape[0]]
        print (image.size)
        print(image.shape)
        image, signal = getSignal(image)
        cv2.imwrite('./static/output{}.bmp'.format(iterations), image)
        finnKlokka(signal)
        print("finished")
        
        

    except Exception as e:
        print(e.args)

    return render_template('processedImage.html', iterations = iterations)
    


def grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   #fjern fargedimensjonen
    cv2.imwrite('./static/grayscale{}.bmp'.format(iterations), image) #Skriv ut fil
    median = np.median(image)
    threshold = int(min(255, (1.0 + 0.33) * median)+20)
    return threshold, image

def cannyEdge(threshold, image):
    image = cv2.Canny(image, threshold/2, threshold)
    cv2.imwrite('./static/cannyEdge{}.bmp'.format(iterations), image) #Skriv ut fil

def houghCircle(threshold, image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1,750, param1=threshold,param2=30,minRadius=200, maxRadius=300)
    image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 3, y - 3), (x + 3, y + 3), (0, 0, 255), -1)
            cv2.imwrite('./static/houghCircle{}.bmp'.format(iterations), image) #Skriv ut fil
        if circles.size//3 is not 1:
            raise Exception('Several circles detected')
    else:
        raise Exception('No circles detected')
    return x,y,r, image

def cropClock(x=676, y=324, r=209, size=720):
    r=4*r//5
    image = cv2.imread('./static/grayscale{}.bmp'.format(iterations),0)[y-r:y+r, x-r:x+r]
    mask = np.zeros(image.shape, dtype = "uint8")
    cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
    image = cv2.bitwise_and(image, mask)
    mask = ~mask
    image = cv2.bitwise_xor(image, mask)
    image = cv2.resize(image,(size,size))
    median = np.median(image)
    mean = np.mean(image)
    print("median: ", median, " mean: ", mean)
    cv2.imwrite('./static/cropClock{}.bmp'.format(iterations), image) #Skriv ut fil
    return image

def clockDepolarize(image, size = 720):
    radius = size //2
    image = cv2.linearPolar(image, (radius, radius), radius, cv2.WARP_FILL_OUTLIERS)
    cv2.imwrite('./static/depolarizedClock{}.bmp'.format(iterations), image)
    return image


def gaussianBlur(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    cv2.imwrite('./static/gblurredClock{}.bmp'.format(iterations), image)
    return image
    

def thresholdClock(image):
    binc = np.bincount(image.ravel())
    hist = binc[10:-10]
    
    retval,image = cv2.threshold(image,(np.argmax(hist)-20),255, cv2.THRESH_BINARY)
    print("threshold: ", retval)
    cv2.imwrite('./static/thresholdClock{}.bmp'.format(iterations), image)
    return image

def openClock(image):
    kernelerode = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kerneldilate = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    """for n in range (1):
        image = cv2.erode(image, kernelerode, iterations=1)
        image = cv2.dilate(image, kerneldilate, iterations=12)
    """
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kerneldilate, iterations =1)
    cv2.imwrite('./static/openClock{}.bmp'.format(iterations), image)
    return image

def dilateClock(image):
    kerneldilate = cv2.getStructuringElement(cv2.MORPH_RECT,(image.shape[0]//5,1))
    image = cv2.dilate(image, kerneldilate)
    cv2.imwrite('./static/dilateClock{}.bmp'.format(iterations), image)
    return image

def getSignal(image):
    signal = np.zeros(image.shape[0], dtype=int)
    
    for i in range(0, image.shape[0]):
        if (image[i][0]!=0):
            j = 0
            while image[i][j] != 0:
                signal[i]+=1
                j+=1
    return image, signal
        
def finnKlokka(signal):
    peaks, heights = find_peaks(signal, 10, distance=10)
    print ("peaks:  ", peaks, "heights:  ", heights)
    print (peaks.size)
    print (signal)
    if peaks.size == 1:
        peaks = peaks.astype(int)
        heights = heights['peak_heights'].astype(int)
        timeviser = peaks[0]
        minuttviser = timeviser
        believable = False
        print(" here " , peaks[0], heights[0])
        for time in range(12):
            print(60*time - 5, timeviser , 60*time + 5)
            if (60*time - 5 < timeviser + timeviser//720 < 60*time + 5)==True:
                believable = True
                break
        if (not believable):
            print('mangler en viser!')
    elif peaks.size == 2:
        peaks = peaks.astype(int)
        heights = heights['peak_heights'].astype(int)
        minuttviser = peaks[np.where(heights == np.amax(heights))]
        timeviser = peaks[np.where(heights == np.amin(heights))]
        
    elif peaks is None:
        print('finner ingen visere')
    else:
        print('finner for mange visere')
        

    
    print(minuttviser, timeviser)
    timer = int((timeviser.astype(int)/720)*12+3)
    if timer >= 13:
        timer = timer - 12
    minutter = int((minuttviser.astype(int)/720)*60+15)
    if minutter >=60:
        minutter = minutter -60

    klokka = ("klokka: {:02d}:{:02d}".format(timer, minutter))

    image = cv2.imread('./static/input{}.bmp'.format(iterations))
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100,100)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(image,klokka, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imwrite('./static/klokka{}.bmp'.format(iterations), image)
    return

if __name__ == '__main__':

 app.run(debug=True, host='0.0.0.0')