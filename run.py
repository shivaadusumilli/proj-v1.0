import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from espeak import espeak # text to speech
#### import the necessary packages for ai model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from gtts import gTTS #voice to text
from smbus2 import SMBus #temp sensor
from mlx90614 import MLX90614 #temp sensor
# import lcddriver from lcd#install this from https://github.com/the-raspberry-pi-guy/lcd
from mfrc522 import SimpleMFRC522 #rfid



#hyper-parameters
DEBUG = True
#max temperature to detect fever
MAX_TEMP = 40
#angles limits wrt servos
SERVO_X_LEFT = 0
SERVO_X_RIGHT = 180
SERVO_Y_TOP = 0
SERVO_Y_BOTTOM = 180
#Resolution of image
PIC_RES_X = 100
PIC_RES_Y = 100
#File locations
HIGH_TEMP_LOCATION = "/home/pi/Face-Mask-Detection/data/pics/high_temp/"# location to save high temp ppl pics
MULTIPLE_PPL_LOCATION = "/home/pi/Face-Mask-Detection/data/pics/mult_ppl/" # location to save pics of ppl not following socical distancing
NO_MASK_LOCATION = "/home/pi/Face-Mask-Detection/data/pics/no_mask/"# location to save pics of ppl who dont wear mask
NO_FACE_LOCATION = "/home/pi/Face-Mask-Detection/data/pics/no_ppl/"# location to save images when sensor was triggered but no face detected

#PINS
P_ON_OFF = 18 #trigger pin - this will be middle button of joystick
P_MOTION_DETECTION = 16#trigger pin - motion detector sensor
P_SERVO_X = 35
P_SERVO_Y = 37
P_LAZER = 32
P_DOOR = 12# should be connected to relay

#Addersses
A_TEMP_SENSOR = 0x5A
A_DISPLAY = 0x27

#Variables
SYSTEM_STATE = True
SOUND_LEVEL = 0.5
IS_MOTION_DETECTED = False

#lines in display
L1 = " fill this with "
L2 = "   some  shit   "

def speak(txt):
    language = 'en'
    myobj = gTTS(text=txt, lang=language, slow=False) 
    myobj.save("TempAudioFile.mp3") 
    os.system("mpg321 TempAudioFile.mp3") 

def LogS(txt):
    if DEBUG:
        speak(txt)

def LogP(txt):
    if DEBUG:
        print(txt)
        LogS(txt)


# def updateDisplay():
#     LogP("updating display")
#     #display.lcd_clear()
#     display.lcd_display_string(L1, 1) # Write line of text to first line of display
#     display.lcd_display_string(L2, 2) # Write line of text to second line of display
    



def openDoor():
    LogP("door opened")
    GPIO.output(P_DOOR, GPIO.LOW)

def closeDoor():
    LogP("door closed")
    GPIO.output(P_DOOR, GPIO.HIGH) 

def sensorTriggeredButNoFace(frame):
    # motion sensor triggered but no face was detectd by ai
    # LogP("I thought someone was here, but i can see no one")
    cv2.imwrite(NO_FACE_LOCATION+str(int(time.time()))+".png", frame)
    LogP("image saved")



def noMask(frame):
    #this will be executed when person has no mask
    LogP("NO mask detected")
    cv2.imwrite(NO_MASK_LOCATION+str(int(time.time()))+".png", frame)
    LogP("image saved")


def multipleFaces(frame):
    # this will be executed when multiple faces were found
    LogP("Multiple faces found")
    cv2.imwrite(MULTIPLE_PPL_LOCATION+str(int(time.time()))+".png", frame)
    LogP("image saved")


def highTemp(frame):
    # this will be executed when a person with high temp is dtected
    LogP("HIGH TEMP SCANNED")
    cv2.imwrite(HIGH_TEMP_LOCATION+str(int(time.time()))+".png", frame)
    LogP("image saved")


def scanTemp():
    scannedTempetature = 35
    #scannedTempetature = sensor.get_obj_temp()
    LogP("obj temperature = {}".format(scannedTempetature))
    #LogP("amb temperature = {}".format(sensor.get_amb_temp()))
    return scannedTempetature

def button_callback(channel):
    print("Button was pushed!")
    if SYSTEM_STATE:
        SYSTEM_STATE = False
        return
    SYSTEM_STATE = True

def motionDetected():
    IS_MOTION_DETECTED = True

def setServoAngle(pin, angle):
    if pin==P_SERVO_X:
        duty=angle/18+2
        GPIO.output(pin, True)
        pwm1.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(pin, False)
        pwm1.ChangeDutyCycle(0)
    elif pin==P_SERVO_Y:
        duty = angle / 18 + 2
        GPIO.output(pin, True)
        pwm2.ChangeDutyCycle(duty)
        time.sleep(1)
        GPIO.output(pin, False)
        pwm2.ChangeDutyCycle(0)
    else:
        raise Exception("Unsupported pin received in setServoAngle..")


def orientServoMotors(startX, startY, endX, endY):
    targetX = (startX+endX)/2
    targetY = (startY+endY)/2
    targetX_angle = SERVO_X_LEFT + (SERVO_X_RIGHT - SERVO_X_LEFT) * (targetX)/(PIC_RES_X)
    targetY_angle = SERVO_Y_TOP + (SERVO_Y_BOTTOM - SERVO_Y_TOP) * (targetY)/(PIC_RES_Y)
    setServoAngle(P_SERVO_X,targetX_angle)
    setServoAngle(P_SERVO_Y,targetY_angle)    
    LogP("servos ready in positon")

def DisplayVideoOutput(locs, preds, frame):
    if not DEBUG:
        return
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    print('locs'+str(locs))
    print('preds'+str(preds))
    return (locs, preds)
    

def loop(IS_MOTION_DETECTED):
    if SYSTEM_STATE: #if system is on and motion is detected
        IS_MOTION_DETECTED = True
        if IS_MOTION_DETECTED:
            LogP("Motion is detected")

            ###capture frame from cameraarey
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet) #predict faces and their locations in frame
            DisplayVideoOutput(locs,preds,frame) #display image on hdmi output screen

            noOfFacesDetected = len(locs)
            LogP("{} faced detected".format(noOfFacesDetected))

            ###mask detection
            if noOfFacesDetected == 0: #if there are no faces i.e no people in frame, skip frame
                sensorTriggeredButNoFace(frame)
                #LogP("No faces detected")
                return
            
            (mask, withoutMask) = preds[0]
            
            if noOfFacesDetected > 1: #if there are multiple faces i.e multiple people in frame, call multipleFaces() and skip frame
                multipleFaces(frame)
                #LogP("Multiple faces detected")
                return
            if withoutMask > mask: # if no mask, call noMask(), and skip frame
                noMask(frame)
                #LogP("No mask found")
                return

            ###temp scanning
            (startX, startY, endX, endY) = locs[0] #get cordinates of face in image
            orientServoMotors(startX, startY, endX, endY) #point temp sensor to persons face
            scannedTempetature = scanTemp() #scan temp from IR sensor
            if scannedTempetature > MAX_TEMP: #if person has high temp, then call  highTemp() and skip frame
                highTemp(frame)
                return

            ###if all above conditions passes, then open door
            openDoor()
            sleep(5) #door is kept open for 5 seconds
            closeDoor()

            IS_MOTION_DETECTED = False
        else:
            LogP("Motion detected but system paused..")
    else:
        LogP("System state : OFF")
        

######################SETUP###########################
#GPIO.setwarnings(False)
LogP("System initializing...")
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
#set pins as output and input
GPIO.setup(P_SERVO_X, GPIO.OUT)
GPIO.setup(P_SERVO_Y, GPIO.OUT)
GPIO.setup(P_LAZER, GPIO.OUT)
GPIO.setup(P_DOOR, GPIO.OUT)

pwm1 = GPIO.PWM(P_SERVO_X, 50)
pwm2 = GPIO.PWM(P_SERVO_Y, 50)
pwm1.start(0) # starting duty cycle ( it set the servo to 0 degree )
pwm2.start(0) # starting duty cycle ( it set the servo to 0 degree )
#set event listner for start/stop pin
GPIO.setup(P_ON_OFF, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.add_event_detect(P_ON_OFF,GPIO.RISING,callback=button_callback) # Setup event on pin 10 rising edge
#set event listner for start/stop pin
GPIO.setup(P_MOTION_DETECTION, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)
GPIO.add_event_detect(P_MOTION_DETECTION,GPIO.RISING,callback=motionDetected) # Setup event on pin 10 rising edge
#init temp sensor
bus = SMBus(1)
sensor = MLX90614(bus, address=A_TEMP_SENSOR)
# #init display
# display = lcddriver.lcd()

#turn on laser
if DEBUG:
    pass
    #GPIO.output(P_LAZER, GPIO.HIGH)

### ai stuff
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
LogP("Setup Done...")
######################SETUP_END###########################
LogP("Begining to loop")


while True:
    loop(IS_MOTION_DETECTED)


##TODO##
#sudo apt-get install espeak
#sudo apt-get install espeak python-espeak
# sudo apt install libatlas-base-dev
# pip3 install tensorflow
# pip3 install imutils
# sudo pip3 install opencv-python
# sudo apt update
# sudo apt install gunicorn3
# and reboot
#install mpg321 for audio
#sudo pip3 install spidev
#sudo pip3 install mfrc522
#integrate rfid as last step of authentication


'''
310 display 
60 relay
800 ir temp sen
800 battery
250 servo
400 camera
60 motion
70 buttons
1000 joystick,fan,wires,other
2000 casing
150 rfid

7830

wiring colors:
black - gnd
red - 5v
orange - 12v
green - raspberry power
brown - sda
white - scl

'''
