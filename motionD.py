        
import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3,GPIO.OUT)     #Define pin 3 as an output pin

while True:
GPIO.output(3,1)   #Outputs digital HIGH signal (5V) on pin 3
time.sleep(1)      #Time delay of 1 second

GPIO.output(3,0)   #Outputs digital LOW signal (0V) on pin 3
time.sleep(1)      #Time delay of 1 second





import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
PIR_PIN = 7
GPIO.setup(PIR_PIN, GPIO.IN)

try:
    print(“PIR Module Test (CTRL+C to exit)”)
    time.sleep(2)
    print(“Ready”)
    while True:
        if GPIO.input(PIR_PIN):
            print(“Motion Detected!”)
        time.sleep(1)
except KeyboardInterrupt:
    print(“Quit”)
    GPIO.cleanup()

