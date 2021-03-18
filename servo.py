import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setup(35, GPIO.OUT)

pwm=GPIO.PWM(35, 50)
pwm.start(0)

def SetAngle(angle):
	duty = angle / 18 + 2
	GPIO.output(35, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(35, False)
	pwm.ChangeDutyCycle(0)

SetAngle(0)
sleep(1)
SetAngle(180)
