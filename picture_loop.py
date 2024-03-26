import time
import cv2
import numpy as np
from picamera2 import Picamera2

cv2.startWindowThread()

def start_camera():
	picam2 = Picamera2()
	picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size":(640, 480)}))
	picam2.start()
	
	return picam2

def take_picture(camera, name):
	image = camera.capture_array()
	cv2.imwrite(str(name) + ".jpg", image)
	
def take_picture_directed(camera, name, directory):
	image = camera.capture_array()
	cv2.imwrite( str(directory) + str(name) + ".jpg", image)
	
def photo_reel(camera, num):
	for i in range(num):
		print("picture " + str(i) + "....")
		take_picture(camera, i)
		time.sleep(8)

def photo_reel_directed(camera, num, directory):
	for i in range(num):
		print("picture " + str(i) + ".....")
		take_picture_directed(camera, i, directory)
		time.sleep(5)
	
cam = start_camera()
#photo_reel_directed(cam, 3, "/home/greenleaf/Desktop/testing images/")
take_picture(cam, "placemnet")
