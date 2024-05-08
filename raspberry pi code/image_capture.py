import cv2
import numpy as np
from picamera2 import Picamera2

cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size":(640, 480)}))
picam2.start()

image = picam2.capture_array()
cv2.imwrite("image" +str(1)+".jpg", image)


