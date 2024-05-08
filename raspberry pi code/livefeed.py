
import cv2
import numpy as np
from picamera2 import Picamera2

cv2.startWindowThread()

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size":(640,480)}))
picam2.start()

while True:
	image = picam2.capture_array()
	imageCopy = image.copy()

	grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#grayImage = cv2.medianBlur(grayImage, 5)
	circles = cv2.HoughCircles(grayImage, cv2.HOUGH_GRADIENT, 1, 6, param1=80, param2=25, minRadius=40, maxRadius=60)

	if circles is not None:
		detected_circles = np.uint16(np.around(circles))
		for(x, y, r) in detected_circles[0, :]:
			cv2.circle(imageCopy, (x, y), r, (0, 255, 0), 3)
			cv2.circle(imageCopy, (x, y), 2, (0, 0, 255), 3)

	cv2.imshow("camera", imageCopy)
	if cv2.waitKey(1) == 27:
		break

