#!/usr/bin/env python
import cv2
import numpy as np
from picamera2 import Picamera2
import rpyc

# cv2.startWindowThread()

class MyService(rpyc.Service):
    def on_connect(self, conn):
        print("Client connected")

    def on_disconnect(self, conn):
        print("Client disconnected")

    def test(self):
        print("TESTING")

    def exposed_capture_and_detect_circles(self):
        with Picamera2() as picam2:
            picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size":(4680,2592)}))
            picam2.start()
            image = picam2.capture_array()

        retval, buffer = cv2.imencode('.jpg', image)
        img_str = buffer.tobytes()
        return img_str

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MyService, port=18813)
    t.start()
