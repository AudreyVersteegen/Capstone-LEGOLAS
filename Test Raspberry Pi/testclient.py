import cv2
import numpy as np
import rpyc

conn = rpyc.connect("192.168.42.162", 18861)

while True:
    img_generator = conn.root.exposed_capture_and_detect_circles()
    
    for img_bytes in img_generator:
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow("Circles", image)

        if cv2.waitKey(1) == 27:
            break
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()