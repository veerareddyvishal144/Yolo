import numpy as np
import cv2
from darkflow.net.build import TFNet
import numpy as np
import requests
import os
from urllib.request import urlopen
os.chdir('/projects/yolo')
cap = cv2.VideoCapture(0)
options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
   
    result = tfnet.return_predict(frame)
    for row in result:
        cv2.rectangle(frame,(row["topleft"]["x"],row["topleft"]["y"]),(row["bottomright"]["x"],row["bottomright"]["y"]),(255,254,1),3)
        cv2.putText(frame,row["label"],(row["topleft"]["x"],row["topleft"]["y"]),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255 ), lineType=cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
