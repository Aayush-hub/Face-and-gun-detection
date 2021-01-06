import cv2
import sys
import numpy as np

def apply_Haar_filter(img, haar_cascade,scaleFact = 1.1, minNeigh = 5, minSizeW = 30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFact,
        minNeighbors=minNeigh,
        minSize=(minSizeW, minSizeW),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return features


BLUE = (255,0,0)
YELL = (0,255,255)

#Filters path
haar_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
haar_gun = cv2.CascadeClassifier('cascade.xml')

video_capture = cv2.VideoCapture(0)
cv2.imshow('Video', np.empty((5,5),dtype=float))


while cv2.getWindowProperty('Video', 0) >= 0:
  
    ret, frame = video_capture.read()
    faces = apply_Haar_filter(frame, haar_faces, 1.3 , 5, 30)
    gun = apply_Haar_filter(frame,haar_gun, 6)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 2) #blue
    for (x2, y2, w2, h2) in gun:
        for (x2, y2, w2, h2) in gun:
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), YELL, 2)

        
    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
    	break


video_capture.release()
cv2.destroyAllWindows()