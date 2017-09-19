# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 00:13:18 2017

@author: Charls

This is an eg of face and eyes detection using OpenCV haarCascade classifer api.
Here a Pre-built haar cascade xml is being used for face and eyes detection.
If you want to detect any custom object, you need to build your own classifer model and it generates an xml 
and you can use that xml in the similar manner to detect in an image.

All it does is create a classifer model for face and eyes detection by loading the xml.
pass an face image to the model and it returns with detected x,y axis and creates a borders.

"""

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarfiles/xml/haarcascade_frontalface_default.xml')

eye_cascade =  cv2.CascadeClassifier('haarfiles/xml/haarcascade_eye.xml')

img = cv2.imread('haarfiles/Faces/image.jpg', cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(img, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',img)
cv2.imwrite('haarfiles/Faces/face_eyes_detected_with_borders.png',img)

cv2.waitKey(0)
cv2.destroyAllWindows()