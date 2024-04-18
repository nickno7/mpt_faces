import numpy as np
import cv2 as cv
import os
import gdown
import uuid
import csv
from common import ROOT_FOLDER
# from cascade import create_cascade

# Quellen
#  - How to open the webcam: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#  - How to run the detector: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
#  - How to download files from google drive: https://github.com/wkentaro/gdown
#  - How to save an image with OpenCV: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
#  - How to read/write CSV files: https://docs.python.org/3/library/csv.html
#  - How to create new folders: https://www.geeksforgeeks.org/python-os-mkdir-method/

# This is the data recording pipeline
def record():

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            
        # Display the resulting frame
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    # TODO: Implement the recording stage of your pipeline
    #   Create missing folders before you store data in them (os.mkdir)
    #   Open The OpenCV VideoCapture Device to retrieve live images from your webcam (cv.VideoCapture)
    #   Initialize the Haar feature cascade for face recognition from OpenCV (cv.CascadeClassifier)
    #   If the cascade file (haarcascade_frontalface_default.xml) is missing, download it from google drive
    #   Run the cascade on every image to detect possible faces (CascadeClassifier::detectMultiScale)
    #   If there is exactly one face, write the image and the face position to disk in two seperate files (cv.imwrite, csv.writer)
    #   If you have just saved, block saving for 30 consecutive frames to make sure you get good variance of images.
    # if args.folder is None:
    #     print("Please specify folder for data to be recorded into")
    #     exit()

record()