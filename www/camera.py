from imutils.video import VideoStream
import cv2
import time
import sys
import numpy as np
import os

#cas_path = os.getcwd()
#cas_path += "/data/haarcascade_frontalface_default.xml"
#cas_path = "/home/pi/opencv_samples/data/lbpcascade_frontalface.xml"
#cas_path = "/home/pi/opencv_samples/data/haarcascade_frontalface_alt2.xml"
#cas_path = "~/Downloads/opencv_samples/data/haarcascade_frontalface_alt2.xml"
cas_path = "/Users/isobar/github/opencv_samples/data/lbpcascade_frontalface.xml"

faceCascade = cv2.CascadeClassifier(cas_path)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class VideoCamera(object):

    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = self.VideoCapture()
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def VideoCapture(self):
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        return video

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if faces is not ():
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_simpleFrame(self):
        success, frame = self.video.read()
        rec, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
