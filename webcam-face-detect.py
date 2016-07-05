# -*- coding: utf-8 -*-
import cv2
import sys
import time
from ticket import ticket
from imutils.video import VideoStream


ENABLE_FACE_DETECT = True
ENABLE_FPS = False
ENABLE_VIDEO_STREAM = False

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 依不同的 cascade 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
#MIN_SIZE = 30
MIN_SIZE = 80

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

if ENABLE_VIDEO_STREAM:
    video_capture = VideoStream(usePiCamera=False).start()

else:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

time.sleep(1)
t = ticket()

def faceDetect(gray):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_SIZE, MIN_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces)>0:
        print ("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

while True:
    # Capture frame-by-frame
    _,frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ENABLE_FACE_DETECT:
        resultFrame = faceDetect(gray)
        # Display the resulting frame
        cv2.imshow('Video', resultFrame)

    if ENABLE_FPS:
        print("fps:",t.fps())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
