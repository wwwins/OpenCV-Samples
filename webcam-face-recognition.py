# -*- coding: utf-8 -*-
import cv2
import sys
import time
import numpy as np
from ticket import ticket
from imutils.video import VideoStream

DEBUG = True
ENABLE_FPS = False
ENABLE_VIDEO_STREAM = False

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

SIGHT_W = 3
SIGHT_H = 15
SIGHT_COLOR = (66,66,244)

# 依不同的 cascade 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 3
#MIN_SIZE = 30
MIN_SIZE = 100

if len(sys.argv) < 3:
    print("""
    Usage:
            webcam-face-recognition.py data/haarcascade_frontalface_default.xml lbph-training.yml
    """)
    sys.exit(-1)

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

model_file = sys.argv[2]
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load(model_file)

if ENABLE_VIDEO_STREAM:
    video_capture = VideoStream(usePiCamera=False).start()

else:
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

time.sleep(1)
t = ticket()

def draw_hunting_sight(img, pos1, pos2):
    # lt
    x = pos1[0]
    y = pos1[1]
    pts = np.array([[x,y], [x,y+SIGHT_H], [x+SIGHT_W,y+SIGHT_H],[x+SIGHT_W,y+SIGHT_W],[x+SIGHT_H,y+SIGHT_W],[x+SIGHT_H,y]])
    cv2.fillConvexPoly(img, pts, SIGHT_COLOR)
    # rt
    x = pos2[0]
    y = pos1[1]
    pts = np.array([[x,y], [x,y+SIGHT_H], [x-SIGHT_W,y+SIGHT_H],[x-SIGHT_W,y+SIGHT_W],[x-SIGHT_H,y+SIGHT_W],[x-SIGHT_H,y]])
    cv2.fillConvexPoly(img, pts, SIGHT_COLOR)
    # lb
    x = pos1[0]
    y = pos2[1]
    pts = np.array([[x,y], [x,y-SIGHT_H], [x+SIGHT_W,y-SIGHT_H],[x+SIGHT_W,y-SIGHT_W],[x+SIGHT_H,y-SIGHT_W],[x+SIGHT_H,y]])
    cv2.fillConvexPoly(img, pts, SIGHT_COLOR)
    # rb
    x = pos2[0]
    y = pos2[1]
    pts = np.array([[x,y], [x,y-SIGHT_H], [x-SIGHT_W,y-SIGHT_H],[x-SIGHT_W,y-SIGHT_W],[x-SIGHT_H,y-SIGHT_W],[x-SIGHT_H,y]])
    cv2.fillConvexPoly(img, pts, SIGHT_COLOR)
    # rect
    p1 = pos1
    p2 = pos2
    w1 = int(SIGHT_W*0.5)
    cv2.rectangle(img, (p1[0]+w1,p1[1]+w1), (p2[0]-w1,p2[1]-w1), SIGHT_COLOR)

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

    collector = cv2.face.MinDistancePredictCollector()

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        draw_hunting_sight(frame, (x,y), (x+w,y+h))
        # get a simple prediction
        # prediction_label = recognizer.predict(gray[y:y+h,x:x+w])
        recognizer.predict(gray[y:y+h,x:x+w],collector)
        prediction_label = collector.getLabel()
        prediction_distance = collector.getDist()
        if (DEBUG):
            print("label -> dist: {0} -> {1}".format(prediction_label, prediction_distance))

        if (prediction_distance<100.0):
            showText = "Unknown"
            if(id>=0):
                showText = recognizer.getLabelInfo(prediction_label)
                cv2.putText(frame, str(showText), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)

    return frame

while True:
    # Capture frame-by-frame
    _,frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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