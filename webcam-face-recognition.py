#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: wwwins
# @Date:   2017-08-09 11:15:17
# @Last Modified by:   wwwins
# @Last Modified time: 2017-08-15 12:23:24

import cv2
import sys
import time
import numpy as np
from ticket import ticket
from ImageText import *
from crop_face import *

DEBUG = 0
ENABLE_FPS = False

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# hunting sight
SIGHT_W = 3
SIGHT_H = 15
SIGHT_COLOR = (66,66,244)

# 依不同的 cascade 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 3
#MIN_SIZE = 30
MIN_SIZE = 80

FACE_SIZE = 200

# default tolerance value
TOLERANCE = 0.7

if len(sys.argv) < 3:
    print("""
    Usage:
            webcam-face-recognition.py data/haarcascade_frontalface_default.xml lbph-training.yml [image.jpg]
    """)
    sys.exit(-1)

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascPath = "data/haarcascade_eye_tree_eyeglasses.xml"
eyeCascade = cv2.CascadeClassifier(eyeCascPath)

model_file = sys.argv[2]
recognizer = cv2.face.createLBPHFaceRecognizer()
# recognizer = cv2.face.createFisherFaceRecognizer()
# recognizer = cv2.face.createEigenFaceRecognizer()
recognizer.load(model_file)

arr_images = []
arr_labels = []
label_id = 3

millis = lambda: int(round(time.time() * 1000))
started_waiting_at = millis()
cnt_error = 0
error_max = 30

image = None
if len(sys.argv)>3:
    image = sys.argv[3]
else:
    video_capture = cv2.VideoCapture(1)
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

def faceDetect(frame):
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_SIZE, MIN_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces)>0:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 1)
    return frame

def faceDetectAndCrop(frame):
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_SIZE, MIN_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces)<1:
        return frame,[]
    if len(faces)>0:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 1)
        return frame,faces

def faceTrain(frame):
    global cnt_error,arr_images,arr_labels,label_id
    if cnt_error>error_max:
        print("need training:"+str(label_id+1))
        label_id = label_id + 1
        arr_images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        arr_labels.append(label_id)
        recognizer.update(arr_images, np.array(arr_labels))
        cnt_error = 0

def facePrediction(frame):
    global cnt_error
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    collector = cv2.face.MinDistancePredictCollector()
    prediction_label = recognizer.predict(gray)
    recognizer.predict(gray,collector)
    prediction_label = collector.getLabel()
    prediction_distance = collector.getDist()/100
    if (DEBUG):
        print("label->dist: {0}->{1}".format(prediction_label, prediction_distance))
    if (prediction_distance < TOLERANCE):
        # draw_hunting_sight(frame, (x,y), (x+w,y+h))
        showText = "Unknown"
        if(prediction_label>=0):
            showText = recognizer.getLabelInfo(prediction_label)
            # cv2.putText(frame, str(showText), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            # cv2.putText(frame, str(prediction_label), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            prediction_text(prediction_label, "號大頭照:{:.2f}".format(prediction_distance))
    else:
        cnt_error = cnt_error+1
        prediction_text(999999,"Unknown")

def facePredictionWithEyes(faces, eyes, roi_gray):
    gray_resize = np.array([])
    collector = cv2.face.MinDistancePredictCollector()
    x, y, w, h = faces[0]
    if (len(eyes)==2):
        pos = []
        for ex,ey,ew,eh in eyes:
            center_pos = (ex+int(ew*0.5), ey+int(eh*0.5))
            pos.append(center_pos)
        if pos[0][0]<pos[1][0]:
            eye_left = pos[0]
            eye_right = pos[1]
        else:
            eye_left = pos[1]
            eye_right = pos[0]
        cv2_im = cv2.cvtColor(roi_gray,cv2.COLOR_BGR2RGB)
        cropframe = CropFace(Image.fromarray(cv2_im), eye_left=eye_left, eye_right=eye_right, offset_pct=(0.2,0.2), dest_sz=(FACE_SIZE,FACE_SIZE))
        gray_resize = cv2.cvtColor(np.array(cropframe), cv2.COLOR_RGB2GRAY)
        gray_resize = cv2.equalizeHist(gray_resize)
        # cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),1)
        # get a simple prediction
        # prediction_label = recognizer.predict(gray[y:y+h,x:x+w])
        # recognizer.predict(gray[y:y+h,x:x+w],collector)
        prediction_label = recognizer.predict(gray_resize)
        recognizer.predict(gray_resize,collector)
        prediction_label = collector.getLabel()
        prediction_distance = collector.getDist()/100
        if (DEBUG):
            print("label->dist: {0}->{1}".format(prediction_label, prediction_distance))

        if (prediction_distance < TOLERANCE):
            # draw_hunting_sight(frame, (x,y), (x+w,y+h))
            showText = "Unknown"
            if(prediction_label>=0):
                showText = recognizer.getLabelInfo(prediction_label)
                # cv2.putText(frame, str(showText), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                # cv2.putText(frame, str(prediction_label), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
                prediction_text(prediction_label, showText)
        else:
            prediction_text(999999,"Unknown")
    return gray_resize

def show_image_text(contents):
    global cnt_error
    print('show image text:'+contents)
    frame_title = get_image_text(contents)
    cv2.imshow("Label name",frame_title)

def prediction_text(label, text):
    global started_waiting_at
    if (millis() - started_waiting_at) > 500:
        show_image_text(str(label)+' '+text)
        started_waiting_at = millis()

def main():
    cv2.namedWindow('Video')
    # cv2.moveWindow("Video", 690+640, 750+150)
    cv2.namedWindow("Label name")
    # cv2.moveWindow("Label name", 690+640, 750+150+FRAME_HEIGHT+24)
    cv2.namedWindow('Crop')
    cv2.moveWindow("Crop", 700, 100)

    if image:
        frame = cv2.imread(image)
        faces, eyes, roi_gray = faceDetect(frame)
        if len(faces)<1:
            if DEBUG:
                print ("*** not found faces ***")
            prediction_text(999999,"Unknown")
        result = facePredictionWithEyes(faces, eyes, roi_gray);
        cv2.imshow('Video', frame)
        if(len(result)>0):
            cv2.imshow('Result', result)
        cv2.waitKey(0)
        return

    while True:
        # Capture frame-by-frame
        _,frame = video_capture.read()
        frame = cv2.flip(frame,1)
        frame, faces = faceDetectAndCrop(frame)
        cv2.imshow('Video', frame)
        if len(faces)>0:
            x,y,w,h = faces[0]
            cv2.imshow('Crop', frame[y+1:y+h, x+1:x+w])
            faceTrain(frame[y:y+h, x:x+w])
            facePrediction(frame[y:y+h, x:x+w])
        if ENABLE_FPS:
            print("fps:",t.fps())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()

if __name__ == '__main__':
    main()
    # When everything is done, release the capture
    cv2.destroyAllWindows()
