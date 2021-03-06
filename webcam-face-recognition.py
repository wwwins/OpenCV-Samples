#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: wwwins
# @Date:   2017-08-09 11:15:17
# @Last Modified by:   wwwins
# @Last Modified time: 2017-08-17 12:40:15

import cv2
import sys
import time
import argparse
import numpy as np
from ticket import ticket
from ImageText import *
from crop_face import *

DEBUG = 0
ENABLE_FPS = False
ENABLE_SHOW_VIDEO= True 
ENABLE_SHOW_CROP = False

FRAME_WIDTH = int(1920*1.0)
FRAME_HEIGHT = int(1080*1.0)
RESIZE_FRAME_WIDTH = int(1920*0.5)
RESIZE_FRAME_HEIGHT = int(1080*0.5)

# hunting sight
SIGHT_W = 3
SIGHT_H = 15
SIGHT_COLOR = (66,66,244)

# 依不同的 cascade 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 3
#MIN_SIZE = 30
MIN_SIZE = 200

FACE_SIZE = 200

# default tolerance value
TOLERANCE = 0.6

parser = argparse.ArgumentParser(description='face recognition from image or camera')
parser.add_argument('file', default='lbph-training.yml', help='model file')
parser.add_argument('-f', dest='face_casc_file', metavar='face_casc_file', default='data/haarcascade_frontalface_alt.xml', help='face cascade file')
parser.add_argument('-i', dest='image_file', metavar='image', help='input image file')
parser.add_argument('-x', metavar='x_pos', default='100', type=int, help='x position')
parser.add_argument('-y', metavar='y_pos', default='100', type=int, help='y position')
args = parser.parse_args()

cascPath = args.face_casc_file
faceCascade = cv2.CascadeClassifier(cascPath)

model_file = args.file
recognizer = cv2.face.createLBPHFaceRecognizer()
# recognizer = cv2.face.createFisherFaceRecognizer()
# recognizer = cv2.face.createEigenFaceRecognizer()
recognizer.load(model_file)

pos = (args.x, args.y)

arr_images = []
arr_labels = []
start_label_id = label_id = int(recognizer.getLabels()[-1])
blank_image = np.zeros((320,320,3), np.uint8)

millis = lambda: int(round(time.time() * 1000))
started_waiting_at = millis()
started_waiting_recognition = millis()
train_it = 0

image = args.image_file
if image is None:
    video_capture = cv2.VideoCapture(1)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)    # turn the autofocus off

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
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 4)
        return frame,faces

def faceRecognition(frame):
    global started_waiting_recognition
    label_id = 0
    if (millis() - started_waiting_recognition) > 500:
        label_id = facePrediction(frame)
        if train_it:
            faceTrain(frame)
            label_id = facePrediction(frame)
        started_waiting_recognition = millis()
    return label_id

def faceTrain(frame):
    global arr_images,arr_labels,train_it,arr_labels,label_id
    print("need training:"+str(label_id+1))
    label_id = label_id + 1
    save_frame = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    arr_images.append(save_frame)
    arr_labels.append(label_id)
    recognizer.update(arr_images, np.array(arr_labels))
    cv2.imwrite('mydatasets/{}.jpg'.format(label_id),save_frame)
    arr_images = []
    arr_labels = []

def facePrediction(frame):
    global train_it
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
        train_it = 0
        showText = "Unknown"
        if(prediction_label>=0):
            showText = recognizer.getLabelInfo(prediction_label)
            # cv2.putText(frame, str(showText), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            # cv2.putText(frame, str(prediction_label), (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            prediction_text('',"辨識結果為右方圖像{}({:.2f})".format(prediction_label,(1.0-prediction_distance)))
    else:
        train_it = 1
        prediction_text(999999,"Unknown")
    return prediction_label

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
    print('show image text:'+contents)
    frame_title = get_image_text_with_size(contents,RESIZE_FRAME_WIDTH,50)
    cv2.imshow("Description",frame_title)

def prediction_text(label, text):
    global started_waiting_at
    if (millis() - started_waiting_at) > 500:
        show_image_text(str(label)+' '+text)
        started_waiting_at = millis()

def main():
    global started_waiting_recognition,train_it
    cv2.namedWindow('Video')
    cv2.moveWindow("Video", pos[0], pos[1])
    cv2.namedWindow("Description")
    cv2.moveWindow("Description", pos[0], pos[1]+RESIZE_FRAME_HEIGHT)
    cv2.namedWindow('Face Recognition')
    cv2.moveWindow("Face Recognition", pos[0]+RESIZE_FRAME_WIDTH, pos[1])
    if ENABLE_SHOW_CROP:
        cv2.namedWindow('Crop')
        cv2.moveWindow("Crop", pos[0]+int(FRAME_WIDTH*0.5), pos[1]+FRAME_HEIGHT)

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
        if ENABLE_SHOW_VIDEO:
            resize_frame = cv2.resize(frame, (RESIZE_FRAME_WIDTH,RESIZE_FRAME_HEIGHT))
            cv2.imshow('Video', resize_frame)
        if len(faces)>0:
            x,y,w,h = faces[0]
            if ENABLE_SHOW_CROP:
                cv2.imshow('Crop', frame[y+1:y+h, x+1:x+w])
            label_id = faceRecognition(frame[y:y+h, x:x+w])
            if label_id>0:
                img = cv2.imread('mydatasets/{}.jpg'.format(label_id))
                cv2.imshow('Face Recognition', img)
        else:
            prediction_text('',"臉部偵測中")
            cv2.imshow('Face Recognition', blank_image)
            started_waiting_recognition = millis()
            train_it = 0
        if ENABLE_FPS:
            print("fps:",t.fps())
        if cv2.waitKey(50) & 0xFF == ord('q'):
            recognizer.save(model_file)
            break

    video_capture.release()

if __name__ == '__main__':
    main()
    # When everything is done, release the capture
    cv2.destroyAllWindows()
