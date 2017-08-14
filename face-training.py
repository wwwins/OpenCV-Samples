#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: wwwins
# @Date:   2017-08-08 18:37:00
# @Last Modified by:   wwwins
# @Last Modified time: 2017-08-14 16:40:31

import cv2
import sys
import os
import time
import numpy as np
from glob import glob

IGNORE_FILE_NAME = ['resize']

# SCALE_FACTOR = 1.05

if len(sys.argv) < 4:
    print("""
    Usage:
            face-training.py data/haarcascade_frontalface_default.xml datasets lbph-training.yml
    """)
    sys.exit(-1)

cascPath = sys.argv[1]
image_path = sys.argv[2]

# model_path = dataset
training_file = sys.argv[3]

faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.face.createLBPHFaceRecognizer()
# recognizer = cv2.face.createFisherFaceRecognizer()
# recognizer = cv2.face.createEigenFaceRecognizer()

def faceDetect(gray, fn):
    faces = faceCascade.detectMultiScale(
        gray,
        # scaleFactor=SCALE_FACTOR
    )
    if len(faces)>0:
        print ("{0} found {1} faces!".format(fn, len(faces)))
    else:
        print ("{0} not found faces!".format(fn))
    return faces

def getImagesAndLabels():
    arr_images = []
    arr_labels = []
    arr_labels_info = []
    for f in glob('{0}/*_[0-9]*.png'.format(image_path)):
        frame = cv2.imread(f, 0)
        frame = cv2.equalizeHist(frame)
        fn = f.split('/')[1]
        label = int(fn.split('_')[0])
        label_info = fn.split('_')[2]
        # faces = faceDetect(frame, fn)
        # for (x, y, w, h) in faces:
        #     if label not in arr_labels:
        #         arr_labels_info.append([label,label_info])
        #     arr_images.append(frame[y:y+h,x:x+w])
        #     arr_labels.append(label)
        # reading all datasets
        if label not in arr_labels:
            arr_labels_info.append([label,label_info])
        arr_images.append(frame)
        arr_labels.append(label)
    return arr_images, arr_labels, arr_labels_info

if __name__ == '__main__':
    images,labels,arr_info = getImagesAndLabels()
    # Train the model using the face images and labels
    recognizer.train(images, np.array(labels))
    for info in arr_info:
        print(info)
        recognizer.setLabelInfo(info[0],info[1])
    recognizer.save(training_file)
