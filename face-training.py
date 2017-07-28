#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# face-training.py
#
import cv2
import sys
import os
import time
import numpy as np
from glob import glob

IGNORE_FILE_NAME = ['resize']

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

def faceDetect(gray, fn):
    faces = faceCascade.detectMultiScale(
        gray
    )
    if len(faces)>0:
        print ("{0} found {1} faces!".format(fn, len(faces)))
    else:
        print ("{0} not found faces!".format(fn))
    return faces

def getImagesAndLabels():
    arr_images = []
    arr_labels = []
    for f in glob('{0}/*_[0-9].png'.format(image_path)):
        frame = cv2.imread(f, 0)
        fn = f.split('/')[1]
        label = int(fn.split('_')[0])
        faces = faceDetect(frame, fn)
        for (x, y, w, h) in faces:
            arr_images.append(frame[y:y+h,x:x+w])
            arr_labels.append(label)
    return arr_images, arr_labels

if __name__ == '__main__':
    images,labels = getImagesAndLabels()
    # Train the model using the face images and labels
    recognizer.train(images, np.array(labels))
    recognizer.setLabelInfo(labels[0],'isobar-jacky')
    recognizer.save(training_file)
