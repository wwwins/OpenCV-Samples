#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: wwwins
# @Date:   2017-08-08 18:45:16
# @Last Modified by:   wwwins
# @Last Modified time: 2017-08-09 11:20:21

import cv2
import sys
import os
import time
import numpy as np
from glob import glob

# 依不同的 cascade 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_default: 1.3
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
#MIN_SIZE = 30
MIN_SIZE = 80

FACE_SIZE = 250

if len(sys.argv) < 4:
    print("""
    Usage:
            create-face-dataset.py data/haarcascade_frontalface_default.xml output datasets
    """)
    sys.exit(-1)

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

image_path = sys.argv[2]

# model_path = dataset
model_path = sys.argv[3]

def faceDetect(gray, fn):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=(MIN_SIZE, MIN_SIZE),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces)>0:
        print ("{0} found {1} faces!".format(fn, len(faces)))
    else:
        print ("{0} not found faces!".format(fn))
    return faces

def createDatasets():
    sno = 0
    buf = ''
    for f in glob('{0}/*_[0-9]*.png'.format(image_path)):
        frame = cv2.imread(f, 0)
        fn = f.split('/')[1]
        if buf != fn.split(' ')[0]:
            buf = fn.split(' ')[0]
            sno = sno + 1
        faces = faceDetect(frame, fn)
        for (x, y, w, h) in faces:
            # print "{0}/{1}_{2}".format(model_path, sno-1, fn)
            if (w<FACE_SIZE):
                offset = int((FACE_SIZE-w)*0.5)
                cv2.imwrite("{0}/{1}_{2}".format(model_path, sno-1, fn), frame[y-offset:y-offset+FACE_SIZE,x-offset:x-offset+FACE_SIZE])
            # cv2.imwrite("{0}/{1}_{2}".format(model_path, sno-1, fn), frame[y:y+h,x:x+w])

if __name__ == '__main__':
    createDatasets()
