#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: wwwins
# @Date:   2017-08-08 18:45:16
# @Last Modified by:   wwwins
# @Last Modified time: 2017-08-11 17:52:33

import cv2
import sys
import os
import time
import argparse
import crop_face
import numpy as np
from glob import glob
from crop_face import *

DEBUG = 0

# 依不同的 casca{de} 做調整
# lbpcascade_frontalface: 1.1
# haarcascade_frontalface_default: 1.3
# haarcascade_frontalface_alt2: 1.3
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 3
#MIN_SIZE = 30
MIN_SIZE = 80

# FIX_SIZE = True
FACE_SIZE = 200

parser = argparse.ArgumentParser()
parser.add_argument('src_images', default='output', help='source images folder')
parser.add_argument('datasets', default='datasets', help='datasets folder')
parser.add_argument('-f','--face_casc_file', nargs='?', const='data/haarcascade_frontalface_default.xml', default='data/haarcascade_frontalface_default.xml', help='face cascade file')
parser.add_argument('-e','--eye_casc_file', nargs='?', const='data/haarcascade_eye.xml', default='data/haarcascade_eye.xml', help='eye cascade file')
args = parser.parse_args()

cascPath = args.face_casc_file
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascPath = args.eye_casc_file
eyeCascade = cv2.CascadeClassifier(eyeCascPath)
image_path = args.src_images
model_path = args.datasets

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
        x, y, w, h = faces[0]
        roi_gray = gray[y:y+h,x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
    else:
        print ("{0} not found faces!".format(fn))
    return faces, eyes, roi_gray

def createDatasets():
    sno = 0
    buf = ''
    for f in glob('{0}/*_[0-9]*.png'.format(image_path)):
        frame = cv2.imread(f)
        fn = f.split('/')[1]
        if buf != fn.split(' ')[0]:
            buf = fn.split(' ')[0]
            sno = sno + 1
        faces, eyes, roi_gray = faceDetect(frame, fn)
        x, y, w, h = faces[0]
        if (len(eyes)==2):
            pos = []
            for ex,ey,ew,eh in eyes:
                center_pos = (ex+int(ew*0.5), ey+int(eh*0.5))
                pos.append(center_pos)
                if DEBUG:
                    cv2.rectangle(roi_gray, (ex,ey),(ex+ew,ey+eh),(0,0,255),1)
                    cv2.circle(roi_gray, center_pos, 3, (0,0,255), 1)
            if pos[0][0]<pos[1][0]:
                eye_left = pos[0]
                eye_right = pos[1]
            else:
                eye_left = pos[1]
                eye_right = pos[0]
            cv2_im = cv2.cvtColor(roi_gray,cv2.COLOR_BGR2RGB)
            cropframe = CropFace(Image.fromarray(cv2_im), eye_left=eye_left, eye_right=eye_right, offset_pct=(0.2,0.2), dest_sz=(FACE_SIZE,FACE_SIZE))
            cv2.imshow('window', cv2.cvtColor(np.array(cropframe), cv2.COLOR_RGB2BGR))
            # cv2.imshow('window', roi_gray)
            cv2.waitKey(10)

        if DEBUG:
            print "{0}/{1}_{2}".format(model_path, sno-1, fn)

        # if FIX_SIZE:
        #     if (w<FACE_SIZE):
        #         offset = int((FACE_SIZE-w)*0.5)
        #         cv2.imwrite("{0}/{1}_{2}".format(model_path, sno-1, fn), frame[y-offset:y-offset+FACE_SIZE,x-offset:x-offset+FACE_SIZE])
        # else:
        #     cv2.imwrite("{0}/{1}_{2}".format(model_path, sno-1, fn), frame[y:y+h,x:x+w])
        cropframe.save("{0}/{1}_{2}".format(model_path, sno-1, fn))

if __name__ == '__main__':
    createDatasets()
