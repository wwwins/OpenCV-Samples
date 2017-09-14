#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: wwwins
# @Date:   2017-09-11 14:45:16

import cv2
import sys
import os
import time
import argparse
import dlib
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
FACE_SIZE = 250.0
CROP_IMAGE_WIDTH = 450
CROP_IMAGE_HEIGHT = 600

parser = argparse.ArgumentParser()
parser.add_argument('src_images', default='output', help='source images folder')
parser.add_argument('datasets', default='datasets', help='datasets folder')
parser.add_argument('-f','--face_casc_file', nargs='?', const='data/haarcascade_frontalface_default.xml', default='data/haarcascade_frontalface_default.xml', help='face cascade file')
args = parser.parse_args()

cascPath = args.face_casc_file
faceCascade = cv2.CascadeClassifier(cascPath)
image_path = args.src_images
model_path = args.datasets
blank_image = np.zeros((10,10,3), np.uint8)
detector = dlib.get_frontal_face_detector()

# 框出臉部
def draw_rect(frame, d):
    w = d.right()-d.left()
    h = d.bottom()-d.top()
    if w > FACE_SIZE:
        half_w = (CROP_IMAGE_WIDTH - w)/2
        half_h  = (CROP_IMAGE_HEIGHT - h)/3
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        cv2.rectangle(frame, (d.left()-half_w, d.top()-half_h), (d.right()+half_w, d.bottom()+half_h*2), (10, 10, 250), 2)
        return frame
    return blank_image

# 剪下臉部
def crop_rect(frame, d):
    ow, oh = frame.shape[:2]
    crop_h = CROP_IMAGE_HEIGHT
    crop_w = CROP_IMAGE_WIDTH
    if oh < CROP_IMAGE_HEIGHT:
        crop_h = oh
    if ow < CROP_IMAGE_WIDTH:
        crop_w = ow
    w = d.right()-d.left()
    h = d.bottom()-d.top()
    half_w = (crop_w - w)/2
    half_h = (crop_h - h)/10
    # cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
    # cv2.rectangle(frame, (d.left()-half_w, d.top()-half_h), (d.right()+half_w, d.bottom()+half_h*2), (10, 10, 250), 2)
    # 450x600
    crop_y1 = (d.top()-half_h*4)
    crop_y2 = (d.bottom()+half_h*6)
    if crop_y1 < 0:
        crop_y1 = 0
        crop_y2 = crop_h
    if crop_y2 > oh:
        crop_y1 = oh - crop_h
        crop_y2 = oh

    crop_x1 = (d.left()-half_w)
    crop_x2 = (d.right()+half_w)
    if crop_x1 < 0:
        crop_x1 = 0
        crop_x2 = crop_w
    if crop_x2 > ow:
        crop_x1 = ow - crop_w
        crop_x2 = ow
    return frame[crop_y1:crop_y2, crop_x1:crop_x2]

# 依據FACE_SIZE調整圖片大小
def resize_image(frame,d):
    w = d.right()-d.left()
    h = d.bottom()-d.top()
    k = FACE_SIZE/w
    frame_h, frame_w = frame.shape[:2]
    s_w = int(frame_w*k)
    s_h = int(frame_h*k)
    return cv2.resize(frame,(s_w,s_h))

def resize_and_crop(frame,d):
    # resize
    resize_frame = resize_image(frame,d)
    faces = detector(resize_frame, 0)
    for i, d in enumerate(faces):
        if i==0:
           print('face size:{}x{}'.format((d.right()-d.left()),(d.bottom()-d.top())))
           gray = crop_rect(resize_frame,d)
    return gray

def faceDetect(gray, fn):
    faces = detector(gray, 0)
    if len(faces)>0:
        print("{} dlib Found {} faces!".format(fn,len(faces)))
    # Drawing a rectangle
    for i, d in enumerate(faces):
        if i==0:
            print('face size:{}x{}'.format((d.right()-d.left()),(d.bottom()-d.top())))
            # print("{} Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(fn,i, d.left(), d.top(), d.right(), d.bottom()))
            # gray = draw_rect(gray, d)
            # gray = crop_rect(gray,d)
            # gray = resize_image(gray,d)
            gray = resize_and_crop(gray, d)
    return gray

def createDatasets():
    sno = 0
    buf = ''
    for f in glob('{0}/*.jpg'.format(image_path)):
    # for f in glob('{0}/*.png'.format(image_path)):
        frame = cv2.imread(f)
        fn = f.split('/')[1]
        if buf != fn.split(' ')[0]:
            buf = fn.split(' ')[0]
            sno = sno + 1
        frame = faceDetect(frame, fn)
        # 長寬大於100時才存檔
        if frame.shape[0] > 100:
            # print('Press any key to continus')
            # cv2.imwrite("{0}/{1}_{2}".format(model_path, sno-1, fn), frame)
            crop_image = np.zeros((600,450,3), np.uint8)
            crop_image[:frame.shape[0], :frame.shape[1]] = frame
            cv2.imshow('window', crop_image)
            cv2.imwrite("{0}/{1}".format(model_path, fn), crop_image)
            cv2.waitKey(1)
        else:
            print("File {} error:{}".format(fn, frame.shape))

        if DEBUG:
            print "{0}/{1}_{2}".format(model_path, sno-1, fn)

if __name__ == '__main__':
    createDatasets()
