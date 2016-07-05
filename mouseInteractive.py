# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
import argparse


imagePath = "img.png"
sx = sy = None
previewImage = None

if len(sys.argv) < 3:
    print("""
    Usage:
            python mouseInteractive -i img.png
    """)
    sys.exit(-1)

if sys.argv[1]=="-i":
    imagePath = sys.argv[2]

def createBlankImage(width, height, color=(255,255,255)):
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = color
    return img

def mouseCallback(event,x,y,flags,param):
    global sx,sy,previewImage
    if (event == cv2.EVENT_LBUTTONDOWN):
        print(event,x,y,flags,param)
        bgrColor = frame[y][x]
        previewImage = createBlankImage(200,200,bgrColor)
        hsvColor = cv2.cvtColor(bgrColor.reshape(1,1,3),cv2.COLOR_BGR2HSV)

        print("bgr->hsv:{}->{}".format(bgrColor,hsvColor.tolist()[0][0]))

        cv2.circle(frame,(x,y),6, (0,0,255),-1)
        if (sx != None):
            cv2.line(frame,(sx,sy),(x,y),(0,0,255),3)
        sx = x
        sy = y
        cv2.imshow('demo', frame)
        cv2.imshow('preview', previewImage)

frame = cv2.imread(imagePath)

cv2.namedWindow("demo")
cv2.namedWindow("preview")
cv2.moveWindow("demo", 1500, 300)
cv2.moveWindow("preview", 1500, 80)
cv2.imshow('demo', frame)
cv2.setMouseCallback('demo', mouseCallback)

cv2.waitKey(0)
cv2.destroyAllWindows()
