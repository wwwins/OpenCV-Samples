# -*- coding: utf-8 -*-
import cv2
import sys
import numpy
from collections import deque

DEBUG = False

'''
opencv hsv 定義如下，與一般網頁的 hsv 不同
h:0-179
s:0-255
v:0-255

rgb = 0,19,110
hsv = 115,255,110

rgb = 30,106,198
hsv = 108,217,200
'''
upper_blue = numpy.array([115,255,255])
lower_blue = numpy.array([107,80,80])

MAX_LEN = 32
points = deque(maxlen=MAX_LEN)


# Get user supplied values
imagePath = sys.argv[1]

def getCircleXY(cnts):
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=moments
    # 取得最大輪廓面積
    c = max(cnts, key=cv2.contourArea)
    # 取得最小可包圍輪廓的圓形
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    # 取得此面積的重心
    M = cv2.moments(c)
    if M["m00"] > 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        cx = cy = 0
    return ((int(x),int(y)), int(radius), (cx,cy))

def processImage(frame):
    # BGR->HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 去除雜點
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    if DEBUG:
        result = cv2.bitwise_and(frame,frame,mask=mask)

    # 傳回三個參數: image, contours, hierarchy
    _, cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    # draw all points
    # drawImg = cv2.drawContours(result, cnts, -1, (0,255,0), 3)
    if cnts:
        (position,radius,center) = getCircleXY(cnts)
        if radius > 10:
           cv2.circle(frame, position, radius,(0, 255, 255), 2)
           cv2.circle(frame, center, 5, (0, 0, 255), -1)
           points.append(center)
        else:
           points.append(None)

    if len(points) > 1:
        for i in range(len(points)-1,0,-1):
            if points[i] is None or points[i-1] is None:
                break
            #print("i:",i,",points:",points[i-1],points[i])
            cv2.line(frame, points[i-1], points[i], (200, 100, 30), int(0.4*(1+i)))

    cv2.imshow('demo',frame)
    if DEBUG:
        cv2.imshow('mask',mask)
        cv2.imshow('result',result)

def singleFrame():
    # cv2.imwrite(imagePath,frame)
    frame = cv2.imread(imagePath)
    processImage(frame)

    # cv2.imshow('cnts',drawImg)
    # 按任何鍵就往下執行
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video():
    # 設定 VideoCapture
    v = cv2.VideoCapture(0)
    v.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    v.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        _, frame = v.read()
        # 左右翻轉
        frame = cv2.flip(frame,1)
        processImage(frame)

        # 按下 q 跳離
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()
    # singleFrame()
