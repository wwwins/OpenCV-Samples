# -*- coding: utf-8 -*-
import cv2
import sys
import numpy
from collections import deque

DEBUG = False

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 是否開啟繪圖模式
DRAW = False

'''
opencv hsv 定義如下，與一般網頁的 hsv 不同
h:0-179
s:0-255
v:0-255

rgb = 0,19,110
hsv = 115,255,110

rgb = 30,106,198
hsv = 106,216,198
'''
upper_blue = numpy.array([115,255,255])
lower_blue = numpy.array([107,80,80])

capture_type = "image"
device_id = 0
imagePath = "img.png"

MAX_LEN = 32
points = deque(maxlen=MAX_LEN)

# drawing
sp = (-1, -1)
drawingBoard = None

# Get user supplied values
if len(sys.argv) != 3:
    print("""
    Usage:
            python obj-tracking.py -i img.png
            python obj-tracking.py -s save.png
            python obj-tracking.py -d 0
    """)
    sys.exit(-1)

if sys.argv[1]=="-i":
    capture_type = "image"
    imagePath = sys.argv[2]
if sys.argv[1]=="-s":
    capture_type = "save"
    imagePath = sys.argv[2]
if sys.argv[1]=="-d":
    capture_type = "camera"
    device_id = int(sys.argv[2])

def drawing(img, p):
    global sp
    cv2.line(img,sp, p,(20, 100, 30), 10)
    sp = p

def createBlankImage(width, height, color=(255,255,255)):
    img = numpy.zeros((height, width, 3), numpy.uint8)
    img[:] = color
    return img

def getCircleXY(cnts):
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=moments
    # 取得最大輪廓面積
    c = max(cnts, key=cv2.contourArea)
    # 取得最小可包圍輪廓的圓形
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    # 取得此面積的重心
    # M = cv2.moments(c)
    # if M["m00"] > 0:
    #     cx = int(M['m10']/M['m00'])
    #     cy = int(M['m01']/M['m00'])
    # else:
    #     cx = cy = 0
    # return ((int(x),int(y)), int(radius), (cx,cy))
    return ((int(x),int(y)), int(radius), (int(x),int(y)))

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
            if DRAW:
                drawing(drawingBoard, center)
            cv2.circle(frame, position, radius,(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            points.append(center)
        else:
            points.append(None)

    if len(points) > 1:
        for i in range(len(points)-1,0,-1):
            if points[i] is None or points[i-1] is None:
                break
            # print("i:",i,",points:",points[i-1],points[i])
            cv2.line(frame, points[i-1], points[i], (200, 100, 30), int(0.4*(1+i)))
            # cv2.line(frame, points[i-1], points[i], tuple(numpy.random.randint(0,255,3).tolist()), int(0.4*(1+i)))

    if DRAW and capture_type=="camera":
        frame = cv2.addWeighted(frame,0.7,drawingBoard,0.3,0)

    # frame = cv2.stylization(frame, sigma_s=60, sigma_r=0.07)
    # dst_gray, dst_color = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)

    cv2.imshow('demo',frame)

    if DEBUG:
        cv2.imshow('mask',mask)
        cv2.imshow('result',result)

def saveFrame():
    v = cv2.VideoCapture(device_id)
    v.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    v.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    _, frame = v.read()
    cv2.imwrite(imagePath,frame)

def singleFrame():
    frame = cv2.imread(imagePath)
    processImage(frame)
    # cv2.imshow('cnts',drawImg)
    # 按任何鍵就往下執行
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video():
    global drawingBoard

    # 設定 VideoCapture
    v = cv2.VideoCapture(device_id)
    v.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    v.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    drawingBoard = createBlankImage(FRAME_WIDTH, FRAME_HEIGHT)

    while True:
        _, frame = v.read()
        # 左右翻轉
        frame = cv2.flip(frame,1)
        processImage(frame)
        if DEBUG:
            cv2.imshow("drawingBoard", drawingBoard)

        # 按下 q 跳離
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Moves window to the specified position
    cv2.namedWindow("demo")
    cv2.moveWindow("demo", 1500, 200)

    if capture_type == "camera":
        video()
    elif capture_type == "save":
        saveFrame()
    else:
        singleFrame()
