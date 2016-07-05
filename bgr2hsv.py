import cv2
import numpy
import sys


bgrColor = None

# Get user supplied values
if len(sys.argv) != 3:
    print("""
    Usage:
            python bgr2hsv.py -rgb 0,255,0

    Output:
            OpenCV HSV: [60, 255, 255]
    """)
    sys.exit()

if sys.argv[1]=="-rgb":
    bgrColor = sys.argv[2].split(',')
    bgrColor.reverse()

if bgrColor is None:
    sys.exit()

bgrColor = numpy.uint8([[bgrColor]])
hsvColor = cv2.cvtColor(bgrColor,cv2.COLOR_BGR2HSV)

print "OpenCV HSV:", hsvColor.tolist()[0][0]
