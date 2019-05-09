# -*- coding: utf-8 -*-
#
# Copyright 2019 isobar. All Rights Reserved.
#
# Measure opencv fps
#
# Usage:
#       from ticket import ticket
#       t = ticket()
#       t.display()
#
import sys
import cv2

class ticket(object):

    def __init__(self):
        self._step = 3
        self._frameCount = 0
        self._freq = cv2.getTickFrequency()
        self._prevFrameTime = cv2.getTickCount()

    def fps(self):
        self._frameCount += 1
        nowFrameTime = cv2.getTickCount()
        fps = self._freq / (nowFrameTime - self._prevFrameTime)
        self._prevFrameTime = nowFrameTime
        fpsRounded = round(fps, 1)
        return fpsRounded

    def display(self):
        sys.stdout.write("\rfps:{}".format(self.fps()))
        if (self._frameCount%self._step==0):
            sys.stdout.flush()
