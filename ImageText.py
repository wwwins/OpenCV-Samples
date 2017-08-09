#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Author: wwwins
# @Date:   2017-08-09 11:08:28
# @Last Modified by:   wwwins
# @Last Modified time: 2017-08-09 11:46:32

import cv2
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_gradient_image(image):
    imgsize = image.size
    innerColor = [0, 92, 151]
    outerColor = [54, 55, 149]
    for y in range(imgsize[1]):
        for x in range(imgsize[0]):

            #Find the distance to the center
            distanceToCenter = math.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)
            #Make it on a scale from 0 to 1
            distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * imgsize[0]/2)
            #Calculate r, g, and b values
            r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
            g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)
            b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)

            #Place the pixel
            image.putpixel((x, y), (int(r), int(g), int(b)))
    return image

def get_image_text(contents):

    image = Image.new("RGB",(640,50),(0,0,0))

    image = get_gradient_image(image)
    draw = ImageDraw.Draw(image)
    text = contents

    font = ImageFont.truetype("/Library/Fonts/PingFang.ttc", 32)
    draw.text((10,0),unicode(text, 'UTF-8'), font=font, spacing=20, fill=(240,240,240))

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

