# -*- coding: utf-8 -*-
#!/usr/bin/env python
#
# faceMorph.py
# fork from https://github.com/spmallick/learnopencv/tree/master/FaceMorph
#

import dlib
import numpy as np
import cv2
import sys
import argparse

# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False

    return True

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in xrange(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

def morph(points1,points2,alpha):
    points = []

    # Compute weighted average point coordinates
    for i in xrange(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))

    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

    for pt in dt:
        x, y, z = list(pt)

        x = int(x)
        y = int(y)
        z = int(z)

        try:
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]
            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
        except:
            pass

    return imgMorph

# Calculate delanauy triangle
def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]));

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList();

    # Find the indices of triangles in the points array
    delaunayTri = []
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri

# 從 dlib 取得臉部 68 個特徵點
def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.array([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

if __name__ == '__main__':

    steps = True
    animate = True
    saveFiles = False
    saveVideo = False

    parser = argparse.ArgumentParser(description='face morphing')
    parser.add_argument('src', help='src image file')
    parser.add_argument('dst', help='dst image file')
    parser.add_argument('-f', dest='shape_dat', metavar='shape_dat', default='data/shape_predictor_68_face_landmarks.dat', help='shape predictor face landmarks file')
    parser.add_argument('-o', dest='output_file', metavar='output', help='output file')
    args = parser.parse_args()

    shape_dat = args.shape_dat
    filename1 = args.src
    filename2 = args.dst
    output = args.output_file
    if output:
        saveVideo = True 
    alpha = 0.5

    # Read images
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    # image width and height
    w = img2.shape[1]
    h = img2.shape[0]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_dat)
    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)

    # video
    if saveVideo:
        # out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (w,h))
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc('H','2','6','4'), 10, (w,h)) 

    # boundary points
    # x    x    x
    # 
    # x         x
    #
    # x         x
    #
    # x    x    x
    boundaryPts = np.array([(0,0), (w*0.5,0), (w-1,0), (w-1,h*0.25), (w-1,h*0.75), ( w-1, h-1 ), ( w*0.5, h-1 ), (0, h-1), (0,h*0.25), (0,h*0.75) ]);
    points1 = np.append(points1,boundaryPts,axis=0)
    points2 = np.append(points2,boundaryPts,axis=0)

    rect = (0, 0, w, h);
    dt = calculateDelaunayTriangles(rect, np.array(points2));

    cnt = 0
    if steps:
        for a in xrange(0,21):
            imgMorph = morph(points1,points2,a*0.05)
            if saveFiles:
                fn = "img{}.png".format(cnt)
                print ("processing:"+fn)
                cv2.imwrite("images/"+fn,imgMorph)
            if saveVideo:
                out.write(imgMorph)
            if animate:
                cv2.imshow("Morphed Face", np.uint8(imgMorph))
                if cnt<1:
                    print("press any key to continue...")
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(50)
            cnt = cnt + 1

    # Display Result
    cv2.imshow("Morphed Face", np.uint8(imgMorph))
    cv2.waitKey(0)
