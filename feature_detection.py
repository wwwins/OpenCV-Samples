# -*- coding: utf-8 -*-
import cv2
import numpy
import sys
import imutils


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
device_id = 0

"""
    圖形比對
    - [Detect]用不同的演算法找出特徵點(SIFT/SURF/ORB/FAST)
    - [Match]用不同的比對方式找出你要的點(match/knnMatch/FlannBasedMatcher)
    - [Filter]matches points distances 愈小愈好
    - [Draw]
"""

MIN_MATCH_COUNT = 10

queryImagePath = None
imagePath = "img.png"

if len(sys.argv) < 3:
    print("""
    Usage:
            python feature_detection -i img.png
            python feature_detection -m logo.png img.png
            python feature_detection -s left.png right.png
    """)
    sys.exit(-1)

if sys.argv[1]=="-i":
    type = "features"
    imagePath = sys.argv[2]
if sys.argv[1]=="-m":
    type = "matches"
    queryImagePath = sys.argv[2]
    imagePath = sys.argv[3]
if sys.argv[1]=="-s":
    type = "stitch"
    queryImagePath = sys.argv[2]
    imagePath = sys.argv[3]

def video(queryFrame):
    # 設定 VideoCapture
    v = cv2.VideoCapture(device_id)
    v.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    v.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        _, frame = v.read()
        processImagesknnMatch(queryFrame,frame)
        # 按下 q 跳離
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v.release()
    cv2.destroyAllWindows()

def stitch(imageLeft, imageRight):
    detector = cv2.xfeatures2d.SURF_create(300)
    kpLeft, descLeft = detector.detectAndCompute(imageLeft, None)
    kpRight, descRight = detector.detectAndCompute(imageRight, None)
    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.knnMatch(descLeft, descRight, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    if len(good)>MIN_MATCH_COUNT:
        src_pts = numpy.float32([ kpRight[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = numpy.float32([ kpLeft[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)

        # get homography
        # http://finalfrank.pixnet.net/blog/post/30022271-%5B影像處理%5D-環場影像拼接的視差問題
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        # stitching
        panorama = cv2.warpPerspective(imageRight, M, (imageRight.shape[1] + imageLeft.shape[1], imageRight.shape[0]+200))
        panorama[0:imageLeft.shape[0], 0:imageLeft.shape[1]] = imageLeft
        cv2.imshow('panorama',panorama)

def processImage(frame):
    # 先將圖形轉成灰階
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """
    SURF Hessian Threshold: 800
    """
    # detector = cv2.xfeatures2d.SURF_create(800)
    # 特徵點不考慮方向性
    # detector.setUpright(True)
    """
    SIFT
    """
    # detector = cv2.xfeatures2d.SIFT_create()
    """
    ORB
    fixed: disable OpenCL https://github.com/Itseez/opencv/issues/6081
    """
    cv2.ocl.setUseOpenCL(False)
    detector = cv2.ORB_create(500)
    kp, desc = detector.detectAndCompute(frame, None)
    """
    FAST
    detector = cv2.FastFeatureDetector_create()
    kp = detector.detect(frame)
    """
    print("Keypoints:",len(kp))
    img = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('demo',img)

def processImagesMatch(frame1,frame2):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cv2.ocl.setUseOpenCL(False)
    detector = cv2.ORB_create(500)
    kp1, desc1 = detector.detectAndCompute(frame1, None)
    kp2, desc2 = detector.detectAndCompute(frame2, None)
    """
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher
    簡單說
    SIFT, SURF 用 cv2.NORM_L2 或是 cv2.NORM_L1
    ORB, BRIEF, BRISK 用 cv2.NORM_HAMMING
    ORB(VTA_K == 3 or 4) 用 cv2.NORM_HAMMING2
    """
    bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfmatcher.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)
    img = cv2.drawMatches(frame1,kp1,frame2,kp2,matches[:10], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('demo',img)

def processImagesknnMatch(frame1,frame2):
    # frame2_orig = frame2
    # frame1_orig = frame1
    # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # detector = cv2.xfeatures2d.SIFT_create()
    detector = cv2.xfeatures2d.SURF_create(300)
    kp1, desc1 = detector.detectAndCompute(frame1, None)
    kp2, desc2 = detector.detectAndCompute(frame2, None)
    bfmatcher = cv2.BFMatcher()
    """
    k = 2 (indicating the top two matches for each feature vector are returned)
    Count of best matches found per each query descriptor or
    less if a query descriptor has less than k possible matches in total.
    """
    matches = bfmatcher.knnMatch(desc1, desc2, k=2)
    """
    http://docs.opencv.org/master/d4/de0/classcv_1_1DMatch.html#gsc.tab=0
    DMatch:
            float   distance
            int     imgIdx
            int     queryIdx
            int     trainIdx
    """
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # 畫出對應區塊
    if len(good)>MIN_MATCH_COUNT:
        """
        reshape(-1,1,2)
        row,h = -1(取到完)
        column, w = 1
        dimensions = 2
        output:
                array([[[ 0.,  1.]],
                       [[ 1.,  2.]],
                       [[ 2.,  3.]],
                       [[ 3.,  4.]],
                       [[ 4.,  5.]],
                       [[ 5.,  6.]]], dtype=float32)
        """
        src_pts = numpy.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = numpy.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        # matchesMask = mask.ravel().tolist()

        h,w,_ = frame1.shape
        pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        # frame2 = cv2.polylines(frame2_orig,[numpy.int32(dst)],True,(200,100,50),3, cv2.LINE_AA)
        frame2 = cv2.polylines(frame2,[numpy.int32(dst)],True,(200,100,50),3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        # matchesMask = None

    img = cv2.drawMatchesKnn(frame1,kp1,frame2,kp2,good,None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('demo',img)


"""
    文件上說效能好不少，但目前版本的 OpenCV 3.1.0 有 bug。
    Bug: https://github.com/Itseez/opencv/issues/5667
"""
def processImagesFlannBasedMatcher(frame1,frame2):
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # detector = cv2.xfeatures2d.SIFT_create()
    detector = cv2.xfeatures2d.SURF_create(500)
    kp1, desc1 = detector.detectAndCompute(frame1, None)
    kp2, desc2 = detector.detectAndCompute(frame2, None)

    # FlannBasedMatcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # or pass empty dictionary
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in xrange(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = 0)

    img = cv2.drawMatchesKnn(rame1,kp1,frame2,kp2,matches,None,**draw_params)
    cv2.imshow('demo',img)

def main():
    # cv2.imwrite(imagePath,frame)
    frame = cv2.imread(imagePath)
    if type == "features":
        processImage(frame)
    elif type == "matches":
        queryFrame = cv2.imread(queryImagePath)
        # processImagesMatch(queryFrame, frame)
        # processImagesknnMatch(imutils.resize(queryFrame,width=400), imutils.resize(frame,width=400))
        processImagesknnMatch(queryFrame, frame)
        # processImagesFlannBasedMatcher(queryFrame, frame)
        # video(queryFrame)
    elif type == "stitch":
        queryFrame = cv2.imread(queryImagePath)
        stitch(queryFrame, frame)

    # cv2.imshow('cnts',drawImg)
    # 按任何鍵就往下執行
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Moves window to the specified position
    cv2.namedWindow("demo")
    cv2.moveWindow("demo", 1500, 200)

    main()
