# OpenCV-Samples
My OpenCV Samples

## 人臉偵測/追蹤
img-face-detect.py
```python
python img-face-detect.py images/MonaLisa.jpg data/haarcascade_frontalface_alt2.xml
```
webcam-face-detect.py
```python
python webcam-face-detect.py data/haarcascade_frontalface_alt2.xml
```
![img-face-detect.png](https://raw.githubusercontent.com/wwwins/OpenCV-Samples/master/screenshots/img-face-detect.png)

## 物體追蹤
obj-tracking.py
```python
// for image
python obj-tracking.py -i img.png
// for webcam
python obj-tracking.py -i 0
```
![obj-tracking.png](https://raw.githubusercontent.com/wwwins/OpenCV-Samples/master/screenshots/obj-tracking.png)

## 特偵點應用
feature_detection.py

```python
python feature_detection -i img.png -d [SIFT/SURF/ORB]
python feature_detection -m logo.png img.png
python feature_detection -s left.png right.png
```
![feature-surf.png](https://raw.githubusercontent.com/wwwins/OpenCV-Samples/master/screenshots/feature-surf.png)

![feature-pi-2.png](https://raw.githubusercontent.com/wwwins/OpenCV-Samples/master/screenshots/feature-pi-2.png)

## Face morph
facemorph.py
```python
python facemorph.py src.jpg dst.jpg -o output.avi
```
[video](https://youtu.be/rPn_D4v4Iko)

## Face recognition
webcam-face-recognition.py
Dynamic LBPH model file training and updating
```python
python webcam-face-recognition.py lbph-training.yml
```
[add later](https://)