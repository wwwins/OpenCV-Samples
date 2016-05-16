# -*- coding: utf-8 -*-
from bottle import route, run, template, get, post, request, response
from random import randint
from camera import VideoCamera


@route('/stream')
def stream():
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    v = VideoCamera()
    while True:
        r = yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + v.get_frame() + b'\r\n\r\n')

run(host='localhost', port=8080, debug=True)
