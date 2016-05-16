import socket
import time
import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (1920, 1080)
    camera.framerate = 20 

    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 2222))
    server_socket.listen(0)

    # Accept a single connection and make a file-like object out of it
    connection = server_socket.accept()[0].makefile('wb')
    try:
        camera.start_recording(connection, format='h264')
        camera.wait_recording(300)
        camera.stop_recording()
    finally:
        connection.close()
        server_socket.close()

