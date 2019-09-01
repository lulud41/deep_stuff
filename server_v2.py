#!/usr/bin/env python3


import socket
import numpy as np
import cv2
import io
import time

from picamera import PiCamera



host =""
port=3001


msg_recu=b''
size_data_sent=0

def initCamera():
        camera = PiCamera()
        time.sleep(2)
        camera.resolution = (640, 480)
        camera.framerate =30
        return camera

def init_socket(host, port):

	connexion = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	connexion.bind(( host,port))
	connexion.listen(5)
	print('serveur dispo')
	socket_cree, infos_co = connexion.accept()
	print("la connexion est lance")

	return socket_cree,connexion

def sendFrame(stream):
    size_data_sent=0
  
    taille = stream.tell()
    sock.sendall(str(taille).encode())
    print("taille frame "+str(taille))
  
    while sock.recv(10).decode() != 'ok':
        time.sleep(0.1)
        print("waiting for ack")
    
    while size_data_sent < taille:  #pas sur
        size_data_sent += sock.send(stream.getvalue())
    
#    print("frame envoyee "+str(size_data_sent))
    

camera = initCamera()

sock,connexion = init_socket(host,port)
stream = io.BytesIO()

for frame in camera.capture_continuous(stream, format="jpeg", use_video_port=True):
    
    sendFrame(stream)

    stream.truncate(0)
    stream.seek(0)
   

print("fermeture de la co")
sock.close()
connexion.close()
