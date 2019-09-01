#!/usr/bin/env python3

import socket
import numpy as np
import cv2
import io

class Client:
    def __init__(self, host="10.42.0.209",port=3001):
        self.port=port
        self.host=host
        self.sock=None
        self.camera=None
        self.stream=io.BytesIO()
    
    def initConnexion(self):
   # gerer l'exception pas de serveur
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host,self.port))
            print("Connection lancée avec le serveur")
        except (ConnectionRefusedError):
            print("/!\\ Erreur: serveur non lancé    !!!! ")

    def displayData(self,data):
        cv2.imshow("frame ",data)
        key = cv2.waitKey(1)
    
    def readFrame(self):
        
        data_recieved=0
        data_size = int(self.sock.recv(10).decode())
        self.sock.send(b'ok')
        self.stream.truncate(0)
        self.stream.seek(0)

        while data_recieved < data_size:

            data_recieved += self.stream.write(self.sock.recv(data_size))
        self.stream.seek(0)
        frame = cv2.imdecode(np.fromstring(self.stream.read(), np.uint8), 1)
        return frame
