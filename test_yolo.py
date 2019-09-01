#!/usr/bin/env python3

from darknet.python import darknet2
import cv2
#dll en C donc on passe des string encodées en argument : b''
import client


path = "/home/lucien/Documents/projet_dl_camera/camera_streamer_v2/darknet/"

image_path = path+"data/dog.jpg"

def drawBoundingBoxes(predictions,image):

	#print(predictions)
	for i in range(0,len(predictions)):
		(center_x, center_y, heigth,length) = predictions[i][2]

		x1 = int(center_x - heigth/2)
		y1 = int(center_y - length/2)
		x2 = int(x1 + heigth)
		y2 = int(y1 + length)

		cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
		cv2.putText(image,predictions[i][0].decode(),(x1,y1),cv2.FONT_HERSHEY_DUPLEX,1,1)

	cv2.imshow("data",image)
	key  = cv2.waitKey(1)
	return image


net = darknet2.load_net((path+"cfg/yolov3-tiny.cfg").encode(), (path+"yolov3-tiny.weights").encode(), 0)
meta = darknet2.load_meta((path+"cfg/coco.data").encode())


camera = client.Client()
camera.initConnexion()


for i in range(1,100):
	print(i)
	frame = camera.readFrame()
	predictions = darknet2.detect(net, meta, frame)
	print(predictions)
	drawBoundingBoxes(predictions, frame)
	
camera.sock.close()
cv2.destroyAllWindows()


"""
for i in range(1,5):
	image = cv2.imread("morceau"+str(i)+".jpg")
	#predictions = darknet2.detect(net, meta, frame)
	


	image = drawBoundingBoxes(predictions, frame)
	print(type(image))
	name = "prediction"+str(i)+".jpg"
	cv2.imwrite(name,image)



"""