import socket, threading
from typing import ByteString
import cv2
import pickle
import numpy as np
import base64
import io
from PIL import Image

# import keras



# import keras_retinanet

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import os
import time
import tensorflow as tf
from tensorflow import keras

import sys

import serial






# 1. PC IP 입력
serverIp='172.30.1.57'


# 2. NMEA 신호발생기. 포트번호도 수정
aisSerialPort = serial.Serial(
    port='COM1',
    baudrate=38400,
)


def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(get_session())
#tf.keras.backend.set_sessionn(get_session())

# 3. 경로 수정. 만약 Visual Studio Code에서 실행하는경우 절대경로를 해주셔야 합니다.
# 아나콘다나 cmd등에서 실행하는거면 상대경로도 됩니다.
model_path = R'C:\Users\rian\Desktop\AICamera\Server\resnet50_coco_best_v2.1.0.h5'

model = models.load_model(model_path, backbone_name='resnet50')


print(model)

labels_to_names_no = {0: 'Breakwater', 1: 'Container_Ship', 2: 'Ferry_Coastal', 
        3: 'Ferry_Hsc', 4: 'Ferry_Ropax', 5: 'Fishing_Bridge_Closed',
                     6: 'Fishing_Bridge_Open', 7: 'pier', 8: 'Buoy'}

labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}



def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


# AI 데이터 수신 / 전송
def aiData(client_socket, addr):
    while True:
        # socket의 recv함수는 연결된 소켓으로부터 데이터를 받을 대기하는 함수입니다. 최초 4바이트를 대기합니다.
        data = client_socket.recv(4)
        length = int.from_bytes(data, "big")
        receiveData = recvall(client_socket, int(length))

        sitename_base64_str = receiveData.decode('utf-8')


        decoded_data = base64.b64decode(sitename_base64_str)
        np_data = np.fromstring(decoded_data,np.uint8)
        img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

        frame, scale = resize_image(img)
        # 4. 마찬가지로 경로 수정해주세요
        cv2.imwrite(R"C:\Users\rian\Desktop\AICamera\Server\OriginData.jpg",frame)

        draw = img.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(frame, axis=0))

        boxes /= scale


        boundBoxInfo =''
        scoreInfo =''
        labelInfo =''

        sendBoundingBoxData = ''


        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            
            draw_caption(draw, b, caption)\


            # 1. BoundBox 데이터는 ndarray값으로 되어있다.
            # 2. ndarray 값을 list로 변환한다
            # 3. list로 변환된 이후 string 형태로 다시 저장한다.
            # 4. BoundingBox 데이터 정보는 다음과 같다
            # x시작포인트,y시작포인트,x끝포인트,y끝포인트,라벨정보(이름),점수/
            # 5. 2개이상의 데이터가 전송될 경우 /로 구분된다.
            boundBoxInfoArray = b.tolist()
            boundBoxInfo = (','.join(str(e) for e in boundBoxInfoArray))
            scoreInfo = format(score)

            labelInfo = labels_to_names[label]
            sendBoundingBoxData += boundBoxInfo+","+labelInfo+","+scoreInfo+"/"

        # 5. 경로 수정
        cv2.imwrite(R"C:\Users\rian\Desktop\AICamera\Server\Labelling.jpg",draw)

        sendData = sendBoundingBoxData

        client_socket.sendall(sendData.encode())

        print(sendData)
        sys.stdout.flush()


    
# AIS 데이터 전송 (소켓 속도에 따라 전송) 
def aisData(client_socket,addr):
    while  True:
        if aisSerialPort.readable():
            res = aisSerialPort.readline()
            readAisData = res.decode()
            print(readAisData)
            client_socket.sendall(readAisData.encode())

            


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((serverIp, 5000))
server_socket.listen()
print("AI Socket Ready!")

server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket2.bind((serverIp, 5001))
server_socket2.listen()
print("AIS Socket Ready!")

try:
    while True:
        client_socket, addr = server_socket.accept()
        th = threading.Thread(target=aiData, args=(client_socket, addr))
        th.start()
        print("ai Server Connect!")

        client_socket2, addr2 = server_socket2.accept()
        th2 = threading.Thread(target=aisData, args=(client_socket2, addr2))
        th2.start()
        print("ais Server Connect!")
except:
    print("Server Error...")
finally:
    server_socket.close()
    server_socket2.close()