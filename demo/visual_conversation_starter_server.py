import json

import cv2
import zmq
import numpy as np
import uuid
import websocket
import time

import extract_caption

config = {
    "Furhat_IP": "192.168.0.101", # IP of the Furhat
    "detection_period": 50, # How often a frame should be processed
    # (i.e. every for 50 frames that come from the camera stream, 1 will be turned into a question)
    "Dev_IP": "192.168.0.103", # IP of the local machine
    "detection_exposure_port": "9999", # Port to which the Furhat will connect
    "caption_socket_ip":"localhost", # IP at which the caption server is available
    "caption_socket_port":9000, # Port at which the caption server is available
    "question_socket_ip":"localhost", # IP at which the caption server is available
    "question_socket_port": 9090 # Port at which the question-generating server is available
}


def _send_msg(ws, id, text):
    data = {
        "id": id,
        "text": text
    }
    ws.send(json.dumps(data))
    print("Sent message: " + text)


def _recv_msg(ws):
    message = ws.recv()
    incoming_message = json.loads(message)
    return incoming_message["text"]


while True:
    # Setup the sockets
    context = zmq.Context()

    try:
        # Input camera feed from furhat using a SUB socket
        insocket = context.socket(zmq.SUB)
        insocket.setsockopt_string(zmq.SUBSCRIBE, '')
        insocket.connect('tcp://' + config["Furhat_IP"] + ':3000')
        insocket.setsockopt(zmq.RCVHWM, 1)
        insocket.setsockopt(zmq.CONFLATE, 1)  # Only read the last message to avoid lagging behind the stream.
    except ConnectionRefusedError as err:
        print("No connection with Furhat Camera possible right now")
        print(err)
        time.sleep(1)
        continue

    try:
        # Output results using a PUB socket
        context2 = zmq.Context()
        outsocket = context2.socket(zmq.PUB)
        outsocket.bind("tcp://" + config["Dev_IP"] + ":" + config["detection_exposure_port"])
        print("connected to Furhat")
    except ConnectionRefusedError as err:
        print("No connection possible right now with Furhat Output")
        print(err)
        time.sleep(1)
        continue

    try:
        print("connecting to question server")
        id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://" + str(config["question_socket_ip"]) + ":" + str(config["question_socket_port"]) + "/websocket")
        print('connected, entering loop')
    except ConnectionRefusedError as err:
        print("No connection possible right now with question server")
        print(err)
        time.sleep(1)
        continue

    _send_msg(ws, id, "Hello, Server")
    print(_recv_msg(ws))
    _send_msg(ws, id, "begin")
    print(_recv_msg(ws))

    iterations = 0
    detection_period = config[
        "detection_period"]  # Detecting objects is resource intensive, so we try to avoid detecting objects in every frame
    x = True
    while x:
        string = insocket.recv()
        magicnumber = string[0:3]
        #print(magicnumber)
        # check if we have a JPEG image (starts with ffd8ff)
        if magicnumber == b'\xff\xd8\xff':
            buf = np.frombuffer(string, dtype=np.uint8)
            img = cv2.imdecode(buf, flags=1)

            if (iterations % detection_period == 0):

                print("Detecting objects!")
                buf = np.frombuffer(string, dtype=np.uint8)
                img = cv2.imdecode(buf, flags=1)
                height, width, channels = img.shape

                print("Connecting to caption server")
                try:
                    context3 = zmq.Context()
                    caption_socket = context3.socket(zmq.REQ)
                    caption_socket.connect("tcp://" + str(config["caption_socket_ip"]) + ":" + str(config["caption_socket_port"]))
                    caption_socket.send(string)
                    print("Request sent")
                    message = caption_socket.recv()
                    print("Reply received: " + str(message))
                except ConnectionRefusedError as err:
                    print("No connection possible right now")
                    print(err)
                    time.sleep(1)
                    continue

                #cv2.imwrite("im.jpg", img)

                #os.system("python predict_single.py --image im.jpg --output out.txt")

                results = json.loads(message)
                caption = extract_caption.extract_captions(results)
                print(caption)

                _send_msg(ws, id, caption)
                question = _recv_msg(ws)
                print(question)
                _send_msg(ws, id, "[RESET]")
                print(_recv_msg(ws))

                outsocket.send_string(question)

                cv2.imshow("img", img)

            iterations = iterations + 1

        k = cv2.waitKey(1)
        if k % 256 == 27:  # When pressing esc the program stops.
            # ESC pressed
            print("Escape hit, closing...")
            break

    ws.close()

