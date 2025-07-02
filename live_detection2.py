# IMPORTING REQUIRED LIBRARIES
import cv2 as cv
import os
import numpy as np
import time
import torch
import mediapipe as mp
from statistics import mode

capture = cv.VideoCapture(0) # CAPTURING EVERY FRAME PC IN-BUILT CAMERA

# CALLING FUNCTIONS FROM MEDIAPIPE FOR HAND POSE TRACKING
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# LOADING AND INITIALIZING THE YOLOv5 TRAINED MACHINE LEARNING MODEL
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/NEW/yolov5/runs/train/exp24/weights/best.pt', force_reload=True)
model.conf = 0.1
model.iou = 0.45
model.classes = None
model.eval()
names = model.names  

# INITIALIZING VARIABLES FOR CALCULATING FPS
current_time = 0
previous_time = 0

arm_status = # DECLARE ARM STATUS IN REST POSITION

# INITIALIZING A LIST FOR STORING DETECTED VALUES OF EACH FRAME CAPTURED
detection_list = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
                  False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, 
                  False, False, False, False, False, False]

# MODULE TO UPDATE LIST AND CALCULATE MODE
def precise_detection(shakehands_detected) :
    detection_list.pop(0)
    detection_list.append(shakehands_detected)

    return mode(detection_list)

while True:
    isTrue, frame = capture.read() # EACH FRAME BEING CAPTURED FROM VIDEO CAMERA

    # COMPARING THE FRAME WITH TRAINED MODEL AND CALCULATING RESULTS
    results = model(frame)
    # results.print()
    rendered_frame = np.squeeze(results.render())

    # CALCULATING AND DISPLAYING FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv.putText(rendered_frame, f'FPS: {fps}', (500, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    shakehands_detected = False # BOOLEAN VARIABLE FOR DETECTION VALUE

    # DISPLAYING DETECTION VALUE FOR EACH FRAME
    for *box, conf, cls in results.xyxy[0]:
        class_name = names[int(cls)]

        if class_name.lower() == "shake hands":
            shakehands_detected = True

    if shakehands_detected:
        cv.putText(rendered_frame, "DETECTED : SHAKE HANDS", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    else :
        cv.putText(rendered_frame, "NONE DETECTED", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
     # CONTROLLING ROBOT ACTION BASED ON PRECISE DETECTION VALUE
    detection = precise_detection(shakehands_detected)
    
    if detection :
        cv.putText(rendered_frame, "SHAKE HANDS", (10,90), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), 2)
        if arm_status == # ALREADY IN RAISED POSITION :
            arm_status = # REMAIN THE SAME AND DO NOTHING
        else :
            # ROBOT SHAKE HANDS ID = 1
            arm_status = # IN SHAKE HAND POSITION
            
    else :
        cv.putText(rendered_frame, "RELEASE ARM", (10,90), cv.FONT_HERSHEY_COMPLEX, 1.0, (255,0,0), 2)
        if arm_status == # ALREADY RELEASED :
            arm_status = # REMAIN THE SAME AND DO NOTHING
        else :
            # ROBOT RELEASE ARM ID = 0
            arm_status = # IN RELEASED POSITION

    # HAND TRACKING
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results1 = hands.process(imgRGB)

    if results1.multi_hand_landmarks : 
        for handlms in results1.multi_hand_landmarks :
            mpDraw.draw_landmarks(rendered_frame, handlms, mpHands.HAND_CONNECTIONS)

    cv.imshow('LIVE DETECTION', rendered_frame)

    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
