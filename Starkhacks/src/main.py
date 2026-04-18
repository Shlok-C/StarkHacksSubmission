from cv_lib import *
from arm_lib import *

import math
import time
import urllib.request
import os
from collections import namedtuple, deque
from enum import Enum

recognizer = HandGestureRecognizer()
cap = open_camera()

if cap is None:
    raise Exception("Error: No webcam found.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow("GOON — Gesture Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("GOON — Gesture Recognition", 1280, 720)

history = deque(maxlen=10)
last_command = ""
start_time = time.time()

print("GOON Gesture Recognition — Running  (Q to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts_ms  = int((time.time() - start_time) * 1000)
    result = recognizer.process(rgb, ts_ms)

    if result.command != last_command and result.command != "NONE":
        print(f"[gesture] {result.gesture_name:12} → {result.command}  ({result.confidence:.0%})")
        history.append(result.command)
        last_command = result.command
    elif result.hand_count == 0:
        last_command = ""

    recognizer.draw_landmarks(frame, result)
    recognizer.draw_overlay(frame, result, history)
    cv2.imshow("GOON — Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()
print("GOON Gesture Recognition — Stopped")

class State(Enum):
    SLEEP=0
    FREEZE=1
    FOLLOWING=2
    

