import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import math
import time
import urllib.request
import os
from collections import namedtuple, deque



MODEL_PATH = "gesture_recognizer.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Landmark indices
WRIST      = 0
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_TIP  = 8
MIDDLE_MCP = 9

# Model label → (display name, robot command)
# command=None → direction derived from finger vector
GESTURE_MAP = {
    "Closed_Fist": ("FIST",       "GRIP"),
    "Open_Palm":   ("OPEN HAND",  "RELEASE"),
    "Pointing_Up": ("POINT",      None),
    "Thumb_Up":    ("THUMB UP",   "POSITION_1"),
    "Thumb_Down":  ("THUMB DOWN", "MOVE_DOWN"),
    "Victory":     ("VICTORY",    "POSITION_2"),
    "ILoveYou":    ("ILY",        "POSITION_3"),
    "None":        ("---",        "NONE"),
}

GestureResult = namedtuple(
    "GestureResult",
    ["gesture_name", "command", "confidence", "landmarks", "hand_count"]
)


def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading gesture recognizer model (~25 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    except Exception as e:
        print(f"Download failed: {e}")
        exit(1)


def pointing_direction(landmarks):
    mcp   = landmarks[INDEX_MCP]
    tip   = landmarks[INDEX_TIP]
    angle = math.degrees(math.atan2(-(tip.y - mcp.y), tip.x - mcp.x))
    if   -45 <= angle <   45: return "MOVE_RIGHT"
    elif  45 <= angle <  135: return "MOVE_UP"
    elif angle >= 135 or angle < -135: return "MOVE_LEFT"
    else:                      return "MOVE_DOWN"


def pinch_distance(landmarks):
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    wrist = landmarks[WRIST]
    mmcp  = landmarks[MIDDLE_MCP]
    dist      = math.hypot(thumb.x - index.x, thumb.y - index.y)
    hand_size = math.hypot(wrist.x - mmcp.x,  wrist.y - mmcp.y)
    return dist / hand_size if hand_size > 0 else 1.0


def open_camera():
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"Using camera index {idx}")
                return cap
            cap.release()
    return None


class HandGestureRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        download_model()
        opts = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._rec = mp_vision.GestureRecognizer.create_from_options(opts)

    def process(self, rgb_frame, timestamp_ms) -> GestureResult:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        raw    = self._rec.recognize_for_video(mp_img, timestamp_ms)
        hand_count = len(raw.hand_landmarks)

        if not raw.gestures or not raw.hand_landmarks:
            return GestureResult("---", "NONE", 0.0, None, hand_count)

        top       = raw.gestures[0][0]
        label     = top.category_name
        conf      = top.score
        landmarks = raw.hand_landmarks[0]

        # Pinch overrides model output
        pd = pinch_distance(landmarks)
        if pd < 0.15:
            return GestureResult("PINCH", "GRIP_ADJUST",
                                 round(1.0 - pd / 0.15, 2), landmarks, hand_count)

        if label in GESTURE_MAP:
            name, command = GESTURE_MAP[label]
            if command is None:
                command = pointing_direction(landmarks)
            return GestureResult(name, command, conf, landmarks, hand_count)

        return GestureResult(label, "NONE", conf, landmarks, hand_count)

    def draw_landmarks(self, frame, result: GestureResult):
        if result.landmarks is None:
            return frame
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for pt in pts:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, (0, 150, 255), 1)
        return frame

    def draw_overlay(self, frame, result: GestureResult, history: deque):
        h, w = frame.shape[:2]

        # Top-left: gesture + command + confidence
        cv2.rectangle(frame, (10, 10), (440, 85), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture : {result.gesture_name}", (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
        conf_str = f"  ({result.confidence:.0%})" if result.confidence > 0 else ""
        cv2.putText(frame, f"Command : {result.command}{conf_str}", (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Bottom-left: command history
        cv2.rectangle(frame, (10, h - 115), (320, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, "Recent commands:", (15, h - 93),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        for i, cmd in enumerate(list(history)[-3:]):
            cv2.putText(frame, cmd, (15, h - 72 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)

        # Top-right: hand count + quit hint
        cv2.putText(frame, f"Hands: {result.hand_count}", (w - 145, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, "Q to quit", (w - 120, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return frame

    def close(self):
        self._rec.close()


def main():
    recognizer = HandGestureRecognizer()
    cap        = open_camera()
    if cap is None:
        print("Error: No webcam found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("GOON — Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("GOON — Gesture Recognition", 1280, 720)

    history      = deque(maxlen=10)
    last_command = ""
    start_time   = time.time()

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


if __name__ == "__main__":
    main()
