"""
GOON — Gesture-Operated Robotic Arm
3-stage NPU-accelerated gesture recognition (palm detect → landmark → classify),
served as an MJPEG stream over HTTP.

Usage:
    python main.py [camera_index]   # default camera 2
    View stream at http://<device-ip>:8080/
"""

import cv2
import sys
import json
import numpy as np
import torch
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from qai_hub_models.models._shared.mediapipe.utils import preprocess_hand_x64
from qai_hub_models.models.mediapipe_hand_gesture.model import GESTURE_LABELS

from cv_lib.npu_runtime import NPUInterpreter
from cv_lib.gesture_recognition import HAND_CONNECTIONS

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PORT           = 8080
MODELS_DIR     = Path(__file__).parent / "models"
CONF_THRESHOLD = 0.75
LANDMARK_CONF  = 0.2
COMMAND_HISTORY_LEN = 10

# ── Models ────────────────────────────────────────────────────────────────────
PALM_PATH     = str(MODELS_DIR / "PalmDetector_qai.bin")
LANDMARK_PATH = str(MODELS_DIR / "HandLandmarkDetector_qai.bin")
GESTURE_PATH  = str(MODELS_DIR / "GestureClassifier_qai.bin")

print(f"Loading palm detector    : {Path(PALM_PATH).name}")
palm_det = NPUInterpreter(
    PALM_PATH,
    tflite_fallback=str(MODELS_DIR / "HandDetector.tflite"),
)

print(f"Loading landmark detector: {Path(LANDMARK_PATH).name}")
hand_ld = NPUInterpreter(
    LANDMARK_PATH,
    tflite_fallback=str(MODELS_DIR / "HandLandmarkDetector.tflite"),
)

# Gesture classifier: QNN .bin on device, PyTorch on dev machines
print(f"Loading gesture classifier: {Path(GESTURE_PATH).name}")
_gest_clf_npu: NPUInterpreter | None = None
_gest_clf_torch = None

try:
    _gest_clf_npu = NPUInterpreter(GESTURE_PATH)
    print(f"  gesture backend: {_gest_clf_npu.backend}")
except FileNotFoundError:
    print("  No TFLite fallback for gesture classifier — using PyTorch CPU")
    from qai_hub_models.models.mediapipe_hand_gesture.model import CannedGestureClassifier
    _gest_clf_torch = CannedGestureClassifier.from_pretrained()
    _gest_clf_torch.eval()

def _run_gesture_clf(x64_a: np.ndarray, x64_b: np.ndarray) -> np.ndarray:
    """Returns class probability array [8]."""
    if _gest_clf_npu is not None:
        out = _gest_clf_npu.run({"hand": x64_a, "mirrored_hand": x64_b})
        return out["Identity"][0]
    # PyTorch fallback
    with torch.no_grad():
        probs = _gest_clf_torch(
            torch.from_numpy(x64_a), torch.from_numpy(x64_b)
        )
    return probs[0].numpy()

print(f"Backends — palm: {palm_det.backend}  landmark: {hand_ld.backend}  "
      f"gesture: {'qnn/tflite' if _gest_clf_npu else 'pytorch_cpu'}")

# ── Anchor grid (32²×2 + 16²×2 + 8²×6 = 2944 anchors) ───────────────────────
def _make_anchors():
    anchors = []
    for grid, n in [(32, 2), (16, 2), (8, 6)]:
        for y in range(grid):
            for x in range(grid):
                cx, cy = (x + 0.5) / grid, (y + 0.5) / grid
                for _ in range(n):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)

ANCHORS = _make_anchors()

# ── Gesture → robot command map ───────────────────────────────────────────────
# GESTURE_LABELS = ['None','Closed_Fist','Open_Palm','Pointing_Up','Thumb_Down','Thumb_Up','Victory','ILoveYou']
COMMAND_MAP = {
    "Closed_Fist": "GRIP",
    "Open_Palm":   "RELEASE",
    "Pointing_Up": "MOVE_UP",
    "Thumb_Up":    "POSITION_1",
    "Thumb_Down":  "MOVE_DOWN",
    "Victory":     "POSITION_2",
    "ILoveYou":    "POSITION_3",
    "None":        "NONE",
}

# ── Pre/post processing ───────────────────────────────────────────────────────
def _preprocess(img, size=256, backend=None):
    """BGR frame → float32 NHWC [1, H, W, 3] — used by both QNN and TFLite backends."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (cv2.resize(rgb, (size, size)).astype(np.float32) / 255.0)[np.newaxis]


def _decode_box(box_coords, box_scores):
    """Sigmoid + argmax on raw detector outputs → (cx, cy, w, h) in [0,1] or None."""
    scores = box_scores.reshape(-1).clip(-88, 88)   # clip prevents exp overflow
    conf = 1.0 / (1.0 + np.exp(-scores))
    best = int(np.argmax(conf))
    if conf[best] < CONF_THRESHOLD:
        return None
    cx = box_coords[best, 0] / 256.0 + ANCHORS[best, 0]
    cy = box_coords[best, 1] / 256.0 + ANCHORS[best, 1]
    w  = box_coords[best, 2] / 256.0
    h  = box_coords[best, 3] / 256.0
    return float(cx), float(cy), float(w), float(h)


def _crop_hand(frame, cx, cy, w, h, scale=2.2):
    fh, fw = frame.shape[:2]
    side = max(w * fw, h * fh) * scale
    x1 = int(cx * fw - side / 2);  y1 = int(cy * fh - side / 2)
    x2 = int(cx * fw + side / 2);  y2 = int(cy * fh + side / 2)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fw, x2), min(fh, y2)
    crop = frame[y1c:y2c, x1c:x2c]
    return (None, None) if crop.size == 0 else (crop, (x1c, y1c, x2c - x1c, y2c - y1c))


def _run_gesture(frame):
    """
    Full 3-stage pipeline.
    Returns (gesture_label, command, confidence, pts) where pts is a list of
    (px, py) tuples in original frame coordinates, or (None, 'NONE', 0.0, None).
    """
    # Stage 1: Palm detection
    out = palm_det.run(_preprocess(frame))
    coords = out.get("box_coords", next(iter(out.values())))[0]       # [2944, 18]
    scores = out.get("box_scores",  list(out.values())[-1])[0]        # [2944]
    box = _decode_box(coords, scores)
    if box is None:
        return None, "NONE", 0.0, None

    cx, cy, w, h = box
    crop, crop_rect = _crop_hand(frame, cx, cy, w, h)
    if crop is None:
        return None, "NONE", 0.0, None

    # Stage 2: Hand landmark detection
    out2 = hand_ld.run(_preprocess(crop))
    ld_score = float(out2["scores"].flat[0])
    if ld_score < LANDMARK_CONF:
        return None, "NONE", 0.0, None

    lr_val = float(out2["lr"].flat[0])
    lm = out2["landmarks"][0].reshape(21, 3)   # [21, 3] pixel coords in 256×256 crop

    # Map landmark pixel coords (in 256×256 crop) → original frame
    x0, y0, cw, ch = crop_rect
    pts = [
        (int(lm[i, 0] / 256.0 * cw + x0),
         int(lm[i, 1] / 256.0 * ch + y0))
        for i in range(21)
    ]

    # Stage 3: Gesture classification
    hand_t = torch.from_numpy(lm).unsqueeze(0).float()   # [1, 21, 3]
    lr_t   = torch.tensor([[lr_val]])                     # [1, 1]
    x64_a  = preprocess_hand_x64(hand_t, lr_t, mirror=False).numpy()
    x64_b  = preprocess_hand_x64(hand_t, lr_t, mirror=True).numpy()

    probs  = _run_gesture_clf(x64_a, x64_b)               # [8]
    idx    = int(np.argmax(probs))
    label  = GESTURE_LABELS[idx]
    conf   = float(probs[idx])
    cmd    = COMMAND_MAP.get(label, "NONE")

    return label, cmd, conf, pts


# ── Drawing ───────────────────────────────────────────────────────────────────
def _draw(frame, label, cmd, conf, pts, history):
    if pts:
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
        for pt in pts:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, (0, 150, 255), 1)

    h, w = frame.shape[:2]
    scale = w / 640.0
    def fs(s): return max(0.35, s * scale)
    def px(x): return int(x * scale)

    lbl  = label or "---"
    conf_str = f"  ({conf:.0%})" if conf > 0 else ""
    cv2.rectangle(frame, (px(10), px(10)), (px(430), px(85)), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture : {lbl}",
                (px(15), px(38)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.8), (0, 255, 100), 2)
    cv2.putText(frame, f"Command : {cmd}{conf_str}",
                (px(15), px(68)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.7), (0, 200, 255), 2)

    box_h = px(105)
    cv2.rectangle(frame, (px(10), h - box_h), (px(310), h - px(10)), (0, 0, 0), -1)
    cv2.putText(frame, "Recent commands:",
                (px(15), h - box_h + px(22)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.5), (180, 180, 180), 1)
    for i, c in enumerate(list(history)[-3:]):
        cv2.putText(frame, c,
                    (px(15), h - box_h + px(22) + px(24) * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.58), (255, 255, 255), 1)


# ── Camera ────────────────────────────────────────────────────────────────────
_GST_PIPELINE = (
    f"v4l2src device=/dev/video{CAMERA_INDEX} "
    "! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 "
    "! videoconvert ! appsink"
)
cap = cv2.VideoCapture(_GST_PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    # Fallback: plain OpenCV index
    cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: could not open camera {CAMERA_INDEX}")
    sys.exit(1)
print(f"Camera /dev/video{CAMERA_INDEX}: "
      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# ── Shared state ──────────────────────────────────────────────────────────────
_raw_frame  = None
_latest_jpg = None
_frame_lock = threading.Lock()
_command_history: deque = deque(maxlen=COMMAND_HISTORY_LEN)
_last_command = ""


def capture_loop():
    global _raw_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _raw_frame = frame


def inference_loop():
    global _latest_jpg, _last_command
    while True:
        with _frame_lock:
            frame = _raw_frame.copy() if _raw_frame is not None else None
        if frame is None:
            time.sleep(0.001)
            continue

        label, cmd, conf, pts = _run_gesture(frame)

        if cmd != "NONE" and cmd != _last_command:
            print(f"[gesture] {label:15}  →  {cmd}  ({conf:.0%})")
            _command_history.append(cmd)
            _last_command = cmd
        elif cmd == "NONE":
            _last_command = ""

        annotated = frame.copy()
        _draw(annotated, label, cmd, conf, pts, _command_history)
        _, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 50])
        _latest_jpg = jpg.tobytes()


threading.Thread(target=capture_loop,  daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()

# ── HTTP stream ───────────────────────────────────────────────────────────────
class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b'<img src="/stream" style="width:100%">')
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    if _latest_jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(_latest_jpg)
                        self.wfile.write(b"\r\n")
            except BrokenPipeError:
                pass
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            payload = json.dumps({
                "backends": {
                    "palm":     palm_det.backend,
                    "landmark": hand_ld.backend,
                    "gesture":  gest_clf.backend,
                }
            }).encode()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()


server = HTTPServer(("0.0.0.0", PORT), StreamHandler)
print(f"\nGOON running — stream at http://0.0.0.0:{PORT}/")
print(f"Status JSON      : http://0.0.0.0:{PORT}/status")
print("Ctrl+C to stop\n")

try:
    server.serve_forever()
except KeyboardInterrupt:
    pass

cap.release()
server.server_close()
palm_det.close()
hand_ld.close()
if _gest_clf_npu is not None:
    _gest_clf_npu.close()
