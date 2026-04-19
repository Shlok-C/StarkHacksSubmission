import cv2
import sys
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from ai_edge_litert.interpreter import Interpreter, load_delegate

CAMERA_INDEX   = int(sys.argv[1]) if len(sys.argv) > 1 else 0
PORT           = 8080
INFER_EVERY    = 3
CONF_THRESHOLD = 0.7

DET_PATH  = "models/mediapipe_hand_gesture-palm_detector-w8a8.tflite"
LM_PATH   = "models/mediapipe_hand_gesture-hand_landmark_detector-w8a8.tflite"
GEST_PATH = "models/mediapipe_hand_gesture-canned_gesture_classifier-w8a8.tflite"

GESTURE_LABELS = ["Unknown", "Closed_Fist", "Open_Palm", "Pointing_Up",
                  "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]

# ── Quantization params ───────────────────────────────────────────────────────
DET_COORD_SCALE  = 1.590442;  DET_COORD_ZP  = 51
DET_SCORE_SCALE  = 0.003906;  DET_SCORE_ZP  = 0
LM_LM_SCALE      = 0.972075;  LM_LM_ZP     = 33
LM_SCORE_SCALE   = 0.003906;  LM_SCORE_ZP  = 0
GEST_IN_SCALE    = 0.006957;  GEST_IN_ZP   = 111
GEST_OUT_SCALE   = 0.003906;  GEST_OUT_ZP  = 0

# ── Load all models on NPU ────────────────────────────────────────────────────
def load_on_npu(path):
    try:
        d = load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
        interp = Interpreter(model_path=path, experimental_delegates=[d])
        print(f"NPU: {path.split('/')[-1]}")
    except Exception as e:
        print(f"CPU fallback ({e}): {path.split('/')[-1]}")
        interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

det_interp  = load_on_npu(DET_PATH)
lm_interp   = load_on_npu(LM_PATH)
gest_interp = load_on_npu(GEST_PATH)

det_in   = det_interp.get_input_details()
det_out  = {t['name']: t['index'] for t in det_interp.get_output_details()}
lm_in    = lm_interp.get_input_details()
lm_out   = {t['name']: t['index'] for t in lm_interp.get_output_details()}
gest_in  = {t['name']: t['index'] for t in gest_interp.get_input_details()}
gest_out = {t['name']: t['index'] for t in gest_interp.get_output_details()}

# ── Anchor grid ───────────────────────────────────────────────────────────────
def generate_anchors():
    anchors = []
    for grid, n in [(32, 2), (16, 2), (8, 6)]:
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                for _ in range(n):
                    anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)  # (2944, 2)

ANCHORS = generate_anchors()

# ── Preprocessing: scale≈1/255 zp=0 → raw uint8 pixels pass straight through ─
def to_uint8(img, size):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))[np.newaxis]

# ── Stage 1: detect hand bounding box ─────────────────────────────────────────
def detect_hand(frame):
    det_interp.set_tensor(det_in[0]['index'], to_uint8(frame, 256))
    det_interp.invoke()

    coords_q = det_interp.get_tensor(det_out['box_coords'])[0]   # (2944, 18) uint8
    scores_q = det_interp.get_tensor(det_out['box_scores'])[0]   # (2944,)    uint8

    scores = (scores_q.astype(np.float32) - DET_SCORE_ZP) * DET_SCORE_SCALE
    best   = int(np.argmax(scores))
    if scores[best] < CONF_THRESHOLD:
        return None

    coords = (coords_q.astype(np.float32) - DET_COORD_ZP) * DET_COORD_SCALE
    cx = coords[best, 0] / 256.0 + ANCHORS[best, 0]
    cy = coords[best, 1] / 256.0 + ANCHORS[best, 1]
    w  = coords[best, 2] / 256.0
    h  = coords[best, 3] / 256.0
    return float(cx), float(cy), float(w), float(h)

# ── Crop hand with padding ────────────────────────────────────────────────────
def crop_hand(frame, cx, cy, w, h, scale=2.2):
    fh, fw = frame.shape[:2]
    side = max(w * fw, h * fh) * scale
    x1 = max(0, int(cx * fw - side / 2))
    y1 = max(0, int(cy * fh - side / 2))
    x2 = min(fw, int(cx * fw + side / 2))
    y2 = min(fh, int(cy * fh + side / 2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    return crop, (x1, y1, x2 - x1, y2 - y1)

# ── Stage 2: detect 21 landmarks ──────────────────────────────────────────────
def detect_landmarks(crop, crop_rect):
    lm_interp.set_tensor(lm_in[0]['index'], to_uint8(crop, 224))
    lm_interp.invoke()

    score_q = lm_interp.get_tensor(lm_out['scores'])[0, 0]
    score   = float(score_q - LM_SCORE_ZP) * LM_SCORE_SCALE
    if score < 0.5:
        return None, None

    lm_q = lm_interp.get_tensor(lm_out['landmarks'])[0]           # (63,) uint8
    lm_f = (lm_q.astype(np.float32) - LM_LM_ZP) * LM_LM_SCALE    # dequantize
    lm   = lm_f.reshape(21, 3)                                     # (21, 3) [0-1]

    x0, y0, cw, ch = crop_rect
    pts = [(int(lm[i, 0] * cw + x0), int(lm[i, 1] * ch + y0)) for i in range(21)]
    return pts, lm

# ── Stage 3: classify gesture ─────────────────────────────────────────────────
def classify_gesture(lm):
    # Normalize: translate wrist to origin, scale by wrist→middle-MCP distance
    lm = lm.copy()
    lm -= lm[0]
    scale = np.linalg.norm(lm[9])
    if scale < 1e-6:
        return "Unknown"
    lm /= scale

    emb = np.zeros(64, dtype=np.float32)
    emb[:63] = lm.flatten()
    q = np.clip(np.round(emb / GEST_IN_SCALE + GEST_IN_ZP), 0, 255).astype(np.uint8).reshape(1, 64)

    gest_interp.set_tensor(gest_in['hand'],          q)
    gest_interp.set_tensor(gest_in['mirrored_hand'], q)
    gest_interp.invoke()

    out_q  = gest_interp.get_tensor(gest_out['Identity'])[0]       # (8,) uint8
    scores = (out_q.astype(np.float32) - GEST_OUT_ZP) * GEST_OUT_SCALE
    best   = int(np.argmax(scores))
    return GESTURE_LABELS[best] if scores[best] > 0.3 else "Unknown"

# ── Draw skeleton ─────────────────────────────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

def draw(frame, pts, gesture):
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)
    cv2.putText(frame, gesture, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    sys.exit(f"Cannot open camera {CAMERA_INDEX}")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ── Shared state ──────────────────────────────────────────────────────────────
raw_frame = latest_frame = None
latest_pts, latest_gesture = None, "..."
lock = threading.Lock()

def capture_loop():
    global raw_frame
    while True:
        ok, f = cap.read()
        if ok:
            with lock: raw_frame = f

def inference_loop():
    global latest_frame, latest_pts, latest_gesture
    n = 0
    while True:
        with lock: frame = None if raw_frame is None else raw_frame.copy()
        if frame is None: continue

        n += 1
        if n % INFER_EVERY == 0:
            box = detect_hand(frame)
            if box:
                crop, rect = crop_hand(frame, *box)
                if crop is not None:
                    pts, lm = detect_landmarks(crop, rect)
                    if pts:
                        latest_pts     = pts
                        latest_gesture = classify_gesture(lm)
            else:
                latest_pts     = None
                latest_gesture = "no hand"

        if latest_pts:
            draw(frame, latest_pts, latest_gesture)
        else:
            cv2.putText(frame, latest_gesture, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)

        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        latest_frame = jpg.tobytes()

threading.Thread(target=capture_loop,  daemon=True).start()
threading.Thread(target=inference_loop, daemon=True).start()

# ── HTTP stream ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<img src="/stream" style="width:100%">')
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    if latest_frame:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(latest_frame)
                        self.wfile.write(b'\r\n')
            except BrokenPipeError: pass

print(f"Stream at http://10.10.11.97:{PORT}/")
HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()