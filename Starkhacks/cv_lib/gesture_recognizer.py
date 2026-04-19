import cv2
import sys
import mediapipe as mp
import math
import time
import os
import threading
import numpy as np
from collections import namedtuple, deque
from http.server import BaseHTTPRequestHandler, HTTPServer

# AI Edge LiteRT / Qualcomm QNN Imports
from ai_edge_litert.interpreter import Interpreter, load_delegate

# --- Configuration & Constants ---
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 2
PORT = 8080
NPU_MODEL_PATH = "models/old/mediapipe_hand_gesture-canned_gesture_classifier-w8a8.tflite"

# Landmark indices
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Finger Tip/MCP mapping for pointing logic
FINGERS = {
    "THUMB": (4, 2), "INDEX": (8, 5), "MIDDLE": (12, 9), "RING": (16, 13), "PINKY": (20, 17)
}

# The canned classifier usually has specific indices. Update these based on your model's labels:
LABELS = ["Unknown", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Up", "Thumb_Down", "Victory", "ILoveYou"]

GestureResult = namedtuple("GestureResult", ["gesture_name", "command", "confidence", "landmarks", "hand_count"])

# --- Helper Functions ---
def get_extended_finger(landmarks):
    extended = []
    # Thumb logic
    if math.hypot(landmarks[4].x - landmarks[0].x, landmarks[4].y - landmarks[0].y) > \
       math.hypot(landmarks[2].x - landmarks[0].x, landmarks[2].y - landmarks[0].y):
        extended.append("THUMB")
    # Other fingers logic
    for name, (tip, mcp) in list(FINGERS.items())[1:]:
        if landmarks[tip].y < landmarks[mcp].y:
            extended.append(name)
    if len(extended) == 1:
        return extended[0], FINGERS[extended[0]][0]
    return None, None

# --- NPU Powered Recognizer ---
class NPUGestureRecognizer:
    def __init__(self, model_path):
        # 1. Initialize MediaPipe for Landmarks (CPU/GPU side)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )
        
        # 2. Initialize Qualcomm NPU Delegate (HTP side)
        print(f"Loading NPU Model: {model_path}")
        try:
            qnn_delegate = load_delegate("libQnnTFLiteDelegate.so", options={"backend_type": "htp"})
            self.interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[qnn_delegate]
            )
            self.interpreter.allocate_tensors()
        except Exception as e:
            print(f"Failed to load NPU delegate: {e}")
            sys.exit(1)

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        # 3. Get Quantization Params for Float32 -> UInt8 conversion
        # This is critical for w8a8 models on the HTP backend
        self.input_scale, self.input_zero_point = self.input_details['quantization']
        self.output_scale, self.output_zero_point = self.output_details['quantization']

    def process(self, rgb_frame) -> GestureResult:
        results = self.mp_hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return GestureResult("---", "NONE", 0.0, None, 0)

        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Pointing Override (Uses original landmarks)
        pointing_finger, tip_idx = get_extended_finger(landmarks)
        if pointing_finger:
            return GestureResult(f"POINTING ({pointing_finger})", "MOVE_DIRECTED", 1.0, landmarks, 1)

        # --- NPU Inference Path ---
        # 1. Prepare raw features (63 values)
        input_data = []
        for lm in landmarks:
            input_data.extend([lm.x, lm.y, lm.z])
        
        # 2. ADD PADDING TO REACH 64 DIMENSIONS
        # This aligns the vector for the Qualcomm HTP hardware
        input_data.append(0.0) 
        
        input_array = np.array(input_data, dtype=np.float32)

        # 3. Quantize: f32 -> u8
        quant_input = (input_array / self.input_scale) + self.input_zero_point
        quant_input = np.clip(quant_input, 0, 255).astype(np.uint8)
        
        # Reshape to [1, 64] as expected by the model
        quant_input = np.expand_dims(quant_input, axis=0)

        # 4. Invoke NPU
        self.interpreter.set_tensor(self.input_details['index'], quant_input)
        self.interpreter.invoke()
        
        # 5. Get and De-quantize Output
        output_data = self.interpreter.get_tensor(self.output_details['index'])[0]
        if output_data.dtype == np.uint8:
            output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale

        max_idx = np.argmax(output_data)
        conf = float(output_data[max_idx])
        label = LABELS[max_idx] if max_idx < len(LABELS) else "Unknown"

        return GestureResult(label, "NONE", conf, landmarks, 1)
    
    def draw_ui(self, frame, result):
        h, w = frame.shape[:2]
        if result.landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.landmarks]
            
            # Logic for tracking the 'pointing' pixel
            _, tip_idx = get_extended_finger(result.landmarks)
            active_tip_idx = tip_idx if tip_idx else 8
            tip_pt = pts[active_tip_idx]

            # Draw Skeleton
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
            cv2.circle(frame, tip_pt, 8, (0, 0, 255), -1)

            # GUI Box
            cv2.rectangle(frame, (5, 5), (320, 95), (0, 0, 0), -1)
            cv2.putText(frame, f"G: {result.gesture_name}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            cv2.putText(frame, f"Conf: {result.confidence:.2f}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"P: ({tip_pt[0]}, {tip_pt[1]})", (10, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "BACKEND: QUALCOMM NPU (HTP)", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
        return frame
    
# --- Main Logic / Server (Unchanged Architecture) ---
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

raw_frame = latest_frame = None
lock = threading.Lock()

def capture_loop():
    global raw_frame
    while True:
        ok, f = cap.read()
        if ok:
            with lock: raw_frame = f

def inference_loop():
    global latest_frame
    recognizer = NPUGestureRecognizer(NPU_MODEL_PATH)
    while True:
        with lock: frame = None if raw_frame is None else raw_frame.copy()
        if frame is None: continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = recognizer.process(rgb)
        processed_frame = recognizer.draw_ui(frame, result)
        
        _, jpg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        latest_frame = jpg.tobytes()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b'<html><body style="background:#222;color:white;text-align:center;">'
                             b'<h1>RubikPi Robot Vision</h1>'
                             b'<img src="/stream" style="width:80%; border:2px solid #555;">'
                             b'</body></html>')
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
                    time.sleep(0.03) # Limit stream to ~30fps
            except: pass

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    print(f"Robot stream active at http://[PI_IP_ADDRESS]:{PORT}/")
    HTTPServer(('0.0.0.0', PORT), Handler).serve_forever()