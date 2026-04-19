import cv2
import numpy as np
import odrive
from odrive.enums import AxisState, ControlMode, InputMode
import sys
import time
import os
import serial
import serial.tools.list_ports
import threading

try:
    from new_ik import move_to, Z_HOVER
except ImportError:
    print("Error: Could not import 'new_ik.py'. Ensure it's in the same directory.")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
CAMERA_INDEX = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
HOMOGRAPHY_FILE = "homography_matrix.npy"

# ODrive settings
GEAR_RATIO = 45.0
HOME_ANGLE_DEG = 90.0

# Globals
_ser = None
odrv = None
shoulder_axis = None
HOMOGRAPHY_MATRIX = None
target_mm = None
mega_status = {"mode": 0}

# ==========================================
# HARDWARE CONNECTION
# ==========================================
def connect_mega():
    """Auto-detect and connect to Arduino Mega."""
    global _ser
    device = None
    for port in serial.tools.list_ports.comports():
        if (port.vid == 0x2341 and port.pid in (0x0042, 0x0010)) or (port.vid in (0x1A86, 0x10C4)):
            device = port.device
            break
    
    if device:
        _ser = serial.Serial(device, 115200, timeout=0.1)
        time.sleep(2) 
        print(f"[Mega] Connected on {device}")
        threading.Thread(target=mega_reader, daemon=True).start()
    else:
        print("[Mega] Not found! Base and Elbow disabled.")

def mega_reader():
    """Reads status lines from Mega and handles homing handshakes."""
    global _ser
    while _ser and _ser.is_open:
        try:
            line = _ser.readline().decode().strip()
            if not line: continue
            
            parts = line.split(',')
            if len(parts) >= 7:
                mode = int(parts[0])
                mega_status["mode"] = mode
                
                # Check if Mega's physical button requested calibration
                if mode == 1:
                    print("[Mega] Homing requested via button! Sending ACK...")
                    _ser.write(b"2\n") 
        except Exception:
            break

def connect_odrive():
    """Connect to ODrive and prepare axis0."""
    global odrv, shoulder_axis
    print("[ODrive] Searching...")
    try:
        odrv = odrive.find_sync(timeout=5)
        shoulder_axis = odrv.axis0
        shoulder_axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
        shoulder_axis.controller.config.input_mode = InputMode.PASSTHROUGH
        # Hold current position
        shoulder_axis.controller.input_pos = shoulder_axis.encoder.pos_estimate
        print("[ODrive] Connected and holding.")
    except:
        print("[ODrive] Not found! Shoulder disabled.")

# ==========================================
# MOVEMENT DISPATCH
# ==========================================
def execute_move(th_b, th_s, th_e):
    """Sends angles to Mega (Base/Elbow) and ODrive (Shoulder)."""
    
    # Send Base and Elbow to Arduino Mega
    if _ser and _ser.is_open:
        cmd = f"A,{th_b:.2f},{th_e:.2f}\n"
        _ser.write(cmd.encode())
        print(f"[Mega] Sent -> Base: {th_b:.2f}°, Elbow: {th_e:.2f}°")

    # Send Shoulder to ODrive
    if shoulder_axis:
        relative_deg = th_s - HOME_ANGLE_DEG
        target_revs = (relative_deg / 360.0) * GEAR_RATIO
        shoulder_axis.controller.input_pos = target_revs
        print(f"[ODrive] Sent -> Shoulder: {th_s:.2f}° ({target_revs:.2f} revs)")

# ==========================================
# OPENCV MOUSE EVENT
# ==========================================
def on_mouse_click(event, u, v, flags, param):
    global target_mm
    if event == cv2.EVENT_LBUTTONDOWN:
        if HOMOGRAPHY_MATRIX is None: return

        # Transform Pixels (u,v) to MM (x,y)
        pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, HOMOGRAPHY_MATRIX)
        x_mm, y_mm = res[0][0]
        target_mm = (x_mm, y_mm)

        print(f"\n[Click] Target: x={x_mm:.1f}mm, y={y_mm:.1f}mm")

        # Solve Inverse Kinematics
        angles = move_to(x_mm, y_mm, z_above_table=Z_HOVER)
        if angles:
            th_b, th_s, th_e, _ = angles
            threading.Thread(target=execute_move, args=(th_b, th_s, th_e), daemon=True).start()

# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    if os.path.exists(HOMOGRAPHY_FILE):
        HOMOGRAPHY_MATRIX = np.load(HOMOGRAPHY_FILE)
    else:
        sys.exit(f"Error: Missing {HOMOGRAPHY_FILE}. Run your calibration first.")

    connect_mega()
    connect_odrive()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow("Click to Move")
    cv2.setMouseCallback("Click to Move", on_mouse_click)

    print("\n--- INSTRUCTIONS ---")
    print("1. Ensure hardware is powered and homed.")
    print("2. Left-click anywhere on the video feed.")
    print("3. Arm will move Base, Shoulder, and Elbow to the target.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Apply calibration warp
        warped = cv2.warpPerspective(frame, HOMOGRAPHY_MATRIX, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # UI overlays
        color = (0, 255, 0)
        if mega_status["mode"] == 2:
            cv2.putText(warped, "CALIBRATING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            color = (0, 0, 255)

        if target_mm:
            # Draw crosshair using the inverse matrix so it tracks visually
            try:
                M_inv = np.linalg.inv(HOMOGRAPHY_MATRIX)
                pt_mm = np.array([[[float(target_mm[0]), float(target_mm[1])]]], dtype=np.float32)
                pixel_pt = cv2.perspectiveTransform(pt_mm, M_inv)
                u_draw, v_draw = int(pixel_pt[0][0][0]), int(pixel_pt[0][0][1])
                cv2.drawMarker(warped, (u_draw, v_draw), color, cv2.MARKER_CROSS, 20, 2)
            except:
                pass

        cv2.imshow("Click to Move", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if _ser: _ser.close()
    cap.release()
    cv2.destroyAllWindows()
    if shoulder_axis:
        shoulder_axis.requested_state = AxisState.IDLE
