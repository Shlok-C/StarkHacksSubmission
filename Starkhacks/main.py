import threading
import queue
import time
import numpy as np
import cv2
import enum
import sys

from ik_solver import move_to
from serial_bridge import connect, send_angles, home # fix to actual code
from gesture_recognizer import NPUGestureRecognizer, GestureResult

# Shared state
LATEST_RESULT = None
LATEST_FRAME  = None
ARM_BUSY      = False # true while moving
HOMOGRAPHY_MATRIX = None # loaded at startup

# Thread locks
frame_lock = threading.Lock()
result_lock = threading.Lock()
busy_lock = threading.Lock()

command_queue = queue.Queue()

# Define enum states, not sure 
class State(enum.Enum):
    SLEEP = "sleep"
    READY = "ready"
    FOLLOW = "follow"
    HOLD = "hold"
    RECORD = "record"


class ArmController:
    def __init__(self):
        self.state = State.SLEEP
        self.running = False
    
    def transition(self, new_state):
        old = self.state
        self.state = new_state
        print(f"State: {old.value} → {new_state.value}")

    def is_in(self, state):
        return self.state == state
    
    def start(self):
        self.running = True
        self.transition(State.READY)
    
    def stop(self):
        self.running = False
        self.transition(State.SLEEP)

controller = ArmController()  # instantiate class -> everything else calls to this

def startup(): #runs before any threads
    global HOMOGRAPHY_MATRIX

    try:
        HOMOGRAPHY_MATRIX = np.loap("homography_matrix.npy")
        print("Loaded homography matrix")
    except FileNotFoundError:
        print("No homography matrix found")
        sys.exit(1)

    try:
        connect()
        print("Connected to serial port")
    except Exception as e:
        print(f"Failed to connect to serial port: {e}")
        sys.exit(1)

    controller.start()
    print("LFG!")


def pixel_to_mm(u,v):
    if HOMOGRAPHY_MATRIX is None:
        return None
    
    pt = np.array([[[float(u),float(v)]]],dtype=np.float32)
    result = cv2.perspectiveTransform(pt,HOMOGRAPHY_MATRIX)
    x_mm, y_mm = result[0][0]
    return x_mm, y_mm


def capture_loop(cap):  # capture and store latest frame, no processing
    global LATEST_FRAME
    while controller.running:
        ok, frame = cap.read()
        if ok:
            with frame_lock:
                LATEST_FRAME = frame  # ovewrite with newest frame


def inference_loop(recognizer):
    global LATEST_RESULT
    while controller.running:
        # use copy of latest frame
        with frame_lock:
            if LATEST_FRAME is None:
                frame = None
            else:
                frame = LATEST_FRAME.copy()


        if frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # run NPU gesture recognition (SHLOK)
        result = recognizer.recognize(rgb)

        # store results for other threads
        with result_lock:
            LATEST_RESULT = result

        # if a clear movement command is detected, then push to dispatcher
        # only push if not already busy
        if result.command =="MOVE_DIRECTED" and not ARM_BUSY:
            try:
                command_queue.put_nowait("MOVE_DIRECTED", result)
            except queue.Full:
                pass


# Executor, its own thread runs the full pipeline pixel → mm → IK → serial.
def execute_move(result):
    global ARM_BUSY
 
    # Mark arm as busy so inference_loop stops flooding commands
    with busy_lock:
        ARM_BUSY = True
 
    try:
        if result.landmarks is None:
            print("No landmarks in result, skipping")
            return
 
        # Get the index finger tip pixel (landmark 8)
        # landmarks are normalized [0,1], multiply by frame size to get pixels
        lm  = result.landmarks[8]
        u   = int(lm.x * 640)
        v   = int(lm.y * 480)
 
        # Convert pixel to real-world mm using homography
        coords = pixel_to_mm(u, v)
        if coords is None:
            print("Homography not loaded, cannot convert pixel to mm")
            return
 
        x_mm, y_mm = coords
        print(f"Target pixel ({u}, {v}) → ({x_mm:.1f} mm, {y_mm:.1f} mm)")
 
        # Run IK solver, returns angles or None if unreachable
        angles = move_to(x_mm, y_mm)
        if angles is None:
            print("Target unreachable, no command sent")
            return
 
        th_b, th_s, th_e, th_w = angles
 
        # Send angles to Arduino over serial, IMPLEMENT THIS!!
        send_angles(th_b, th_s, th_e, th_w)
        print(f"Angles sent, waiting for DONE")
 
    finally:
        # Always release busy flag, even if something went wrong above, prevents locking up
        with busy_lock:
            ARM_BUSY = False
 

# FOLLOW Mode
# Every 100ms, reads the latest hand position and sends a move command.
def follow_loop():
    print("Follow mode started")
    last_x, last_y = None, None
    MOVE_THRESHOLD_MM = 10   # only move if target shifted more than this, NEED TUNING
 
    while controller.is_in(State.FOLLOW):
 
        with result_lock:
            result = LATEST_RESULT
 
        # Only act on pointing gestures in follow mode
        if result is None or result.command != "MOVE_DIRECTED" or result.landmarks is None:
            time.sleep(0.1)
            continue
 
        lm = result.landmarks[8]
        u  = int(lm.x * 640)
        v  = int(lm.y * 480)
 
        coords = pixel_to_mm(u, v)
        if coords is None:
            time.sleep(0.1)
            continue
 
        x_mm, y_mm = coords
 
        # Only send a new command if the hand has moved enough, avoids jitter
        if last_x is not None:
            dist = ((x_mm - last_x)**2 + (y_mm - last_y)**2) ** 0.5
            if dist < MOVE_THRESHOLD_MM:
                time.sleep(0.1)
                continue
 
        last_x, last_y = x_mm, y_mm
 
        # Move the arm, reuse execute_move but only if not already busy
        if not ARM_BUSY:
            threading.Thread(target=execute_move, args=(result,), daemon=True).start()
 
        time.sleep(0.1)
 
    print("Follow mode ended")
 
 

# Command Dispatcher
# Decides what to do based on current state and incoming command, transitions happen here
def dispatcher_loop():
    print("Dispatcher running")
 
    while controller.running:
 
        try:
            # Block for up to 0.5s waiting for a command
            # Timeout lets the loop check controller.running periodically
            cmd, payload = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue
 
        print(f"Command received: {cmd}")
 
        if cmd == "MOVE_DIRECTED":
            if controller.is_in(State.READY):
                # Run in a separate thread so dispatcher stays responsive
                threading.Thread(target=execute_move, args=(payload,), daemon=True).start()
            else:
                print(f"Ignoring MOVE, state is {controller.state.value}")
 
        elif cmd == "FOLLOW":
            if controller.is_in(State.READY):
                controller.transition(State.FOLLOW)
                # follow_loop runs in its own thread and stops when state changes
                threading.Thread(target=follow_loop, daemon=True).start()
 
        elif cmd == "STOP_FOLLOW":
            if controller.is_in(State.FOLLOW):
                controller.transition(State.READY)
 
        elif cmd == "HOLD":
            # Freeze the arm in place, stop accepting move commands
            if not controller.is_in(State.SLEEP):
                controller.transition(State.HOLD)
 
        elif cmd == "RELEASE":
            if controller.is_in(State.HOLD):
                controller.transition(State.READY)
 
        elif cmd == "RECORD":
            if controller.is_in(State.READY):
                controller.transition(State.RECORD)
                print("Recording started, implement record_loop here")
                # code here! lol
 
        elif cmd == "STOP_RECORD":
            if controller.is_in(State.RECORD):
                controller.transition(State.READY)
                print("Recording stopped")
 
        elif cmd == "SLEEP":
            controller.stop()
 
        elif cmd == "WAKE":
            if controller.is_in(State.SLEEP):
                controller.start()
 
        elif cmd == "HOME":
            print("Sending home command to Arduino")
            try:
                home()
            except Exception as e:
                print(f"Failed: {e}")
 
        else:
            print(f"Unknown command: {cmd}")
 
 
# overlays on camera feed
def display_loop():
    while controller.running:
 
        with frame_lock:
            if LATEST_FRAME is None:
                frame = None
            else:
                LATEST_FRAME.copy()
 
        if frame is None:
            time.sleep(0.03)
            continue
 
        frame = cv2.flip(frame, 1)
 
        # State overlay, top left
        state_color = {
            State.SLEEP:  (100, 100, 100),
            State.READY:  (0,   200,  0),
            State.FOLLOW: (0,   200, 255),
            State.HOLD:   (0,   120, 255),
            State.RECORD: (0,     0, 255),
        }.get(controller.state, (255, 255, 255))
 
        cv2.rectangle(frame, (0, 0), (200, 36), (0, 0, 0), -1)
        cv2.putText(frame, f"STATE: {controller.state.value.upper()}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
 
        # Arm busy indicator
        if ARM_BUSY:
            cv2.putText(frame, "MOVING", (frame.shape[1] - 100, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)
 
        # Latest gesture
        with result_lock:
            result = LATEST_RESULT
 
        if result is not None and result.gesture_name != "---":
            cv2.putText(frame, result.gesture_name,
                        (8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
 
        cv2.imshow("Arm Controller", frame)
 
        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            command_queue.put(("SLEEP", None))
            break
 

if __name__ == "__main__":
    print("ARM CONTROLLER FOR ARIEL, STARKHACKS 2026")
 
    # 1. Load matrix, connect hardware, transition to READY
    startup()
 
    # 2. Open camera
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Camera not found")
        sys.exit(1)
 
    # 3. Initialize gesture recognizer
    recognizer = NPUGestureRecognizer("models/old/mediapipe_hand_gesture-canned_gesture_classifier-w8a8.tflite")
 
    # 4. Send arm to home position
    print("Homing arm...")
    command_queue.put(("HOME", None))
 
    # 5. Launch background threads
    threading.Thread(target=capture_loop,    args=(cap,),        daemon=True).start()
    threading.Thread(target=inference_loop,  args=(recognizer,), daemon=True).start()
    threading.Thread(target=dispatcher_loop,                     daemon=True).start()
 
    print("All threads started")
    print("Press Q in the camera window to quit\n")
 
    # 6. Display loop runs on main thread (OpenCV needs this)
    try:
        display_loop()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        # Clean shutdown, stop state machine, release camera
        controller.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete")

