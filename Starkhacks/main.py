import threading
import queue
import time
import numpy as np
import cv2
import enum
import sys
import os
import serial
import serial.tools.list_ports
import lcm

# new_ik.py lives two directories above this file (project root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from StarkHacksSubmission.new_ik import move_to
from cv_lib.gesture_recognizer import NPUGestureRecognizer, GestureResult
from cv_lib.voice_recognition import listen_transcripts, take_picture
from mytypes.arm_angles import arm_angles as ArmAngles

# Shared state
LATEST_RESULT     = None
LATEST_FRAME      = None
ARM_BUSY          = False
AI_BUSY           = False
HOMOGRAPHY_MATRIX = None
FRAME_W           = 640   # updated from first real frame
FRAME_H           = 480   # CALIBRATE: match actual camera resolution
assistant         = None

# Thread locks
frame_lock    = threading.Lock()
result_lock   = threading.Lock()
busy_lock     = threading.Lock()
ai_busy_lock  = threading.Lock()

command_queue = queue.Queue()

# Serial connection to Arduino Mega (set in connect())
_ser = None

# LCM publisher (shared — LCM is thread-safe for publish)
lc = lcm.LCM()


# ── Serial helpers ─────────────────────────────────────────────────────────────

def _find_mega():
    """Auto-detect Arduino Mega by USB VID/PID."""
    for port in serial.tools.list_ports.comports():
        vid, pid = port.vid, port.pid
        print(f"  {port.device} | VID:{vid:#06x} PID:{pid:#06x} | {port.description}")
        if vid == 0x2341 and pid in (0x0042, 0x0010):   # official Mega
            print(f"  → official Mega")
            return port.device
        if vid in (0x1A86, 0x10C4):                      # CH340 / CP210x clone
            print(f"  → clone Mega (CH340/CP210x)")
            return port.device
    return None


def connect():
    global _ser
    device = _find_mega()
    if device is None:
        raise RuntimeError("No Arduino Mega found on any USB port")
    _ser = serial.Serial(device, 115200, timeout=1)
    time.sleep(2)                    # wait for Arduino bootloader reset
    _ser.reset_input_buffer()
    print(f"[serial] connected: {device}")


def send_angles(th_b, th_e):
    """Send base and elbow angles to Arduino Mega.

    Protocol (ASCII, newline-terminated):
        A,<base_deg>,<elbow_deg>\\n

    Arduino drives:
      TMC2208 stepper → base  (th_b)
      L298N stepper   → elbow (th_e)
    Shoulder → ODrive via LCM only, not here.
    # WRIST: pass th_w here and append to line once wrist hardware is ready.
    """
    if _ser is None:
        print("[serial] not connected — skipping send_angles")
        return
    line = f"A,{th_b:.2f},{th_e:.2f}\n"
    _ser.write(line.encode())


def home():
    """Tell Arduino to execute homing sequence. Protocol: 'H\\n'"""
    if _ser is None:
        print("[serial] not connected — skipping home")
        return
    _ser.write(b"H\n")


# ── LCM publisher ──────────────────────────────────────────────────────────────

def _publish_lcm(th_b, th_s, th_e):
    """Publish angles on TARGET_ANGLE so odrive_control.py drives the shoulder."""
    msg          = ArmAngles()
    msg.base     = float(th_b)
    msg.shoulder = float(th_s)
    msg.elbow    = float(th_e)
    lc.publish("TARGET_ANGLE", msg.encode())


# ── State machine ──────────────────────────────────────────────────────────────

class State(enum.Enum):
    SLEEP        = "sleep"
    READY        = "ready"
    FOLLOW       = "follow"
    HOLD         = "hold"
    RECORD       = "record"
    CONVERSATION = "conversation"


class ArmController:
    def __init__(self):
        self.state   = State.SLEEP
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


controller = ArmController()


# ── Voice routing ──────────────────────────────────────────────────────────────
# Order matters: longer phrases first so "stop following" wins over "stop".
VOICE_CMD_MAP = [
    (("stop following", "stop follow"),             "STOP_FOLLOW",  None),
    (("follow me", "follow", "track my finger"),    "FOLLOW",       None),
    (("stop recording", "end recording"),           "STOP_RECORD",  None),
    (("start recording", "record"),                 "RECORD",       None),
    (("hold", "freeze", "stop"),                    "HOLD",         None),
    (("release", "resume", "continue", "unfreeze"), "RELEASE",      None),
    (("go home", "home position", "home"),          "HOME",         None),
    (("wake up", "wake"),                           "WAKE",         None),
    (("go to sleep", "sleep"),                      "SLEEP",        None),
    (("take picture", "cheese", "snap"),            "TAKE_PICTURE", None),
]

WAKE_PHRASES = ("hey ariel", "you're crazy")
EXIT_PHRASES = ("done", "stop talking", "thanks ariel", "goodbye", "shut up", "i'ma kill you")


def _run_ai(question: str):
    """Spawned per ASK_AI. Multi-turn WorkbenchAssistant retains history."""
    global AI_BUSY, assistant
    with ai_busy_lock:
        AI_BUSY = True
    try:
        if assistant is None:
            from cv_lib import WorkbenchAssistant
            assistant = WorkbenchAssistant()
        print(f"[ai] asking: {question}")
        assistant.ask(question)
    except Exception as e:
        print(f"[ai] failed: {e}")
    finally:
        with ai_busy_lock:
            AI_BUSY = False


def _match_voice_command(text: str):
    for phrases, cmd, payload in VOICE_CMD_MAP:
        if any(p in text for p in phrases):
            return cmd, payload
    return None


def _dispatch_or_take_picture(cmd: str, payload):
    if cmd == "TAKE_PICTURE":
        threading.Thread(target=take_picture, daemon=True).start()
    else:
        command_queue.put((cmd, payload))


def _voice_cmd_on_text(text: str) -> bool:
    hit = _match_voice_command(text)
    if hit is None:
        return False
    cmd, payload = hit
    print(f"[voice→dispatch] '{text}' → {cmd}")
    _dispatch_or_take_picture(cmd, payload)
    return True


def _voice_ai_on_text(text: str) -> bool:
    if controller.is_in(State.CONVERSATION) and any(p in text for p in EXIT_PHRASES):
        print(f"[voice→ai] exit: '{text}'")
        command_queue.put(("END_CONVERSATION", None))
        return True

    wake_hit = next((p for p in WAKE_PHRASES if p in text), None)
    if wake_hit:
        if not controller.is_in(State.CONVERSATION):
            command_queue.put(("START_CONVERSATION", None))
        remainder = text.split(wake_hit, 1)[1].strip(" ,.?!")
        if remainder:
            command_queue.put(("ASK_AI", remainder))
        print(f"[voice→ai] wake: '{text}' remainder='{remainder}'")
        return True

    if controller.is_in(State.CONVERSATION):
        hit = _match_voice_command(text)
        if hit is not None:
            cmd, payload = hit
            print(f"[voice→ai/cmd] '{text}' → {cmd}")
            _dispatch_or_take_picture(cmd, payload)
            return True
        print(f"[voice→ai] follow-up: '{text}'")
        command_queue.put(("ASK_AI", text))
        return True

    return False


def _voice_on_text(text: str):
    if _voice_ai_on_text(text):
        return
    if _voice_cmd_on_text(text):
        return
    print(f"[voice] ignored: '{text}'")


# ── Startup ────────────────────────────────────────────────────────────────────

def startup():
    global HOMOGRAPHY_MATRIX
    try:
        HOMOGRAPHY_MATRIX = np.load("homography_matrix.npy")
        print("Loaded homography matrix")
    except FileNotFoundError:
        print("No homography matrix found — run calibration first")
        sys.exit(1)

    try:
        connect()
    except Exception as e:
        print(f"Failed to connect to serial port: {e}")
        sys.exit(1)

    controller.start()
    print("LFG!")


# ── Pixel → mm ─────────────────────────────────────────────────────────────────

def pixel_to_mm(u, v):
    if HOMOGRAPHY_MATRIX is None:
        return None
    pt     = np.array([[[float(u), float(v)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, HOMOGRAPHY_MATRIX)
    x_mm, y_mm = result[0][0]
    return x_mm, y_mm


# ── Camera threads ─────────────────────────────────────────────────────────────

def capture_loop(cap):
    global LATEST_FRAME, FRAME_W, FRAME_H
    while controller.running:
        ok, frame = cap.read()
        if ok:
            with frame_lock:
                LATEST_FRAME = frame
                FRAME_H, FRAME_W = frame.shape[:2]


def inference_loop(recognizer):
    global LATEST_RESULT
    while controller.running:
        with frame_lock:
            frame = LATEST_FRAME.copy() if LATEST_FRAME is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = recognizer.process(rgb)

        with result_lock:
            LATEST_RESULT = result

        if result.command == "MOVE_DIRECTED":
            with busy_lock:
                arm_busy = ARM_BUSY
            if not arm_busy:
                try:
                    command_queue.put_nowait(("MOVE_DIRECTED", result))
                except queue.Full:
                    pass


# ── Move execution: pixel → mm → IK → LCM + serial ───────────────────────────

def execute_move(result):
    global ARM_BUSY

    with busy_lock:
        ARM_BUSY = True

    try:
        if result.landmarks is None:
            print("[move] no landmarks, skipping")
            return

        # Index finger tip = landmark 8, normalized [0,1] → actual pixels
        lm = result.landmarks[8]
        with frame_lock:
            w, h = FRAME_W, FRAME_H
        u = int(lm.x * w)
        v = int(lm.y * h)

        coords = pixel_to_mm(u, v)
        if coords is None:
            print("[move] homography not loaded")
            return

        x_mm, y_mm = coords
        print(f"[move] pixel ({u},{v}) → ({x_mm:.1f} mm, {y_mm:.1f} mm)")

        # CALIBRATE: Z is fixed at Z_HOVER defined in arm_lib/ik.py
        angles = move_to(x_mm, y_mm)
        if angles is None:
            print("[move] target unreachable")
            return

        th_b, th_s, th_e, th_w = angles  # th_w kept for when wrist is ready
        print(f"[move] base={th_b:.1f}° shoulder={th_s:.1f}° elbow={th_e:.1f}°"
              # f" wrist={th_w:.1f}°"  # uncomment when wrist hardware is active
              )

        # 1. Publish shoulder angle to ODrive via LCM (base/elbow included for logging)
        _publish_lcm(th_b, th_s, th_e)

        # 2. Send base + elbow to Arduino Mega; shoulder → ODrive only
        send_angles(th_b, th_e)
        # send_angles(th_b, th_e, th_w)  # uncomment when wrist hardware is active

    finally:
        with busy_lock:
            ARM_BUSY = False


# ── Follow mode ────────────────────────────────────────────────────────────────

# CALIBRATE: minimum mm movement before sending a new command; tune to balance
# responsiveness vs. jitter from hand tremor. Start high (15–20) then lower.
MOVE_THRESHOLD_MM = 10

def follow_loop():
    print("Follow mode started")
    last_x, last_y = None, None

    while controller.is_in(State.FOLLOW):
        with result_lock:
            result = LATEST_RESULT

        if result is None or result.command != "MOVE_DIRECTED" or result.landmarks is None:
            time.sleep(0.1)
            continue

        lm = result.landmarks[8]
        with frame_lock:
            w, h = FRAME_W, FRAME_H
        u = int(lm.x * w)
        v = int(lm.y * h)

        coords = pixel_to_mm(u, v)
        if coords is None:
            time.sleep(0.1)
            continue

        x_mm, y_mm = coords

        if last_x is not None:
            dist = ((x_mm - last_x)**2 + (y_mm - last_y)**2) ** 0.5
            if dist < MOVE_THRESHOLD_MM:
                time.sleep(0.1)
                continue

        last_x, last_y = x_mm, y_mm

        with busy_lock:
            arm_busy = ARM_BUSY
        if not arm_busy:
            threading.Thread(target=execute_move, args=(result,), daemon=True).start()

        time.sleep(0.1)

    print("Follow mode ended")


# ── Command dispatcher ─────────────────────────────────────────────────────────

def dispatcher_loop():
    print("Dispatcher running")

    while controller.running:
        try:
            cmd, payload = command_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        print(f"[dispatch] {cmd}")

        if cmd == "MOVE_DIRECTED":
            if controller.is_in(State.READY):
                threading.Thread(target=execute_move, args=(payload,), daemon=True).start()
            else:
                print(f"  ignoring MOVE — state is {controller.state.value}")

        elif cmd == "FOLLOW":
            if controller.is_in(State.READY):
                controller.transition(State.FOLLOW)
                threading.Thread(target=follow_loop, daemon=True).start()

        elif cmd == "STOP_FOLLOW":
            if controller.is_in(State.FOLLOW):
                controller.transition(State.READY)

        elif cmd == "HOLD":
            if not controller.is_in(State.SLEEP):
                controller.transition(State.HOLD)

        elif cmd == "RELEASE":
            if controller.is_in(State.HOLD):
                controller.transition(State.READY)

        elif cmd == "RECORD":
            if controller.is_in(State.READY):
                controller.transition(State.RECORD)
                print("  record_loop not yet implemented — TODO")
                # TODO: launch record_loop thread here

        elif cmd == "STOP_RECORD":
            if controller.is_in(State.RECORD):
                controller.transition(State.READY)
                print("  recording stopped")

        elif cmd == "SLEEP":
            controller.stop()

        elif cmd == "WAKE":
            if controller.is_in(State.SLEEP):
                controller.start()

        elif cmd == "HOME":
            print("  homing...")
            try:
                home()
            except Exception as e:
                print(f"  home failed: {e}")

        elif cmd == "START_CONVERSATION":
            if controller.is_in(State.READY):
                controller.transition(State.CONVERSATION)
            else:
                print(f"  ignoring START_CONVERSATION — state is {controller.state.value}")

        elif cmd == "END_CONVERSATION":
            if controller.is_in(State.CONVERSATION):
                controller.transition(State.READY)

        elif cmd == "ASK_AI":
            if not payload:
                print("  [ai] empty payload, skipping")
            else:
                with ai_busy_lock:
                    busy = AI_BUSY
                if busy:
                    print(f"  [ai] busy, dropping '{payload}'")
                else:
                    threading.Thread(target=_run_ai, args=(payload,), daemon=True).start()

        elif cmd == "RESET_AI":
            if assistant is not None:
                assistant.reset()
                print("  [ai] conversation memory cleared")

        else:
            print(f"  unknown command: {cmd}")


# ── Display (must run on main thread) ─────────────────────────────────────────

def display_loop():
    while controller.running:
        with frame_lock:
            frame = LATEST_FRAME.copy() if LATEST_FRAME is not None else None

        if frame is None:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)

        state_color = {
            State.SLEEP:        (100, 100, 100),
            State.READY:        (0,   200,   0),
            State.FOLLOW:       (0,   200, 255),
            State.HOLD:         (0,   120, 255),
            State.RECORD:       (0,     0, 255),
            State.CONVERSATION: (255, 120, 200),
        }.get(controller.state, (255, 255, 255))

        cv2.rectangle(frame, (0, 0), (200, 36), (0, 0, 0), -1)
        cv2.putText(frame, f"STATE: {controller.state.value.upper()}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)

        with busy_lock:
            arm_busy = ARM_BUSY
        if arm_busy:
            cv2.putText(frame, "MOVING", (frame.shape[1] - 100, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2)

        with result_lock:
            result = LATEST_RESULT

        if result is not None and result.gesture_name != "---":
            cv2.putText(frame, result.gesture_name,
                        (8, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Arm Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            controller.stop()   # signals capture_loop + inference_loop to exit too
            break


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("ARM CONTROLLER FOR ARIEL, STARKHACKS 2026")

    startup()

    # CALIBRATE: camera index — try 0 or 1 if this doesn't open
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Camera not found at index 2")
        sys.exit(1)

    recognizer = NPUGestureRecognizer(
        "models/old/mediapipe_hand_gesture-canned_gesture_classifier-w8a8.tflite"
    )

    print("Homing arm...")
    command_queue.put(("HOME", None))

    threading.Thread(target=capture_loop,    args=(cap,),        daemon=True).start()
    threading.Thread(target=inference_loop,  args=(recognizer,), daemon=True).start()
    threading.Thread(target=dispatcher_loop,                     daemon=True).start()
    threading.Thread(
        target=listen_transcripts,
        args=(_voice_on_text, lambda: not controller.running),
        daemon=True,
    ).start()

    print("All threads started")
    print("Press Q in the camera window to quit\n")

    try:
        display_loop()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt")
        controller.stop()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete")
