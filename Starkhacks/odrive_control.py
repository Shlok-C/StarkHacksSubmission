import serial
import serial.tools.list_ports
import time

import lcm

from mytypes import arm_angles

# =========================
# MODES
# =========================
JOG = 0
GET_HOME = 1
POS_MODE = 2

SPEED = 0.3

# =========================
# GEAR SETTINGS
# =========================
GEAR_RATIO = 45.0
HOME_ANGLE_DEG = 90.0

def deg_to_odrive_pos(angle_deg):
    relative_deg = angle_deg - HOME_ANGLE_DEG
    return (relative_deg / 360.0) * GEAR_RATIO

# =========================
# AUTO-DETECT MEGA
# =========================
def find_mega():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        vid = port.vid
        pid = port.pid

        print(f"Checking: {port.device} | {port.description} | VID:{vid} PID:{pid}")

        # Official Arduino Mega
        if vid == 0x2341 and pid in (0x0042, 0x0010):
            print(f"✅ Found OFFICIAL Mega: {port.device}")
            ser = serial.Serial(port.device, 115200, timeout=1)
            time.sleep(2)
            ser.reset_input_buffer()
            return ser

        # Clone Megas (CH340 / CP210x)
        if vid in (0x1A86, 0x10C4):
            print(f"⚠️ Found clone Mega: {port.device}")
            ser = serial.Serial(port.device, 115200, timeout=1)
            time.sleep(2)
            ser.reset_input_buffer()
            return ser

    return None

print("Searching for Arduino Mega...")
ser = find_mega()

if ser is None:
    raise RuntimeError("No Mega found!")

import odrive
from odrive.enums import AxisState, ControlMode, InputMode
from odrive.utils import dump_errors, request_state

# =========================
# LCM SETUP
# =========================
lc = lcm.LCM()
latest_target_deg = None

def handle_lcm(channel, data):
    global latest_target_deg
    msg = arm_angles.decode(data)
    latest_target_deg = msg.shoulder  # arm_angles has base/shoulder/elbow fields

lc.subscribe("TARGET_ANGLE", handle_lcm)

# =========================
# CONNECT ODRIVE
# =========================
print("Looking for ODrive...")
odrv = odrive.find_sync(timeout=10)

if odrv is None:
    raise RuntimeError("No ODrive found.")

print(f"Connected | Serial: {odrv.serial_number}")

axis = odrv.axis0
axis.clear_errors()
time.sleep(0.2)

# =========================
# CALIBRATION
# =========================
def calibrate():
    print("Starting calibration...")
    axis.requested_state = AxisState.FULL_CALIBRATION_SEQUENCE

    t0 = time.time()
    while time.time() - t0 < 30:
        if axis.current_state == AxisState.IDLE:
            break
        time.sleep(0.1)

    print("Calibration done.")
    dump_errors(odrv)

# =========================
# SERIAL DECODER
# =========================
def decode_msg(line):
    parts = line.split(',')
    if len(parts) != 7:
        return None

    try:
        return (
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            int(parts[4]),
            int(parts[5]),
            int(parts[6])
        )
    except:
        return None

# =========================
# SET CONTROL MODE
# =========================
axis.controller.config.control_mode = ControlMode.POSITION_CONTROL
axis.controller.config.input_mode = InputMode.PASSTHROUGH

request_state(axis, AxisState.CLOSED_LOOP_CONTROL)
time.sleep(1)

# =========================
# JOG STATE
# =========================
scale = 10.0

jog_initialized = False
val2_start = 0
motor_start_pos = 0

def reset_jog():
    global jog_initialized
    jog_initialized = False

def jog(val2):
    global jog_initialized, val2_start, motor_start_pos

    if not jog_initialized:
        val2_start = val2
        motor_start_pos = axis.encoder.pos_estimate
        jog_initialized = True

    relative = val2 - val2_start
    target = motor_start_pos + (relative / scale)

    axis.controller.input_pos = target

# =========================
# HOMING
# =========================
def get_home():
    print("Starting homing...")

    ser.write(b'2\n')

    while True:
        line = ser.readline().decode().strip()
        decoded = decode_msg(line)
        if decoded is None:
            continue

        if decoded[0] == 2:
            break

    print("Moving toward limit...")

    axis.controller.config.input_mode = InputMode.PASSTHROUGH

    while True:
        line = ser.readline().decode().strip()
        decoded = decode_msg(line)
        if decoded is None:
            continue

        _, _, _, _, _, shoulder_lim, _ = decoded

        if shoulder_lim == 0:
            print("Limit hit")
            break

        axis.controller.input_pos = axis.encoder.pos_estimate - SPEED

    # Backoff with trajectory
    axis.controller.config.input_mode = InputMode.TRAP_TRAJ

    axis.trap_traj.config.vel_limit = 3
    axis.trap_traj.config.accel_limit = 5

    target = axis.encoder.pos_estimate + 3
    axis.controller.input_pos = target

    while abs(axis.encoder.pos_estimate - target) > 0.05:
        time.sleep(0.01)

    # Set home = 0
    axis.encoder.set_linear_count(0)
    axis.controller.input_pos = 0

    axis.controller.config.input_mode = InputMode.PASSTHROUGH
    reset_jog()

    print("Homing done")

# =========================
# LCM POSITION MODE
# =========================
def move_to_angle(angle_deg):
    target = deg_to_odrive_pos(angle_deg)

    axis.controller.config.input_mode = InputMode.TRAP_TRAJ

    axis.trap_traj.config.vel_limit = 5
    axis.trap_traj.config.accel_limit = 10

    axis.controller.input_pos = target

# =========================
# MAIN LOOP
# =========================
calibrate()

print("Running...\n")

last_mode = None

while True:
    try:
        lc.handle_timeout(0)

        line = ser.readline().decode().strip()
        if not line:
            continue

        decoded = decode_msg(line)
        if decoded is None:
            continue

        mode, val1, val2, val3, base_lim, shoulder_lim, elbow_lim = decoded

        # Mode change
        if mode != last_mode:
            print(f"Mode {last_mode} -> {mode}")

            if mode == JOG:
                reset_jog()

            elif mode == GET_HOME:
                get_home()

            last_mode = mode

        # Jog
        if mode == JOG:
            jog(val2)

        # LCM control
        elif mode == POS_MODE:
            if latest_target_deg is not None:
                move_to_angle(latest_target_deg)
                latest_target_deg = None

    except KeyboardInterrupt:
        print("Stopping")
        axis.controller.input_pos = axis.encoder.pos_estimate
        axis.requested_state = AxisState.IDLE
        break

    except Exception as e:
        print(f"Error: {e}")
        break