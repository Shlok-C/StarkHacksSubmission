import serial
import time
import odrive
from odrive.enums import AxisState, ControlMode, InputMode
from odrive.utils import dump_errors, request_state

JOG = 0
GET_HOME = 1
SPEED = 0.05

# -------------------------
# Serial setup (Arduino)
# -------------------------
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

# -------------------------
# Connect to ODrive
# -------------------------
print("Looking for ODrive...")
odrv = odrive.find_any(timeout=10)
if odrv is None:
    raise RuntimeError("No ODrive found.")

print(f"Connected | Serial: {odrv.serial_number} | Bus voltage: {odrv.vbus_voltage:.2f} V")

axis = odrv.axis0
axis.clear_errors()
time.sleep(0.2)

# -------------------------
# Calibration
# -------------------------
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

# -------------------------
# Decode new 7-value packet:
# mode, enc1, enc2, enc3, lim1, lim2, lim3
# -------------------------
def decode_msg(line):
    parts = line.split(',')
    if len(parts) != 7:
        return None

    try:
        mode = int(parts[0])
        val1 = float(parts[1])
        val2 = float(parts[2])   # encoder 2 value
        val3 = float(parts[3])
        base_lim = int(parts[4])
        shoulder_lim = int(parts[5])
        elbow_lim = int(parts[6])

        return mode, val1, val2, val3, base_lim, shoulder_lim, elbow_lim

    except ValueError:
        return None

# -------------------------
# Read only the newest serial message
# This prevents us from reacting to stale buffered packets
# -------------------------
def read_latest_message():
    latest = None

    # Read at least one line
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if line:
        latest = line

    # Drain anything else already waiting and keep only the newest line
    while ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            latest = line

    return latest



# -------------------------
# Jog tuning
# -------------------------
scale = 10.0   # higher = less motion per encoder tick

# -------------------------
# Jog state
# -------------------------
jog_initialized = False
val2_start = 0.0
motor_start_pos = 0.0

def reset_jog():
    global jog_initialized
    jog_initialized = False

def jog(val2):
    global jog_initialized, val2_start, motor_start_pos

    # First time entering jog mode:
    # capture the current encoder input as zero reference
    # and capture the current motor position
    if not jog_initialized:
        val2_start = val2
        motor_start_pos = axis.encoder.pos_estimate
        jog_initialized = True

        print(f"Initial val2 = {val2_start}")
        print(f"Motor start pos = {motor_start_pos:.3f}")
        print("Jog control active...\n")

    # Make the first encoder value act like zero
    relative_input = val2 - val2_start

    # Convert encoder movement into motor position
    target_pos = motor_start_pos + (relative_input / scale)

    # Send position command to ODrive
    axis.controller.input_pos = target_pos

    print(
        f"val2={val2:.2f} | "
        f"rel={relative_input:.2f} | "
        f"target={target_pos:.3f} | "
        f"pos={axis.encoder.pos_estimate:.3f}"
    )

# =========================
# HOMING
# =========================
def get_home():
    print("Starting homing...")

    ser.write(b'2\n')

    # wait for confirmation
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


# -------------------------
# Main
# -------------------------
calibrate()

# Calibration ends in IDLE, so re-enter closed loop here
axis.controller.config.control_mode = ControlMode.POSITION_CONTROL
axis.controller.config.input_mode = InputMode.PASSTHROUGH
request_state(axis, AxisState.CLOSED_LOOP_CONTROL)
time.sleep(1)

print("Listening for data from Arduino...\n")

last_mode = None

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue

        decoded = decode_msg(line)
        if decoded is None:
            continue

        mode, val1, val2, val3, base_lim, shoulder_lim, elbow_lim = decoded

        # If mode changed, handle the transition
        if mode != last_mode:
            print(f"Mode changed: {last_mode} -> {mode}")

            # If we are entering jog mode, re-zero the jog reference
            if mode == JOG:
                reset_jog()

            # If we are entering home mode, run homing ONCE
            elif mode == GET_HOME:
                print("going home")
                get_home()

            last_mode = mode

        # While mode is 0, keep jogging continuously
        if mode == JOG:
            print("still in jog mode")
            jog(val2)

        if axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
            print("Axis exited closed loop!")
            dump_errors(odrv)
            break

    except ValueError:
        print("Error: Could not convert values")
    except KeyboardInterrupt:
        print("\nStopping motor...")
        axis.controller.input_pos = axis.encoder.pos_estimate
        axis.requested_state = AxisState.IDLE
        break
    except Exception as e:
        print(f"Serial error: {e}")
        break
