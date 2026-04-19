import math

# ── ARM PHYSICAL DIMENSIONS ──────────────────────────────────────────────────
L1 = 13.612 * 25.4  # 345.7 mm  lower arm (shoulder to elbow)
L2 = 11.130 * 25.4  # 282.7 mm  upper arm (elbow to wrist)
L3 = 4.000  * 25.4  # 101.6 mm  end effector (wrist to tool tip)

# ── GEOMETRY ─────────────────────────────────────────────────────────────────
SHOULDER_HEIGHT = 6.105 * 25.4  # 155.1 mm — shoulder joint above table surface
TABLE_Z = -SHOULDER_HEIGHT       # -155.1 mm — table surface in IK coordinates
                                 # Z=0 means end effector at shoulder height
                                 # Z=-155 means end effector on the table

# ── WORKSPACE OFFSET (measure with ruler before testing) ─────────────────────
SHOULDER_X = 0  # mm from rectangle (0,0) corner to shoulder joint in X
SHOULDER_Y = 0  # mm from rectangle (0,0) corner to shoulder joint in Y

# ── SAFE OPERATING PARAMETERS ────────────────────────────────────────────────
Z_HOVER = 12 * 25.4   # 304.8 mm — default hover height ABOVE TABLE
                       # IK z for hover = TABLE_Z + Z_HOVER = 149.7 mm
Z_MIN   = 4  * 25.4   # 101.6 mm — minimum hover height above table (safety floor)
                       # IK z for minimum = TABLE_Z + Z_MIN = -53.5 mm


# ── HELPERS ───────────────────────────────────────────────────────────────────

def table_to_ik_z(mm_above_table):
    """Convert a height above the table into IK Z (relative to shoulder)."""
    return TABLE_Z + mm_above_table


def correction(x_raw, y_raw):
    """Shift homography coordinates to be relative to the shoulder joint."""
    return x_raw - SHOULDER_X, y_raw - SHOULDER_Y


def base_angle(x, y):
    """θ_base in degrees — how far the base rotates to face the target."""
    return math.degrees(math.atan2(y, x))


# ── REACHABILITY CHECK ────────────────────────────────────────────────────────

def is_reachable(x, y, z):
    """
    Returns True if (x, y, z) is within the arm's reachable workspace.
    x, y must already be shoulder-corrected.
    z is IK Z (relative to shoulder, not table).
    """
    r       = math.sqrt(x**2 + y**2)
    r_wrist = r - L3

    if r_wrist < 0:
        return False   # target closer than end effector length

    d = math.sqrt(r_wrist**2 + z**2)

    if d > (L1 + L2):
        return False   # too far

    if d < abs(L1 - L2):
        return False   # too close (inside minimum reach)

    return True


# ── IK SOLVER ────────────────────────────────────────────────────────────────

def solve(x, y, z):
    """
    Compute all four joint angles for a given target.
    x, y : shoulder-corrected horizontal position (mm)
    z    : IK Z, height relative to shoulder (mm). Use table_to_ik_z() to convert.

    Returns: (theta_base, theta_shoulder, theta_elbow, theta_wrist) all in degrees.
    """
    r       = math.sqrt(x**2 + y**2)
    r_wrist = r - L3
    d       = math.sqrt(r_wrist**2 + z**2)

    theta_base = base_angle(x, y)

    # Clamp acos inputs to [-1, 1] to prevent crash on floating point edge cases
    cos_elbow = max(-1.0, min(1.0, (L1**2 + L2**2 - d**2) / (2 * L1 * L2)))
    theta_elbow = math.degrees(math.acos(cos_elbow))

    beta      = math.atan2(z, r_wrist)
    cos_alpha = max(-1.0, min(1.0, (L1**2 + d**2 - L2**2) / (2 * L1 * d)))
    alpha     = math.acos(cos_alpha)

    theta_shoulder = math.degrees(alpha + beta)
    theta_wrist    = (theta_elbow + theta_shoulder) * -1

    return theta_base, theta_shoulder, theta_elbow, theta_wrist


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def move_to(x_raw, y_raw, z_above_table=Z_HOVER):
    """
    Full pipeline: raw homography coordinates → joint angles.

    x_raw, y_raw   : mm coordinates from homography (relative to rectangle corner)
    z_above_table  : how high above the TABLE the end effector should hover (mm)
                     defaults to Z_HOVER (304.8 mm)

    Returns: (theta_base, theta_shoulder, theta_elbow, theta_wrist) in degrees,
             or None if target is unreachable.
    """
    x, y = correction(x_raw, y_raw)
    z    = table_to_ik_z(z_above_table)

    if not is_reachable(x, y, z):
        print(f"  [IK] NOT REACHABLE  x={x:.1f}  y={y:.1f}  z={z:.1f}")
        return None

    th_b, th_s, th_e, th_w = solve(x, y, z)

    print(f"  [IK] Base: {th_b:7.2f}°  Shoulder: {th_s:7.2f}°  "
          f"Elbow: {th_e:7.2f}°  Wrist: {th_w:7.2f}°")
    return th_b, th_s, th_e, th_w


# ── LOW-LEVEL TEST BLOCK ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("IK SOLVER — LOW LEVEL TESTS")
    print(f"  L1={L1:.1f}mm  L2={L2:.1f}mm  L3={L3:.1f}mm")
    print(f"  Shoulder height: {SHOULDER_HEIGHT:.1f}mm above table")
    print(f"  TABLE_Z = {TABLE_Z:.1f}mm  (Z=0 is shoulder level)")
    print(f"  Z_HOVER = {Z_HOVER:.1f}mm above table → IK z = {table_to_ik_z(Z_HOVER):.1f}mm")
    print("=" * 60)

    tests = [
        # (label,            x_raw, y_raw, z_above_table)
        ("straight ahead",   200,   0,     Z_HOVER),
        ("right side",       0,     200,   Z_HOVER),
        ("diagonal",         200,   200,   Z_HOVER),
        ("close to arm",     120,   0,     Z_HOVER),
        ("low hover",        200,   0,     Z_MIN),
        ("too far",          700,   0,     Z_HOVER),
        ("too close",        10,    0,     Z_HOVER),
    ]

    for label, xr, yr, zh in tests:
        print(f"\n  [{label}]  raw=({xr}, {yr})  hover={zh:.0f}mm above table")
        move_to(xr, yr, zh)

    print("\n" + "=" * 60)
    print("If all reachable tests printed angles, the solver is working.")
    print("Next step: connect Arduino and send angles over serial.")