import math

# arm physical dimensions
L1 = 13.612 * 25.4  # 345.7 mm
L2 = 11.130 * 25.4  # 282.7 mm
L3 = 4 * 25.4       # 101.6 mm


# workspace offset
SHOULDER_X = 0 # will adjust later
SHOULDER_Y = 0 # will adjust later

# safe operating parameters
Z_HOVER = (12 * 25.4)
Z_MIN = (4 * 25.4)


def correction(x_raw, y_raw):
    x_from_shoulder = x_raw - SHOULDER_X
    y_from_shoulder = y_raw - SHOULDER_Y

    return x_from_shoulder, y_from_shoulder

def base_angle(x, y):
    degrees = math.degrees(math.atan2(y, x))
    return degrees

def is_reachable(x,y,z):
    r = math.sqrt(x**2 + y**2)
    r_wrist = r - L3

    d = math.sqrt(r_wrist**2 + z**2)

    if d > (L1 + L2):
        return False    #too far
    
    if d < abs(L1 - L2):
        return False    #too close

    if r_wrist < 0:
        return False
    
    return True

def solve(x,y,z):
    r = math.sqrt(x**2 + y**2)
    r_wrist = r - L3

    d = math.sqrt(r_wrist**2 + z**2)

    theta_base = base_angle(x,y)
    theta_elbow = math.degrees(math.acos((L1**2 + L2**2 - d**2) / (2 * L1 * L2)))

    beta = math.atan2(z, r_wrist)
    alpha = math.acos((L1**2 + d**2 - L2**2) / (2 * L1 * d))

    theta_shoulder = math.degrees(alpha + beta)
    theta_wrist = (theta_elbow + theta_shoulder) * -1

    return theta_base, theta_shoulder, theta_elbow, theta_wrist


def move_to(x_raw, y_raw, z=Z_HOVER):
    x, y = correction(x_raw, y_raw)

    if not is_reachable(x, y, z):
        print("Not reachable")
        return None
    
    th_b, th_s, th_e, th_w = solve(x,y,z)

    print(f"Base: {th_b:.1f}°  Shoulder: {th_s:.1f}°  Elbow: {th_e:.1f}°  Wrist: {th_w:.1f}°")
    return th_b, th_s, th_e, th_w
