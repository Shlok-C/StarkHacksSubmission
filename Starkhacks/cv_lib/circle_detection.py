import cv2
import numpy as np

# --- CONFIGURATION ---
MATRIX_FILE   = "homography_matrix.npy"
CAMERA_INDEX  = 1
DISPLAY_SCALE = 3
RECT_WIDTH_MM  = 200  # must match your calibration script
RECT_HEIGHT_MM = 150

# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_mm(px, py, matrix):
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)
    return result[0][0]  # (x_mm, y_mm)


def detect_black_circle(frame):
    """
    Finds the largest black circle in the frame.
    Returns (cx, cy, radius) in pixel space, or None if not found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold: keep only dark pixels (the black circle)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep the largest contour that looks circular
    best = None
    best_score = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:   # ignore tiny specks
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.6 and area > best_score:   # 1.0 = perfect circle
            best = c
            best_score = area

    if best is None:
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(best)
    return int(cx), int(cy), int(radius)


def main():
    # Load homography matrix
    try:
        matrix = np.load(MATRIX_FILE)
        print(f"Loaded homography matrix from '{MATRIX_FILE}'")
    except FileNotFoundError:
        print(f"ERROR: '{MATRIX_FILE}' not found.")
        print("Run homography_calibration.py first and click the 4 corners.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Detecting black circle — press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        result = detect_black_circle(frame)

        if result is not None:
            cx, cy, radius = result

            # Convert pixel centroid to real-world mm
            x_mm, y_mm = pixel_to_mm(cx, cy, matrix)

            # Draw circle and centroid on live feed
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5,      (0, 255, 0), -1)

            # Label on live feed
            label = f"({x_mm:.1f} mm, {y_mm:.1f} mm)"
            cv2.putText(frame, label, (cx + 12, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            print(f"Circle centroid: pixel ({cx}, {cy})  ->  {x_mm:.1f} mm, {y_mm:.1f} mm")
        else:
            cv2.putText(frame, "No circle detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Also show warped top-down view with the detected point
        warped_raw = cv2.warpPerspective(
            frame, matrix, (RECT_WIDTH_MM, RECT_HEIGHT_MM)
        )
        warped_big = cv2.resize(
            warped_raw,
            (RECT_WIDTH_MM * DISPLAY_SCALE, RECT_HEIGHT_MM * DISPLAY_SCALE),
            interpolation=cv2.INTER_LINEAR
        )

        # Draw dot on warped view at the detected position
        if result is not None:
            wx = int(x_mm * DISPLAY_SCALE)
            wy = int(y_mm * DISPLAY_SCALE)
            cv2.circle(warped_big, (wx, wy), 8, (0, 255, 0), -1)
            cv2.putText(warped_big, f"({x_mm:.1f}, {y_mm:.1f}) mm",
                        (wx + 10, wy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Live Feed", frame)
        cv2.imshow("Warped Top-Down", warped_big)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()