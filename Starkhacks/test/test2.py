import cv2
import numpy as np
import os

# --- CONFIGURATION ---
RECT_WIDTH_MM  = 200   # Physical width of your taped rectangle (mm)
RECT_HEIGHT_MM = 150   # Physical height of your taped rectangle (mm)
DISPLAY_SCALE  = 3     # Scale-up factor for the warped window (3x = 900x600 px display)
MATRIX_FILE    = "homography_matrix.npy"
CAMERA_INDEX   = 1

# Corner labels in click order: TL, TR, BL, BR
CORNER_LABELS = ["1: Top-Left", "2: Top-Right", "3: Bottom-Left", "4: Bottom-Right"]
CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (0, 0, 255)]

# --- GLOBALS ---
clicked_points  = []   # pixel coords clicked on the live feed
matrix          = None # computed or loaded homography matrix
warped_cursor   = (0, 0)  # current cursor position on the WARPED window (in mm)

# ─────────────────────────────────────────────────────────────────────────────
# MOUSE CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def live_feed_click(event, x, y, flags, param):
    """Records corner clicks on the live camera feed."""
    global clicked_points, matrix
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        label = CORNER_LABELS[len(clicked_points) - 1]
        print(f"  Clicked {label}: pixel ({x}, {y})")
        if len(clicked_points) == 4:
            compute_and_save_matrix()


def warped_mouse_move(event, x, y, flags, param):
    """
    Tracks cursor position on the WARPED (scaled) window.
    Converts from display pixels back to real-world mm.
    """
    global warped_cursor
    # x, y are in the SCALED display window — divide by DISPLAY_SCALE to get mm
    mm_x = x / DISPLAY_SCALE
    mm_y = y / DISPLAY_SCALE
    warped_cursor = (mm_x, mm_y)

# ─────────────────────────────────────────────────────────────────────────────
# HOMOGRAPHY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_save_matrix():
    global matrix
    pts_src = np.float32(clicked_points)
    pts_dst = np.float32([
        [0,              0             ],   # TL
        [RECT_WIDTH_MM,  0             ],   # TR
        [0,              RECT_HEIGHT_MM],   # BL
        [RECT_WIDTH_MM,  RECT_HEIGHT_MM],   # BR
    ])
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    np.save(MATRIX_FILE, matrix)
    print(f"\n  Matrix saved to '{MATRIX_FILE}' — won't need to click corners again.")


def pixel_to_mm(px, py):
    """
    Transforms a single pixel coordinate from the RAW camera frame
    into real-world mm using the current homography matrix.
    """
    if matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, matrix)
    return result[0][0]  # (x_mm, y_mm)


def build_warped_display(frame):
    """
    Warps the frame and scales it up for comfortable viewing.
    Draws a crosshair + coordinate readout at the cursor position.
    """
    warped_raw = cv2.warpPerspective(
        frame, matrix, (RECT_WIDTH_MM, RECT_HEIGHT_MM)
    )
    # Scale up for display
    warped_big = cv2.resize(
        warped_raw,
        (RECT_WIDTH_MM * DISPLAY_SCALE, RECT_HEIGHT_MM * DISPLAY_SCALE),
        interpolation=cv2.INTER_LINEAR
    )

    # Cursor position in display pixels
    cx = int(warped_cursor[0] * DISPLAY_SCALE)
    cy = int(warped_cursor[1] * DISPLAY_SCALE)

    # Crosshair
    color = (0, 255, 255)
    cv2.line(warped_big, (cx - 12, cy), (cx + 12, cy), color, 1)
    cv2.line(warped_big, (cx, cy - 12), (cx, cy + 12), color, 1)
    cv2.circle(warped_big, (cx, cy), 5, color, 1)

    # Coordinate label — show in mm AND cm
    mm_x, mm_y = warped_cursor
    cm_x, cm_y = mm_x / 10, mm_y / 10
    label = f"({mm_x:6.1f} mm, {mm_y:6.1f} mm)  |  ({cm_x:.2f} cm, {cm_y:.2f} cm)"

    # Background pill so text is readable over any surface
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx, ly = 8, 8
    cv2.rectangle(warped_big, (lx - 4, ly - 2), (lx + tw + 4, ly + th + 4), (0, 0, 0), -1)
    cv2.putText(warped_big, label, (lx, ly + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # Draw a small dot at every 50mm grid intersection for reference
    for gx in range(0, RECT_WIDTH_MM + 1, 50):
        for gy in range(0, RECT_HEIGHT_MM + 1, 50):
            px = int(gx * DISPLAY_SCALE)
            py = int(gy * DISPLAY_SCALE)
            cv2.circle(warped_big, (px, py), 3, (180, 180, 180), -1)
            if gx > 0 or gy > 0:  # skip origin label
                grid_label = f"{gx},{gy}"
                cv2.putText(warped_big, grid_label, (px + 4, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)

    return warped_big


def draw_live_overlay(frame):
    """
    Draws clicked corners, next-corner guidance, and a status bar
    on the live camera feed.
    """
    h, w = frame.shape[:2]

    # Clicked corners
    for i, pt in enumerate(clicked_points):
        cv2.circle(frame, (pt[0], pt[1]), 8, CORNER_COLORS[i], -1)
        cv2.putText(frame, CORNER_LABELS[i], (pt[0] + 10, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLORS[i], 2, cv2.LINE_AA)

    # Next corner prompt
    if len(clicked_points) < 4:
        next_label = CORNER_LABELS[len(clicked_points)]
        next_color = CORNER_COLORS[len(clicked_points)]
        prompt = f"Click next: {next_label}"
        cv2.putText(frame, prompt, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, next_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Calibrated  |  R=reset  Q=quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

    # Draw lines between clicked corners once all 4 are in
    if len(clicked_points) == 4:
        order = [0, 1, 3, 2, 0]  # TL -> TR -> BR -> BL -> TL
        for i in range(4):
            p1 = tuple(clicked_points[order[i]])
            p2 = tuple(clicked_points[order[i + 1]])
            cv2.line(frame, p1, p2, (0, 255, 100), 2)

    return frame

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global clicked_points, matrix

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("Brio Live Feed")
    cv2.setMouseCallback("Brio Live Feed", live_feed_click)

    warped_win = "Warped Top-Down View  (move cursor for coordinates)"

    # ── Try to load a previously saved matrix ──
    if os.path.exists(MATRIX_FILE):
        matrix = np.load(MATRIX_FILE)
        print(f"Loaded existing matrix from '{MATRIX_FILE}'.")
        print("Press R to recalibrate, Q to quit.\n")
        # Pre-fill clicked_points so the UI knows we're calibrated
        # (we don't know the original pixels, so just mark as done)
        clicked_points = [[0, 0], [0, 0], [0, 0], [0, 0]]
        cv2.namedWindow(warped_win)
        cv2.setMouseCallback(warped_win, warped_mouse_move)
    else:
        print("No saved matrix found. Click the 4 corners to calibrate.")
        print("Order: 1=Top-Left  2=Top-Right  3=Bottom-Left  4=Bottom-Right")
        print("R=reset  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break

        draw_live_overlay(frame)
        cv2.imshow("Brio Live Feed", frame)

        if matrix is not None:
            warped_display = build_warped_display(frame)
            cv2.imshow(warped_win, warped_display)
            # Ensure mouse callback is registered (first time the window appears)
            cv2.setMouseCallback(warped_win, warped_mouse_move)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\nResetting calibration...")
            clicked_points = []
            matrix = None
            if os.path.exists(MATRIX_FILE):
                os.remove(MATRIX_FILE)
                print(f"  Deleted '{MATRIX_FILE}'.")
            cv2.destroyWindow(warped_win)
            print("Click 4 corners again.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()