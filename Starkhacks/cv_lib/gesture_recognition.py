import cv2
import math
from collections import namedtuple, deque
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Landmark indices
WRIST      = 0
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_TIP  = 8
MIDDLE_MCP = 9

# Gesture label → (display name, robot command)
# command=None means direction is derived from finger vector at runtime
GESTURE_MAP = {
    "Closed_Fist": ("FIST",      "GRIP"),
    "Open_Palm":   ("OPEN HAND", "RELEASE"),
    "Pointing_Up": ("POINT",     None),
    "Thumb_Up":    ("THUMB UP",  "POSITION_1"),
    "Thumb_Down":  ("THUMB DOWN","MOVE_DOWN"),
    "Victory":     ("VICTORY",   "POSITION_2"),
    "ILoveYou":    ("ILY",       "POSITION_3"),
    "None":        ("---",       "NONE"),
    # Heuristic labels from classify_gesture()
    "OPEN":  ("OPEN HAND", "RELEASE"),
    "FIST":  ("FIST",      "GRIP"),
    "POINT": ("POINT",     None),
    "PEACE": ("PEACE",     "POSITION_2"),
}

GestureResult = namedtuple(
    "GestureResult",
    ["gesture_name", "command", "confidence", "landmarks", "hand_count"]
)


def pointing_direction(landmarks) -> str:
    """Return MOVE_* based on index finger vector direction."""
    mcp   = landmarks[INDEX_MCP]
    tip   = landmarks[INDEX_TIP]

    # landmarks can be MediaPipe NormalizedLandmark objects (with .x/.y)
    # or plain (px, py) tuples from classify_gesture path
    if hasattr(mcp, "x"):
        dx, dy = tip.x - mcp.x, tip.y - mcp.y
    else:
        dx, dy = tip[0] - mcp[0], tip[1] - mcp[1]

    angle = math.degrees(math.atan2(-dy, dx))
    if   -45 <= angle <   45: return "MOVE_RIGHT"
    elif  45 <= angle <  135: return "MOVE_UP"
    elif angle >= 135 or angle < -135: return "MOVE_LEFT"
    else:                      return "MOVE_DOWN"


def pinch_distance(landmarks) -> float:
    """Thumb-to-index distance normalised by hand size (0 = touching)."""
    thumb = landmarks[THUMB_TIP]
    index = landmarks[INDEX_TIP]
    wrist = landmarks[WRIST]
    mmcp  = landmarks[MIDDLE_MCP]

    if hasattr(thumb, "x"):
        dist      = math.hypot(thumb.x - index.x, thumb.y - index.y)
        hand_size = math.hypot(wrist.x - mmcp.x,  wrist.y - mmcp.y)
    else:
        dist      = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
        hand_size = math.hypot(wrist[0] - mmcp[0],  wrist[1] - mmcp[1])

    return dist / hand_size if hand_size > 0 else 1.0


def resolve_command(label: str, landmarks, confidence: float) -> GestureResult:
    """
    Convert a raw gesture label + landmarks into a GestureResult.
    Applies pinch override and pointing-direction resolution.
    """
    # Pinch override (fine gripper control)
    if landmarks is not None:
        pd = pinch_distance(landmarks)
        if pd < 0.15:
            return GestureResult(
                "PINCH", "GRIP_ADJUST",
                round(1.0 - pd / 0.15, 2),
                landmarks, 1,
            )

    if label in GESTURE_MAP:
        name, command = GESTURE_MAP[label]
        if command is None and landmarks is not None:
            command = pointing_direction(landmarks)
        elif command is None:
            command = "NONE"
        return GestureResult(name, command, confidence, landmarks, 1)

    return GestureResult(label, "NONE", confidence, landmarks, 1)


def classify_gesture(pts: list[tuple]) -> str:
    """
    Heuristic gesture name from raw (px, py) landmark list.
    Used when model output is landmarks-only (no class head).
    """
    fingers_up = [
        pts[8][1]  < pts[6][1],
        pts[12][1] < pts[10][1],
        pts[16][1] < pts[14][1],
        pts[20][1] < pts[18][1],
    ]
    count = sum(fingers_up)

    if count == 4:                                       return "OPEN"
    if count == 0:                                       return "FIST"
    if fingers_up[0] and count == 1:                    return "POINT"
    if fingers_up[0] and fingers_up[1] and count == 2:  return "PEACE"
    return "None"


def draw_landmarks(frame, landmarks, is_pixel_coords: bool = False):
    """Draw hand skeleton. landmarks: list of (px,py) or NormalizedLandmark."""
    if landmarks is None:
        return frame
    h, w = frame.shape[:2]

    if is_pixel_coords:
        pts = [(int(lm[0]), int(lm[1])) for lm in landmarks]
    else:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 5, (0, 150, 255), 1)
    return frame


def draw_overlay(frame, result: GestureResult, history: deque):
    """Render gesture name, command, confidence, and command history onto frame."""
    h, w = frame.shape[:2]
    scale = w / 640.0  # scale text/boxes relative to a 640-wide reference

    def fs(s): return max(0.35, s * scale)   # font scale
    def px(x): return int(x * scale)         # x pixel
    def py(y): return int(y * scale)         # y pixel

    # Top-left info box
    cv2.rectangle(frame, (px(10), py(10)), (px(430), py(85)), (0, 0, 0), -1)
    cv2.putText(frame, f"Gesture : {result.gesture_name}",
                (px(15), py(38)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.8), (0, 255, 100), 2)
    conf_str = f"  ({result.confidence:.0%})" if result.confidence > 0 else ""
    cv2.putText(frame, f"Command : {result.command}{conf_str}",
                (px(15), py(68)), cv2.FONT_HERSHEY_SIMPLEX, fs(0.7), (0, 200, 255), 2)

    # Bottom-left history box
    box_h = py(105)
    cv2.rectangle(frame, (px(10), h - box_h), (px(310), h - px(10)), (0, 0, 0), -1)
    cv2.putText(frame, "Recent commands:",
                (px(15), h - box_h + py(22)), cv2.FONT_HERSHEY_SIMPLEX,
                fs(0.5), (180, 180, 180), 1)
    for i, cmd in enumerate(list(history)[-3:]):
        cv2.putText(frame, cmd,
                    (px(15), h - box_h + py(22) + py(24) * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs(0.58), (255, 255, 255), 1)

    # Top-right hand count
    cv2.putText(frame, f"Hands: {result.hand_count}",
                (w - px(145), py(38)), cv2.FONT_HERSHEY_SIMPLEX,
                fs(0.7), (200, 200, 200), 2)
    return frame
