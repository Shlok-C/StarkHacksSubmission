from .gesture_recognition import (
    GESTURE_MAP, GestureResult,
    classify_gesture, resolve_command, pointing_direction, pinch_distance,
    draw_landmarks, draw_overlay,
)
from .npu_runtime import NPUInterpreter
from .context import workbench_context, WorkbenchContext, WorkbenchSnapshot
from .gemini import WorkbenchAssistant, ask_question
from .comp_recognition import *
try:
    from .voice_recognition import *
except Exception:
    pass  # Vosk model not installed
try:
    from .tts import talk
except Exception:
    pass  # elevenlabs package not installed

