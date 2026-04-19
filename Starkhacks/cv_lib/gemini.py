# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION NOTE — wiring live state into this assistant
#
# To feed live camera frames and gesture results to WorkbenchAssistant, add
# the following to Starkhacks/main.py:
#
#     from cv_lib.context import workbench_context
#
# ...then inside inference_loop(), immediately after:
#
#     result = recognizer.process(rgb)
#
# add:
#
#     workbench_context.update(frame_bgr=frame, gesture=result)
#
# WorkbenchContext has its own internal lock, independent of main.py's `lock`.
# Until this wiring is added, WorkbenchAssistant still runs but the [STATE]
# block will say "unknown" and no image will be attached.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional

import cv2
import PIL.Image
from google import genai
from google.genai import types

from .context import workbench_context, WorkbenchSnapshot


# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

SYSTEM_INSTRUCTION = (
    "You are a AI robotic workbench assistant for engineering tasks "
    "such as soldering, building PCBs and circuits, and assembling parts."
    "Be terse, objective, and act as a teacher and learning assistant for"
    "any projects that your builder might be undertaking."
)

# context dict contains CV info (gesture and workbench picture?)
# component recognition?
# arm position?


def _format_state_from_dict(context: dict) -> str:
    lines = ["[STATE]"]

    g = context.get("gesture")
    if g is None:
        lines.append("gesture: unknown")
    else:
        tip = "n/a"
        lms = getattr(g, "landmarks", None)
        if lms and len(lms) > 8:
            lm = lms[8]
            tip = f"({lm.x:.2f}, {lm.y:.2f})"
        lines.append(
            f"gesture: name={g.gesture_name} command={g.command} "
            f"conf={g.confidence:.2f} hands={g.hand_count} index_tip={tip}"
        )

    arm = context.get("arm_pose")
    if arm is None:
        lines.append("arm_pose: unknown")
    else:
        b, s, e, w = arm
        lines.append(
            f"arm_pose: base={b:.1f} shoulder={s:.1f} "
            f"elbow={e:.1f} wrist={w:.1f} (deg)"
        )

    lines.append(
        "image: attached" if context.get("frame_bgr") is not None else "image: none"
    )
    return "\n".join(lines)


def _snapshot_to_dict(snap: WorkbenchSnapshot) -> dict:
    return {
        "frame_bgr": snap.frame_bgr,
        "gesture": snap.gesture,
        "arm_pose": snap.arm_pose,
    }


def _build_parts(context: dict, question: str, include_image: bool) -> list:
    parts: list = []
    frame_bgr = context.get("frame_bgr") if include_image else None
    if frame_bgr is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        parts.append(PIL.Image.fromarray(rgb))
    parts.append(_format_state_from_dict(context))
    parts.append(question)
    return parts


def _speak(text: str) -> None:
    try:
        from .tts import talk
        talk(text)
    except Exception as e:
        print(f"[gemini] TTS failed, returning text only: {e}")


def ask_question(question: str, context: Optional[dict] = None, speak: bool = True):
    """One-shot (no chat history). `context` may hold:
        - frame_bgr: np.ndarray (BGR, as from cv2)
        - gesture:   GestureResult
        - arm_pose:  (base, shoulder, elbow, wrist) in degrees
    Omit `context` to pull the latest from workbench_context.
    When `speak` is True (default), the reply is spoken via ElevenLabs TTS.
    """
    if context is None:
        context = _snapshot_to_dict(workbench_context.snapshot())

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
        contents=_build_parts(context, question, include_image=True),
    )
    if speak:
        _speak(response.text)
    return response


class WorkbenchAssistant:
    """Multi-turn chat session. Pulls latest frame/gesture/arm from
    workbench_context on every ask()."""

    def __init__(self, model: str = "gemini-3-flash-preview"):
        self._model = model
        self._chat = self._new_chat()

    def _new_chat(self):
        return client.chats.create(
            model=self._model,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
        )

    def reset(self) -> None:
        self._chat = self._new_chat()

    def ask(self, question: str, include_image: bool = True, speak: bool = True) -> str:
        context = _snapshot_to_dict(workbench_context.snapshot())
        parts = _build_parts(context, question, include_image=include_image)
        text = self._chat.send_message(parts).text
        if speak:
            _speak(text)
        return text
