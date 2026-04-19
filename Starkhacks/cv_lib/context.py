from dataclasses import dataclass
from threading import Lock
from typing import Optional, Tuple
import numpy as np

from .gesture_recognition import GestureResult


@dataclass
class WorkbenchSnapshot:
    frame_bgr: Optional[np.ndarray] = None
    gesture: Optional[GestureResult] = None
    arm_pose: Optional[Tuple[float, float, float, float]] = None


class WorkbenchContext:
    def __init__(self):
        self._lock = Lock()
        self._snap = WorkbenchSnapshot()

    def update(self, *, frame_bgr=None, gesture=None, arm_pose=None):
        with self._lock:
            if frame_bgr is not None:
                self._snap.frame_bgr = frame_bgr
            if gesture is not None:
                self._snap.gesture = gesture
            if arm_pose is not None:
                self._snap.arm_pose = arm_pose

    def snapshot(self) -> WorkbenchSnapshot:
        with self._lock:
            return WorkbenchSnapshot(
                frame_bgr=self._snap.frame_bgr,
                gesture=self._snap.gesture,
                arm_pose=self._snap.arm_pose,
            )


workbench_context = WorkbenchContext()
