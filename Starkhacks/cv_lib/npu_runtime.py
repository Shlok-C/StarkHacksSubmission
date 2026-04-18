"""
NPU-accelerated inference wrapper for RUBIK Pi 3 (Qualcomm QCS6490).

Backend selection (automatic, in priority order):
  1. QAIRT / QNN — .bin serialized model, full Hexagon NPU via QAIRT 2.42.x
  2. TFLite CPU  — .tflite fallback (used until QNN binary is available)

Usage:
    interp = NPUInterpreter("models/HandGestureRecognizer_qai.bin")
    outputs = interp.run(input_array)   # dict[str, np.ndarray]
    print(interp.backend)               # "qnn" | "tflite_cpu"
"""

from __future__ import annotations

import numpy as np
from pathlib import Path


class NPUInterpreter:
    def __init__(self, model_path: str, tflite_fallback: str | None = None):
        self._path = Path(model_path)
        self._tflite_fallback = Path(tflite_fallback) if tflite_fallback else None
        self._interp = None
        self._input_details = None
        self._output_details = None
        self.backend: str = "none"

        if self._path.suffix == ".bin":
            self._init_qnn()
        else:
            self._init_tflite()

    # ── QNN / QAIRT backend ───────────────────────────────────────────────────

    def _init_qnn(self):
        try:
            from qai_appbuilder import QNNContext, QNNConfig, Runtime
            QNNConfig.Config(qnn_lib_path="/usr/lib", runtime=Runtime.HTP)
            self._qnn_ctx = QNNContext(
                model_name=self._path.stem,
                model_path=str(self._path),
            )
            self._input_names   = self._qnn_ctx.getInputName()
            self._output_names  = self._qnn_ctx.getOutputName()
            self._output_shapes = self._qnn_ctx.getOutputShapes()
            self.backend = "qnn"
            print(f"[NPU] QNN/HTP backend loaded: {self._path.name}")
        except (ImportError, Exception) as e:
            print(f"[NPU] qai_appbuilder unavailable ({e}) — falling back to TFLite")
            candidates = [
                self._tflite_fallback,
                self._path.with_suffix(".tflite"),
            ]
            tflite_path = next((p for p in candidates if p and p.exists()), None)
            if tflite_path is None:
                raise FileNotFoundError(
                    f"QNN runtime unavailable and no TFLite fallback found for {self._path}"
                )
            self._path = tflite_path
            self._init_tflite()

    def _run_qnn(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        input_list = [inputs.get(n, next(iter(inputs.values()))) for n in self._input_names]
        raw = self._qnn_ctx.Inference(input_list)
        return {
            name: out.reshape(shape)
            for name, out, shape in zip(self._output_names, raw, self._output_shapes)
        }

    # ── TFLite backend (NNAPI delegate → CPU fallback) ────────────────────────

    def _init_tflite(self):
        from ai_edge_litert.interpreter import Interpreter
        self._interp = Interpreter(model_path=str(self._path))
        self.backend = "tflite_cpu"
        print(f"[NPU] TFLite CPU loaded: {self._path.name}")

        self._interp.allocate_tensors()
        self._input_details = self._interp.get_input_details()
        self._output_details = self._interp.get_output_details()
        self._out_idx = {t["name"]: t["index"] for t in self._output_details}

    def _run_tflite(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for detail in self._input_details:
            key = detail["name"]
            tensor = inputs.get(key, next(iter(inputs.values())))
            # Quantize float input to int8 if model expects it
            if detail["dtype"] == np.int8 and tensor.dtype != np.int8:
                scale, zp = detail["quantization"]
                tensor = (tensor / scale + zp).clip(-128, 127).astype(np.int8)
            self._interp.set_tensor(detail["index"], tensor)
        self._interp.invoke()
        return {
            name: self._interp.get_tensor(idx)
            for name, idx in self._out_idx.items()
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, inputs: dict[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
        """
        Run inference. `inputs` can be a dict of named tensors or a single
        array (auto-assigned to the first input).
        """
        if isinstance(inputs, np.ndarray):
            inputs = {"input": inputs}

        if self.backend == "qnn":
            return self._run_qnn(inputs)
        return self._run_tflite(inputs)

    @property
    def input_details(self):
        return self._input_details

    @property
    def output_details(self):
        return self._output_details

    def close(self):
        if self.backend == "qnn" and hasattr(self, "_qnn_model"):
            del self._qnn_model
