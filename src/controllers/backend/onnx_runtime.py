from __future__ import annotations

import numpy as np
import onnxruntime as ort

class OnnxBackend:
    def __init__(
            self,
            model: str = "src/resources/weights/yolo11s.onnx",
            cpu_threads: int = 2,
            input_size: int = 640,
            confidence_threshold: float = 0.25,
            iou_threshold: float = 0.45,
            class_ids: list[int] | None = None,
    ):
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_ids = class_ids

        options = ort.SessionOptions()
        if cpu_threads >= 0:
            options.intra_op_num_threads = max(1, cpu_threads)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            model,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Stater backend requires 1 input from model")
        self._input_name = inputs[0].name

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self._session.run(
            None,
            {self._input_name: tensor}
        )
    
    def warmup(self, input_shape: tuple[int, ...], iterations: int) -> None:
        dummy = np.zeros(input_shape, dtype=np.float32)
        for _ in range(iterations):
            self.infer(dummy)
