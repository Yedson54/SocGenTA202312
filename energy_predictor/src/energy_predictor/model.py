import numpy as np
import onnxruntime as rt
from pathlib import Path

class EnergyPredictor:
    """Wrapper around an ONNX model for energy reduction prediction."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.sess: rt.InferenceSession | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.sess = rt.InferenceSession(str(self.model_path))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.sess is None:
            raise RuntimeError("Model not loaded")
        input_name = self.sess.get_inputs()[0].name
        outputs = self.sess.run(None, {input_name: X.astype(np.float32)})
        return outputs[0]
