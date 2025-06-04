import numpy as np
import pytest
from energy_predictor.model import EnergyPredictor


def test_predict_without_loading():
    predictor = EnergyPredictor('model/rf_clf.onnx')
    with pytest.raises(RuntimeError):
        predictor.predict(np.zeros((1, 1)))
