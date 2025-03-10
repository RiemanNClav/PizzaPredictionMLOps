import pytest
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


@pytest.fixture
def predict_pipeline():
    return PredictPipeline()

@pytest.fixture
def custom_data():
    return CustomData(40,
                  "A",
                  "vegetables",
                  "thai_veggie",
                  "small",
                  "no",
                  "no",
                  "no")
@pytest.fixture
def features(custom_data):
    return custom_data.get_data_as_data_frame()

def test_predict_pipeline_real(predict_pipeline, features):
    preds = predict_pipeline.predict(features)
    assert preds is not None  # Verifica que la predicción no sea None
    assert isinstance(preds[0], np.float64)  # Verifica el tipo de la predicción
