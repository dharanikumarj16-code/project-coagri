import pytest
import pandas as pd
from prediction_engine import PredictionEngine

@pytest.fixture
def dummy_historical_data():
    return pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "crop": ["rice", "rice", "wheat", "wheat", "rice"],
        "temp": [28.0, 30.0, 20.0, 22.0, 29.0],
        "min_temp": [24.0, 26.0, 15.0, 17.0, 25.0],
        "max_temp": [32.0, 34.0, 25.0, 27.0, 33.0],
        "rainfall": [150.0, 180.0, 50.0, 60.0, 160.0],
        "humidity": [80.0, 85.0, 60.0, 65.0, 82.0],
        "result": [1, 1, 1, 1, 0]
    })

def test_prediction_engine_fit_and_predict(dummy_historical_data):
    engine = PredictionEngine(n_neighbors=2)
    assert not engine.fitted
    
    engine.fit(dummy_historical_data)
    assert engine.fitted
    
    query = pd.DataFrame([{
        "date": "2026-05-01",
        "temp": 29.0,
        "min_temp": 25.0,
        "max_temp": 33.0,
        "rainfall": 155.0,
        "humidity": 81.0
    }])
    
    results = engine.predict_viability(query, target_crop="Rice")
    assert len(results) == 1
    res = results[0]
    assert res["crop"] == "Rice"
    assert "similarity_score" in res
    assert "success_probability" in res
    assert res["risk_level"] in ["low", "medium", "high"]

def test_prediction_engine_crop_comparisons(dummy_historical_data):
    engine = PredictionEngine(n_neighbors=2)
    engine.fit(dummy_historical_data)
    
    query = pd.DataFrame([{
        "temp": 29.0, "min_temp": 25.0, "max_temp": 33.0,
        "rainfall": 155.0, "humidity": 81.0
    }])
    
    comparisons = engine.analyze_crop_comparisons(query, ["rice", "wheat"])
    assert isinstance(comparisons, list)
    assert len(comparisons) > 0
    assert "similarity_score" in comparisons[0]
    assert "risk_level" in comparisons[0]

def test_unfitted_prediction_engine_raises():
    engine = PredictionEngine()
    query = pd.DataFrame([{"temp": 25.0, "min_temp": 20.0, "max_temp": 30.0, "rainfall": 100.0, "humidity": 70.0}])
    with pytest.raises(ValueError, match="PredictionEngine must be fitted"):
        engine.predict_viability(query)
