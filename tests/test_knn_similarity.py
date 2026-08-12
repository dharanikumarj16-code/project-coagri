import pytest
import pandas as pd
from knn_similarity_engine import KNNSimilarityEngine

@pytest.fixture
def dummy_patterns_data():
    return pd.DataFrame({
        "crop": ["rice", "rice", "wheat"],
        "temp": [28.0, 30.0, 20.0],
        "min_temp": [24.0, 26.0, 15.0],
        "max_temp": [32.0, 34.0, 25.0],
        "rainfall": [150.0, 180.0, 50.0],
        "humidity": [80.0, 85.0, 60.0],
        "result": [1, 1, 1]
    })

def test_knn_similarity_engine_find_matches(dummy_patterns_data):
    engine = KNNSimilarityEngine(n_neighbors=2)
    assert not engine.fitted
    
    engine.fit(dummy_patterns_data)
    assert engine.fitted
    
    query = pd.DataFrame([{
        "temp": 29.0, "min_temp": 25.0, "max_temp": 33.0,
        "rainfall": 160.0, "humidity": 82.0
    }])
    
    matches = engine.find_matches(query)
    assert len(matches) == 1
    match_entry = matches[0]
    assert "query_index" in match_entry
    assert "best_match" in match_entry
    assert "similarity" in match_entry["best_match"]
    assert match_entry["best_match"]["similarity"] > 0

def test_unfitted_knn_raises():
    engine = KNNSimilarityEngine()
    query = pd.DataFrame([{"temp": 25.0, "min_temp": 20.0, "max_temp": 30.0, "rainfall": 100.0, "humidity": 70.0}])
    with pytest.raises(ValueError, match="KNNSimilarityEngine has not been fitted"):
        engine.find_matches(query)
