import pytest
import pandas as pd
import numpy as np
from feature_scaler import FeatureScaler

def test_feature_scaler_fit_transform():
    df = pd.DataFrame({
        "temp": [20.0, 25.0, 30.0],
        "min_temp": [15.0, 20.0, 25.0],
        "max_temp": [25.0, 30.0, 35.0],
        "rainfall": [100.0, 150.0, 200.0],
        "humidity": [80.0, 85.0, 90.0]
    })
    
    scaler = FeatureScaler()
    assert not scaler.fitted
    
    scaled = scaler.fit_transform(df)
    assert scaler.fitted
    assert scaled.shape == (3, 5)
    # Check zero mean for standardized features
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-5)

def test_feature_scaler_unfitted_raises_error():
    scaler = FeatureScaler()
    df = pd.DataFrame({
        "temp": [20.0], "min_temp": [15.0], "max_temp": [25.0],
        "rainfall": [100.0], "humidity": [80.0]
    })
    with pytest.raises(ValueError, match="FeatureScaler has not been fitted yet"):
        scaler.transform(df)

def test_feature_scaler_handles_non_numeric():
    df = pd.DataFrame({
        "temp": ["20.0", "invalid", None],
        "min_temp": [15.0, 20.0, 25.0],
        "max_temp": [25.0, 30.0, 35.0],
        "rainfall": [100.0, 150.0, 200.0],
        "humidity": [80.0, 85.0, 90.0]
    })
    scaler = FeatureScaler()
    scaled = scaler.fit_transform(df)
    assert scaled.shape == (3, 5)
    assert not np.isnan(scaled).any()
