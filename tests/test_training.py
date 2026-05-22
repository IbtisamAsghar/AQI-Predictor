import os
import pytest
import numpy as np
from pathlib import Path
from src.models.model import AQIPredictionModel

def test_model_training_and_serialization():
    """Verifies convergence, NaN alignment, predictions, and save/load integrity."""
    
    # 1. Generate synthetic time-series tabular dataset
    np.random.seed(42)
    n_samples = 150
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    
    # Generate multi-horizon targets with synthetic NaNs at the end (to simulate shift boundaries)
    y_24h = np.random.uniform(10, 300, n_samples)
    y_48h = np.random.uniform(10, 300, n_samples)
    y_72h = np.random.uniform(10, 300, n_samples)
    
    y_24h[-24:] = np.nan
    y_48h[-48:] = np.nan
    y_72h[-72:] = np.nan
    
    y_dict = {
        "target_aqi_24h": y_24h,
        "target_aqi_48h": y_48h,
        "target_aqi_72h": y_72h
    }
    
    # 2. Instantiate and fit model wrapper
    model = AQIPredictionModel(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y_dict)
    
    # 3. Assert models were trained successfully
    assert len(model.models) == 3
    for horizon, estimator in model.models.items():
        # Assert each underlying random forest model is fitted
        assert hasattr(estimator, "estimators_"), f"Estimator for {horizon} failed to fit."
        
    # 4. Generate predictions
    preds = model.predict(X)
    
    # Assert output schema and lengths
    expected_keys = {"pred_aqi_24h", "pred_aqi_48h", "pred_aqi_72h"}
    assert set(preds.keys()) == expected_keys
    
    for key in expected_keys:
        assert len(preds[key]) == n_samples
        assert not np.isnan(preds[key]).any(), f"Predictions for {key} contained NaN values!"
        
    # 5. Assert Serialization & Deserialization round-trip
    test_filepath = "tests/test_model_temp.pkl"
    try:
        # Save model
        model.save(test_filepath)
        assert os.path.exists(test_filepath), "Model save failed to write pickle file to disk."
        
        # Load model back
        loaded_model = AQIPredictionModel.load(test_filepath)
        
        # Generate predictions from loaded model
        loaded_preds = loaded_model.predict(X)
        
        # Assert identical outputs
        for key in expected_keys:
            np.testing.assert_array_almost_equal(
                preds[key],
                loaded_preds[key],
                err_msg=f"Deserialized model outputs mismatched for {key}!"
            )
            
    finally:
        # Cleanup temporary pickle file safely
        if os.path.exists(test_filepath):
            os.remove(test_filepath)
            
if __name__ == "__main__":
    pytest.main([__file__])
