import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from src.pipelines.feature_pipeline import engineer_features

def test_engineer_features_pipeline():
    """Asserts that our feature engineering engine transforms raw tables without errors."""
    # 1. Create a dummy raw dataset (80 hours of records to allow rolling 24h & shifts 72h)
    base_time = pd.to_datetime("2026-05-20T00:00:00Z")
    records = []
    
    for i in range(80):
        timestamp = (base_time + pd.Timedelta(hours=i)).isoformat()
        records.append({
            "timestamp": timestamp,
            "location": "Islamabad",
            "latitude": 33.6844,
            "longitude": 73.0479,
            "pm2_5": float(20.0 + i % 10),
            "pm10": float(50.0 + i % 20),
            "no2": 5.0,
            "so2": 2.0,
            "co": 150.0,
            "o3": 60.0,
            "temperature": float(30.0 + i % 5),
            "humidity": 40.0,
            "wind_speed": 5.0,
            "wind_direction": 180.0,
            "aqi": float(60.0 + i % 10)
        })
        
    df_raw = pd.DataFrame(records)
    
    # 2. Run feature transformations under INFERENCE mode (is_training = False)
    df_inference = engineer_features(df_raw, is_training=False)
    
    # Check that shapes were preserved and features were added
    assert df_inference.shape[0] == 80
    assert "sin_hour" in df_inference.columns
    assert "cos_hour" in df_inference.columns
    assert "rolling_mean_24h_pm2_5" in df_inference.columns
    assert "lag_48h_aqi" in df_inference.columns
    assert "aqi_change_rate_1h" in df_inference.columns
    
    # Verify cyclical mappings reside correctly inside unit boundary limits [-1.0, 1.0]
    assert df_inference["sin_hour"].min() >= -1.0
    assert df_inference["sin_hour"].max() <= 1.0
    assert df_inference["cos_hour"].min() >= -1.0
    assert df_inference["cos_hour"].max() <= 1.0
    
    # Verify lag values (e.g. index 1 lag should be index 0 actual value)
    # df_raw index 0 AQI is 60.0, so df_inference index 1 lag_1h_aqi should be 60.0
    assert df_inference.loc[1, "lag_1h_aqi"] == 60.0
    assert pd.isna(df_inference.loc[0, "lag_1h_aqi"]) # First row has no lag
    
    # Assert targets are NOT generated under inference mode to prevent leakage/crashes
    assert "target_aqi_24h" not in df_inference.columns
    
    # 3. Run feature transformations under TRAINING mode (is_training = True)
    df_training = engineer_features(df_raw, is_training=True)
    
    # Verify multi-horizon future targets are generated
    assert "target_aqi_24h" in df_training.columns
    assert "target_aqi_48h" in df_training.columns
    assert "target_aqi_72h" in df_training.columns
    
    # Verify shift target shifts correctly (e.g. row 0 target_24h is row 24 AQI)
    # Row 24 AQI is (60 + 24 % 10) = 64.0
    assert df_training.loc[0, "target_aqi_24h"] == df_training.loc[24, "aqi"]
    assert pd.isna(df_training.loc[70, "target_aqi_24h"]) # Shifting by -24 at index 70 (>80-24=56) yields NaN
