import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logging import get_logger

logger = get_logger("feature_pipeline")

def engineer_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Stateless transformation function to engineer time-series features.
    
    Args:
        df: Pandas DataFrame containing raw weather & air quality records.
            Must contain: 'timestamp', 'aqi', 'pm2_5', 'pm10', 'temperature', 
            'humidity', 'wind_speed', 'wind_direction'.
        is_training: If True, shifts future AQI values backward to generate 
            targets (+24h, +48h, +72h). If False, skips target generation (for inference).
            
    Returns:
        Pandas DataFrame containing raw features alongside all engineered fields,
        sorted chronologically.
    """
    # 1. Sort chronologically by timestamp
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Engineering features over dataset of shape {df.shape}...")
    
    # 2. Time-Based Cyclical Encodings (sine/cosine mapping)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    
    # Map cyclical behaviors into continuous [-1.0, 1.0] spaces
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    
    # 3. Time-Series Historical Lags
    # Compute lags to capture past trajectories of target pollutant (PM2.5) and AQI
    df["lag_1h_aqi"] = df["aqi"].shift(1)
    df["lag_2h_aqi"] = df["aqi"].shift(2)
    df["lag_24h_aqi"] = df["aqi"].shift(24)
    df["lag_48h_aqi"] = df["aqi"].shift(48)
    
    df["lag_1h_pm2_5"] = df["pm2_5"].shift(1)
    df["lag_24h_pm2_5"] = df["pm2_5"].shift(24)
    
    # 4. Rolling Window Statistics
    # rolling statistics over short (3h), medium (12h), and long (24h) periods
    df["rolling_mean_3h_pm2_5"] = df["pm2_5"].rolling(window=3, min_periods=1).mean()
    df["rolling_std_3h_pm2_5"] = df["pm2_5"].rolling(window=3, min_periods=1).std().fillna(0.0)
    
    df["rolling_mean_24h_pm2_5"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    df["rolling_std_24h_pm2_5"] = df["pm2_5"].rolling(window=24, min_periods=1).std().fillna(0.0)
    
    df["rolling_mean_12h_aqi"] = df["aqi"].rolling(window=12, min_periods=1).mean()
    df["rolling_mean_24h_aqi"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    
    # 5. Derived Ratios and Rates
    # AQI rate of change over the last hour (epsilon added to prevent division-by-zero)
    df["aqi_change_rate_1h"] = (df["aqi"] - df["lag_1h_aqi"]) / (df["lag_1h_aqi"] + 1e-5)
    
    # 6. Forecasting Multi-Horizon Targets
    # Shift FUTURE AQI values backward in time to serve as forecasting targets
    if is_training:
        logger.info("Generating multi-step forecasting targets (+24h, +48h, +72h)...")
        df["target_aqi_24h"] = df["aqi"].shift(-24)
        df["target_aqi_48h"] = df["aqi"].shift(-48)
        df["target_aqi_72h"] = df["aqi"].shift(-72)
        
    logger.info("Feature engineering completed successfully.")
    return df

def run_feature_pipeline() -> bool:
    """
    Active pipeline runner triggered by scheduled cron jobs.
    In later phases, this fetches live values, queries past records,
    calculates features, and commits back into MongoDB.
    """
    logger.info("Triggering active feature pipeline execution...")
    
    # Local placeholder demonstration data mimicking ingestion engine
    try:
        from src.data.ingest import ingest_data
        raw_record = ingest_data()
        
        # Mimic a historical series by copying the record with hourly timestamps
        base_time = pd.to_datetime(raw_record["timestamp"])
        records = []
        
        # Generate 49 dummy hours of history leading to this record for testing rolling windows
        for i in range(49):
            time_delta = pd.Timedelta(hours=(48 - i))
            rec = raw_record.copy()
            rec["timestamp"] = (base_time - time_delta).isoformat()
            # Add small random noise to mimic time series changes
            rec["aqi"] = max(0.0, rec["aqi"] + np.random.uniform(-10, 10))
            rec["pm2_5"] = max(0.0, rec["pm2_5"] + np.random.uniform(-5, 5))
            records.append(rec)
            
        # Convert to DataFrame
        df_raw = pd.DataFrame(records)
        
        # Run feature transformation
        df_features = engineer_features(df_raw, is_training=False)
        
        # Display the newest engineered row (the latest feature vector)
        latest_row = df_features.iloc[-1]
        
        print("\n" + "="*70)
        print("         PEARLS AQI PREDICTOR - FEATURE ENGINE PIPELINE RUN")
        print("="*70)
        print(f"  Processed History size: {df_features.shape[0]} records")
        print(f"  Target Datetime:        {latest_row['timestamp']}")
        print(f"  Calculated AQI:         {latest_row['aqi']:.1f}")
        print(f"  Cyclical Cos Hour:      {latest_row['cos_hour']:.3f}")
        print(f"  Rolling 24h Mean AQI:   {latest_row['rolling_mean_24h_aqi']:.1f}")
        print(f"  1-Hour Lag PM2.5:       {latest_row['lag_1h_pm2_5']:.1f}")
        print(f"  1-Hour AQI Change Rate: {latest_row['aqi_change_rate_1h']:.3f}")
        print("="*70)
        print("  >>> FEATURE ENGINE PIPELINE: 100% OPERATIONAL <<<\n")
        
        return True
    except Exception as e:
        logger.critical(f"Feature pipeline crashed during execution: {e}")
        raise e

if __name__ == "__main__":
    run_feature_pipeline()
