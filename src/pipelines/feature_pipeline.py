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
    Retrieves real-time meteorological and air quality measurements,
    joins them with historical logs from the MongoDB Atlas Feature Store,
    engineers time-series features incrementally, and upserts the
    new engineered record back into MongoDB.
    """
    logger.info("Triggering active feature pipeline execution...")
    
    try:
        # 1. Fetch real-time weather & air quality records
        from src.data.ingest import ingest_data
        raw_record = ingest_data()
        if not raw_record:
            raise RuntimeError("Data ingestion failed to retrieve real-time record.")
            
        new_timestamp = raw_record["timestamp"]
        logger.info(f"Ingested raw record timestamp: {new_timestamp}")
        
        # 2. Fetch the latest historical records for lag/rolling computations
        # 72 hours is sufficient to compute 48-hour lags and 24-hour rolling stats.
        from src.data.feature_store import get_latest_features, insert_records
        
        try:
            df_history = get_latest_features(limit=72)
        except Exception as e:
            logger.warning(f"Could not retrieve history from Feature Store: {e}. Fallback to empty df.")
            df_history = pd.DataFrame()
            
        # 3. Standardize and merge raw record with history
        raw_columns = [
            "timestamp", "location", "latitude", "longitude",
            "pm2_5", "pm10", "no2", "so2", "co", "o3",
            "temperature", "humidity", "wind_speed", "wind_direction", "aqi"
        ]
        
        if not df_history.empty:
            # Filter history to keep only raw columns for re-engineering
            df_history_raw = df_history[[c for c in raw_columns if c in df_history.columns]].copy()
            df_new_raw = pd.DataFrame([raw_record])
            df_new_raw["timestamp"] = pd.to_datetime(df_new_raw["timestamp"])
            
            # Combine history with new raw record
            df_combined = pd.concat([df_history_raw, df_new_raw], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["timestamp", "location"])
        else:
            logger.warning("Feature Store is empty or unreachable. Running engineering on single record.")
            df_combined = pd.DataFrame([raw_record])
            df_combined["timestamp"] = pd.to_datetime(df_combined["timestamp"])
            
        # 4. Run feature transformations
        df_engineered = engineer_features(df_combined, is_training=False)
        
        # 5. Extract only the newly calculated record
        latest_engineered_row = df_engineered.iloc[-1]
        latest_ts = latest_engineered_row["timestamp"]
        if isinstance(latest_ts, pd.Timestamp):
            latest_ts = latest_ts.isoformat()
            
        logger.info(f"Target timestamp for upsert: {latest_ts}")
        
        # Convert row to native python types (prevents numpy serialization errors in MongoDB BSON)
        record_to_insert = latest_engineered_row.to_dict()
        if isinstance(record_to_insert["timestamp"], pd.Timestamp):
            record_to_insert["timestamp"] = record_to_insert["timestamp"].isoformat()
            
        for k, v in record_to_insert.items():
            if isinstance(v, (np.integer, np.int64)):
                record_to_insert[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                record_to_insert[k] = float(v)
            elif pd.isna(v):
                record_to_insert[k] = None
                
        # 6. Commit the new engineered record into MongoDB Atlas Feature Store
        logger.info("Upserting new engineered record into MongoDB Atlas...")
        upsert_count = insert_records([record_to_insert])
        logger.info(f"Feature Store update finished. Upserted count: {upsert_count}")
        
        # Safe formatting helper
        def fmt_val(v, fmt):
            return f"{v:{fmt}}" if v is not None else "N/A"

        print("\n" + "="*70)
        print("         PEARLS AQI PREDICTOR - FEATURE ENGINE PIPELINE RUN")
        print("="*70)
        print(f"  Ingestion Source:       Open-Meteo APIs / AQICN Fallback")
        print(f"  Processed History size: {df_engineered.shape[0]} records")
        print(f"  Target Datetime:        {record_to_insert['timestamp']}")
        print(f"  Calculated AQI:         {fmt_val(record_to_insert.get('aqi'), '.1f')}")
        print(f"  Rolling 24h Mean AQI:   {fmt_val(record_to_insert.get('rolling_mean_24h_aqi'), '.1f')}")
        print(f"  1-Hour Lag PM2.5:       {fmt_val(record_to_insert.get('lag_1h_pm2_5'), '.1f')}")
        print(f"  1-Hour AQI Change Rate: {fmt_val(record_to_insert.get('aqi_change_rate_1h'), '.3f')}")
        print("="*70)
        print("  >>> FEATURE ENGINE PIPELINE: 100% OPERATIONAL & COMMITTED <<<\n")
        
        return True
        
    except Exception as e:
        logger.critical(f"Feature pipeline crashed during execution: {e}")
        raise e

if __name__ == "__main__":
    run_feature_pipeline()
