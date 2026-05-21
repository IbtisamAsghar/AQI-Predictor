import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.utils.logging import get_logger
from src.data.ingest import calculate_pm25_aqi
from src.pipelines.feature_pipeline import engineer_features
from src.data.feature_store import insert_records

logger = get_logger("historical_backfiller")

def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Queries the Open-Meteo historical archive API for meteorological measurements with retries."""
    logger.info(f"Fetching historical weather archives from {start_date} to {end_date}...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": settings.LATITUDE,
        "longitude": settings.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Karachi"
    }
    
    retries = 3
    delay = 2
    hourly_raw = {}
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=25)
            response.raise_for_status()
            hourly_raw = response.json().get("hourly", {})
            break
        except Exception as e:
            logger.warning(f"Weather fetch attempt {attempt} failed: {e}")
            if attempt == retries:
                logger.error("Failed to query historical weather archive after maximum retries.")
                raise e
            time.sleep(delay)
            delay *= 2
            
    df = pd.DataFrame({
        "time": hourly_raw.get("time", []),
        "temperature": hourly_raw.get("temperature_2m", []),
        "humidity": hourly_raw.get("relative_humidity_2m", []),
        "wind_speed": hourly_raw.get("wind_speed_10m", []),
        "wind_direction": hourly_raw.get("wind_direction_10m", [])
    })
    logger.info(f"Successfully fetched {df.shape[0]} historical weather records.")
    return df

def fetch_historical_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Queries the Open-Meteo air quality API for historical pollutant measurements with retries."""
    logger.info(f"Fetching historical air quality logs from {start_date} to {end_date}...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": settings.LATITUDE,
        "longitude": settings.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "timezone": "Asia/Karachi"
    }
    
    retries = 3
    delay = 2
    hourly_raw = {}
    
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=25)
            response.raise_for_status()
            hourly_raw = response.json().get("hourly", {})
            break
        except Exception as e:
            logger.warning(f"Air quality fetch attempt {attempt} failed: {e}")
            if attempt == retries:
                logger.error("Failed to query historical air quality after maximum retries.")
                raise e
            time.sleep(delay)
            delay *= 2
            
    df = pd.DataFrame({
        "time": hourly_raw.get("time", []),
        "pm2_5": hourly_raw.get("pm2_5", []),
        "pm10": hourly_raw.get("pm10", []),
        "no2": hourly_raw.get("nitrogen_dioxide", []),
        "so2": hourly_raw.get("sulphur_dioxide", []),
        "co": hourly_raw.get("carbon_monoxide", []),
        "o3": hourly_raw.get("ozone", [])
    })
    logger.info(f"Successfully fetched {df.shape[0]} historical air quality records.")
    return df

def run_backfill(start_date: str = "2023-06-01", end_date: str = None) -> bool:
    """
    Unified orchestrator to fetch historical weather/pollution data, 
    merge them, compute engineered features, and seed MongoDB Atlas.
    """
    if end_date is None:
        # Default to 3 days ago to ensure complete archive availability
        end_date = (datetime.now(timezone.utc) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        
    logger.info("="*60)
    logger.info("     STARTING MLOPS FEATURE STORE HISTORICAL BACKFILL")
    logger.info(f"     Coordinates: lat={settings.LATITUDE}, lon={settings.LONGITUDE} ({settings.CITY})")
    logger.info(f"     Target Window: [{start_date}] -> [{end_date}]")
    logger.info("="*60)
    
    try:
        # 1. Fetch meteorological archive
        df_weather = fetch_historical_weather(start_date, end_date)
        
        # 2. Fetch air quality logs
        df_aq = fetch_historical_air_quality(start_date, end_date)
        
        if df_weather.empty or df_aq.empty:
            logger.error("Fetched historical datasets were empty. Backfill aborted.")
            return False
            
        # 3. Merge weather and air quality on time
        logger.info("Merging datasets and standardizing columns...")
        df_merged = pd.merge(df_weather, df_aq, on="time", how="inner")
        
        # 4. Standardize timezone and structure
        # Parse Islamabad timezone (Asia/Karachi) and convert to UTC ISO format strings
        df_merged["timestamp"] = pd.to_datetime(df_merged["time"]).dt.tz_localize("Asia/Karachi").dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df_merged["location"] = settings.CITY
        df_merged["latitude"] = settings.LATITUDE
        df_merged["longitude"] = settings.LONGITUDE
        
        # 5. Dynamically calculate target US EPA AQI
        df_merged["aqi"] = df_merged["pm2_5"].apply(calculate_pm25_aqi)
        
        # Drop temporary 'time' column
        df_merged = df_merged.drop(columns=["time"])
        
        # 6. Feed historical dataframe through feature pipeline (training mode generates forecasting targets)
        df_engineered = engineer_features(df_merged, is_training=True)
        
        # 7. Convert NaN target forecasts (occurring at the end of the history) into null for MongoDB
        # Pandas NaN represents as Float('nan') which MongoDB saves as NaN. We replace with None for clean JSON.
        df_engineered = df_engineered.replace({float('nan'): None, np.nan: None})
        
        # 8. Convert to dict list and seed MongoDB Atlas
        records = df_engineered.to_dict(orient="records")
        
        logger.info(f"Writing {len(records)} engineered features to MongoDB Atlas...")
        written_count = insert_records(records)
        
        logger.info("="*60)
        logger.info(f"  [SUCCESS] Seeding complete! {written_count} records inserted/upserted.")
        logger.info("  >>> FEATURE STORE HISTORICAL BACKFILL: 100% OPERATIONAL <<<")
        logger.info("="*60)
        return True
        
    except Exception as e:
        logger.critical(f"Historical backfill operation crashed: {e}")
        raise e

if __name__ == "__main__":
    # Test-run backfiller over a short 2-week slice to verify complete pipeline sanity
    # This prevents hitting API rate limits during local driver testing
    test_start = "2026-05-01"
    test_end = "2026-05-14"
    run_backfill(start_date=test_start, end_date=test_end)
