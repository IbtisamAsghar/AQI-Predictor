import sys
from pathlib import Path

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import time
import requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.config import settings
from src.utils.logging import get_logger

logger = get_logger("ingestion_engine")

def calculate_pm25_aqi(pm25: float) -> float:
    """Calculates the US EPA AQI value from a raw PM2.5 concentration (ug/m3)."""
    if pm25 < 0:
        return 0.0
    
    # Standard EPA AQI breakpoints for PM2.5: (C_low, C_high, I_low, I_high)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    
    for c_l, c_h, i_l, i_h in breakpoints:
        if c_l <= pm25 <= c_h:
            aqi = ((i_h - i_l) / (c_h - c_l)) * (pm25 - c_l) + i_l
            return round(aqi, 1)
            
    return 500.0  # Cap at maximum US EPA AQI level if hazardous

def fetch_open_meteo() -> Optional[Dict[str, Any]]:
    """Fetches real-time weather & air quality data from Open-Meteo APIs."""
    logger.info("Attempting to ingest meteorological and pollutant data from Open-Meteo...")
    
    # 1. Query Air Quality API
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": settings.LATITUDE,
        "longitude": settings.LONGITUDE,
        "current": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "timezone": "Asia/Karachi"
    }
    
    # 2. Query Weather Forecast API
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": settings.LATITUDE,
        "longitude": settings.LONGITUDE,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Karachi"
    }
    
    try:
        # Air Quality Request
        aq_response = requests.get(aq_url, params=aq_params, timeout=10)
        aq_response.raise_for_status()
        aq_data = aq_response.json().get("current", {})
        
        # Weather Request
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        weather_response.raise_for_status()
        weather_data = weather_response.json().get("current", {})
        
        # Parse timestamp from API (or fallback to current UTC)
        api_time = aq_data.get("time") or weather_data.get("time")
        if api_time:
            # Parse ISO time (e.g. 2026-05-21T14:00) and localize as UTC
            timestamp = datetime.strptime(api_time, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).isoformat()
        else:
            timestamp = datetime.now(timezone.utc).isoformat()
            
        pm2_5 = float(aq_data.get("pm2_5", 0.0))
        
        merged_data = {
            "timestamp": timestamp,
            "location": settings.CITY,
            "latitude": settings.LATITUDE,
            "longitude": settings.LONGITUDE,
            "pm2_5": pm2_5,
            "pm10": float(aq_data.get("pm10", 0.0)),
            "no2": float(aq_data.get("nitrogen_dioxide", 0.0)),
            "so2": float(aq_data.get("sulphur_dioxide", 0.0)),
            "co": float(aq_data.get("carbon_monoxide", 0.0)),
            "o3": float(aq_data.get("ozone", 0.0)),
            "temperature": float(weather_data.get("temperature_2m", 25.0)),
            "humidity": float(weather_data.get("relative_humidity_2m", 50.0)),
            "wind_speed": float(weather_data.get("wind_speed_10m", 0.0)),
            "wind_direction": float(weather_data.get("wind_direction_10m", 0.0)),
            "aqi": calculate_pm25_aqi(pm2_5)
        }
        
        logger.info("Successfully fetched data from Open-Meteo APIs.")
        return merged_data
        
    except Exception as e:
        logger.warning(f"Failed to ingest from Open-Meteo APIs: {e}")
        return None

def fetch_aqicn_fallback() -> Optional[Dict[str, Any]]:
    """Fallback fetcher querying AQICN API if Open-Meteo fails."""
    if not settings.AQICN_TOKEN:
        logger.error("AQICN fallback requested but AQICN_TOKEN is empty in environment.")
        return None
        
    logger.info("Executing fallback data ingestion via AQICN JSON API...")
    url = f"https://api.waqi.info/feed/geo:{settings.LATITUDE};{settings.LONGITUDE}/"
    params = {"token": settings.AQICN_TOKEN}
    
    try:
        response = requests.get(url, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
        
        if payload.get("status") != "ok":
            logger.error(f"AQICN API returned an error status: {payload.get('data')}")
            return None
            
        data = payload.get("data", {})
        iaqi = data.get("iaqi", {})
        
        # AQICN yields parameters directly in iaqi format as {'v': value}
        pm2_5 = float(iaqi.get("pm25", {}).get("v", 0.0))
        
        # Parse API timestamp, fallback to UTC
        s_time = data.get("time", {}).get("iso")
        if s_time:
            # Parse standard ISO timestamp
            timestamp = datetime.fromisoformat(s_time.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        else:
            timestamp = datetime.now(timezone.utc).isoformat()
            
        merged_data = {
            "timestamp": timestamp,
            "location": settings.CITY,
            "latitude": settings.LATITUDE,
            "longitude": settings.LONGITUDE,
            "pm2_5": pm2_5,
            "pm10": float(iaqi.get("pm10", {}).get("v", 0.0)),
            "no2": float(iaqi.get("no2", {}).get("v", 0.0)),
            "so2": float(iaqi.get("so2", {}).get("v", 0.0)),
            "co": float(iaqi.get("co", {}).get("v", 0.0)),
            "o3": float(iaqi.get("o3", {}).get("v", 0.0)),
            "temperature": float(iaqi.get("t", {}).get("v", 25.0)),
            "humidity": float(iaqi.get("h", {}).get("v", 50.0)),
            "wind_speed": float(iaqi.get("w", {}).get("v", 0.0)) * 3.6,  # Convert m/s to km/h to match Open-Meteo
            "wind_direction": float(iaqi.get("wd", {}).get("v", 0.0)),
            "aqi": calculate_pm25_aqi(pm2_5)
        }
        
        logger.info("Successfully fetched fallback data from AQICN API.")
        return merged_data
        
    except Exception as e:
        logger.error(f"AQICN Fallback API request failed: {e}")
        return None

def ingest_data() -> Dict[str, Any]:
    """Unified entrypoint that runs data ingestion with automated fallback and retries."""
    retries = 3
    delay = 2
    
    for attempt in range(1, retries + 1):
        data = fetch_open_meteo()
        if data:
            return data
            
        logger.warning(f"Ingestion attempt {attempt} failed. Retrying in {delay} seconds...")
        time.sleep(delay)
        delay *= 2
        
    # If primary fails, trigger fallback
    fallback_data = fetch_aqicn_fallback()
    if fallback_data:
        return fallback_data
        
    raise RuntimeError("Ingestion failed: Both primary (Open-Meteo) and fallback (AQICN) APIs are unreachable.")

if __name__ == "__main__":
    # Self-test validation
    try:
        record = ingest_data()
        print("\n" + "="*60)
        print("          PEARLS AQI PREDICTOR - INGESTION TEST RUN")
        print("="*60)
        for key, value in record.items():
            print(f"  {key:<16}: {value}")
        print("="*60)
        print("  >>> INGESTION TEST STATUS: 100% OPERATIONAL <<<\n")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Ingestion Self-test failed: {e}")
