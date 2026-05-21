import sys
from pathlib import Path

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from src.data.ingest import calculate_pm25_aqi, ingest_data, fetch_open_meteo

def test_calculate_pm25_aqi():
    """Asserts EPA AQI calculations align precisely with US EPA mathematical standards."""
    # Boundary 1: PM2.5 = 0.0 ug/m3 -> AQI = 0
    assert calculate_pm25_aqi(0.0) == 0.0
    
    # Boundary 2: PM2.5 = 12.0 ug/m3 -> AQI = 50
    assert calculate_pm25_aqi(12.0) == 50.0
    
    # Boundary 3: PM2.5 = 35.4 ug/m3 -> AQI = 100
    assert calculate_pm25_aqi(35.4) == 100.0
    
    # Boundary 4: Extreme hazardous -> caps at 500
    assert calculate_pm25_aqi(600.0) == 500.0

def test_fetch_open_meteo_schema():
    """Asserts live Open-Meteo returns map exactly to our operational dictionary schemas."""
    data = fetch_open_meteo()
    if data is None:
        pytest.skip("Open-Meteo API is currently offline or local DNS resolution failed. Skipping live schema check.")
        
    required_keys = [
        "timestamp", "location", "latitude", "longitude",
        "pm2_5", "pm10", "no2", "so2", "co", "o3",
        "temperature", "humidity", "wind_speed", "wind_direction", "aqi"
    ]
    
    for key in required_keys:
        assert key in data, f"Required key '{key}' was missing from Open-Meteo schema!"
        assert data[key] is not None, f"Key '{key}' was returned as None!"
        
    assert data["location"] == "Islamabad"
    assert isinstance(data["aqi"], (int, float))

def test_ingest_data_fallback_resilience():
    """Asserts unified ingestion triggers successfully, skipping if offline."""
    try:
        record = ingest_data()
        assert isinstance(record, dict)
        assert record["pm2_5"] >= 0.0
        assert 0.0 <= record["aqi"] <= 500.0
    except RuntimeError as re:
        if "Ingestion failed" in str(re):
            pytest.skip("Both Open-Meteo and AQICN fallback are offline/unreachable due to network connection. Skipping live validation.")
        else:
            raise re
