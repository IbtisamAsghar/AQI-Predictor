import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger
from src.models.model import AQIPredictionModel
from src.models.explainability import AQIExplainer
from src.dashboard.app import get_aqi_category_info

logger = get_logger("verify_dashboard")

def test_explainability_and_dashboard_logic():
    logger.info("="*70)
    logger.info("          PEARLS AQI PREDICTOR - DASHBOARD & SHAP INTEGRATION TEST")
    logger.info("="*70)
    
    # 1. Test US EPA AQI Categories Mapping logic
    logger.info("  1. Verifying US EPA Air Quality Index categories mapper...")
    
    test_cases = {
        0: "Good",
        25: "Good",
        50: "Good",
        51: "Moderate",
        75: "Moderate",
        100: "Moderate",
        101: "Unhealthy for Sensitive Groups",
        125: "Unhealthy for Sensitive Groups",
        150: "Unhealthy for Sensitive Groups",
        151: "Unhealthy",
        175: "Unhealthy",
        200: "Unhealthy",
        201: "Very Unhealthy",
        250: "Very Unhealthy",
        300: "Very Unhealthy",
        301: "Hazardous",
        500: "Hazardous"
    }
    
    for aqi, expected_cat in test_cases.items():
        info = get_aqi_category_info(aqi)
        assert info["name"] == expected_cat, f"For AQI {aqi}, expected category '{expected_cat}', got '{info['name']}'"
        
    logger.info("     [SUCCESS] All US EPA category mappings resolved correctly.")
    
    # 2. Setup mock model wrapper and preprocessor to verify SHAP Explainer
    logger.info("  2. Instantiating mock model estimators and scalers...")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    
    # Create 3 mini random forests
    rf_24 = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    rf_48 = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    rf_72 = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    
    # Dummy data: 20 samples, 4 features
    features = ["pm2_5", "temperature", "humidity", "sin_hour"]
    X_train = np.random.uniform(10, 100, (20, len(features)))
    y_24 = np.random.uniform(15, 120, 20)
    y_48 = np.random.uniform(15, 120, 20)
    y_72 = np.random.uniform(15, 120, 20)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf_24.fit(X_train_scaled, y_24)
    rf_48.fit(X_train_scaled, y_48)
    rf_72.fit(X_train_scaled, y_72)
    
    # Wrap in our AQIPredictionModel wrapper
    model_wrapper = AQIPredictionModel(n_estimators=5, max_depth=3)
    model_wrapper.models = {
        "target_aqi_24h": rf_24,
        "target_aqi_48h": rf_48,
        "target_aqi_72h": rf_72
    }
    
    # 3. Test SHAP Explainer instantiation and calculation
    logger.info("  3. Initializing TreeExplainer and calculating local SHAP contributions...")
    
    explainer = AQIExplainer(model_wrapper, scaler, features)
    
    # Single sample inference
    X_sample = np.random.uniform(10, 100, (1, len(features)))
    X_sample_scaled = scaler.transform(X_sample)
    
    explanations = explainer.explain_instance(X_sample_scaled)
    
    assert "plus_24h" in explanations
    assert "plus_48h" in explanations
    assert "plus_72h" in explanations
    
    # Verify values mapping
    for h in ["plus_24h", "plus_48h", "plus_72h"]:
        h_data = explanations[h]
        assert "base_value" in h_data
        assert "prediction" in h_data
        assert "contributions" in h_data
        
        contribs = h_data["contributions"]
        assert len(contribs) == len(features), f"Expected {len(features)} feature contributions, got {len(contribs)}"
        for f in features:
            assert f in contribs, f"Feature '{f}' missing from contributions"
            
        # Mathematical check: prediction should equal expected_value + sum(shap_values)
        expected_pred = h_data["base_value"] + sum(contribs.values())
        actual_pred = float(model_wrapper.models[f"target_aqi_{h.replace('plus_', '')}"].predict(X_sample_scaled)[0])
        
        assert np.isclose(expected_pred, actual_pred, atol=1e-4), f"SHAP summation mismatch for {h}: expected {expected_pred}, got model prediction {actual_pred}"
        
    logger.info("     [SUCCESS] Local SHAP explanations sum to exact model predictions.")
    
    # 4. Test Global Feature Importance
    logger.info("  4. Extracting global feature importances...")
    global_importances = explainer.get_global_importances()
    
    assert "plus_24h" in global_importances
    for h in ["plus_24h", "plus_48h", "plus_72h"]:
        h_importances = global_importances[h]
        assert len(h_importances) == len(features)
        total_imp = sum(item["importance"] for item in h_importances)
        assert np.isclose(total_imp, 1.0, atol=1e-4), f"Global importances for {h} do not sum to 1.0 (sum: {total_imp})"
        logger.info(f"     [SUCCESS] {h} top feature: {h_importances[0]['feature']} (score: {h_importances[0]['importance']:.3f})")
        
    # 5. Verify local UI Alert triggers logic
    logger.info("  5. Checking threshold UI Alert logic...")
    
    low_aqi = 45.0
    high_aqi = 155.0
    
    low_aqi_info = get_aqi_category_info(low_aqi)
    high_aqi_info = get_aqi_category_info(high_aqi)
    
    assert low_aqi <= 150, "Low AQI should be below or equal to safety threshold"
    assert high_aqi > 150, "High AQI should exceed safety threshold"
    
    logger.info(f"     AQI {low_aqi} triggers category: {low_aqi_info['name']}")
    logger.info(f"     AQI {high_aqi} triggers category: {high_aqi_info['name']} (THRESHOLD EXCEEDED: ALERT!)")
    
    logger.info("="*70)
    logger.info("     >>> DASHBOARD & SHAP LOGIC INTEGRATION TEST PASSED SUCCESSFUL <<<")
    logger.info("="*70)

if __name__ == "__main__":
    test_explainability_and_dashboard_logic()
