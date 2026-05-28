import sys
import os
import requests
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from huggingface_hub import hf_hub_download
from src.config import settings
from src.utils.database import get_db
from src.utils.logging import get_logger
from src.models.model import AQIPredictionModel
from src.models.explainability import AQIExplainer

# Setup logging
logger = get_logger("dashboard")

# Streamlit Page Configurations

# Helper function to map AQI values to official US EPA categories
def get_aqi_category_info(aqi: float) -> dict:
    if aqi <= 50:
        return {"name": "Good", "color": "#10b981", "pulse": "pulse-good", "desc": "Air quality is satisfactory, and air pollution poses little or no risk.", "guideline": "Enjoy outdoor activities. No precautions are necessary."}
    elif aqi <= 100:
        return {"name": "Moderate", "color": "#f59e0b", "pulse": "pulse-moderate", "desc": "Air quality is acceptable. However, there may be a risk for some people.", "guideline": "Sensitive individuals should consider reducing prolonged or heavy outdoor exertion."}
    elif aqi <= 150:
        return {"name": "Unhealthy for Sensitive Groups", "color": "#f97316", "pulse": "pulse-sensitive", "desc": "Members of sensitive groups may experience health effects.", "guideline": "Sensitive groups (children, elderly, asthmatics) should limit prolonged outdoor exertion."}
    elif aqi <= 200:
        return {"name": "Unhealthy", "color": "#ef4444", "pulse": "pulse-unhealthy", "desc": "Everyone may begin to experience health effects.", "guideline": "Active adults and children should limit prolonged outdoor exertion. Sensitive groups should avoid outdoor activity."}
    elif aqi <= 300:
        return {"name": "Very Unhealthy", "color": "#8b5cf6", "pulse": "pulse-veryunhealthy", "desc": "Health alert: everyone may experience more serious health effects.", "guideline": "Avoid all outdoor physical activity. Keep windows closed and run air purifiers."}
    else:
        return {"name": "Hazardous", "color": "#7f1d1d", "pulse": "pulse-hazardous", "desc": "Health warning of emergency conditions: everyone is more likely to be affected.", "guideline": "STAY INDOORS. Keep indoor air clean. Wear N95 masks if outdoor travel is mandatory."}

# REST Client predictions fetcher
def fetch_predictions_from_api() -> dict:
    """Attempts to query predictions from the FastAPI REST endpoint."""
    try:
        url = f"{settings.API_URL}/predict/live"
        response = requests.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            data["source"] = "FastAPI REST API"
            return data
    except Exception as e:
        logger.warning(f"Failed to query FastAPI REST API at {settings.API_URL}: {e}")
    return None

# Local prediction engine fallback when REST service is offline
@st.cache_resource(ttl=3600)  # Cache local loading for 1 hour to prevent redundant DB reads
def load_champion_artifacts_locally():
    """Fallback method to connect to MongoDB Atlas and Hugging Face Hub directly to serve predictions."""
    logger.info("REST API offline. Resolving champion model version locally from MongoDB Atlas...")
    db = get_db()
    collection = db["model_registry"]
    
    champion_doc = collection.find_one({"is_champion": True})
    if not champion_doc:
        raise RuntimeError("No model is currently promoted as 'is_champion = True' in model registry collection.")
        
    version = champion_doc["version"]
    repo_id = champion_doc.get("hf_repo_id", settings.HF_REPO_ID)
    
    logger.info(f"Local Fallback Resolved: Version={version} from HF Repository={repo_id}")
    
    # Download weights
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=f"versions/{version}/model.pkl", token=settings.HF_TOKEN)
        preprocessor_path = hf_hub_download(repo_id=repo_id, filename=f"versions/{version}/preprocessor.pkl", token=settings.HF_TOKEN)
        features_path = hf_hub_download(repo_id=repo_id, filename=f"versions/{version}/features.pkl", token=settings.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not download versioned files ({version}). Downloading root fallback path: {e}")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl", token=settings.HF_TOKEN)
        preprocessor_path = hf_hub_download(repo_id=repo_id, filename="preprocessor.pkl", token=settings.HF_TOKEN)
        features_path = hf_hub_download(repo_id=repo_id, filename="features.pkl", token=settings.HF_TOKEN)
        
    # Deserialize
    model_wrapper = AQIPredictionModel.load(model_path)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(features_path, "rb") as f:
        active_features = pickle.load(f)
        
    return model_wrapper, preprocessor, active_features, champion_doc

def compute_predictions_locally() -> dict:
    """Retrieves newest feature store record and runs predictions locally using cached weights."""
    model_wrapper, preprocessor, active_features, champion_doc = load_champion_artifacts_locally()
    
    db = get_db()
    latest_record = db["features_hourly"].find_one(
        {"location": settings.CITY},
        sort=[("timestamp", -1)]
    )
    
    if not latest_record:
        raise ValueError(f"No features found in MongoDB collection 'features_hourly' for location: {settings.CITY}")
        
    # Align features
    feature_vector = []
    for f in active_features:
        if f not in latest_record:
            raise KeyError(f"Feature Store record is missing active schema feature: '{f}'")
        feature_vector.append(latest_record[f])
        
    # Preprocess & Predict
    X = np.array([feature_vector])
    X_scaled = preprocessor.transform(X)
    preds = model_wrapper.predict(X_scaled)
    
    current_aqi = latest_record.get("aqi")
    aqi_24 = max(0.0, float(preds["pred_aqi_24h"][0]))
    aqi_48 = max(0.0, float(preds["pred_aqi_48h"][0]))
    aqi_72 = max(0.0, float(preds["pred_aqi_72h"][0]))
    
    # Resolve health categories
    def get_cat(v):
        return get_aqi_category_info(v)["name"]
        
    return {
        "location": latest_record.get("location"),
        "timestamp": latest_record.get("timestamp"),
        "current_aqi": current_aqi,
        "current_health_status": get_cat(current_aqi) if current_aqi is not None else "Unknown",
        "predictions": {
            "plus_24h": {"aqi": round(aqi_24, 2), "status": get_cat(aqi_24)},
            "plus_48h": {"aqi": round(aqi_48, 2), "status": get_cat(aqi_48)},
            "plus_72h": {"aqi": round(aqi_72, 2), "status": get_cat(aqi_72)}
        },
        "model_version": champion_doc.get("version", "N/A"),
        "source": "Local Fallback (Direct Connection)",
        "features": feature_vector,
        "feature_names": active_features,
        "model_wrapper": model_wrapper,
        "preprocessor": preprocessor
    }

def main():
    st.set_page_config(
        page_title="Pearls AQI Forecaster & MLOps Dashboard",
        page_icon="💨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom premium dark-slate styling with glassmorphism CSS
    st.markdown("""
    <style>
        /* Base theme override */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Transparent and Blur Glassmorphic Cards */
        .glass-card {
            background: rgba(17, 24, 39, 0.7);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        .status-header {
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        
        /* Ring pulsing animation for risk bands */
        .pulsing-ring {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        
        .pulse-good { background: #10b981; animation: pulse 2s infinite; }
        .pulse-moderate { background: #f59e0b; animation: pulse 2s infinite; }
        .pulse-sensitive { background: #f97316; animation: pulse 2s infinite; }
        .pulse-unhealthy { background: #ef4444; animation: pulse 2s infinite; }
        .pulse-veryunhealthy { background: #8b5cf6; animation: pulse 2s infinite; }
        .pulse-hazardous { background: #7f1d1d; animation: pulse 2s infinite; }
        
        /* Styled metric tiles */
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 4px;
        }
        
        /* Title glow effect */
        .title-glow {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
            margin-bottom: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ----------------- SIDEBAR CONTROLS -----------------
    st.sidebar.markdown(f"<div style='text-align: center; padding: 10px;'><h2 style='color:#60a5fa;'>MLOps Controller</h2></div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")

    # Quick connection diagnostics widget
    connection_mode = st.sidebar.empty()
    api_status_tile = st.sidebar.empty()

    # Refresh triggers
    if st.sidebar.button("🔄 Refresh Live Forecasts", use_container_width=True):
        st.cache_resource.clear()

    st.sidebar.markdown("### Location Settings")
    st.sidebar.info(f"📍 **City**: {settings.CITY}\n🌐 **Lat**: {settings.LATITUDE}\n🌐 **Lon**: {settings.LONGITUDE}")

    st.sidebar.markdown("### US EPA Hazard Bands")
    for band, props in {
        "0-50": {"name": "Good", "color": "#10b981"},
        "51-100": {"name": "Moderate", "color": "#f59e0b"},
        "101-150": {"name": "Unhealthy for Sensitive", "color": "#f97316"},
        "151-200": {"name": "Unhealthy", "color": "#ef4444"},
        "201-300": {"name": "Very Unhealthy", "color": "#8b5cf6"},
        "301+": {"name": "Hazardous", "color": "#7f1d1d"}
    }.items():
        st.sidebar.markdown(
            f"<span style='color:{props['color']}; font-weight:bold;'>■</span> **{band}**: {props['name']}",
            unsafe_allow_html=True
        )

    # ----------------- INGEST INFERENCE METRICS -----------------
    import time
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Checkpoint 1
    status_text.markdown("🔄 **Step 1/4 (25%)**: Pinging FastAPI serving registry endpoint...")
    progress_bar.progress(25)
    time.sleep(0.3)
    
    # Try REST API first
    pred_payload = fetch_predictions_from_api()
    
    if pred_payload:
        # Checkpoint 2
        status_text.markdown("🔌 **Step 2/4 (50%)**: Connected to REST API. Retrieving live forecasts...")
        progress_bar.progress(50)
        time.sleep(0.3)
        
        # Checkpoint 3
        status_text.markdown("🛰️ **Step 3/4 (75%)**: Unpacking multi-horizon predictions & EPA hazard bands...")
        progress_bar.progress(75)
        time.sleep(0.3)
        
        connection_mode.success("🔌 Connected to FastAPI REST API")
        api_status_tile.metric("Serving Mode", "REST API (Port 7860)")
    else:
        # FastAPI is offline, trigger local database & Hugging Face Hub direct engine fallback
        status_text.warning("⚠️ FastAPI serving offline. Activating direct DB/Registry local fallback...")
        time.sleep(0.5)
        
        # Checkpoint 2
        status_text.markdown("🔄 **Step 2/4 (50%)**: Connecting to MongoDB Atlas & resolving champion registry version...")
        progress_bar.progress(50)
        
        try:
            pred_payload = compute_predictions_locally()
            
            # Checkpoint 3
            status_text.markdown("🛰️ **Step 3/4 (75%)**: Downloading champion model weights from Hugging Face Hub LFS...")
            progress_bar.progress(75)
            time.sleep(0.4)
            
            connection_mode.warning("⚠️ FastAPI Offline - Using Local Fallback")
            api_status_tile.metric("Serving Mode", "Direct Local Engine")
        except Exception as ex:
            status_text.empty()
            progress_bar.empty()
            st.error(f"Failed to fetch predictions from both REST API and Local database fallback: {ex}")
            st.stop()
            
    # Checkpoint 4
    status_text.success("📈 **Step 4/4 (100%)**: Rendering interactive Plotly MLOps dashboard...")
    progress_bar.progress(100)
    time.sleep(0.4)
    
    # Clean up loading widgets for visual excellence
    status_text.empty()
    progress_bar.empty()

    # Header Display
    col_header_title, col_header_source = st.columns([4, 1])
    with col_header_title:
        st.markdown("<h1 class='title-glow'>Pearls AQI Forecasting System</h1>", unsafe_allow_html=True)
        st.markdown(f"**Real-time MLOps Dashboard for {settings.CITY}, Pakistan** | Model Version: `{pred_payload.get('model_version')}`")
    with col_header_source:
        st.markdown(
            f"<div style='text-align:right; margin-top:20px;'>"
            f"<span style='background:rgba(59,130,246,0.2); border:1px solid #3b82f6; padding:6px 12px; border-radius:20px; font-size:0.85rem; font-weight:600; color:#93c5fd;'>"
            f"Data Source: {pred_payload.get('source')}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Main Content Layout: Tabs
    tab_overview, tab_forecast, tab_shap, tab_history = st.tabs([
        "📊 Real-time Overview",
        "📈 3-Day Forecast Visuals",
        "🧠 Explainable AI (SHAP)",
        "🕰️ Historical Insights (EDA)"
    ])

    # ----------------- TAB 1: OVERVIEW -----------------
    with tab_overview:
        current_aqi = pred_payload.get("current_aqi")

        if current_aqi is not None:
            cat_info = get_aqi_category_info(current_aqi)

            # UI alert threshold warning (>150 AQI)
            if current_aqi > 150:
                st.markdown(
                    f"<div style='background:rgba(239,68,68,0.15); border:1px solid #ef4444; border-radius:12px; padding:16px; margin-bottom:24px; color:#fca5a5;'>"
                    f"🚨 <strong style='color:#ffffff;'>CRITICAL PUBLIC HEALTH WARNING:</strong> The current air quality index ({current_aqi}) "
                    f"has crossed the safety threshold of 150 (Category: {cat_info['name']}). Active adults, children, and sensitive individuals "
                    f"should avoid all outdoor exertion. Keep windows closed and filter indoor air.</div>",
                    unsafe_allow_html=True
                )

            col_widget_left, col_widget_right = st.columns([1, 1])

            with col_widget_left:
                st.markdown(
                    f"<div class='glass-card' style='text-align: center; border-left: 6px solid {cat_info['color']};'>"
                    f"<div class='status-header'>CURRENT AIR QUALITY</div>"
                    f"<div style='font-size: 5rem; font-weight: 900; line-height: 1; color:#ffffff;'>{current_aqi:.0f}</div>"
                    f"<div style='margin-top: 15px; font-size: 1.5rem; font-weight: 700; color:{cat_info['color']};'>"
                    f"<span class='pulsing-ring {cat_info['pulse']}'></span>{cat_info['name']}</div>"
                    f"<div style='margin-top: 15px; color:#94a3b8; font-size: 0.95rem; font-weight: 500;'>{cat_info['desc']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with col_widget_right:
                st.markdown(
                    f"<div class='glass-card' style='height: 100%;'>"
                    f"<div class='status-header'>RECOMMENDED HEALTH GUIDELINES</div>"
                    f"<div style='font-size: 1.15rem; font-weight:600; color:#38bdf8; margin-bottom:12px;'>Action Protocol:</div>"
                    f"<p style='font-size: 1.05rem; line-height:1.6; color:#e2e8f0;'>{cat_info['guideline']}</p>"
                    f"<div style='margin-top: 25px; border-top:1px solid rgba(255,255,255,0.05); padding-top:15px; font-size:0.85rem; color:#64748b;'>"
                    f"Measurements synchronized for coordinates: {settings.LATITUDE}°N, {settings.LONGITUDE}°E.<br/>"
                    f"Timestamp: {pred_payload.get('timestamp')}"
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("Current AQI measurement is not available.")

        # Display weather features list
        st.markdown("### Core Meteorological Indicators")
        col_feat1, col_feat2, col_feat3, col_feat4 = st.columns(4)

        # Query database to grab the exact latest raw weather values
        db = get_db()
        latest_raw = db["features_hourly"].find_one(
            {"location": settings.CITY},
            sort=[("timestamp", -1)]
        )

        if latest_raw:
            with col_feat1:
                st.markdown(
                    f"<div class='glass-card' style='text-align:center; padding:15px;'>"
                    f"<div class='status-header'>Temperature</div>"
                    f"<div class='metric-value'>{latest_raw.get('temperature', 0.0):.1f}°C</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_feat2:
                st.markdown(
                    f"<div class='glass-card' style='text-align:center; padding:15px;'>"
                    f"<div class='status-header'>Relative Humidity</div>"
                    f"<div class='metric-value'>{latest_raw.get('humidity', 0.0):.0f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_feat3:
                st.markdown(
                    f"<div class='glass-card' style='text-align:center; padding:15px;'>"
                    f"<div class='status-header'>Wind Speed</div>"
                    f"<div class='metric-value'>{latest_raw.get('wind_speed', 0.0):.1f} km/h</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col_feat4:
                st.markdown(
                    f"<div class='glass-card' style='text-align:center; padding:15px;'>"
                    f"<div class='status-header'>Particulate PM2.5</div>"
                    f"<div class='metric-value'>{latest_raw.get('pm2_5', 0.0):.1f} µg/m³</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ----------------- TAB 2: FORECASTS -----------------
    with tab_forecast:
        st.markdown("### Multi-Horizon Forecast Projection")

        predictions = pred_payload.get("predictions", {})
        aqi_24 = predictions.get("plus_24h", {}).get("aqi")
        aqi_48 = predictions.get("plus_48h", {}).get("aqi")
        aqi_72 = predictions.get("plus_72h", {}).get("aqi")

        if current_aqi is not None and aqi_24 is not None:
            # Forecast alerts trigger
            max_forecasted_aqi = max(aqi_24, aqi_48, aqi_72)
            if max_forecasted_aqi > 150:
                st.markdown(
                    f"<div style='background:rgba(245,158,11,0.12); border:1px solid #f59e0b; border-radius:12px; padding:16px; margin-bottom:24px; color:#fef08a;'>"
                    f"⚠️ <strong style='color:#ffffff;'>CRITICAL EXPOSURE DETECTED:</strong> Forecast models project a major surge in "
                    f"pollution reaching {max_forecasted_aqi:.1f} AQI within the next 3 days. Recommend pre-planning sensitive/outdoor "
                    f"excursions around peak pollution intervals.</div>",
                    unsafe_allow_html=True
                )

            col_fc1, col_fc2, col_fc3 = st.columns(3)

            for col, horizon, val, status in zip(
                [col_fc1, col_fc2, col_fc3],
                ["Tomorrow (+24h)", "Day 2 (+48h)", "Day 3 (+72h)"],
                [aqi_24, aqi_48, aqi_72],
                [predictions["plus_24h"]["status"], predictions["plus_48h"]["status"], predictions["plus_72h"]["status"]]
            ):
                cat_info = get_aqi_category_info(val)
                with col:
                    st.markdown(
                        f"<div class='glass-card' style='text-align: center; border-bottom: 5px solid {cat_info['color']};'>"
                        f"<div class='status-header'>{horizon}</div>"
                        f"<div style='font-size: 3rem; font-weight: 800; color:#ffffff;'>{val:.0f}</div>"
                        f"<div style='margin-top: 5px; font-weight: 700; color:{cat_info['color']}; font-size:1.15rem;'>"
                        f"<span class='pulsing-ring {cat_info['pulse']}'></span>{cat_info['name']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # Interactive Plotly Forecast Trend Line
            fig = go.Figure()

            # Plotly layout annotations / shapes for EPA bands
            fig.add_trace(go.Scatter(
                x=["Current", "24-Hours", "48-Hours", "72-Hours"],
                y=[current_aqi, aqi_24, aqi_48, aqi_72],
                mode="lines+markers+text",
                text=[f"{current_aqi:.0f}", f"{aqi_24:.0f}", f"{aqi_48:.0f}", f"{aqi_72:.0f}"],
                textposition="top center",
                textfont=dict(color='#ffffff', size=11, family="sans-serif"),
                name="Projected AQI Trend",
                line=dict(color="#3b82f6", width=4, shape="spline"),
                marker=dict(size=10, color="#60a5fa", line=dict(color="#1d4ed8", width=2))
            ))

            # Add beautiful shaded horizontal zones marking AQI bands
            fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.08, line_width=0, annotation_text="Good (0-50)", annotation_position="left", annotation_font_color="#10b981")
            fig.add_hrect(y0=50, y1=100, fillcolor="#f59e0b", opacity=0.08, line_width=0, annotation_text="Moderate (51-100)", annotation_position="left", annotation_font_color="#f59e0b")
            fig.add_hrect(y0=100, y1=150, fillcolor="#f97316", opacity=0.08, line_width=0, annotation_text="Sensitive (101-150)", annotation_position="left", annotation_font_color="#f97316")
            fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.08, line_width=0, annotation_text="Unhealthy (151-200)", annotation_position="left", annotation_font_color="#ef4444")
            fig.add_hrect(y0=200, y1=300, fillcolor="#8b5cf6", opacity=0.08, line_width=0, annotation_text="Very Unhealthy (201-300)", annotation_position="left", annotation_font_color="#8b5cf6")

            fig.update_layout(
                title="3-Day Forecast Chronology Trend vs EPA Limits",
                xaxis_title="Forecasting Horizon Steps",
                yaxis_title="US EPA Air Quality Index (AQI)",
                yaxis_range=[0, max(max_forecasted_aqi + 40, 160)],
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient forecast payload to render Plotly chart.")

    # ----------------- TAB 3: SHAP EXPLAINABILITY -----------------
    with tab_shap:
        st.markdown("### Explainable ML Insights (SHAP Value Indicators)")
        st.write("This panel dissects the exact drivers influencing your model's AQI predictions, explaining feature contribution weightings.")

        # Check if we can compute SHAP (either locally or through custom logic)
        # We will instantiate SHAP explainer here on the fly
        try:
            if "model_wrapper" not in pred_payload:
                # Re-load artifacts to fetch local variables
                model_wrapper, preprocessor, active_features, champion_doc = load_champion_artifacts_locally()
            else:
                model_wrapper = pred_payload["model_wrapper"]
                preprocessor = pred_payload["preprocessor"]
                active_features = pred_payload["feature_names"]

            # Get latest record feature vector
            db = get_db()
            latest_record = db["features_hourly"].find_one(
                {"location": settings.CITY},
                sort=[("timestamp", -1)]
            )

            if latest_record:
                feature_vector = []
                for f in active_features:
                    feature_vector.append(latest_record[f])

                X = np.array([feature_vector])
                X_scaled = preprocessor.transform(X)

                # Setup explainer
                explainer = AQIExplainer(model_wrapper, preprocessor, active_features)

                # Local explanation
                shap_progress = st.empty()
                with shap_progress.status("🧠 Computing local SHAP feature impacts...", expanded=False) as status:
                    explanations = explainer.explain_instance(X_scaled)
                    status.update(label="🧠 SHAP calculations completed!", state="complete", expanded=False)
                time.sleep(0.3)
                shap_progress.empty()

                horizon_select = st.selectbox(
                    "Select Forecast Step to Explain:",
                    ["plus_24h", "plus_48h", "plus_72h"]
                )

                exp_data = explanations.get(horizon_select, {})
                contributions = exp_data.get("contributions", {})
                base_val = exp_data.get("base_value", 0.0)
                prediction_val = exp_data.get("prediction", 0.0)

                if contributions:
                    st.write(f"**Reigning Base Bias (Average Train Target)**: `{base_val:.2f}` AQI")
                    st.write(f"**Predicted Target Forecast (Base + SHAP sum)**: `{prediction_val:.2f}` AQI")

                    # Sort contributions
                    sorted_attrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                    features_plot = [x[0] for x in sorted_attrib]
                    values_plot = [x[1] for x in sorted_attrib]
                    colors_plot = ["#ef4444" if val > 0 else "#10b981" for val in values_plot]

                    fig_shap = go.Figure(go.Bar(
                        x=values_plot,
                        y=features_plot,
                        orientation="h",
                        marker_color=colors_plot
                    ))

                    fig_shap.update_layout(
                        title=f"Local SHAP Feature Contributions Impacting Prediction ({horizon_select})",
                        xaxis_title="SHAP Value Contribution Weighting (AQI +/-)",
                        yaxis_title="Engineered Input Feature",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(autorange="reversed")
                    )

                    st.plotly_chart(fig_shap, use_container_width=True)
                    st.info("🔴 Crimson Red bars show features raising the AQI forecast. 🟢 Green bars show features lowering the AQI forecast.")
                else:
                    st.error("No SHAP contributions computed. Falling back to global importances.")

                # Global Importance Fallback Chart
                st.markdown("---")
                st.markdown("### Global Model Feature Importance (All Targets)")
                global_imp = explainer.get_global_importances()

                # Select target
                horizon_imp_select = st.selectbox(
                    "Select Target Step for Global Feature Importance:",
                    ["plus_24h", "plus_48h", "plus_72h"],
                    key="global_imp_select"
                )

                features_imp = global_imp.get(horizon_imp_select, [])
                if features_imp:
                    df_imp = pd.DataFrame(features_imp).head(15)  # top 15

                    fig_imp = px.bar(
                        df_imp,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title=f"Top 15 Global Random Forest Feature Importances for {horizon_imp_select}"
                    )

                    fig_imp.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="Relative Feature Importance Score",
                        yaxis_title="Feature Name",
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Could not fetch latest feature record from Feature Store for explanations.")
        except Exception as shap_ex:
            st.error(f"Failed to load SHAP explainers: {shap_ex}")

    # ----------------- TAB 4: HISTORICAL EDA -----------------
    with tab_history:
        st.markdown("### Historical Trends and Exploratory Insights")
        st.write("Analyzes past air quality records stored inside the MongoDB Atlas Feature Store database.")

        # Days selector
        days_range = st.slider("Select Historical Window Timeline:", min_value=7, max_value=90, value=30, step=7)

        # Fetch historical data
        db = get_db()
        collection = db["features_hourly"]

        start_time = datetime.now(timezone.utc) - timedelta(days=days_range)
        cursor = collection.find(
            {"location": settings.CITY, "timestamp": {"$gte": start_time.isoformat()}},
            {"_id": 0}
        ).sort("timestamp", 1)

        df_history = pd.DataFrame(list(cursor))

        if not df_history.empty:
            df_history["timestamp"] = pd.to_datetime(df_history["timestamp"])
            st.write(f"Loaded `{df_history.shape[0]}` hourly records inside the target window.")

            # Chronological rolling plot
            fig_hist = px.line(
                df_history,
                x="timestamp",
                y=["aqi", "pm2_5"],
                title=f"Islamabad Chronological AQI & PM2.5 Trends (Last {days_range} Days)"
            )
            fig_hist.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Date Timeline",
                yaxis_title="Index / Concentration (µg/m³)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Diurnal Cycle
            df_history["hour"] = df_history["timestamp"].dt.hour
            df_hourly_avg = df_history.groupby("hour")[["aqi", "pm2_5", "temperature"]].mean().reset_index()

            fig_diurnal = px.bar(
                df_hourly_avg,
                x="hour",
                y="aqi",
                color="pm2_5",
                title="Diurnal Air Quality Pattern (Average AQI by Hour of Day)",
                labels={"hour": "Hour of Day (0-23)", "aqi": "Average AQI", "pm2_5": "PM2.5 Level (µg/m³)"}
            )
            fig_diurnal.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickmode="linear", tick0=0, dtick=1)
            )
            st.plotly_chart(fig_diurnal, use_container_width=True)

            # Correlation Heatmap
            st.markdown("### Pollutant and Meteorological Core Correlations")
            st.write("This matrix shows the linear correlation coefficient between weather features and air pollutants.")

            corr_cols = [c for c in ["aqi", "pm2_5", "pm10", "temperature", "humidity", "wind_speed"] if c in df_history.columns]
            corr = df_history[corr_cols].corr()

            fig_corr = px.imshow(
                corr,
                labels=dict(color="Correlation"),
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale="RdBu_r",
                zmin=-1.0, zmax=1.0
            )
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No historical features found in the selected range.")


if __name__ == '__main__':
    main()
