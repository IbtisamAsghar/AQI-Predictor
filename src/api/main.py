import sys
import pickle
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.utils.database import get_db
from src.utils.logging import get_logger
from src.models.model import AQIPredictionModel

logger = get_logger("serving_api")

# Global state variables for caching models in memory
model_wrapper: AQIPredictionModel = None
preprocessor = None
active_features: List[str] = []
champion_metadata: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup model artifact pulling from Hugging Face Hub and cleanups on shutdown."""
    global model_wrapper, preprocessor, active_features, champion_metadata
    logger.info("============================================================")
    logger.info("      INITIALIZING PEARLS AQI FORECASTING SERVING LAYER")
    logger.info("============================================================")
    
    try:
        # 1. Connect to MongoDB and retrieve champion metadata
        logger.info("Connecting to MongoDB Atlas to resolve champion model version...")
        db = get_db()
        collection = db["model_registry"]
        
        champion_doc = collection.find_one({"is_champion": True})
        if not champion_doc:
            logger.error("MLOps Registry Violation: No model version is currently promoted as 'is_champion = True'.")
            raise RuntimeError("No promoted champion model found in model registry database.")
            
        champion_metadata = champion_doc
        version = champion_metadata["version"]
        repo_id = champion_metadata.get("hf_repo_id", settings.HF_REPO_ID)
        
        logger.info(f"Promoted Champion resolved: Version={version}")
        logger.info(f"Target Hugging Face Hub Source: '{repo_id}'")
        
        # 2. Programmatically download weights and scalers from Hugging Face Hub via hf_hub_download
        logger.info(f"Downloading model binary weights from Hugging Face Hub...")
        
        # Try downloading the version-locked files first; fall back to root-level files if needed
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"versions/{version}/model.pkl",
                token=settings.HF_TOKEN
            )
            preprocessor_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"versions/{version}/preprocessor.pkl",
                token=settings.HF_TOKEN
            )
            features_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"versions/{version}/features.pkl",
                token=settings.HF_TOKEN
            )
        except Exception as e:
            logger.warning(f"Could not download versioned files ({version}). Trying root fallback path: {e}")
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.pkl",
                token=settings.HF_TOKEN
            )
            preprocessor_path = hf_hub_download(
                repo_id=repo_id,
                filename="preprocessor.pkl",
                token=settings.HF_TOKEN
            )
            features_path = hf_hub_download(
                repo_id=repo_id,
                filename="features.pkl",
                token=settings.HF_TOKEN
            )
            
        logger.info("Successfully fetched weights from Hugging Face Hub LFS storage.")
        
        # 3. Deserialize model and scaling pipeline into server RAM
        logger.info("Deserializing pickle binaries into RAM...")
        model_wrapper = AQIPredictionModel.load(model_path)
        
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
            
        with open(features_path, "rb") as f:
            active_features = pickle.load(f)
            
        logger.info(f"Serving layer is fully operational. Loaded {len(active_features)} engineered features schema.")
        logger.info("============================================================")
        
    except Exception as e:
        logger.critical(f"FATAL ERROR during serving initialization: {e}")
        raise RuntimeError("FastAPI lifespan startup failed.") from e
        
    yield
    
    # Shutdown actions
    logger.info("Shutting down Pearls AQI Serving Layer.")
    model_wrapper = None
    preprocessor = None
    active_features = []
    champion_metadata = {}

# Instantiate FastAPI application with lifespan context manager
app = FastAPI(
    title="Pearls AQI Prediction Serving API",
    description="REST serving layer for multi-horizon Air Quality Index forecasting in Islamabad.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomPredictionRequest(BaseModel):
    features: Dict[str, float]

def get_aqi_health_category(aqi: float) -> str:
    """Maps numerical AQI values to official US EPA Air Quality Index health bands."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

@app.get("/health", status_code=status.HTTP_200_OK)
def get_health() -> Dict[str, Any]:
    """Retrieves dynamic health status of MongoDB Atlas and local memory layers."""
    db_status = "disconnected"
    try:
        db = get_db()
        db.command("ping")
        db_status = "connected"
    except Exception:
        pass
        
    model_status = "loaded" if model_wrapper is not None and preprocessor is not None else "missing"
    
    overall_status = "green"
    if db_status != "connected" or model_status != "loaded":
        overall_status = "red"
        
    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "model_weights": model_status,
        "champion_version": champion_metadata.get("version", "N/A")
    }

@app.get("/metrics", status_code=status.HTTP_200_OK)
def get_metrics() -> Dict[str, Any]:
    """Serves the active champion's training metrics, features list, and parameters."""
    if not champion_metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Champion model metadata not loaded. Serving layer is offline."
        )
    return {
        "version": champion_metadata.get("version"),
        "timestamp": champion_metadata.get("timestamp"),
        "average_rmse": champion_metadata.get("average_rmse"),
        "metrics": champion_metadata.get("metrics"),
        "features_count": len(active_features),
        "active_features": active_features,
        "parameters": champion_metadata.get("parameters")
    }

@app.get("/predict/live", status_code=status.HTTP_200_OK)
def predict_live() -> Dict[str, Any]:
    """
    Fetches the absolute latest engineered feature document from MongoDB Atlas Feature Store,
    runs scaling and model inference, and yields a comprehensive 3-day AQI forecast.
    """
    if model_wrapper is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model estimators are not loaded in memory."
        )
        
    # 1. Fetch latest feature record from Feature Store
    db = get_db()
    latest_record = db["features_hourly"].find_one(
        {"location": settings.CITY},
        sort=[("timestamp", -1)]
    )
    
    if not latest_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No engineered hourly features found in MongoDB Atlas Feature Store for location: {settings.CITY}."
        )
        
    # 2. Extract and align feature list vectors in exact sequence
    feature_vector = []
    for f in active_features:
        if f not in latest_record:
            logger.error(f"Inference schema misalignment: active schema feature '{f}' not found in latest Feature Store record.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature Store record is missing active schema feature: '{f}'"
            )
        feature_vector.append(latest_record[f])
        
    # 3. Apply standard scaler preprocessing transformations
    X = np.array([feature_vector])
    try:
        X_scaled = preprocessor.transform(X)
    except Exception as e:
        logger.error(f"Preprocessing transform scaling failed during inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data scaling preprocessing error."
        )
        
    # 4. Generate forecasts for all three horizons
    try:
        preds = model_wrapper.predict(X_scaled)
    except Exception as e:
        logger.error(f"Random forest inference model predict failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forecasting model inference execution error."
        )
        
    current_aqi = latest_record.get("aqi")
    
    aqi_24 = max(0.0, float(preds["pred_aqi_24h"][0]))
    aqi_48 = max(0.0, float(preds["pred_aqi_48h"][0]))
    aqi_72 = max(0.0, float(preds["pred_aqi_72h"][0]))
    
    return {
        "location": latest_record.get("location"),
        "timestamp": latest_record.get("timestamp"),
        "current_aqi": current_aqi,
        "current_health_status": get_aqi_health_category(current_aqi) if current_aqi is not None else "Unknown",
        "predictions": {
            "plus_24h": {
                "aqi": round(aqi_24, 2),
                "status": get_aqi_health_category(aqi_24)
            },
            "plus_48h": {
                "aqi": round(aqi_48, 2),
                "status": get_aqi_health_category(aqi_48)
            },
            "plus_72h": {
                "aqi": round(aqi_72, 2),
                "status": get_aqi_health_category(aqi_72)
            }
        },
        "model_version": champion_metadata.get("version", "N/A"),
        "hf_commit_sha": champion_metadata.get("hf_commit_sha", "N/A")
    }

@app.post("/predict/custom", status_code=status.HTTP_200_OK)
def predict_custom(request: CustomPredictionRequest) -> Dict[str, Any]:
    """
    Accepts customized manual engineered inputs matching active features schemas,
    runs scaling and inferences, and yields forecasts.
    """
    if model_wrapper is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model estimators are not loaded in memory."
        )
        
    payload = request.features
    feature_vector = []
    
    for f in active_features:
        if f not in payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Custom feature payload is missing required schema feature input: '{f}'"
            )
        feature_vector.append(payload[f])
        
    X = np.array([feature_vector])
    try:
        X_scaled = preprocessor.transform(X)
        preds = model_wrapper.predict(X_scaled)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Custom prediction inference failed: {e}"
        )
        
    aqi_24 = max(0.0, float(preds["pred_aqi_24h"][0]))
    aqi_48 = max(0.0, float(preds["pred_aqi_48h"][0]))
    aqi_72 = max(0.0, float(preds["pred_aqi_72h"][0]))
    
    return {
        "predictions": {
            "plus_24h": {
                "aqi": round(aqi_24, 2),
                "status": get_aqi_health_category(aqi_24)
            },
            "plus_48h": {
                "aqi": round(aqi_48, 2),
                "status": get_aqi_health_category(aqi_48)
            },
            "plus_72h": {
                "aqi": round(aqi_72, 2),
                "status": get_aqi_health_category(aqi_72)
            }
        },
        "model_version": champion_metadata.get("version", "N/A")
    }

if __name__ == "__main__":
    import uvicorn
    # Local runtime serving
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
