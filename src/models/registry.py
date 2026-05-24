import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from huggingface_hub import HfApi

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import settings
from src.utils.database import get_db
from src.utils.logging import get_logger

logger = get_logger("model_registry")

# Collection names
COLLECTION_NAME = "model_registry"

def get_hf_api() -> HfApi:
    """Instantiates the Hugging Face Hub client wrapper."""
    if not settings.HF_TOKEN:
        raise ValueError("Hugging Face API write token (HF_TOKEN) is not configured in settings.")
    return HfApi(token=settings.HF_TOKEN)

def register_model(
    version: str,
    metrics: Dict[str, Any],
    parameters: Dict[str, Any],
    features: List[str]
) -> Dict[str, Any]:
    """
    Registers model metadata in MongoDB and uploads model weight binaries to Hugging Face Hub.
    Executes automated Champion Promotion logic by comparing candidate validation RMSE with the
    current champion.
    
    Args:
        version: Unique version string identifier (e.g. 'v_20260524_153952').
        metrics: Dictionary of evaluated metrics (MAE, RMSE, R2) for each horizon.
        parameters: Training parameters and hyperparameters.
        features: Active engineered features list.
        
    Returns:
        The dictionary document stored in the metadata registry database.
    """
    # 1. Connect to Hugging Face Hub and ensure model repo exists
    api = get_hf_api()
    repo_id = settings.HF_REPO_ID
    if not repo_id:
        raise ValueError("Hugging Face Repository ID (HF_REPO_ID) is not configured in settings.")
        
    logger.info(f"Connecting to Hugging Face Model Weights Registry: '{repo_id}'...")
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        logger.warning(f"Did not explicitly create repository (may already exist or belong to user): {e}")
        
    # 2. Upload model files (Dual uploads: root for serving inference, version folders for history)
    files_to_upload = {
        "model.pkl": settings.MODEL_DIR / "model.pkl",
        "preprocessor.pkl": settings.MODEL_DIR / "preprocessor.pkl",
        "features.pkl": settings.MODEL_DIR / "features.pkl"
    }
    
    commit_sha = None
    
    for filename, local_path in files_to_upload.items():
        if not local_path.exists():
            raise FileNotFoundError(f"Required model artifact not found locally at: {local_path}")
            
        # Upload versioned tracking path
        version_repo_path = f"versions/{version}/{filename}"
        logger.info(f"  Uploading artifact to versioned path: {version_repo_path}...")
        
        # Exponential retry loop for Hugging Face network uploads
        retries = 3
        delay = 2
        for attempt in range(1, retries + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=version_repo_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                break
            except Exception as e:
                logger.warning(f"    Failed to upload versioned {filename} on attempt {attempt}: {e}")
                if attempt == retries:
                    logger.critical(f"    Hugging Face upload failed permanently for versioned {filename}.")
                    raise
                time.sleep(delay)
                delay *= 2
                
        # Upload root serving path
        logger.info(f"  Uploading artifact to root serving path: {filename}...")
        delay = 2
        for attempt in range(1, retries + 1):
            try:
                res = api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model"
                )
                commit_sha = res.oid  # Extract git commit SHA
                break
            except Exception as e:
                logger.warning(f"    Failed to upload serving {filename} on attempt {attempt}: {e}")
                if attempt == retries:
                    logger.critical(f"    Hugging Face upload failed permanently for serving {filename}.")
                    raise
                time.sleep(delay)
                delay *= 2

    logger.info(f"Successfully uploaded model binaries to Hugging Face (Git Commit SHA: {commit_sha[:8]}).")
    
    # 3. Calculate Average RMSE across evaluated horizons
    rmse_values = []
    for h_key, h_metrics in metrics.items():
        if isinstance(h_metrics, dict) and "rmse" in h_metrics:
            val = h_metrics["rmse"]
            if val is not None:
                rmse_values.append(val)
                
    avg_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else float('inf')
    
    # 4. Fetch metadata registry database and run Champion Promotion check
    db = get_db()
    collection = db[COLLECTION_NAME]
    
    is_champion = False
    
    try:
        current_champion = collection.find_one({"is_champion": True})
    except Exception as e:
        logger.warning(f"Could not search model metadata index (may be first run): {e}")
        current_champion = None
        
    if current_champion is None:
        is_champion = True
        logger.info("No active champion found in model registry. Automatically promoting new model to CHAMPION!")
    else:
        curr_avg_rmse = current_champion.get("average_rmse", float('inf'))
        logger.info(f"Reigning Champion: version={current_champion['version']}, Average RMSE={curr_avg_rmse:.4f}")
        logger.info(f"Candidate Model:   version={version}, Average RMSE={avg_rmse:.4f}")
        
        if avg_rmse < curr_avg_rmse:
            is_champion = True
            logger.info(f">>> SUCCESS! Candidate model is superior (RMSE {avg_rmse:.4f} < {curr_avg_rmse:.4f}).")
            logger.info(f"Promoting version '{version}' to CHAMPION!")
        else:
            logger.info(f"Reigning champion remains undefeated. Candidate version '{version}' registered as candidate only.")
            
    # 5. Insert model log to MongoDB Atlas
    model_doc = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hf_repo_id": repo_id,
        "hf_commit_sha": commit_sha,
        "features": features,
        "metrics": metrics,
        "average_rmse": avg_rmse,
        "parameters": parameters,
        "is_champion": is_champion
    }
    
    try:
        if is_champion:
            # Demote all existing champions atomically
            logger.info("Demoting all existing models from champion status...")
            collection.update_many({"is_champion": True}, {"$set": {"is_champion": False}})
            
        # Write metadata record
        collection.insert_one(model_doc)
        # Drop the mongodb internal _id key to keep standard python dictionaries returning
        model_doc.pop("_id", None)
        logger.info(f"Successfully serialized model metadata and committed to MongoDB Atlas version logs.")
        
    except Exception as e:
        logger.critical(f"Failed to record model metadata inside MongoDB Atlas registry: {e}")
        raise RuntimeError("Model registry metadata write error.") from e
        
    return model_doc

if __name__ == "__main__":
    # Sanity display
    print("Pearls Model Registry Setup wrapper initialized.")
