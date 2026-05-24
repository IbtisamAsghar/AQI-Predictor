import sys
from pathlib import Path

# Add root folder to search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.utils.database import get_db
from src.utils.logging import get_logger
from src.models.registry import register_model

logger = get_logger("verify_registry")

def run_registry_audit():
    logger.info("="*70)
    logger.info("          PEARLS AQI PREDICTOR - MODEL REGISTRY INTEGRATION AUDIT")
    logger.info("="*70)
    
    # 1. Check Configuration Settings
    logger.info("  1. Auditing local credentials and Hugging Face targets...")
    if not settings.HF_TOKEN:
        logger.error("     [FAIL] Hugging Face write token (HF_TOKEN) is not set!")
        sys.exit(1)
    if not settings.HF_REPO_ID:
        logger.error("     [FAIL] Hugging Face repo target (HF_REPO_ID) is not set!")
        sys.exit(1)
        
    logger.info(f"     [SUCCESS] Hugging Face Token: Configured.")
    logger.info(f"     [SUCCESS] Hugging Face Repo:  {settings.HF_REPO_ID}")
    
    # 2. Check if local model artifacts exist
    logger.info("  2. Verifying local serialized model artifacts exist...")
    required_files = ["model.pkl", "preprocessor.pkl", "features.pkl"]
    for f in required_files:
        path = settings.MODEL_DIR / f
        if not path.exists():
            logger.error(f"     [FAIL] Required model file '{f}' not found in models/ directory.")
            logger.error("     Please execute 'python src/pipelines/training_pipeline.py' before running this test.")
            sys.exit(1)
        logger.info(f"     [SUCCESS] Located required file: {f} ({path.stat().st_size / 1024:.2f} KB)")
        
    # 3. Trigger integration test upload
    test_version = "v_test_registry_verify"
    test_metrics = {
        "horizon_24h": {"mae": 15.2, "rmse": 19.8, "r2": 0.62},
        "horizon_48h": {"mae": 16.8, "rmse": 21.5, "r2": 0.52},
        "horizon_72h": {"mae": 17.5, "rmse": 22.1, "r2": 0.48}
    }
    test_parameters = {"n_estimators": 100, "random_state": 42}
    test_features = ["pm2_5", "pm10", "temperature_2m", "relative_humidity_2m"]
    
    logger.info(f"  3. Invoking model registry publisher for version '{test_version}'...")
    try:
        doc = register_model(
            version=test_version,
            metrics=test_metrics,
            parameters=test_parameters,
            features=test_features
        )
        
        # Verify returned document
        assert doc["version"] == test_version, "Document version mismatch!"
        assert doc["hf_repo_id"] == settings.HF_REPO_ID, "HF Repo ID mismatch!"
        assert doc["hf_commit_sha"] is not None, "HF Commit SHA was not captured!"
        assert "average_rmse" in doc, "Average RMSE was not computed!"
        
        logger.info("     [SUCCESS] Hugging Face Hub upload completed successfully.")
        logger.info(f"     [SUCCESS] Commit SHA Captured: {doc['hf_commit_sha']}")
        logger.info(f"     [SUCCESS] MongoDB Version Log: Created.")
        logger.info(f"     [SUCCESS] Champion Promotion State: is_champion={doc['is_champion']}")
        
    except Exception as e:
        logger.error(f"     [FAIL] Model Registry publisher execution crashed: {e}")
        sys.exit(1)
        
    # 4. Clean up the test log inside MongoDB Atlas
    logger.info("  4. Cleaning up temporary verification log from MongoDB Atlas...")
    try:
        db = get_db()
        collection = db["model_registry"]
        
        # Delete test record
        result = collection.delete_one({"version": test_version})
        if result.deleted_count > 0:
            logger.info("     [SUCCESS] Self-cleaning complete! Deleted temporary test record.")
        else:
            logger.warning("     [WARNING] Test record was not found for deletion. Clean-up skipped.")
            
    except Exception as e:
        logger.error(f"     [FAIL] Clean-up operation failed: {e}")
        sys.exit(1)
        
    logger.info("="*70)
    logger.info("     >>> INTEGRATION AUDIT SUCCESSFUL - MODEL REGISTRY STACK READY <<<")
    logger.info("="*70)

if __name__ == "__main__":
    run_registry_audit()
